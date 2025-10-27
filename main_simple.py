#!/usr/bin/env python3
"""
PaSST (Patchout Spectrogram Transformer) 音響イベント検出API
シンプル版 - 基本機能のみ実装
"""

import os
import io
import json
from typing import List, Dict, Optional
import traceback

import torch
import numpy as np
import librosa
import soundfile as sf
from hear21passt.base import get_basic_model
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# AudioSetのラベルマッピング（主要なもののみ抜粋）
# 完全版は後で追加
AUDIOSET_LABELS = {
    0: "Speech",
    137: "Music",
    35: "Laughter",
    42: "Crying, sobbing",
    47: "Sneeze",
    50: "Cough",
    288: "Silence",
    # 他のラベルは後で追加
}

# グローバル変数でモデルを保持
model = None
device = None

# モデル名
MODEL_NAME = "PaSST (passt_s_swa_p16_128_ap476)"

# FastAPIアプリケーション
app = FastAPI(
    title="PaSST Audio Event Detection API",
    description="Patchout Spectrogram Transformer を使用した音響イベント検出API",
    version="3.0.0"
)

def load_model():
    """PaSSTモデルを読み込む"""
    global model, device

    print(f"モデルをロード中: {MODEL_NAME}")
    try:
        # PaSSTモデルの読み込み（logitsモード = 527クラス分類）
        model = get_basic_model(mode="logits")

        # デバイスの設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        print(f"✅ モデルのロードに成功しました")
        print(f"   - デバイス: {device}")
        print(f"   - クラス数: 527 (AudioSet)")
        print(f"   - サンプリングレート: 32000 Hz")

    except Exception as e:
        print(f"❌ モデルのロードに失敗しました: {str(e)}")
        traceback.print_exc()
        raise

def process_audio_for_passt(audio_data: np.ndarray, sample_rate: int) -> torch.Tensor:
    """
    音声データをPaSST用に前処理する
    PaSSTは32kHzの音声を期待
    """
    # モノラルに変換
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # PaSSTが期待するサンプリングレート（32kHz）にリサンプリング
    target_sr = 32000
    if sample_rate != target_sr:
        audio_data = librosa.resample(
            y=audio_data,
            orig_sr=sample_rate,
            target_sr=target_sr
        )

    # float32に変換
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # 正規化（-1.0 〜 1.0）
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val

    # Tensorに変換（バッチ次元を追加）
    audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)

    return audio_tensor

def predict_audio_events(audio_tensor: torch.Tensor, top_k: int = 5, threshold: float = 0.1) -> List[Dict[str, any]]:
    """
    音声からイベントを予測
    """
    global model, device

    # デバイスに転送
    audio_tensor = audio_tensor.to(device)

    # 推論実行
    with torch.no_grad():
        logits = model(audio_tensor)

    # Softmaxで確率に変換
    probs = torch.softmax(logits, dim=-1)

    # Top-kの予測を取得
    top_probs, top_indices = torch.topk(probs[0], min(top_k, 527))

    # 結果をリスト化
    predictions = []
    for idx, prob in zip(top_indices.cpu().numpy(), top_probs.cpu().numpy()):
        if prob >= threshold:  # 閾値以上のみ
            # ラベルを取得（マッピングがない場合はClass_XXX形式）
            label = AUDIOSET_LABELS.get(int(idx), f"Class_{idx}")
            predictions.append({
                "label": label,
                "score": float(prob),
                "class_id": int(idx)
            })

    return predictions

@app.on_event("startup")
async def startup_event():
    """サーバー起動時にモデルを読み込む"""
    print("=" * 50)
    print("PaSST Audio Event Detection API")
    print(f"Model: {MODEL_NAME}")
    print("=" * 50)
    load_model()

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "name": "PaSST Audio Event Detection API",
        "version": "3.0.0",
        "model": MODEL_NAME,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }

@app.post("/analyze_sound")
async def analyze_sound(
    file: UploadFile = File(...),
    top_k: int = 5,
    threshold: float = 0.1
):
    """
    音声ファイルを分析して音響イベントを検出
    ASTのv2と互換性のあるエンドポイント
    """
    try:
        # ファイルを読み込み
        audio_bytes = await file.read()

        # 音声データを読み込み
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"音声ファイルの読み込みに失敗: {str(e)}")

        # 音声の情報
        duration = len(audio_data) / sample_rate

        # PaSST用に前処理
        audio_tensor = process_audio_for_passt(audio_data, sample_rate)

        # 予測実行
        predictions = predict_audio_events(audio_tensor, top_k, threshold)

        # レスポンス作成（v2互換）
        response = {
            "predictions": predictions,
            "audio_info": {
                "filename": file.filename,
                "duration_seconds": round(duration, 2),
                "sample_rate": sample_rate,
                "original_sample_rate": sample_rate,
                "resampled_to": 32000
            },
            "model_info": {
                "name": MODEL_NAME,
                "version": "3.0.0",
                "backend": "PaSST"
            }
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        print(f"エラー: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"分析中にエラーが発生: {str(e)}")

@app.post("/analyze_timeline")
async def analyze_timeline(
    file: UploadFile = File(...),
    segment_duration: float = 10.0,
    overlap: float = 0.0,
    top_k: int = 3,
    threshold: float = 0.1
):
    """
    音声を時系列で分析（v2互換）
    """
    try:
        # ファイルを読み込み
        audio_bytes = await file.read()

        # 音声データを読み込み
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

        # モノラルに変換
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # PaSSTの32kHzにリサンプリング
        target_sr = 32000
        if sample_rate != target_sr:
            audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr

        # セグメント化のパラメータ計算
        segment_samples = int(segment_duration * sample_rate)
        step_samples = int(segment_samples * (1 - overlap))

        # タイムライン分析
        timeline = []
        for start in range(0, len(audio_data) - segment_samples + 1, step_samples):
            # セグメントを抽出
            segment = audio_data[start:start + segment_samples]

            # Tensorに変換
            segment_tensor = torch.from_numpy(segment.astype(np.float32)).unsqueeze(0)

            # 予測
            predictions = predict_audio_events(segment_tensor, top_k, threshold)

            # 時刻を計算
            time_seconds = start / sample_rate

            timeline.append({
                "time": round(time_seconds, 1),
                "events": predictions
            })

        # サマリー作成
        duration = len(audio_data) / sample_rate

        response = {
            "timeline": timeline,
            "summary": {
                "total_segments": len(timeline),
                "duration_seconds": round(duration, 2),
                "segment_duration": segment_duration,
                "overlap": overlap
            },
            "audio_info": {
                "filename": file.filename,
                "duration_seconds": round(duration, 2),
                "sample_rate": 32000  # PaSSTは32kHz
            },
            "model_info": {
                "name": MODEL_NAME,
                "version": "3.0.0",
                "backend": "PaSST"
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        print(f"エラー: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"タイムライン分析中にエラー: {str(e)}")

if __name__ == "__main__":
    # ポート8017で起動（v2と同じ）
    port = int(os.environ.get("PORT", 8017))
    uvicorn.run(app, host="0.0.0.0", port=port)