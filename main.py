#!/usr/bin/env python3
"""
PaSST (Patchout Spectrogram Transformer) 音響イベント検出API
本番対応版 - v2との互換性を維持
"""

import os
import io
import json
import ssl
from typing import List, Dict, Optional
import traceback

# SSL証明書の検証を無効化（開発環境のみ）
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

import torch
import numpy as np
import librosa
import soundfile as sf
from hear21passt.base import get_basic_model
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# AudioSetのラベルマッピング（主要な527クラス）
# v2/audioset_labels.json と同じ形式で管理可能
AUDIOSET_LABELS = {
    0: "Speech",
    1: "Male speech, man speaking",
    2: "Female speech, woman speaking",
    3: "Child speech, kid speaking",
    4: "Conversation",
    5: "Narration, monologue",
    6: "Babbling",
    7: "Speech synthesizer",
    8: "Shout",
    9: "Bellow",
    10: "Whoop",
    11: "Yell",
    12: "Children shouting",
    13: "Screaming",
    14: "Whispering",
    15: "Laughter",
    16: "Baby laughter",
    17: "Giggle",
    18: "Snicker",
    19: "Belly laugh",
    20: "Chuckle, chortle",
    21: "Crying, sobbing",
    22: "Baby cry, infant cry",
    23: "Whimper",
    24: "Wail, moan",
    35: "Laughter",
    42: "Crying, sobbing",
    47: "Sneeze",
    50: "Cough",
    51: "Throat clearing",
    52: "Belch, burp",
    137: "Music",
    288: "Silence",
    # 残りのラベルは後で完全版を追加
}

# グローバル変数でモデルを保持
model = None
device = None

# モデル名
MODEL_NAME = "PaSST-S SWA (passt_s_swa_p16_128_ap476)"
MODEL_DESCRIPTION = "Patchout Spectrogram Transformer - AudioSet (mAP: 0.476)"

# FastAPIアプリケーション
app = FastAPI(
    title="PaSST Audio Event Detection API",
    description="Patchout Spectrogram Transformer を使用した音響イベント検出API - v3",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS設定（v2と同じ）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        print(f"   - モデル: {MODEL_NAME}")
        print(f"   - デバイス: {device}")
        print(f"   - クラス数: 527 (AudioSet)")
        print(f"   - サンプリングレート: 32000 Hz")
        print(f"   - 性能: mAP 0.476 (AudioSet)")

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
    print("PaSST Audio Event Detection API v3")
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
        "description": MODEL_DESCRIPTION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze_sound": "/analyze_sound",
            "analyze_timeline": "/analyze_timeline",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント（v2互換）"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model": MODEL_NAME,
        "device": str(device) if device else None,
        "version": "3.0.0"
    }

@app.post("/analyze_sound")
async def analyze_sound(
    file: UploadFile = File(...),
    top_k: int = Query(5, description="返す予測結果の数"),
    threshold: float = Query(0.1, description="最小確率しきい値")
):
    """
    音声ファイルを分析して音響イベントを検出
    ASTのv2と完全互換のエンドポイント
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

        # レスポンス作成（v2完全互換）
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
                "backend": "PaSST",
                "performance": "mAP: 0.476"
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
    segment_duration: float = Query(10.0, description="セグメントの長さ（秒）"),
    overlap: float = Query(0.0, description="オーバーラップ率（0-1）"),
    top_k: int = Query(3, description="各時刻で返すイベント数"),
    threshold: float = Query(0.1, description="最小確率しきい値")
):
    """
    音声を時系列で分析（v2完全互換）
    デフォルト: 10秒セグメント、オーバーラップなし（最適設定）
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
        event_counter = {}  # イベントの出現回数をカウント

        for start in range(0, len(audio_data) - segment_samples + 1, step_samples):
            # セグメントを抽出
            segment = audio_data[start:start + segment_samples]

            # Tensorに変換
            segment_tensor = torch.from_numpy(segment.astype(np.float32)).unsqueeze(0)

            # 予測
            predictions = predict_audio_events(segment_tensor, top_k, threshold)

            # 時刻を計算
            time_seconds = start / sample_rate

            # イベントカウント
            for pred in predictions:
                label = pred["label"]
                if label not in event_counter:
                    event_counter[label] = {"count": 0, "total_score": 0}
                event_counter[label]["count"] += 1
                event_counter[label]["total_score"] += pred["score"]

            timeline.append({
                "time": round(time_seconds, 1),
                "events": predictions
            })

        # 最も頻出するイベントを計算
        most_common_events = []
        for label, stats in sorted(event_counter.items(), key=lambda x: x[1]["count"], reverse=True)[:5]:
            most_common_events.append({
                "label": label,
                "occurrences": stats["count"],
                "average_score": round(stats["total_score"] / stats["count"], 3)
            })

        # サマリー作成
        duration = len(audio_data) / sample_rate

        response = {
            "timeline": timeline,
            "summary": {
                "total_segments": len(timeline),
                "duration_seconds": round(duration, 2),
                "segment_duration": segment_duration,
                "overlap": overlap,
                "most_common_events": most_common_events
            },
            "audio_info": {
                "filename": file.filename,
                "duration_seconds": round(duration, 2),
                "sample_rate": 32000  # PaSSTは32kHz
            },
            "model_info": {
                "name": MODEL_NAME,
                "version": "3.0.0",
                "backend": "PaSST",
                "performance": "mAP: 0.476"
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
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)