#!/usr/bin/env python3
"""
PaSST (Patchout Spectrogram Transformer) 音響イベント検出API - Supabase統合版
file_pathsベースの処理でaudio_filesテーブルと連携

=============================================================================
🔊 重要: サンプリングレートの違い (v2 AST → v3 PaSST)
=============================================================================

【v2 (AST)】
- サンプリングレート: 16kHz (16000 Hz)
- モデル: MIT/ast-finetuned-audioset-10-10-0.4593
- ライブラリ: transformers (Hugging Face)

【v3 (PaSST)】
- サンプリングレート: 32kHz (32000 Hz) ⚠️ v2の2倍
- モデル: passt_s_swa_p16_128_ap476
- ライブラリ: hear21passt

【なぜ32kHzなのか？】
PaSSTモデルは学習時に32kHzの音声データで訓練されているため、
推論時も32kHzにリサンプリングする必要があります。

【影響範囲】
- 入力音声が何Hzであっても、内部で自動的に32kHzにリサンプリングされます
- ユーザー側で特別な対応は不要（APIインターフェースは変更なし）
- 処理時間はほぼ同じ（リサンプリングのオーバーヘッドは軽微）

【精度向上】
32kHzにより、より高周波数帯域の音響特徴を捉えることができ、
結果として音響イベント検出精度が向上しています (mAP 0.459 → 0.476)

=============================================================================
"""

import os
import io
import json
import tempfile
import traceback
from typing import List, Dict, Optional
from datetime import datetime, timezone
import time
import ssl

# SSL証明書の検証を無効化（開発環境のみ）
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

import torch
import numpy as np
import librosa
import soundfile as sf
from hear21passt.base import get_basic_model
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# AWS S3とSupabase
import boto3
from botocore.exceptions import ClientError
from supabase import create_client, Client
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

# グローバル変数でモデルを保持
model = None
device = None
labels_map = None  # AudioSetラベルマッピング

# モデル情報
MODEL_NAME = "PaSST-S SWA (passt_s_swa_p16_128_ap476)"
MODEL_DESCRIPTION = "Patchout Spectrogram Transformer - AudioSet (mAP: 0.476)"
SAMPLING_RATE = 32000  # ⚠️ v2の16kHzから32kHzに変更

# Supabaseクライアントの初期化
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URLおよびSUPABASE_KEYが設定されていません")

supabase: Client = create_client(supabase_url, supabase_key)
print(f"✅ Supabase接続設定完了: {supabase_url}")

# AWS S3クライアントの初期化
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
s3_bucket_name = os.getenv('S3_BUCKET_NAME', 'watchme-vault')
aws_region = os.getenv('AWS_REGION', 'ap-southeast-2')

if not aws_access_key_id or not aws_secret_access_key:
    raise ValueError("AWS_ACCESS_KEY_IDおよびAWS_SECRET_ACCESS_KEYが設定されていません")

s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)
print(f"✅ AWS S3接続設定完了: バケット={s3_bucket_name}, リージョン={aws_region}")

# FastAPIアプリケーション
app = FastAPI(
    title="PaSST Audio Event Detection API with Supabase",
    description="Patchout Spectrogram Transformer を使用した音響イベント検出API（Supabase統合版）- v3",
    version="3.0.0"
)

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# リクエストモデル
class FetchAndProcessPathsRequest(BaseModel):
    file_paths: List[str]
    threshold: Optional[float] = 0.1
    top_k: Optional[int] = 3
    analyze_timeline: Optional[bool] = True
    segment_duration: Optional[float] = 10.0  # 10秒が最適
    overlap: Optional[float] = 0.0  # オーバーラップなしが最適

def load_audioset_labels():
    """AudioSetラベルマッピングをJSONから読み込む"""
    global labels_map

    try:
        labels_file = os.path.join(os.path.dirname(__file__), 'audioset_labels.json')
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels_map = json.load(f)
        print(f"✅ AudioSetラベル読み込み完了: {len(labels_map)}クラス")
    except Exception as e:
        print(f"⚠️ AudioSetラベルファイルが見つかりません: {str(e)}")
        labels_map = {}

def load_model():
    """PaSSTモデルを読み込む"""
    global model, device

    print(f"🔄 モデルをロード中: {MODEL_NAME}")
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
        print(f"   - サンプリングレート: {SAMPLING_RATE} Hz (32kHz)")
        print(f"   - 性能: mAP 0.476 (AudioSet)")

        # ラベルマッピングを読み込む
        load_audioset_labels()

    except Exception as e:
        print(f"❌ モデルのロードに失敗しました: {str(e)}")
        traceback.print_exc()
        raise

def extract_info_from_file_path(file_path: str) -> Dict[str, str]:
    """
    ファイルパスからデバイスID、日付、時間ブロックを抽出

    Args:
        file_path: S3ファイルパス (例: files/device-id/2025-07-20/14-30/audio.wav)

    Returns:
        device_id, date, time_block を含む辞書
    """
    parts = file_path.split('/')

    if len(parts) >= 2:
        device_id = parts[1]
        return {
            'device_id': device_id
        }
    else:
        return {
            'device_id': 'unknown'
        }

async def update_audio_files_status(file_path: str, status: str = 'completed'):
    """
    audio_filesテーブルのbehavior_features_statusを更新

    Args:
        file_path: ファイルパス
        status: ステータス ('pending', 'processing', 'completed', 'error')
    """
    try:
        update_response = supabase.table('audio_files') \
            .update({'behavior_features_status': status}) \
            .eq('file_path', file_path) \
            .execute()

        if update_response.data:
            print(f"✅ ステータス更新成功: {file_path} -> {status}")
            return True
        else:
            print(f"⚠️ 対象レコードが見つかりません: {file_path}")
            return False

    except Exception as e:
        print(f"❌ ステータス更新エラー: {str(e)}")
        return False

async def save_to_spot_features(device_id: str, recorded_at: str,
                                 timeline_data: List[Dict]):
    """
    spot_featuresテーブルにタイムライン形式の結果を保存

    Args:
        device_id: デバイスID
        recorded_at: 録音日時 (UTC timestamp)
        timeline_data: タイムライン形式のイベントデータ
    """
    try:
        processed_at = datetime.now(timezone.utc).isoformat()

        data = {
            'device_id': device_id,
            'recorded_at': recorded_at,
            'behavior_extractor_result': timeline_data,  # JSONB形式
            'behavior_extractor_status': 'completed',
            'behavior_extractor_processed_at': processed_at
        }

        response = supabase.table('spot_features') \
            .upsert(data) \
            .execute()

        if response.data:
            print(f"✅ spot_features保存成功: {device_id}/{recorded_at}")
            return True
        else:
            print(f"⚠️ データ保存失敗: レスポンスが空です")
            return False

    except Exception as e:
        print(f"❌ データ保存エラー: {str(e)}")
        traceback.print_exc()
        return False

def download_from_s3(file_path: str, local_path: str) -> bool:
    """S3から音声ファイルをダウンロード"""
    try:
        print(f"📥 S3からダウンロード中: {file_path}")
        s3_client.download_file(s3_bucket_name, file_path, local_path)
        print(f"✅ ダウンロード完了: {file_path}")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"❌ ファイルが見つかりません: {file_path}")
        else:
            print(f"❌ S3ダウンロードエラー: {error_code} - {str(e)}")
        return False
    except Exception as e:
        print(f"❌ 予期しないエラー: {str(e)}")
        return False

def process_audio_for_passt(audio_data: np.ndarray, sample_rate: int) -> torch.Tensor:
    """
    音声データをPaSST用に前処理

    ⚠️ 重要: PaSSTは32kHzの音声を期待します（v2のASTは16kHz）

    Args:
        audio_data: 音声データ
        sample_rate: 元のサンプリングレート

    Returns:
        PaSST用に処理されたTensor
    """
    # モノラルに変換
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # PaSSTが期待する32kHzにリサンプリング
    if sample_rate != SAMPLING_RATE:
        audio_data = librosa.resample(
            y=audio_data,
            orig_sr=sample_rate,
            target_sr=SAMPLING_RATE
        )
        print(f"🔄 リサンプリング: {sample_rate}Hz → {SAMPLING_RATE}Hz")

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

def predict_audio_events(audio_tensor: torch.Tensor, top_k: int = 5,
                        threshold: float = 0.1) -> List[Dict]:
    """
    音声データから音響イベントを予測

    Args:
        audio_tensor: PaSST用に処理済みのTensor
        top_k: 返す上位予測の数
        threshold: 最小確率しきい値

    Returns:
        予測結果のリスト
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
        if prob >= threshold:
            class_id = str(int(idx))
            # ラベルマッピングから取得（なければClass_XXX形式）
            label = labels_map.get(class_id, f"Class_{class_id}")
            predictions.append({
                "label": label,
                "score": round(float(prob), 4)
            })

    return predictions

def analyze_timeline(audio_data: np.ndarray, sample_rate: int,
                    segment_duration: float = 10.0,
                    overlap: float = 0.0,
                    top_k: int = 3,
                    threshold: float = 0.1) -> Dict:
    """
    音声データを時系列で分析

    Args:
        audio_data: 音声データ
        sample_rate: サンプリングレート
        segment_duration: セグメントの長さ（秒）- デフォルト10秒が最適
        overlap: オーバーラップ率 (0-1) - デフォルト0が最適
        top_k: 各時刻で返すイベント数
        threshold: 最小確率しきい値

    Returns:
        時系列分析結果
    """
    # モノラルに変換
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # 32kHzにリサンプリング
    if sample_rate != SAMPLING_RATE:
        audio_data = librosa.resample(
            y=audio_data,
            orig_sr=sample_rate,
            target_sr=SAMPLING_RATE
        )
        sample_rate = SAMPLING_RATE

    # セグメント設定
    segment_samples = int(segment_duration * sample_rate)
    hop_samples = int(segment_samples * (1 - overlap))

    # タイムライン結果を格納
    timeline = []
    all_events = {}

    # 音声が短い場合は全体を1セグメントとして処理
    if len(audio_data) < segment_samples:
        audio_tensor = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0)
        events = predict_audio_events(audio_tensor, top_k, threshold)
        timeline.append({
            "time": 0.0,
            "events": events
        })
        for event in events:
            label = event["label"]
            if label not in all_events:
                all_events[label] = {"count": 0, "total_score": 0}
            all_events[label]["count"] += 1
            all_events[label]["total_score"] += event["score"]
    else:
        # 通常のセグメント処理
        for i in range(0, len(audio_data) - segment_samples + 1, hop_samples):
            segment = audio_data[i:i + segment_samples]
            time_position = i / sample_rate

            # Tensorに変換
            segment_tensor = torch.from_numpy(segment.astype(np.float32)).unsqueeze(0)

            # 予測
            events = predict_audio_events(segment_tensor, top_k, threshold)

            # タイムラインに追加
            timeline.append({
                "time": round(time_position, 1),
                "events": events
            })

            # イベントの集計
            for event in events:
                label = event["label"]
                if label not in all_events:
                    all_events[label] = {"count": 0, "total_score": 0}
                all_events[label]["count"] += 1
                all_events[label]["total_score"] += event["score"]

    # 最も頻繁なイベントを集計
    most_common = []
    for label, stats in sorted(all_events.items(), key=lambda x: x[1]["count"], reverse=True)[:5]:
        most_common.append({
            "label": label,
            "occurrences": stats["count"],
            "average_score": round(stats["total_score"] / stats["count"], 4)
        })

    return {
        "timeline": timeline,
        "summary": {
            "total_segments": len(timeline),
            "duration_seconds": round(len(audio_data) / sample_rate, 1),
            "segment_duration": segment_duration,
            "overlap": overlap,
            "most_common_events": most_common
        }
    }

async def process_single_file(file_path: str, threshold: float = 0.1, top_k: int = 3,
                             analyze_timeline_flag: bool = True,
                             segment_duration: float = 10.0,
                             overlap: float = 0.0) -> Dict:
    """
    単一ファイルを処理（タイムライン形式で保存）
    """
    temp_file = None
    try:
        # audio_filesテーブルからrecorded_atを取得
        audio_file_response = supabase.table('audio_files') \
            .select('device_id, recorded_at') \
            .eq('file_path', file_path) \
            .single() \
            .execute()

        if not audio_file_response.data:
            return {"status": "error", "file_path": file_path, "error": "Audio file record not found"}

        device_id = audio_file_response.data['device_id']
        recorded_at = audio_file_response.data['recorded_at']

        # ステータスを処理中に更新
        await update_audio_files_status(file_path, 'processing')

        # 一時ファイルを作成してダウンロード
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_file = tmp.name

        if not download_from_s3(file_path, temp_file):
            await update_audio_files_status(file_path, 'error')
            return {"status": "error", "file_path": file_path, "error": "Download failed"}

        # 音声データを読み込む
        audio_data, sample_rate = sf.read(temp_file)
        print(f"🎵 音声ロード完了: {len(audio_data)/sample_rate:.2f}秒, {sample_rate}Hz")

        # タイムライン分析を実行
        timeline_result = analyze_timeline(
            audio_data, sample_rate,
            segment_duration, overlap, top_k, threshold
        )

        # spot_featuresテーブルに保存
        save_success = await save_to_spot_features(
            device_id,
            recorded_at,
            timeline_result['timeline']
        )

        if save_success:
            await update_audio_files_status(file_path, 'completed')
            return {
                "status": "success",
                "file_path": file_path,
                "device_id": device_id,
                "recorded_at": recorded_at,
                "timeline": timeline_result
            }
        else:
            await update_audio_files_status(file_path, 'error')
            return {"status": "error", "file_path": file_path, "error": "Save failed"}

    except Exception as e:
        print(f"❌ ファイル処理エラー: {file_path} - {str(e)}")
        traceback.print_exc()
        await update_audio_files_status(file_path, 'error')
        return {"status": "error", "file_path": file_path, "error": str(e)}

    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

@app.on_event("startup")
async def startup_event():
    """サーバー起動時にモデルをロード"""
    load_model()

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "PaSST Audio Event Detection API with Supabase Integration",
        "model": MODEL_NAME,
        "version": "3.0.0",
        "sampling_rate": f"{SAMPLING_RATE} Hz (32kHz)",
        "status": "ready" if model is not None else "not ready",
        "endpoints": {
            "/fetch-and-process-paths": "Process audio files from S3 via file paths",
            "/health": "Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "sampling_rate": SAMPLING_RATE,
        "supabase_connected": supabase is not None,
        "s3_connected": s3_client is not None
    }

@app.post("/fetch-and-process-paths")
async def fetch_and_process_paths(request: FetchAndProcessPathsRequest):
    """
    file_pathsベースの音響イベント検出エンドポイント（v2完全互換）

    Args:
        request: file_paths配列とオプションパラメータ

    Returns:
        処理結果のサマリーと詳細
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    processed_files = []
    error_files = []

    print(f"🚀 処理開始: {len(request.file_paths)}個のファイル")

    for file_path in request.file_paths:
        result = await process_single_file(
            file_path,
            request.threshold,
            request.top_k,
            request.analyze_timeline,
            request.segment_duration,
            request.overlap
        )

        if result["status"] == "success":
            processed_files.append(file_path)
        else:
            error_files.append({
                "file_path": file_path,
                "error": result.get("error", "Unknown error")
            })

    execution_time = time.time() - start_time

    total_files = len(request.file_paths)
    success_count = len(processed_files)
    error_count = len(error_files)

    response = {
        "status": "success" if error_count == 0 else "partial",
        "summary": {
            "total_files": total_files,
            "processed": success_count,
            "errors": error_count
        },
        "processed_files": processed_files,
        "error_files": error_files if error_files else None,
        "execution_time_seconds": round(execution_time, 1),
        "message": f"{total_files}件中{success_count}件を正常に処理しました"
    }

    print(f"✅ 処理完了: {success_count}/{total_files}件成功 (実行時間: {execution_time:.1f}秒)")

    return JSONResponse(content=response)

if __name__ == "__main__":
    print("=" * 50)
    print("PaSST Audio Event Detection API with Supabase")
    print(f"Model: {MODEL_NAME}")
    print(f"Sampling Rate: {SAMPLING_RATE} Hz (32kHz)")
    print("=" * 50)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8017,
        log_level="info"
    )
