#!/usr/bin/env python3
"""
AST (Audio Spectrogram Transformer) Sound Event Detection API - Supabase Integration
file_paths-based processing with audio_files table integration

Model: MIT/ast-finetuned-audioset-10-10-0.4593
Sampling Rate: 16kHz
Library: transformers (Hugging Face)
"""

import os
import io
import json
import tempfile
import traceback
from typing import List, Dict, Optional
from datetime import datetime, timezone
import time

import torch
import numpy as np
import librosa
import soundfile as sf
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# AWS S3 and Supabase
import boto3
from botocore.exceptions import ClientError
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables for model
model = None
feature_extractor = None
id2label = None

# Model information
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODEL_DESCRIPTION = "Audio Spectrogram Transformer - AudioSet (mAP: 0.459)"
SAMPLING_RATE = 16000

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

# FastAPI application
app = FastAPI(
    title="AST Audio Event Detection API with Supabase",
    description="Audio Spectrogram Transformer for sound event detection (Supabase integration) - v3",
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

def load_model():
    """Load AST model and feature extractor"""
    global model, feature_extractor, id2label

    print(f"🔄 Loading model: {MODEL_NAME}")
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        model = ASTForAudioClassification.from_pretrained(MODEL_NAME)

        # Get label mapping
        id2label = model.config.id2label

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        print(f"✅ Model loaded successfully")
        print(f"   - Model: {MODEL_NAME}")
        print(f"   - Device: {device}")
        print(f"   - Classes: {len(id2label)} (AudioSet)")
        print(f"   - Sampling Rate: {SAMPLING_RATE} Hz (16kHz)")
        print(f"   - Performance: mAP 0.459 (AudioSet)")

    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
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

        # Get local_date and local_time from audio_files table
        local_date = None
        local_time = None
        try:
            audio_file_response = supabase.table('audio_files').select('local_date, local_time').eq(
                'device_id', device_id
            ).eq(
                'recorded_at', recorded_at
            ).execute()

            if audio_file_response.data and len(audio_file_response.data) > 0:
                local_date = audio_file_response.data[0].get('local_date')
                local_time = audio_file_response.data[0].get('local_time')
                print(f"Retrieved local_date from audio_files: {local_date}")
                print(f"Retrieved local_time from audio_files: {local_time}")
            else:
                print(f"⚠️ No audio_files record found for device_id={device_id}, recorded_at={recorded_at}")
        except Exception as e:
            print(f"❌ Error fetching local_date/local_time from audio_files: {e}")

        data = {
            'device_id': device_id,
            'recorded_at': recorded_at,
            'local_date': local_date,  # Local date from audio_files
            'local_time': local_time,  # Local time from audio_files
            'behavior_extractor_result': timeline_data  # JSONB形式
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

def process_audio(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Preprocess audio data for AST model

    Args:
        audio_data: Audio data (numpy array)
        sample_rate: Original sampling rate

    Returns:
        Processed audio data
    """
    # Convert to mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample to model's expected sampling rate (16kHz)
    target_sr = feature_extractor.sampling_rate
    if sample_rate != target_sr:
        audio_data = librosa.resample(
            audio_data,
            orig_sr=sample_rate,
            target_sr=target_sr
        )

    # Convert to float32
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Normalize (-1.0 to 1.0)
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val

    return audio_data

def predict_audio_events(audio_data: np.ndarray, top_k: int = 5,
                        threshold: float = 0.1) -> List[Dict]:
    """
    Predict audio events from audio data

    Args:
        audio_data: Preprocessed audio data
        top_k: Number of top predictions to return
        threshold: Minimum probability threshold

    Returns:
        List of predicted events
    """
    # Extract features
    inputs = feature_extractor(
        audio_data,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt"
    )

    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

    # Format results
    predictions = []
    for prob, idx in zip(top_probs.cpu(), top_indices.cpu()):
        score = prob.item()
        if score >= threshold:
            label_id = idx.item()
            label = id2label.get(label_id) or id2label.get(str(label_id)) or f"Event_{label_id}"
            predictions.append({
                "label": label,
                "score": round(score, 4)
            })

    return predictions

def analyze_timeline(audio_data: np.ndarray, sample_rate: int,
                    segment_duration: float = 10.0,
                    overlap: float = 0.0,
                    top_k: int = 3,
                    threshold: float = 0.1) -> Dict:
    """
    Analyze audio data in timeline segments

    Args:
        audio_data: Audio data
        sample_rate: Sampling rate
        segment_duration: Segment length in seconds (default 10s)
        overlap: Overlap ratio (0-1, default 0)
        top_k: Number of events to return per segment
        threshold: Minimum probability threshold

    Returns:
        Timeline analysis results
    """
    # Preprocess audio
    processed_audio = process_audio(audio_data, sample_rate)
    target_sr = feature_extractor.sampling_rate

    # Segment configuration
    segment_samples = int(segment_duration * target_sr)
    hop_samples = int(segment_samples * (1 - overlap))

    # Store timeline results
    timeline = []
    all_events = {}

    # Handle short audio (less than segment_duration)
    if len(processed_audio) < segment_samples:
        events = predict_audio_events(processed_audio, top_k, threshold)
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
        # Normal segment processing
        for i in range(0, len(processed_audio) - segment_samples + 1, hop_samples):
            segment = processed_audio[i:i + segment_samples]
            time_position = i / target_sr

            # Predict events for segment
            events = predict_audio_events(segment, top_k, threshold)

            # Add to timeline
            timeline.append({
                "time": round(time_position, 1),
                "events": events
            })

            # Aggregate events
            for event in events:
                label = event["label"]
                if label not in all_events:
                    all_events[label] = {"count": 0, "total_score": 0}
                all_events[label]["count"] += 1
                all_events[label]["total_score"] += event["score"]

    # Get most common events
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
            "duration_seconds": round(len(processed_audio) / target_sr, 1),
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
    """Root endpoint"""
    return {
        "message": "AST Audio Event Detection API with Supabase Integration",
        "model": MODEL_NAME,
        "version": "3.0.0",
        "sampling_rate": f"{SAMPLING_RATE} Hz (16kHz)",
        "status": "ready" if model is not None else "not ready",
        "endpoints": {
            "/fetch-and-process-paths": "Process audio files from S3 via file paths",
            "/health": "Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
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
    print("AST Audio Event Detection API with Supabase")
    print(f"Model: {MODEL_NAME}")
    print(f"Sampling Rate: {SAMPLING_RATE} Hz (16kHz)")
    print("=" * 50)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8017,
        log_level="info"
    )
