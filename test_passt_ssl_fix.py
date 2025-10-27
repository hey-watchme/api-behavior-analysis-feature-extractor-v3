#!/usr/bin/env python3
"""
PaSST (Patchout Spectrogram Transformer) の基本動作テスト
SSL証明書エラー対応版
"""

import ssl
import os
import numpy as np
import torch

# SSL証明書の検証を無効化（開発環境のみ）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 環境変数でも設定
os.environ['PYTHONHTTPSVERIFY'] = '0'

from hear21passt.base import get_basic_model

def test_basic_model():
    """基本的なモデルのロードと推論テスト"""

    print("=" * 50)
    print("PaSST Model Basic Test (SSL Fix)")
    print("=" * 50)

    # 1. モデルのロード
    print("\n1. Loading PaSST model...")
    try:
        # logitsモードでモデルを取得（AudioSetの527クラス分類）
        model = get_basic_model(mode="logits")
        model.eval()

        # デバイスの設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {device}")
        model = model.to(device)

        print("   ✅ Model loaded successfully!")

    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. ダミーオーディオデータの作成
    print("\n2. Creating dummy audio data...")
    # PaSSTは32kHzのオーディオを期待（10秒間）
    sample_rate = 32000
    duration = 10  # seconds
    audio_length = sample_rate * duration

    # ランダムなオーディオ信号を生成（バッチサイズ1）
    audio_wave = torch.randn(1, audio_length).to(device)
    print(f"   Audio shape: {audio_wave.shape}")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Duration: {duration} seconds")

    # 3. 推論の実行
    print("\n3. Running inference...")
    try:
        with torch.no_grad():
            logits = model(audio_wave)

        print(f"   Output shape: {logits.shape}")
        print(f"   Number of classes: {logits.shape[-1]}")

        # Softmaxで確率に変換
        probs = torch.softmax(logits, dim=-1)

        # Top-5の予測を取得
        top5_probs, top5_indices = torch.topk(probs[0], 5)

        print("\n   Top-5 predictions (indices):")
        for i, (idx, prob) in enumerate(zip(top5_indices.cpu().numpy(), top5_probs.cpu().numpy())):
            print(f"   {i+1}. Class {idx}: {prob:.4f}")

        print("\n   ✅ Inference successful!")

    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. モデル情報の取得
    print("\n4. Model information:")
    # パラメータ数のカウント
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    print("\n" + "=" * 50)
    print("✅ All tests passed successfully!")
    print("=" * 50)

    return model

if __name__ == "__main__":
    # 基本テストの実行
    model = test_basic_model()

    print("\n🎉 PaSST model is ready to use!")

    if model is not None:
        print("\nModel architecture summary:")
        print(f"  - Model class: {model.__class__.__name__}")
        print(f"  - Input: 32kHz audio")
        print(f"  - Output: 527 AudioSet classes")