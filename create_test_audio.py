#!/usr/bin/env python3
"""
テスト用音声ファイルの作成
"""

import numpy as np
import soundfile as sf

def create_test_audio():
    """簡単なテスト音声を作成"""

    # パラメータ
    sample_rate = 16000  # 16kHz
    duration = 5  # 5秒

    # 時間軸
    t = np.linspace(0, duration, sample_rate * duration)

    # 複数の周波数を混ぜた信号を作成
    # 440Hz (A4音) + 880Hz (A5音) + ホワイトノイズ
    signal = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4音
        0.2 * np.sin(2 * np.pi * 880 * t) +  # A5音
        0.1 * np.random.randn(len(t))        # ホワイトノイズ
    )

    # 音量を正規化
    signal = signal / np.max(np.abs(signal)) * 0.8

    # ファイルに保存
    filename = "test_audio.wav"
    sf.write(filename, signal, sample_rate)

    print(f"✅ テスト音声ファイルを作成しました: {filename}")
    print(f"   - サンプリングレート: {sample_rate} Hz")
    print(f"   - 長さ: {duration} 秒")
    print(f"   - 内容: 440Hz + 880Hz + ノイズ")

    return filename

if __name__ == "__main__":
    create_test_audio()