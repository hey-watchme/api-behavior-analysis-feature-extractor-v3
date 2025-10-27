#!/usr/bin/env python3
"""
音声ファイルの情報を確認
"""

import sys
import soundfile as sf

def check_audio_file(filepath):
    """音声ファイルの情報を表示"""
    try:
        data, samplerate = sf.read(filepath)
        duration = len(data) / samplerate

        print(f"ファイル: {filepath}")
        print(f"サンプリングレート: {samplerate} Hz")
        print(f"長さ: {duration:.2f} 秒")
        print(f"チャンネル数: {1 if len(data.shape) == 1 else data.shape[1]}")
        print(f"サンプル数: {len(data)}")
        print(f"データ形状: {data.shape}")

        return data, samplerate
    except Exception as e:
        print(f"エラー: {e}")
        return None, None

if __name__ == "__main__":
    filepath = "/Users/kaya.matsumoto/Desktop/children_3people_001.wav"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]

    check_audio_file(filepath)