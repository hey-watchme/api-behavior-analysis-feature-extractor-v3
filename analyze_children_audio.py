#!/usr/bin/env python3
"""
子供3人の音声ファイルの詳細分析
AudioSetのすべてのイベントを検出
"""

import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

import torch
import numpy as np
import librosa
import soundfile as sf
from hear21passt.base import get_basic_model
import json

# AudioSetラベル（主要なものを定義）
AUDIOSET_LABELS = {
    0: "Speech (会話)",
    1: "Male speech, man speaking (男性の話し声)",
    2: "Female speech, woman speaking (女性の話し声)",
    3: "Child speech, kid speaking (子供の話し声)",
    4: "Conversation (会話)",
    5: "Narration, monologue",
    6: "Babbling (喃語)",
    7: "Speech synthesizer",
    8: "Shout (叫び声)",
    9: "Bellow",
    10: "Whoop",
    11: "Yell (大声)",
    12: "Children shouting (子供の叫び声)",
    13: "Screaming (悲鳴)",
    14: "Whispering (ささやき声)",
    15: "Laughter (笑い声)",
    16: "Baby laughter (赤ちゃんの笑い声)",
    17: "Giggle (くすくす笑い)",
    18: "Snicker",
    19: "Belly laugh (大笑い)",
    20: "Chuckle, chortle",
    21: "Crying, sobbing (泣き声)",
    22: "Baby cry, infant cry (赤ちゃんの泣き声)",
    23: "Whimper (すすり泣き)",
    24: "Wail, moan",
    34: "Child singing (子供の歌声)",
    35: "Synthetic singing",
    47: "Sneeze (くしゃみ)",
    50: "Sniff (鼻をすする音)",
    51: "Throat clearing (咳払い)",
    52: "Belch, burp",
    54: "Chewing, mastication (咀嚼音)",
    71: "Children playing (子供の遊び声)",
    137: "Music (音楽)",
    288: "Silence (静寂)",
    506: "Inside, small room (室内・小部屋)",
    # 環境音
    74: "Mechanisms",
    75: "Door (ドア)",
    76: "Doorbell",
    77: "Knock (ノック)",
    78: "Tap (タップ音)",
    79: "Squeak",
    80: "Cupboard open or close",
    81: "Drawer open or close",
    82: "Dishes, pots, and pans",
    83: "Cutlery, silverware",
    84: "Chopping (food)",
    85: "Frying (food)",
    86: "Microwave oven",
    87: "Blender",
    88: "Water tap, faucet",
    89: "Sink (filling or washing)",
    90: "Bathtub (filling or washing)",
    91: "Hair dryer",
    92: "Toilet flush",
    93: "Toothbrush",
    94: "Electric toothbrush",
    95: "Vacuum cleaner",
    96: "Mechanical fan",
    97: "Air conditioning",
    98: "Clothes dryer",
    99: "Dishwasher",
    100: "Sewing machine",
    101: "Cash register",
    102: "Computer keyboard",
    103: "Printer",
    104: "Writing",
    105: "Alarm",
    106: "Telephone",
    107: "Telephone bell ringing",
    288: "Silence (無音)",
    300: "Domestic sounds, home sounds (家庭内の音)",
    400: "Television (テレビ)",
}

def analyze_audio_detailed(filepath, segment_duration=10.0):
    """音声ファイルを詳細に分析"""

    print("モデルをロード中...")
    model = get_basic_model(mode="logits")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"音声ファイルを読み込み中: {filepath}")
    audio_data, sample_rate = sf.read(filepath)

    # モノラル化
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # 32kHzにリサンプリング
    if sample_rate != 32000:
        audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=32000)
        sample_rate = 32000

    duration = len(audio_data) / sample_rate
    print(f"音声長さ: {duration:.1f}秒")

    # セグメント分析
    segment_samples = int(segment_duration * sample_rate)
    results = []

    print(f"\n{segment_duration}秒ごとに分析中...")
    print("=" * 60)

    for start in range(0, len(audio_data) - segment_samples + 1, segment_samples):
        segment = audio_data[start:start + segment_samples]
        time_sec = start / sample_rate

        # 正規化
        max_val = np.max(np.abs(segment))
        if max_val > 0:
            segment = segment / max_val

        # テンソルに変換
        audio_tensor = torch.from_numpy(segment.astype(np.float32)).unsqueeze(0).to(device)

        # 推論
        with torch.no_grad():
            logits = model(audio_tensor)

        # 確率に変換
        probs = torch.softmax(logits, dim=-1)

        # Top-20の予測を取得（より多くの情報を取得）
        top_probs, top_indices = torch.topk(probs[0], 20)

        print(f"\n【{time_sec:.0f}-{time_sec + segment_duration:.0f}秒】")
        print("-" * 40)

        segment_events = []
        for idx, prob in zip(top_indices.cpu().numpy(), top_probs.cpu().numpy()):
            if prob >= 0.01:  # 1%以上の確率のイベントをすべて表示
                label = AUDIOSET_LABELS.get(int(idx), f"Unknown Event (ID:{idx})")
                segment_events.append({
                    "label": label,
                    "score": float(prob),
                    "id": int(idx)
                })
                if prob >= 0.05:  # 5%以上は強調表示
                    print(f"  ★ {label}: {prob*100:.1f}%")
                else:
                    print(f"     {label}: {prob*100:.1f}%")

        results.append({
            "time": time_sec,
            "events": segment_events
        })

    # サマリー統計
    print("\n" + "=" * 60)
    print("【全体サマリー】")
    print("=" * 60)

    event_stats = {}
    for segment in results:
        for event in segment["events"]:
            label = event["label"]
            if label not in event_stats:
                event_stats[label] = {"count": 0, "total_score": 0}
            event_stats[label]["count"] += 1
            event_stats[label]["total_score"] += event["score"]

    # 頻出イベント
    print("\n最も頻繁に検出された音響イベント:")
    for label, stats in sorted(event_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:15]:
        avg_score = stats["total_score"] / stats["count"]
        print(f"  • {label}")
        print(f"    出現回数: {stats['count']}回, 平均確率: {avg_score*100:.1f}%")

    return results

if __name__ == "__main__":
    filepath = "/Users/kaya.matsumoto/Desktop/children_3people_001.wav"
    results = analyze_audio_detailed(filepath)

    # 結果をJSONファイルに保存
    with open("children_audio_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n詳細な分析結果を children_audio_analysis.json に保存しました。")