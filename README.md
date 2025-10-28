# PaSST Audio Event Detection API (v3)

**Patchout Spectrogram Transformer (PaSST)** を使用した高性能音響イベント検出API

## 🚀 v3の新機能とアップグレード内容

### モデルアップグレード: AST → PaSST
- **従来（v2）**: AST (Audio Spectrogram Transformer)
- **新版（v3）**: **PaSST** (Patchout Spectrogram Transformer)

### 主な改善点
| 項目 | v2 (AST) | v3 (PaSST) | 改善率 |
|------|----------|------------|--------|
| **精度 (mAP)** | 0.459 | **0.476** | +3.7% |
| **サンプリングレート** | 16kHz | **32kHz** | 2倍 |
| **メモリ使用量** | 基準 | **約1/10** | -90% |
| **処理速度** | 基準 | **高速化** | Patchout効果 |
| **商用利用** | MIT License | Apache-2.0 | ✅ 両方OK |

### 🎵 重要: サンプリングレートの違い

**v2 (AST) と v3 (PaSST) の最も重要な違い**

| モデル | サンプリングレート | 説明 |
|--------|-------------------|------|
| **v2 (AST)** | **16kHz (16000 Hz)** | CD音質の約1/3 |
| **v3 (PaSST)** | **32kHz (32000 Hz)** | **v2の2倍** - より高周波数帯域を捉える |

#### なぜ32kHzなのか？

1. **学習データの違い**: PaSSTモデルは学習時に32kHzの音声データで訓練されています
2. **精度向上**: 32kHzにより、より高周波数帯域の音響特徴（子音、環境音など）を正確に捉えられます
3. **互換性**: 入力音声が何Hzであっても、APIが自動的に32kHzにリサンプリングします

#### ユーザー側の影響

✅ **対応不要**: APIインターフェースは変更なし
✅ **自動変換**: 8kHz、16kHz、44.1kHz、48kHzなど、どの音声でも自動的に32kHzに変換
✅ **処理時間**: リサンプリングのオーバーヘッドは軽微（ほぼ影響なし）
✅ **精度向上**: 結果として音響イベント検出精度が向上 (mAP 0.459 → 0.476)

## 📊 PaSSTの特徴

### 技術的優位性
- **Patchout技術**: 学習時に一部パッチを除去することで効率化
- **SWA (Stochastic Weight Averaging)**: より安定した予測
- **高効率**: GPUメモリ使用量を大幅削減しながら精度向上
- **527種類**の音響イベント検出（AudioSetベース）

## 🔧 セットアップ

### 1. 依存ライブラリのインストール

```bash
# 仮想環境の作成（推奨）
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# 依存ライブラリのインストール
pip install -r requirements.txt
```

### 2. 環境設定（Supabase連携時）

```bash
cp .env.example .env
# .envファイルを編集してSupabase/AWSの認証情報を設定
```

## 🏃 サーバーの起動

### ⚠️ 重要：本番環境はDockerコンテナで稼働

**このAPIは本番環境（EC2）でDockerコンテナとして自動デプロイされています。**

- **コンテナ名**: `behavior-analysis-feature-extractor-v2`
- **ポート**: 8017
- **稼働環境**: AWS EC2 (Docker)
- **デプロイ方法**: GitHub Actionsによる自動CI/CD

### ローカル開発時の起動

#### 基本起動（Supabase統合なし）
```bash
# ポート8017で起動（v2と同じポート）
python3 main.py
```

#### Supabase統合版の起動（本番と同じ環境）
```bash
# .envファイルを設定してから実行
python3 main_supabase.py
```

## 📡 APIエンドポイント

### ✅ 実装済みエンドポイント

#### 1. ヘルスチェック
```bash
curl http://localhost:8017/health
```

レスポンス:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model": "PaSST-S SWA (passt_s_swa_p16_128_ap476)",
  "device": "cpu",
  "version": "3.0.0",
  "supabase_connected": true,
  "s3_configured": true
}
```

#### 2. 音声ファイル分析（ファイルアップロード）
```bash
curl -X POST "http://localhost:8017/analyze_sound" \
  -F "file=@audio.wav" \
  -F "top_k=5" \
  -F "threshold=0.1"
```

#### 3. タイムライン分析（ファイルアップロード）
```bash
curl -X POST "http://localhost:8017/analyze_timeline" \
  -F "file=@audio.wav" \
  -F "segment_duration=10.0" \
  -F "overlap=0.0" \
  -F "top_k=3"
```

**最適設定**: 10秒セグメント、オーバーラップなし
- 処理速度と精度のベストバランス
- v2との完全互換性

#### 4. **S3統合エンドポイント（Lambdaから呼ばれる）** ⭐ 重要
```bash
curl -X POST "http://localhost:8017/fetch-and-process-paths" \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["files/device-id/2025-10-20/14-30/audio.wav"],
    "threshold": 0.1,
    "top_k": 3,
    "segment_duration": 10.0,
    "overlap": 0.0
  }'
```

**このエンドポイントの機能**:
- ✅ S3からファイルをダウンロード
- ✅ タイムライン分析を実行
- ✅ Supabaseの`behavior_yamnet`テーブルに結果を保存
- ✅ Supabaseの`audio_files`テーブルのステータスを更新
- ✅ 複数ファイルの一括処理対応

**使用場所**:
- Lambda関数: `watchme-audio-worker`
- エンドポイント: `/behavior-analysis/features/fetch-and-process-paths`

## 🔄 v2からの移行

### 完全な後方互換性
- **同じポート番号**: 8017
- **同じエンドポイント**: `/analyze_sound`, `/analyze_timeline`, `/fetch-and-process-paths`
- **同じレスポンス形式**: v2と完全互換
- **同じコンテナ名**: `behavior-analysis-feature-extractor-v2`（後方互換のため名前は維持）
- **ECRリポジトリ**: `watchme-behavior-analysis-feature-extractor`

### 移行手順
1. ✅ **完了**: v3コードがGitHubにコミット済み
2. ✅ **完了**: CI/CD設定完了（GitHub Actions）
3. ✅ **完了**: Docker設定完了（Dockerfile.prod）
4. ✅ **完了**: Supabase統合（main_supabase.py）

## 🐳 Docker対応（本番環境）

### ⚠️ 重要：手動でのDockerビルドは不要

**本番環境へのデプロイは GitHub Actions による CI/CD で自動実行されます。**

### CI/CDによる自動デプロイフロー

```bash
# ローカルでコードを変更したら、pushするだけ
git add .
git commit -m "your changes"
git push origin main

# 以降は自動実行される：
# 1. GitHub Actionsが起動
# 2. Dockerイメージをビルド（ARM64、PaSSTモデルプリロード）
# 3. ECRへプッシュ
# 4. EC2に設定ファイルをコピー
# 5. .envファイルを自動作成/更新
# 6. 既存コンテナを削除
# 7. 新しいコンテナを起動
# 8. ヘルスチェック実行
```

### GitHub Actionsの設定

- **ファイル**: `.github/workflows/deploy-to-ecr.yml`
- **トリガー**: `main`ブランチへのpush
- **実行時間**: 約5-10分
- **確認URL**: `https://github.com/hey-watchme/api-behavior-analysis-feature-extractor-v3/actions`

### 手動でのローカルDockerビルド（開発用）

```bash
# ローカルテスト用のビルド
docker build -f Dockerfile.prod -t passt-api:local .

# ローカルで起動
docker run -p 8017:8017 \
  --env-file .env \
  passt-api:local
```

### EC2での確認コマンド

```bash
# EC2にSSH接続
ssh -i ~/watchme-key.pem ubuntu@3.24.16.82

# コンテナの状態確認
docker ps | grep behavior-analysis-feature-extractor

# ログ確認
docker logs behavior-analysis-feature-extractor-v2 --tail 100

# ヘルスチェック
curl http://localhost:8017/health
```

## 📈 パフォーマンス比較

### ベンチマーク結果（1分間の音声）
| 指標 | AST (v2) | PaSST (v3) |
|------|----------|------------|
| 処理時間 | ~30秒 | **~25秒** |
| メモリ使用量 | 2GB | **200MB** |
| 検出精度 | Good | **Better** |

## 🎯 検出可能な音響イベント（主要カテゴリ）

### 人間の音声・音
- Speech（会話）
- Laughter（笑い声）
- Crying（泣き声）
- Cough（咳）
- Sneeze（くしゃみ）
- Singing（歌声）
- Shouting（叫び声）

### 環境音
- Music（音楽）
- Silence（静寂）
- Applause（拍手）
- Door（ドアの音）
- Footsteps（足音）

### その他
- 動物の鳴き声（犬、猫、鳥など）
- 楽器の音
- 機械音・電子音
- 自然音（雨、風など）

**合計527種類**のAudioSetクラスに対応

## 🔬 技術仕様

- **モデル**: PaSST-S SWA (passt_s_swa_p16_128_ap476)
- **アーキテクチャ**: Patchout Spectrogram Transformer
- **入力**: 32kHz サンプリングレート（自動リサンプリング対応）
- **出力**: 527クラスの確率分布
- **フレームワーク**: PyTorch + Timm

## 📝 ライセンス

- **コード**: Apache-2.0 License
- **モデル**: Apache-2.0 License（商用利用可能）
- **参考論文**: "Efficient Training of Audio Transformers with Patchout" (INTERSPEECH 2022)

## 🙏 謝辞

- PaSST開発者: Khaled Koutini, Gerhard Widmer (JKU, Austria)
- GitHub: https://github.com/kkoutini/PaSST

## ✅ 実装完了済み

- ✅ **Supabase完全統合**（`fetch-and-process-paths`エンドポイント）
  - S3からのファイルダウンロード
  - behavior_yamnetテーブルへのデータ保存
  - audio_filesテーブルのステータス更新
- ✅ **CI/CD設定**（GitHub Actions）
  - 自動Dockerビルド（ARM64）
  - ECRへの自動プッシュ
  - EC2への自動デプロイ
  - ヘルスチェック実行
- ✅ **PaSSTモデルのプリロード**（起動時間を数秒に短縮）
- ✅ **v2との完全な後方互換性**

## 🚧 今後の実装予定

- [ ] AudioSetラベルの完全版追加（527クラス → 現在は主要クラスのみ）
- [ ] バッチ処理の最適化
- [ ] GPU対応の強化
- [ ] モニタリングダッシュボード（CloudWatch統合）

## 📞 サポート

問題が発生した場合は、GitHubのIssueまたは開発チームまでご連絡ください。