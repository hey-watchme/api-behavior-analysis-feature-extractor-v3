# Behavior Features API | AST Audio Event Detection (v3)

**Audio Spectrogram Transformer (AST)** を使用した音響イベント検出API

> **モデル変更履歴**: 2024-10 PaSST導入 → 2025-11-21 **ASTに戻す**（感情イベント検出精度向上のため）

---

## 🗺️ ルーティング詳細

| 項目 | 値 | 説明 |
|------|-----|------|
| **🏷️ サービス名** | Behavior Features API | 音響イベント検出（527種類） |
| **📦 モデル** | AST | Audio Spectrogram Transformer (MIT) |
| | | |
| **🌐 外部アクセス（Nginx）** | | |
| └ 公開エンドポイント | `https://api.hey-watch.me/behavior-analysis/features/` | Lambdaから呼ばれるパス |
| └ Nginx設定ファイル | `/etc/nginx/sites-available/api.hey-watch.me` | 152-174行目 |
| └ proxy_pass先 | `http://localhost:8017/` | 内部転送先 |
| └ タイムアウト | 180秒 | read/connect/send |
| | | |
| **🔌 API内部エンドポイント** | | |
| └ ヘルスチェック | `/health` | GET |
| └ ファイル分析 | `/analyze_sound` | POST |
| └ タイムライン分析 | `/analyze_timeline` | POST |
| └ **S3統合（重要）** | `/fetch-and-process-paths` | POST - Lambdaが呼ぶ |
| | | |
| **🐳 Docker/コンテナ** | | |
| └ コンテナ名 | `behavior-analysis-feature-extractor` | `docker ps`で表示される名前 |
| └ ポート（内部） | 8017 | コンテナ内 |
| └ ポート（公開） | `127.0.0.1:8017:8017` | ローカルホストのみ |
| └ ヘルスチェック | `/health` | Docker healthcheck |
| | | |
| **☁️ AWS ECR** | | |
| └ リポジトリ名 | `watchme-behavior-analysis-feature-extractor` | イメージ保存先 |
| └ リージョン | ap-southeast-2 (Sydney) | |
| └ URI | `754724220380.dkr.ecr.ap-southeast-2.amazonaws.com/watchme-behavior-analysis-feature-extractor:latest` | |
| | | |
| **⚙️ systemd** | | |
| └ サービス名 | `watchme-behavior-yamnet.service` | ※名前が不統一 |
| └ 起動コマンド | `docker-compose up -d` | |
| └ 自動起動 | enabled | サーバー再起動時に自動起動 |
| | | |
| **📂 ディレクトリ** | | |
| └ ソースコード | `/Users/kaya.matsumoto/projects/watchme/api/behavior-analysis/feature-extractor-v3` | ローカル |
| └ GitHubリポジトリ | `hey-watchme/api-behavior-analysis-feature-extractor-v3` | |
| └ EC2配置場所 | Docker内部のみ（ディレクトリなし） | ECR経由デプロイ |
| | | |
| **🔗 呼び出し元** | | |
| └ Lambda関数 | `watchme-audio-worker` | 30分ごと |
| └ 呼び出しURL | `https://api.hey-watch.me/behavior-analysis/features/fetch-and-process-paths` | フルパス |
| └ 環境変数 | `API_BASE_URL=https://api.hey-watch.me` | Lambda内 |

---

## 🎯 現在使用中のモデル: AST

### モデル情報
- **モデル名**: `MIT/ast-finetuned-audioset-10-10-0.4593`
- **開発元**: MIT CSAIL
- **精度 (mAP)**: 0.459
- **サンプリングレート**: 16kHz
- **ライセンス**: Apache-2.0（商用利用可能）
- **ライブラリ**: transformers (Hugging Face)

### ASTの特徴・強み

#### ✅ 実運用での優位性
- **Speech detection**: 非常に高精度
- **Laughter detection**: 優秀（感情分析に重要）
- **Cough detection**: 優秀（健康状態の指標）
- **人間の感情イベント全般**: 実績が豊富

#### 技術仕様
- **アーキテクチャ**: Vision Transformer (ViT) のオーディオ版
- **学習データ**: AudioSet (200万件の音声、527クラス)
- **パラメータ数**: 約86M
- **Transformer層**: 12 layers
- **入力形式**: Mel-spectrogram (128 bins)

#### コミュニティ・実績
- ✅ Hugging Faceで最も人気のAudioSetモデル（ダウンロード数50,000+）
- ✅ 多くの研究・プロダクトで採用実績あり
- ✅ ドキュメント・サンプルコードが充実

### 🔄 PaSSTからASTへ戻した理由（2025-11-21）

**感情イベント検出精度の優先**:
- PaSSTは全体的なmAP（0.476）は高いが、Speech/Laughter/Coughなど**人間の感情イベント検出**でASTに劣る
- WatchMeのユースケースでは感情イベント検出が最優先
- 実績のあるASTモデルに戻すことで、より確実な検出を実現

**検出可能な527種類**の音響イベント（AudioSetベース）

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
  "model_name": "MIT/ast-finetuned-audioset-10-10-0.4593",
  "sampling_rate": 16000,
  "version": "3.0.0",
  "supabase_connected": true,
  "s3_connected": true
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
- ✅ Supabaseの`audio_features`テーブルに結果を保存
  - `behavior_extractor_result`: JSONB形式でタイムラインデータ
  - `behavior_extractor_status`: 'completed'
  - `behavior_extractor_processed_at`: 処理完了時刻
- ✅ Supabaseの`audio_files`テーブルのステータスを更新
- ✅ 複数ファイルの一括処理対応

**使用場所**:
- Lambda関数: `watchme-audio-worker`
- エンドポイント: `/behavior-analysis/features/fetch-and-process-paths`

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
# 2. Dockerイメージをビルド（ARM64、ASTモデルプリロード）
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
docker build -f Dockerfile.prod -t ast-api:local .

# ローカルで起動
docker run -p 8017:8017 \
  --env-file .env \
  ast-api:local
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

## 📈 パフォーマンス

### 処理性能（1分間の音声）
| 指標 | 値 |
|------|-----|
| 処理時間 | ~30秒 |
| メモリ使用量 | ~2GB |
| サンプリングレート | 16kHz |
| 検出精度 (mAP) | 0.459 |

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

- **モデル**: MIT/ast-finetuned-audioset-10-10-0.4593
- **アーキテクチャ**: Vision Transformer (ViT) for Audio
- **入力**: 16kHz サンプリングレート（自動リサンプリング対応）
- **出力**: 527クラスの確率分布
- **フレームワーク**: PyTorch + Transformers (Hugging Face)

## 📝 ライセンス

- **コード**: Apache-2.0 License
- **モデル**: Apache-2.0 License（商用利用可能）
- **参考論文**: "AST: Audio Spectrogram Transformer" (2021)

## 🙏 謝辞

- AST開発者: Yuan Gong, Yu-An Chung, James Glass (MIT CSAIL)
- Hugging Face: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593

## ✅ 実装完了済み

- ✅ **Supabase完全統合**（`fetch-and-process-paths`エンドポイント）
  - S3からのファイルダウンロード
  - spot_featuresテーブルへのデータ保存（behavior_extractor_result）
  - audio_filesテーブルのステータス更新
- ✅ **CI/CD設定**（GitHub Actions）
  - 自動Dockerビルド（ARM64）
  - ECRへの自動プッシュ
  - EC2への自動デプロイ
  - ヘルスチェック実行
- ✅ **ASTモデルのプリロード**（起動時間を数秒に短縮）
- ✅ **感情イベント検出に最適化**（Speech, Laughter, Cough）

## 📄 関連ドキュメント

### SED モデル比較・選定ガイド
**→ [SED_MODEL_COMPARISON.md](./SED_MODEL_COMPARISON.md)**

v2 (AST) と v3 (PaSST) の詳細比較、最新モデル候補（BEATs, HTS-ATなど）の情報をまとめた資料です。モデル切り替えを検討する際は必ずこのドキュメントを参照してください。

**内容**:
- v2 (AST) と v3 (PaSST) の技術仕様・性能比較
- 実運用での検出精度（Speech, Laughter, Coughなど）
- 2024-2025年の最新モデル候補（BEATs, HTS-AT, Audio-MAE）
- ネット上の評判・コミュニティの評価
- モデル切り替え作業の見積もり
- 今後の検証ロードマップ

## 🚧 今後の実装予定

- [ ] **フィルタリング機能の再実装**（AST用のラベル閾値設定）
- [ ] AudioSetラベルの日本語翻訳追加
- [ ] バッチ処理の最適化
- [ ] GPU対応の強化
- [ ] モニタリングダッシュボード（CloudWatch統合）

## 📞 サポート

問題が発生した場合は、GitHubのIssueまたは開発チームまでご連絡ください。