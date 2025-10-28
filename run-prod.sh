#!/bin/bash
# PaSST Audio Event Detection API (v3) - 本番環境デプロイスクリプト

set -e  # エラーが発生したら即座に終了

echo "🚀 Starting PaSST Audio Event Detection API (v3) deployment..."

# 環境変数の設定
export AWS_REGION="ap-southeast-2"
export ECR_REGISTRY="754724220380.dkr.ecr.ap-southeast-2.amazonaws.com"
export ECR_REPOSITORY="watchme-behavior-analysis-feature-extractor"
export CONTAINER_NAME="behavior-analysis-feature-extractor"

# ECRログイン
echo "🔐 Logging into Amazon ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# 最新のイメージをプル
echo "📦 Pulling latest image from ECR..."
docker pull ${ECR_REGISTRY}/${ECR_REPOSITORY}:latest

# 既存のコンテナを停止・削除（存在する場合）
echo "🛑 Stopping existing container (if any)..."
docker-compose -f docker-compose.prod.yml down 2>/dev/null || true

# 環境変数ファイルの確認
if [ ! -f .env ]; then
    echo "⚠️ Warning: .env file not found in current directory"
    # 共有の環境変数ファイルから作成
    if [ -f /home/ubuntu/.env ]; then
        echo "Copying shared .env file..."
        cp /home/ubuntu/.env .env
        echo "✅ .env file copied from shared location"
    else
        echo "Creating .env file with production credentials..."
        cat > .env << EOF
# Supabase設定
SUPABASE_URL=https://qvtlwotzuzbavrzqhyvt.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF2dGx3b3R6dXpiYXZyenFoeXZ0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTEzODAzMzAsImV4cCI6MjA2Njk1NjMzMH0.g5rqrbxHPw1dKlaGqJ8miIl9gCXyamPajinGCauEI3k

# AWS S3設定
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
S3_BUCKET_NAME=watchme-audio-files
AWS_REGION=ap-southeast-2
EOF
        echo "✅ .env file created with production credentials"
    fi
fi

# Docker Composeで新しいコンテナを起動
echo "🚀 Starting new container with docker-compose..."
docker-compose -f docker-compose.prod.yml up -d

# コンテナが正常に起動したか確認
echo "⏳ Waiting for container to start..."
sleep 10

# コンテナの状態を確認
if docker ps | grep -q ${CONTAINER_NAME}; then
    echo "✅ Container is running!"

    # ヘルスチェック
    echo "🏥 Performing health check..."
    sleep 5
    if curl -f http://localhost:8017/health > /dev/null 2>&1; then
        echo "✅ Health check passed!"
        curl -s http://localhost:8017/health | python3 -m json.tool 2>/dev/null || curl http://localhost:8017/health
    else
        echo "⚠️ Health check failed, but container is running"
    fi

    # コンテナログの最後の20行を表示（PaSSTモデルロード確認のため）
    echo "📋 Recent container logs:"
    docker logs --tail 20 ${CONTAINER_NAME}
else
    echo "❌ Container failed to start!"
    echo "📋 Container logs:"
    docker logs ${CONTAINER_NAME}
    exit 1
fi

# 古いイメージのクリーンアップ
echo "🧹 Cleaning up old images..."
docker image prune -f

echo "✅ Deployment completed successfully!"
echo "🌐 API is available at: https://api.hey-watch.me/behavior-analysis/features/"
echo "📍 Health endpoint: https://api.hey-watch.me/behavior-analysis/features/health"
echo "🎯 Model: PaSST-S SWA (v3) - 32kHz sampling rate"
