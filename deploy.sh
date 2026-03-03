#!/bin/bash
set -e

# 建议 cron: 工作日 16:30 运行（A股15:00收盘，Tushare约15:30-16:00数据就绪）
# 30 16 * * 1-5 cd /Users/catalpa/Downloads/OB/cn/sz && ./deploy.sh >> deploy.log 2>&1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE_DIR="$SCRIPT_DIR/site"
REPO_URL=$(cd "$SCRIPT_DIR" && git remote get-url origin)

# --- 数据可用性检查：重试最多 3 次，每次间隔 10 分钟 ---
MAX_RETRIES=3
RETRY_INTERVAL=600  # 秒

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 尝试第 $attempt 次生成网站..."
    cd "$SCRIPT_DIR"
    if python3 run.py web --output "$SITE_DIR"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 网站生成成功"
        break
    else
        if [ "$attempt" -eq "$MAX_RETRIES" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 已达最大重试次数 ($MAX_RETRIES)，放弃部署"
            exit 1
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 生成失败，等待 ${RETRY_INTERVAL}s 后重试..."
        sleep $RETRY_INTERVAL
    fi
done

# 如果配置了自定义域名，取消下行注释
# echo "yourdomain.com" > "$SITE_DIR/CNAME"

# 使用临时目录部署到 gh-pages（避免污染主仓库）
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

cp -R "$SITE_DIR/" "$TMPDIR/site"
cd "$TMPDIR/site"
git init
git config http.postBuffer 524288000
git checkout -b gh-pages
git add -A
git commit -m "Deploy $(date '+%Y-%m-%d %H:%M')"
git remote add origin "$REPO_URL"
git push -f origin gh-pages

echo "Deployed successfully!"
