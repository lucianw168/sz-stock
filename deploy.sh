#!/bin/bash
set -e

# 建议 cron: 工作日 16:30 运行（A股15:00收盘，Tushare约15:30-16:00数据就绪）
# 30 16 * * 1-5 cd /Users/catalpa/Downloads/OB/cn/sz && ./deploy.sh >> deploy.log 2>&1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE_DIR="$SCRIPT_DIR/site"
REPO_URL=$(cd "$SCRIPT_DIR" && git remote get-url origin)

# --- Step 1: 尝试更新日线数据（失败则用已有数据继续） ---
cd "$SCRIPT_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 更新日线数据..."
if python3 run.py daily 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 日线数据更新成功"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 日线数据更新失败（Tushare 不可用？），使用已有数据继续部署"
fi

# --- Step 2: 生成网站 ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 生成网站..."
if ! python3 run.py web --output "$SITE_DIR"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 网站生成失败，放弃部署"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 网站生成成功"

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
