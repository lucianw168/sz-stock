#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE_DIR="$SCRIPT_DIR/site"
REPO_URL=$(cd "$SCRIPT_DIR" && git remote get-url origin)

# 生成最新网站
cd "$SCRIPT_DIR"
python3 run.py web --output "$SITE_DIR"

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
