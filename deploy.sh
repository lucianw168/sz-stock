#!/bin/bash
set -e

SITE_DIR="site"
REPO_URL=$(git remote get-url origin)

# 生成最新网站
python3 run.py web --output "$SITE_DIR"

# 如果配置了自定义域名，取消下行注释
# echo "yourdomain.com" > "$SITE_DIR/CNAME"

# 部署到 gh-pages 分支
cd "$SITE_DIR"
git init
git checkout -b gh-pages
git add -A
git commit -m "Deploy $(date '+%Y-%m-%d %H:%M')"
git remote add origin "$REPO_URL"
git push -f origin gh-pages
cd ..

echo "Deployed successfully!"
