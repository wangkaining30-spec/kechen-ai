#!/bin/bash
# run_demo.sh — 客尘AI集群v1.0 路由器实测 (5 专家 + 指定路由)
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "############ 客尘AI集群v1.0 · 路由器实测 ############"
echo
echo "---- 数学专家 (新) ----"
./router.sh "3×3等于几"
./router.sh "1+1等于几"
./router.sh "什么是加法"
echo "---- 旧专家回归 ----"
./router.sh "床前明月"
./router.sh "负一元素是什么"
./router.sh "你好"
echo "---- 附加测试: 论语/名言路由 ----"
./router.sh "学而时习之"
./router.sh "宇宙的循环"
