#!/bin/bash
# run_demo.sh — 客尘AI集群v1.0 路由器实测 (4 专家 + 3 条指定路由)
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "############ 客尘AI集群v1.0 · 路由器实测 ############"
echo
./router.sh "床前明月"
./router.sh "负一元素是什么"
./router.sh "你好"
echo "---- 附加测试: 论语/名言路由 ----"
./router.sh "学而时习之"
./router.sh "宇宙的循环"
