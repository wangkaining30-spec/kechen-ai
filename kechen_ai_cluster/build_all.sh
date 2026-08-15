#!/bin/bash
# ============================================================================
#  build_all.sh — 客尘AI集群v1.0 编译 5 个专家 (v0.8 架构, 同一份源码)
# ============================================================================
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
CXX="${CXX:-clang++}"
FLAGS="-O2 -std=c++17"

echo "== 客尘AI集群v1.0 · 编译 5 个专家 (v0.8 架构) =="
echo "编译器: $CXX"
echo

build() {
    local name="$1" corpus="$2" weights="$3" demos="$4"
    echo "---- 编译 ai_brain_$name (语料 $corpus, 权重 $weights) ----"
    $CXX $FLAGS \
        -D'EXPERT_NAME="'"$name"'"' \
        -D'CORPUS_FILE="'"$corpus"'"' \
        -D'WEIGHTS_FILE="'"$weights"'"' \
        -D'DEMOS="'"$demos"'"' \
        ai_brain_v08.cpp -o "ai_brain_$name"
    echo "    OK -> ai_brain_$name"
    echo
}

build poet    corpus_poet.txt    ai_brain_poet_weights.bin    "床前明月|白日依山|秦时明月|春眠不觉"
build lunyu   corpus_lunyu.txt   ai_brain_lunyu_weights.bin   "学而时习之|温故而知|君子坦荡荡|见贤思齐"
build chat    corpus_chat.txt    ai_brain_chat_weights.bin    "你好|你叫什么|再见|你会什么"
build cosmos  corpus_cosmos.txt  ai_brain_cosmos_weights.bin  "负一元素是|宇宙的本原|归源衰变|引力元素"
build math    corpus_math.txt    ai_brain_math_weights.bin    "1+1等于几|3×3等于几|什么是加法|圆周率"

echo "== 全部编译完成 =="
ls -la ai_brain_poet ai_brain_lunyu ai_brain_chat ai_brain_cosmos ai_brain_math
