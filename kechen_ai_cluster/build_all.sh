#!/bin/bash
# ============================================================================
#  build_all.sh — 客尘AI集群v2.0 编译 6 个专家 (v0.8 架构) + 快答层 ai_brain_fast
# ============================================================================
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
CXX="${CXX:-clang++}"
FLAGS="-O2 -std=c++17"

echo "== 客尘AI集群v2.0 · 编译 6 个专家 (v0.8 架构) + 快答层 =="
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

build poet    corpus_poet.txt    ai_brain_poet_weights.bin    "床前明月|题西林壁|泊船瓜洲|小池"
build lunyu   corpus_lunyu.txt   ai_brain_lunyu_weights.bin   "学而时习之|君子和而不同|欲速则不达|逝者如斯夫"
build chat    corpus_chat.txt    ai_brain_chat_weights.bin    "你好|你喜欢什么|你累吗|你怕什么"
build cosmos  corpus_cosmos.txt  ai_brain_cosmos_weights.bin  "负一元素是|第1条能量底基律|第49项三面能量律|引力元素是什么"
build math    corpus_math.txt    ai_brain_math_weights.bin    "1+1等于几|6×7等于几|什么是加法|正方形周长怎么算"
build history corpus_history.txt ai_brain_history_weights.bin "中国第一个朝代是什么|大禹治水|万里长城有多长|历史是什么"

echo "---- 编译 ai_brain_fast (快答层: 零训练零权重, 加载6语料建索引) ----"
$CXX $FLAGS ai_brain_fast.cpp -o ai_brain_fast
echo "    OK -> ai_brain_fast"
echo
echo "== 全部编译完成 =="
ls -la ai_brain_poet ai_brain_lunyu ai_brain_chat ai_brain_cosmos ai_brain_math ai_brain_history ai_brain_fast
