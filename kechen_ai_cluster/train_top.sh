#!/bin/bash
# ============================================================================
#  train_top.sh — 客尘AI集群v3.0 顶尖化训练（分批，防卡死）
#  用法: bash train_top.sh [专家名]
#   不带参数=全部6个分批训练；带参数=只训指定专家（poet/lunyu/chat/math/history/cosmos）
# ============================================================================
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# ---------- 1. 合并超长语料 ----------
echo "== 合并顶尖语料 =="
cat ~/kechen_poems_full.txt >> corpus_poet.txt && echo "  +poet(古诗66首)"
cat ~/kechen_history_full.txt >> corpus_history.txt && echo "  +history(59条)"
grep '^数-' ~/kechen_math_guoxue.txt >> corpus_math.txt && echo "  +math(54条)"
grep '^国学-' ~/kechen_math_guoxue.txt >> corpus_lunyu.txt && echo "  +lunyu(30条)"
grep '^聊-' ~/kechen_chat_cosmos.txt >> corpus_chat.txt && echo "  +chat(40条)"
grep '^宇宙-' ~/kechen_chat_cosmos.txt >> corpus_cosmos.txt && echo "  +cosmos(38条)"

# ---------- 2. 删权重+编译 ----------
echo "== 删旧权重+重新编译 =="
rm -f ai_brain_*_weights.bin
bash build_all.sh

# ---------- 3. 分批训练 ----------
train_one() {
    local name="$1" q="$2"
    echo ""
    echo "======== 训练专家: $name（Ctrl+C可中断，已完成的会保存）========"
    ./ai_brain_$name -q "$q"
    echo "---- $name 训练完成 ✅ ----"
}

if [ -n "${1:-}" ]; then
    case "$1" in
        poet) train_one poet "床前明月" ;;
        lunyu) train_one lunyu "学而时习之" ;;
        chat) train_one chat "你好" ;;
        math) train_one math "3×3等于几" ;;
        history) train_one history "中国第一个朝代" ;;
        cosmos) train_one cosmos "负一元素是什么" ;;
        *) echo "未知专家: $1" ;;
    esac
else
    train_one poet "床前明月"
    train_one lunyu "学而时习之"
    train_one chat "你好"
    train_one math "3×3等于几"
    train_one history "中国第一个朝代"
    train_one cosmos "负一元素是什么"
    echo ""
    echo "== 全部训练完成！测试: ./router.sh =="
fi