#!/bin/bash
# ============================================================================
#  ask.sh — 客尘AI自主问答总入口（查表→直算/搜索→思考→记忆闭环）
#  用法: bash ask.sh "问题"
#  功能:
#    1. 快答层查表 (ai_brain_fast 自动扫描全部 corpus_*.txt, 命中→秒答)
#    2. 未命中 → 数学问题本地直算 (python3, 精确秒答, 无需联网)
#    3. 其他未命中 → 自主搜索 (Bing) + 本地 Qwen3B 思考整理
#    4. 学到的问答追加到 corpus_learn.txt (fast 自动扫描 → 下次秒答!)
#       记忆闭环: 问→学→记→(再问)快答层直接命中
# ============================================================================
Q="$1"
if [ -z "$Q" ]; then
    echo "用法: bash ask.sh \"问题\""
    exit 1
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "🤖 客尘AI思考中..."

# ========== 第1步：快答层查表 (fast 启动时自动加载全部 corpus_*.txt,
#            含 corpus_learn.txt 学习记忆库 → 新知识秒答) ==========
FAST_ANS=$(./ai_brain_fast -q "$Q" 2>/dev/null)
if [ -n "$FAST_ANS" ]; then
    echo "✅ 本地知识直接回答（${Q}）:"
    echo "$FAST_ANS"
    exit 0
fi

# ========== 第2步：数学问题本地直算 (无需联网, 结果精确, 直接进入记忆) ==========
#   支持: N的M次方/次幂、N乘(以)M、N加(上)M、N减(去)M、N除以M、N^M、N×M 等
MATH_ANS="$(python3 - "$Q" <<'PYEOF' 2>/dev/null
import re, sys
q = sys.argv[1] if len(sys.argv) > 1 else ''
# 归一化: 去空白/常见标点/语气词 ("等于几/等于多少/是多少/是几" 等)
s = re.sub(r'[\s，。？！!?、；;：:等于多少等于几是多少是几请问]', '', q)
ops = [
    (re.compile(r'(\d+(?:\.\d+)?)\s*的\s*(\d+(?:\.\d+)?)\s*(?:次方|次幂)'), lambda a, b: a ** b),
    (re.compile(r'(\d+(?:\.\d+)?)\s*\^\s*(\d+(?:\.\d+)?)'), lambda a, b: a ** b),
    (re.compile(r'(\d+(?:\.\d+)?)\s*(?:乘以|乘|×|\*)\s*(\d+(?:\.\d+)?)'), lambda a, b: a * b),
    (re.compile(r'(\d+(?:\.\d+)?)\s*(?:加上|加|＋|\+)\s*(\d+(?:\.\d+)?)'), lambda a, b: a + b),
    (re.compile(r'(\d+(?:\.\d+)?)\s*(?:减去|减|－|-)\s*(\d+(?:\.\d+)?)'), lambda a, b: a - b),
    (re.compile(r'(\d+(?:\.\d+)?)\s*(?:除以|除|÷|/)\s*(\d+(?:\.\d+)?)'), lambda a, b: a / b if b else None),
]
for pat, fn in ops:
    m = pat.search(s)
    if not m:
        continue
    try:
        a, b = float(m.group(1)), float(m.group(2))
        r = fn(a, b)
        if r is None:
            continue
        if abs(r - round(r)) < 1e-9:
            print(int(round(r)))              # 整数结果
        else:
            print(('%.6f' % r).rstrip('0').rstrip('.'))  # 小数结果去尾零
    except Exception:
        continue
    sys.exit(0)
sys.exit(1)
PYEOF
)"

if [ -n "$MATH_ANS" ]; then
    echo "🧮 数学问题本地直算（${Q}）:"
    ANS="$MATH_ANS"
else
    echo "🔍 本地没学过，自主搜索中..."
    # ========== 第3步：自主搜索 ==========
    bash search.sh "$Q" > $HOME/ke_search_result.txt 2>&1
    if [ ! -s $HOME/ke_search_result.txt ]; then
        echo "❌ 搜索失败，换个问法试试"
        exit 1
    fi
    cat $HOME/ke_search_result.txt | head -5

    echo ""
    echo "🧠 本地思考中（Qwen3B整理）..."
    # ========== 第4步：本地思考 ==========
    ANS=$(bash think.sh "$Q" $HOME/ke_search_result.txt 2>/dev/null | tail -5)
fi

echo "💡 客尘AI的回答："
echo "$ANS"

# ========== 第5步：记忆（追加到语料 corpus_learn.txt） ==========
#   ai_brain_fast 启动时自动扫描当前目录全部 corpus_*.txt (含 corpus_learn.txt),
#   因此学习到的问答写入 corpus_learn.txt 后, 下次提问直接快答层命中 (秒答)。
#   语料行格式: 问题=答案 (每行须单行、不含 '=' 以免破坏按行解析)
if [ -n "$ANS" ]; then
    # 清洗: 去换行→压缩连续空格→去首尾空格→把 '=' 换全角冒号(避免破坏 Q=A 解析)→限长
    ONE=$(echo "$ANS" | tr '\n' ' ' | tr -s ' ' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/=/：/g' | cut -c1-200)
    QN=$(echo "$Q"   | tr '\n' ' ' | tr -s ' ' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/=/：/g' | cut -c1-80)
    if [ -n "$ONE" ]; then
        if grep -qF "$QN=" corpus_learn.txt 2>/dev/null; then
            echo ""
            echo "ℹ️ 这个问题之前已记住，跳过重复记忆"
        else
            echo "$QN=$ONE" >> corpus_learn.txt
            echo ""
            echo "🧠 已记住！下次直接答（不用搜了）"
        fi
    fi
fi
