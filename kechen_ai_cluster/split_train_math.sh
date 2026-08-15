#!/bin/bash
# ============================================================================
#  split_train_math.sh — 数学三副本: 语料均分 + 集群编译 + 训练
#
#  集群架构: 把 corpus_math.txt 按行均分成 3 份, 用同一份 ai_brain_v08.cpp
#  编译出 3 个数学副本 (ai_brain_math_a/b/c), 各自用一份语料训练,
#  权重分别保存为 ai_brain_math_a_weights.bin / _b_ / _c_。
#  配合 router.sh v3.0: 数学问题 -> 三副本投票 (答案一致->输出; 不一致
#  -> 回退 ai_brain_fast 查表; 再不行 -> 提示联网搜索)。
#
#  流程:
#    1. 按行均分 corpus_math.txt
#         -> corpus_math_a.txt / corpus_math_b.txt / corpus_math_c.txt
#    2. 集群编译 3 个数学副本 (ai_brain_v08.cpp + -D 宏注入身份/语料/权重)
#    3. 首次运行自动训练 (训练完成自动保存权重; 已有权重则加载跳过训练)
#
#  用法:
#    bash split_train_math.sh            = split + build + train (完整流程)
#    bash split_train_math.sh split     只均分语料
#    bash split_train_math.sh build     均分 + 编译
#    bash split_train_math.sh train     均分 + 编译 + 训练
#    bash split_train_math.sh retrain   删旧三副本权重后强制重训
#
#  注意: 不修改原 corpus_math.txt; 重复运行会覆盖三副本语料/权重。
#        训练耗时较长, 属正常现象 (每副本最多 150 epochs)。
# ============================================================================
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

CXX="${CXX:-clang++}"
FLAGS="-O2 -std=c++17"
SRC="corpus_math.txt"
DEMOS="1+1等于几|6×7等于几|什么是加法|正方形周长怎么算"

# ---------- 1. 按行均分语料 ----------
split_parts() {
    echo "== [1/3] 按行均分语料: $SRC -> corpus_math_a/b/c.txt =="
    [ -s "$SRC" ] || { echo "[错误] 找不到语料文件: $SRC"; exit 1; }
    local total
    total="$(wc -l < "$SRC")"
    if [ "$total" -lt 3 ]; then
        echo "[错误] 语料只有 $total 行, 不足 3 份, 无法均分"; exit 1
    fi
    # 均分: 每份 base 行, 余数 rem 行分给前 rem 份 (如 10 行 -> 4/3/3)
    awk -v n=3 '
        { lines[NR] = $0 }
        END {
            base = int(NR / n); rem = NR % n
            start = 1
            for (i = 1; i <= n; i++) {
                cnt = base + (i <= rem ? 1 : 0)
                file = "corpus_math_" substr("abc", i, 1) ".txt"
                for (j = start; j < start + cnt; j++) print lines[j] > file
                start += cnt
            }
        }' "$SRC"
    echo "  共 $total 行, 均分 3 份:"
    wc -l corpus_math_a.txt corpus_math_b.txt corpus_math_c.txt
    echo
}

# ---------- 2. 集群编译 3 个数学副本 ----------
build_copies() {
    echo "== [2/3] 集群编译 3 个数学副本 (ai_brain_v08.cpp, 编译器: $CXX) =="
    local i
    for i in a b c; do
        echo "---- 编译 ai_brain_math_$i (语料 corpus_math_$i.txt, 权重 ai_brain_math_${i}_weights.bin) ----"
        $CXX $FLAGS \
            -D"EXPERT_NAME=\"math_$i\"" \
            -D"CORPUS_FILE=\"corpus_math_$i.txt\"" \
            -D"WEIGHTS_FILE=\"ai_brain_math_${i}_weights.bin\"" \
            -D"DEMOS=\"$DEMOS\"" \
            ai_brain_v08.cpp -o "ai_brain_math_$i"
        echo "    OK -> ai_brain_math_$i"
    done
    echo
}

# ---------- 3. 训练 3 个数学副本 ----------
train_copies() {
    echo "== [3/3] 训练 3 个数学副本 (首次运行自动训练+保存权重) =="
    local i
    for i in a b c; do
        if [ -s "ai_brain_math_${i}_weights.bin" ]; then
            echo "---- ai_brain_math_$i 已有权重 -> 跳过训练 (加载验证) ----"
            if ./ai_brain_math_$i -q "1+1等于几" >/dev/null 2>&1; then
                echo "    加载 OK: ai_brain_math_${i}_weights.bin"
            else
                echo "    警告: 加载验证失败 (可执行 bash split_train_math.sh retrain)"
            fi
        else
            echo "======== 训练 ai_brain_math_$i (语料 corpus_math_$i.txt) ========"
            ./ai_brain_math_$i -q "1+1等于几"
            if [ -s "ai_brain_math_${i}_weights.bin" ]; then
                echo "---- ai_brain_math_$i 训练完成, 权重已保存 ✅ ----"
            else
                echo "[错误] ai_brain_math_$i 权重未生成, 训练失败"; exit 1
            fi
        fi
        echo
    done
}

# ---------- 入口 ----------
case "${1:-all}" in
    split)   split_parts ;;
    build)   split_parts; build_copies ;;
    train)   split_parts; build_copies; train_copies ;;
    retrain) split_parts; build_copies
             rm -f ai_brain_math_a_weights.bin ai_brain_math_b_weights.bin ai_brain_math_c_weights.bin
             echo "== 已删除旧三副本权重, 开始重训 =="
             train_copies ;;
    all|"")  split_parts; build_copies; train_copies ;;
    *) echo "用法: bash split_train_math.sh [split|build|train|retrain|all]"; exit 1 ;;
esac

echo "============================================================"
echo "完成! 数学三副本已就绪, 可测试:"
echo "  ./router.sh \"1+1等于几\"     # 数学问题 -> 三副本投票"
echo "  ./router.sh                  # 交互模式"
echo "============================================================"
