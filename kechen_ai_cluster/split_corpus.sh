#!/bin/bash
# ============================================================================
#  split_corpus.sh — 客尘AI集群v1.0 语料拆分器
#
#  输入:
#    kechen_corpus.txt   普通语料 (行前缀: 无前缀=古诗 / 论语- / 名言- /
#                       对话- / 字典-)
#    kechen_cosmos.txt   宇宙语料 (行前缀: 宇宙-)
#    kechen_math.txt     数学语料 (每行一段: Q=A 问答对 / 纯陈述)
#    kechen_history.txt  历史语料 (每行一段: Q=A 问答对, 原样拷贝)
#   (优先读 $HOME 下的原文件; 若不可读则回退到 /sdcard/Download 副本)
#
#  输出 (每行一段, 已剥离 "前缀:标题：" 只留正文):
#    corpus_poet.txt     古诗            (无前缀行, 22 段)
#    corpus_lunyu.txt    论语+名言       (论语- 前缀 + 名言- 前缀, 22 段)
#    corpus_chat.txt     对话+字典       (对话- 前缀 + 字典- 前缀, 37 段)
#    corpus_cosmos.txt   宇宙模型        (kechen_cosmos.txt 全部 35 段)
#    corpus_math.txt     数学知识        (kechen_math.txt 全部 18 段)
#    corpus_history.txt  历史知识        (kechen_history.txt 全部 20 段)
# ============================================================================
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# ---- 定位语料源文件 (原文件优先, sdcard 副本兜底) ----
CORPUS_SRC=""
for p in "$HOME/kechen_corpus.txt" /sdcard/Download/kechen_corpus.txt; do
    if [ -r "$p" ]; then CORPUS_SRC="$p"; break; fi
done
COSMOS_SRC=""
for p in "$HOME/kechen_cosmos.txt" /sdcard/Download/kechen_cosmos.txt; do
    if [ -r "$p" ]; then COSMOS_SRC="$p"; break; fi
done
MATH_SRC=""
for p in "$HOME/kechen_math.txt" /sdcard/Download/kechen_math.txt; do
    if [ -r "$p" ]; then MATH_SRC="$p"; break; fi
done
HISTORY_SRC=""
for p in "$HOME/kechen_history.txt" /sdcard/Download/kechen_history.txt; do
    if [ -r "$p" ]; then HISTORY_SRC="$p"; break; fi
done
[ -n "$CORPUS_SRC" ] || { echo "[错误] 找不到 kechen_corpus.txt"; exit 1; }
[ -n "$COSMOS_SRC" ] || { echo "[错误] 找不到 kechen_cosmos.txt"; exit 1; }
[ -n "$MATH_SRC" ] || { echo "[错误] 找不到 kechen_math.txt"; exit 1; }
[ -n "$HISTORY_SRC" ] || { echo "[错误] 找不到 kechen_history.txt"; exit 1; }

echo "== 客尘AI集群v1.0 语料拆分 =="
echo "普通语料源: $CORPUS_SRC"
echo "宇宙语料源: $COSMOS_SRC"
echo "数学语料源: $MATH_SRC"
echo "历史语料源: $HISTORY_SRC"
echo

# ---- 按前缀分类: 剥离 "前缀:标题：" 只保留正文 ----
: > corpus_poet.txt
: > corpus_lunyu.txt
: > corpus_chat.txt
: > corpus_cosmos.txt
: > corpus_math.txt
: > corpus_history.txt

n_poet=0; n_lunyu=0; n_chat=0

while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    key="${line%%：*}"                        # "XX：标题" 中的标题 (键)
    body="${line#*：}"                        # 正文
    case "$line" in
        论语-*|名言-*) echo "$body" >> corpus_lunyu.txt; n_lunyu=$((n_lunyu+1));;
        对话-*|字典-*) echo "$body" >> corpus_chat.txt;  n_chat=$((n_chat+1));;
        宇宙-*)        echo "$body" >> corpus_cosmos.txt;;
        *)             # 古诗双份: "标题=诗句" 问答对 + 纯诗句行 (保证诗句开头续写)
                         echo "$key=$body" >> corpus_poet.txt
                         echo "$body"      >> corpus_poet.txt
                         n_poet=$((n_poet+2));;
    esac
done < "$CORPUS_SRC"

# 宇宙语料: 疑问键 (含 什么/怎样/哪来/是谁/为什么/会怎样) 保留为 Q=A 问答对,
#          陈述键只保留正文 (忠实于原语料, 键即问题、正文即答案)
while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    key="${line#宇宙-}"; key="${key%%：*}"
    body="${line#*：}"
    if printf '%s' "$key" | grep -qE '什么|怎样|哪来|是谁|为什么|会怎样'; then
        echo "$key=$body" >> corpus_cosmos.txt
    else
        echo "$body" >> corpus_cosmos.txt
    fi
done < "$COSMOS_SRC"

n_cosmos=$(wc -l < corpus_cosmos.txt)

# 数学语料: 每行一段 (Q=A 问答对 / 纯陈述), 原样拷贝
while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    echo "$line" >> corpus_math.txt
done < "$MATH_SRC"

n_math=$(wc -l < corpus_math.txt)

# 历史语料: 每行一段 (Q=A 问答对), 原样拷贝
while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    echo "$line" >> corpus_history.txt
done < "$HISTORY_SRC"

n_history=$(wc -l < corpus_history.txt)

echo "拆分完成:"
printf "  corpus_poet.txt   (古诗)      %s 段\n" "$n_poet"
printf "  corpus_lunyu.txt  (论语+名言) %s 段\n" "$n_lunyu"
printf "  corpus_chat.txt   (对话+字典) %s 段\n" "$n_chat"
printf "  corpus_cosmos.txt (宇宙模型)  %s 段\n" "$n_cosmos"
printf "  corpus_math.txt   (数学知识)  %s 段\n" "$n_math"
printf "  corpus_history.txt(历史知识)  %s 段\n" "$n_history"
echo
echo "---- 各语料预览 (前 3 行) ----"
for f in corpus_poet.txt corpus_lunyu.txt corpus_chat.txt corpus_cosmos.txt corpus_math.txt corpus_history.txt; do
    echo "----- $f -----"
    head -n 3 "$f"
done
