#!/bin/bash
# ============================================================================
#  think.sh — 客尘AI本地思考模块 v4
#  用法: bash think.sh "问题" "搜索结果文件"
#  v4: 改用 llama-simple 直出生成 (本机 llama-cli 为 server 风格、不输出到 stdout,
#      导致 v3 静默失败); 输出先剥离"提示词回声"再过滤杂项
# ============================================================================
Q="$1"
RESULT_FILE="${2:-$HOME/ke_search_result.txt}"

if [ -z "$Q" ]; then
    echo "用法: bash think.sh \"问题\" [搜索结果文件]"
    exit 1
fi

RESULT=$(head -c 1500 "$RESULT_FILE" 2>/dev/null)
MODEL="$HOME/ds_qwen3b_q8.gguf"

PROMPT="你是客尘AI，一个自主思考的中文AI。根据以下搜索资料，用自己的话回答问题（简洁准确）：
资料：
$RESULT
问题：$Q
回答："

# 本地推理: llama-simple 提示词作位置参数, 输出 = 提示词回声 + 生成文本
# 1) python: 剥离提示词前缀 (精确字符串, 非正则) + 按句去重(防止默认采样循环重复) + 限长
# 2) 去空行  3) 尾部几行 (答案)
if command -v llama-simple >/dev/null 2>&1; then
    llama-simple -m "$MODEL" -n 250 "$PROMPT" 2>/dev/null \
      | python3 -c "
import sys, re
p = sys.argv[1]
out = sys.stdin.read()
if out.startswith(p):
    out = out[len(p):]
# 按句切分 -> 精确去重(保留顺序) -> 合并 -> 限长
parts = re.split(r'(?<=[。！？!?；;])', out)
seen, kept = set(), []
for seg in parts:
    s = seg.strip()
    if not s:
        continue
    if s not in seen:
        seen.add(s)
        kept.append(s)
print(''.join(kept)[:600])
" "$PROMPT" \
      | grep -vE '^\s*$' \
      | tail -6
else
    # 回退: llama-cli (-p 风格, 仅当 llama-simple 不存在时)
    llama-cli -m "$MODEL" -p "$PROMPT" -n 250 --temp 0.5 2>/dev/null \
      | grep -vE 'Loading|^▄|^██|build |model |ftype|modalities|available|/exit|/regen|/clear|/read|/glob|^>|Prompt:|Generation:|llama_' \
      | grep -vE '^\s*$' \
      | tail -6
fi
