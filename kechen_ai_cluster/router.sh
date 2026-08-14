#!/bin/bash
# ============================================================================
#  router.sh — 客尘AI集群v1.0 路由器 (关键词路由 -> 专家AI)
#
#  路由规则 (优先级从高到低):
#    宇宙/负一/能量/混沌/元素/循环/归源 -> cosmos  (宇宙专家)
#    诗/床/月/望/江                    -> poet    (古诗专家)
#    学/君子/论语/名言                 -> lunyu   (论语+名言专家)
#    其他                              -> chat    (对话专家)
#
#  用法:
#    ./router.sh "床前明月"    单次路由问答
#    ./router.sh               交互模式 (输入 quit 退出)
# ============================================================================
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

route_expert() {
    local input="$1"
    if printf '%s' "$input" | grep -qE '宇宙|负一|能量|混沌|元素|循环|归源'; then
        echo "cosmos"
    elif printf '%s' "$input" | grep -qE '诗|床|月|望|江'; then
        echo "poet"
    elif printf '%s' "$input" | grep -qE '学|君子|论语|名言'; then
        echo "lunyu"
    else
        echo "chat"
    fi
}

ask() {
    local input="$1" expert reason
    expert="$(route_expert "$input")"
    case "$expert" in
        cosmos) reason="命中关键词[宇宙/负一/能量/混沌/元素/循环/归源]";;
        poet)   reason="命中关键词[诗/床/月/望/江]";;
        lunyu)  reason="命中关键词[学/君子/论语/名言]";;
        *)      reason="未命中关键词 -> 默认聊天专家";;
    esac
    echo "【路由】输入: $input"
    echo "【路由】→ ai_brain_$expert  ($reason)"
    if [ -x "./ai_brain_$expert" ]; then
        echo "【$expert】$(./ai_brain_$expert -q "$input" 2>&1)"
    else
        echo "【错误】缺少专家程序: ai_brain_$expert"
    fi
    echo
}

if [ "$#" -ge 1 ]; then
    ask "$*"
else
    echo "===== 客尘AI集群v1.0 · 路由交互模式 ====="
    echo "输入内容自动路由到 4 个专家之一 (quit/exit 退出)"
    echo
    while true; do
        printf '你> '
        IFS= read -r line || break
        case "$line" in
            quit|exit|q|退出) echo "再见！"; break;;
            "") continue;;
        esac
        ask "$line"
    done
fi
