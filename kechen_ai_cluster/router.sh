#!/bin/bash
# ============================================================================
#  router.sh — 客尘AI集群v2.0 路由器 (混合架构: Fast快答层 -> 专家AI)
#
#  路由规则 (v2.0 混合架构, 两层):
#    第1层 FAST 快答层 (ai_brain_fast, 零训练零权重):
#       启动时加载 6 个语料文件建立索引, 用户提问 -> 语料相似度检索
#       -> 命中: 直接秒答 (知识问答不再等专家模型生成)
#    第2层 专家模型 (关键词路由 -> ai_brain_* 生成/续写):
#       加/减/乘/除/数学/等于/多少/几      -> math    (数学专家)
#       宇宙/负一/能量/混沌/元素/循环/归源 -> cosmos  (宇宙专家)
#       诗/床/月/望/江/壁                  -> poet    (古诗专家)
#       学/君子/论语/名言                 -> lunyu   (论语+名言专家)
#       朝代/历史/皇帝/明朝/唐朝/宋朝/清朝/
#       战争/公元/年/禹/治水               -> history (历史专家)
#       其他                              -> chat    (对话专家)
#
#  v2.0 变更: 知识问答 (语料内) 由快答层秒答; 快答未命中
#             (语料外/生成类) 才路由专家模型。
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
    if printf '%s' "$input" | grep -qE '加|减|乘|除|数学|等于|多少|几'; then
        echo "math"
    elif printf '%s' "$input" | grep -qE '宇宙|负一|能量|混沌|元素|循环|归源'; then
        echo "cosmos"
    elif printf '%s' "$input" | grep -qE '诗|床|月|望|江|壁'; then
        echo "poet"
    elif printf '%s' "$input" | grep -qE '学|君子|论语|名言'; then
        echo "lunyu"
    elif printf '%s' "$input" | grep -qE '朝代|历史|皇帝|明朝|唐朝|宋朝|清朝|战争|公元|年|禹|治水'; then
        echo "history"
    else
        echo "chat"
    fi
}

ask() {
    local input="$1" ans expert reason t0 t1 fast_ms rc
    # ---- 第1层: FAST 快答层 (知识问答秒答, 零训练零权重) ----
    if [ -x "./ai_brain_fast" ]; then
        t0="$(date +%s%N)"
        ans="$(./ai_brain_fast -q "$input" 2>/dev/null)"
        rc=$?
        t1="$(date +%s%N)"
        fast_ms=$(( (t1 - t0) / 1000000 ))
        if [ "$rc" -eq 0 ] && [ -n "$ans" ]; then
            echo "【路由】输入: $input"
            echo "【路由】→ FAST 快答层命中 (相似度检索·${fast_ms}ms·零训练零权重)"
            echo "【fast】$ans"
            echo
            return
        fi
    else
        fast_ms=0
    fi
    # ---- 第2层: 专家模型 (快答未命中 -> 关键词路由 -> 生成/续写) ----
    expert="$(route_expert "$input")"
    case "$expert" in
        math)    reason="命中关键词[加/减/乘/除/数学/等于/多少/几]";;
        cosmos)  reason="命中关键词[宇宙/负一/能量/混沌/元素/循环/归源]";;
        poet)    reason="命中关键词[诗/床/月/望/江/壁]";;
        lunyu)   reason="命中关键词[学/君子/论语/名言]";;
        history) reason="命中关键词[朝代/历史/皇帝/明朝/唐朝/宋朝/清朝/战争/公元/年/禹/治水]";;
        *)       reason="未命中关键词 -> 默认聊天专家";;
    esac
    echo "【路由】输入: $input"
    echo "【路由】→ FAST未命中(${fast_ms}ms) -> ai_brain_$expert  ($reason)"
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
    echo "===== 客尘AI集群v2.0 · 路由交互模式 (混合架构) ====="
    echo "先走 FAST 快答层(知识问答秒答) → 未命中才路由 6 个专家之一 (quit/exit 退出)"
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
