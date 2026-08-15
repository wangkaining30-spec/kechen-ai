#!/bin/bash
# ============================================================================
#  router.sh — 客尘AI集群v3.2 路由器 (混合架构: Fast快答层优先 -> 专家AI + 数学三副本投票)
#
#  路由规则 (v3.2 混合架构, 快答层确定性优先):
#    数学问题 (fast优先, 三副本投票兜底):
#       1. 第一步先查 ai_brain_fast (-q 模式): 命中 -> 直接输出 (快答层确定性优先)
#       2. fast 未命中 -> ai_brain_math_a/b/c 三个数学副本投票 (语料均分3份各自训练)
#          (答案归一化函数: 剥离问题回声, 去除 答案是/所以/等于/标点/空格 等杂质,
#           提取核心数字或关键词; 三副本归一化后比对: ≥2 个一致 -> 多数答案)
#       3. 投票结果再校验: 与 fast 比对
#          - fast 有且与多数不一致 -> 输出 fast 真值
#          - fast 没有 -> 多数一致输出
#          - 无法多数 -> 提示联网搜索 (bash search.sh "问题")
#    非数学问题 (v2.0 原有, 两层):
#       第1层 FAST 快答层 (ai_brain_fast, 零训练零权重):
#          启动时加载 6 个语料文件建立索引, 用户提问 -> 语料相似度检索
#          -> 命中: 直接秒答 (知识问答不再等专家模型生成)
#       第2层 专家模型 (关键词路由 -> ai_brain_* 生成/续写):
#          加/减/乘/除/数学/等于/多少/几      -> math    (数学专家→fast优先,三副本投票兜底)
#          宇宙/负一/能量/混沌/元素/循环/归源 -> cosmos  (宇宙专家)
#          诗/床/月/望/江/壁                  -> poet    (古诗专家)
#          学/君子/论语/名言                 -> lunyu   (论语+名言专家)
#          朝代/历史/皇帝/明朝/唐朝/宋朝/清朝/
#          战争/公元/年/禹/治水               -> history (历史专家)
#          其他                              -> chat    (对话专家)
#
#  v3.2 变更: 数学问题改为"fast优先" (快答层确定性优先):
#             ①第一步先查 ai_brain_fast (-q 模式): 命中直接输出
#               (语料内已收录的数学问答不再等三副本生成/投票);
#             ②fast 未命中才走三副本投票 (归一化中继);
#             ③投票结果再与 fast 比对: fast 有且不一致 -> 输出 fast 真值;
#               fast 没有 -> 多数一致输出; 无法多数 -> 提示搜索。
#             原 v3.1 的 fast 校准 / trusted_math 可信副本机制随 fast优先 移除。
#  v3.0 变更: 数学问题改走三副本投票 (ai_brain_math_a/b/c, 由
#             bash split_train_math.sh 均分语料并训练)。
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

# ========= v3.2: 数学问题 fast优先 (快答层确定性优先 -> 三副本投票兜底) =========
#   ① 第一步先查 ai_brain_fast (-q 模式): 命中 -> 直接输出 (快答层确定性优先)
#   ② fast 未命中 -> 三副本投票 (归一化中继处理器):
#      - 答案归一化函数: 剥离问题回声, 去除 答案是/所以/等于/标点/空格 等杂质,
#        提取核心数字或关键词
#      - 三副本归一化后比对: ≥2 个一致 -> 多数答案
#   ③ 投票结果再校验 (与 fast 比对):
#      - fast 有且与多数不一致 -> 输出 fast 真值
#      - fast 没有 -> 多数一致输出
#      - 无法多数 -> 提示搜索

# 答案归一化: 返回核心数字(优先, 频率+位置择优)或关键词
normalize_answer() {
    # $1 = 原始答案, $2 = 原始问题 (剥离回声, 可空)
    local raw="$1" q="$2" s nums best
    s="$raw"
    # ① 剥离问题回声 (模型常把问题原样复述在答案开头; 引号内字面匹配,
    #    避免问题里的 ? * [ ] 被当 glob 通配)
    if [ -n "$q" ] && [ -n "$s" ]; then
        s="${s//"$q"/}"
    fi
    # ② 提取核心数字: 在去标点前提取 (避免 16，16 被粘连成 1616),
    #    支持小数/分数; 按出现频率择优, 同频取最后出现者 (答案常在句末)
    nums="$(printf '%s' "$s" | grep -oE '[0-9]+/[0-9]+|[0-9]+([.][0-9]+)?' 2>/dev/null | head -n 64)"
    if [ -n "$nums" ]; then
        best="$(printf '%s\n' $nums | awk '{ cnt[$1]++; last[$1]=NR } END { b=""; mx=0; lp=-1; for (k in cnt) if (cnt[k]>mx || (cnt[k]==mx && last[k]>lp)) { b=k; mx=cnt[k]; lp=last[k] } print b }')"
        # 数字规范化: 去前导零/尾零 (02->2, 2.0->2)
        best="$(printf '%s' "$best" | sed -E 's/^0*([0-9])/\1/; s/\.0+$//; s/(\.[0-9]*[1-9])0+$/\1/')"
        [ -z "$best" ] && best="0"
        echo "$best"
        return
    fi
    # ③ 无数字 -> 去空格/标点/杂质词后取关键词
    #    (用 bash 参数展开逐字删标点: 多字节安全。不能用 tr -d 删中文标点 —
    #     tr 按字节删除会误删中文字符尾段; sed 字符类里 [ ] { } 转义也易出错)
    s="$(printf '%s' "$s" | tr -d '[:space:]')"
    for p in '，' '。' '！' '？' '；' '：' '、' ',' '.' '!' '?' ';' ':' '(' ')' '（' '）' '[' ']' '【' '】' '{' '}' '《' '》' '<' '>' '「' '」' '『' '』' '·' '—' '…' '"' "'" '`' '~' '@' '#' '￥' '^' '|' '\'; do
        # 注意: ${s//"$p"/} 里的引号让 $p 按字面匹配 (否则 ? [ ] \ 会被当 glob 通配)
        s="${s//"$p"/}"
    done
    s="$(printf '%s' "$s" | sed -E 's/答案是|答案就是|答案为|结果就是|结果为|约等于|所以|因此|综上|等于|就是|答案|得数|最后|最终|应该|应当|大概|大约|可能|请问|算式|计算|分别是|是|为|了|的//g')"
    echo "$s" | head -c 24
}

ask_math() {
    local input="$1" a1 a2 a3 n1 n2 n3 nf fast_ans fast_rc majority maj_rep
    echo "【路由】输入: $input"
    # ---- ① FAST 快答层优先: 数学问题第一步先查 ai_brain_fast (-q), 命中直接输出 ----
    #     (快答层确定性优先: 语料内已收录的数学问答不再等三副本生成/投票)
    fast_ans=""; fast_rc=1
    if [ -x "./ai_brain_fast" ]; then
        fast_ans="$(./ai_brain_fast -q "$input" 2>/dev/null)"
        fast_rc=$?
        if [ "$fast_rc" -eq 0 ] && [ -n "$fast_ans" ]; then
            echo "【路由】→ FAST 快答层命中 (数学·快答确定性优先)"
            echo "【fast】$fast_ans"
            echo
            return
        fi
        echo "【路由】→ FAST 快答层未命中 -> 数学三副本投票"
    else
        echo "【路由】→ ai_brain_fast 不可用 -> 数学三副本投票"
    fi
    # ---- ② fast 未命中 -> 三副本投票 (归一化中继处理器) ----
    if [ -x "./ai_brain_math_a" ] && [ -x "./ai_brain_math_b" ] && [ -x "./ai_brain_math_c" ]; then
        a1="$(./ai_brain_math_a -q "$input" 2>/dev/null)"
        a2="$(./ai_brain_math_b -q "$input" 2>/dev/null)"
        a3="$(./ai_brain_math_c -q "$input" 2>/dev/null)"
        echo "【math_a】$a1"
        echo "【math_b】$a2"
        echo "【math_c】$a3"
        # 三副本答案归一化
        n1="$(normalize_answer "$a1" "$input")"
        n2="$(normalize_answer "$a2" "$input")"
        n3="$(normalize_answer "$a3" "$input")"
        echo "【归一化】a→[$n1] | b→[$n2] | c→[$n3]"
        # 归一化后比对: ≥2 个一致 -> 多数答案
        majority=""; maj_rep=""
        if [ -n "$n1" ] && [ -n "$n2" ] && [ "$n1" = "$n2" ]; then
            majority="$n1"; maj_rep="a/b"
        elif [ -n "$n1" ] && [ -n "$n3" ] && [ "$n1" = "$n3" ]; then
            majority="$n1"; maj_rep="a/c"
        elif [ -n "$n2" ] && [ -n "$n3" ] && [ "$n2" = "$n3" ]; then
            majority="$n2"; maj_rep="b/c"
        fi
        if [ -n "$majority" ]; then
            echo "【投票】✅ 归一化后 ≥2 副本一致 ($maj_rep) -> 多数答案 [$majority]"
        else
            echo "【投票】⚠️ 归一化后无多数一致"
        fi
        # ---- ③ 投票结果再校验: 与 fast 比对 ----
        #     (第①步命中时已直接返回, 此处 fast_ans 通常为空; 完整保留
        #      fast 有结果的决策分支, 保证"fast 有且不一致 -> fast 真值")
        if [ "$fast_rc" -eq 0 ] && [ -n "$fast_ans" ]; then
            nf="$(normalize_answer "$fast_ans" "$input")"
            echo "【fast】$fast_ans (归一化: [$nf])"
            if [ -n "$nf" ] && [ -n "$majority" ] && [ "$nf" != "$majority" ]; then
                # fast 有且与多数不一致 -> fast 是真值 (快答层确定性优先)
                echo "【校验】⚠️ fast [$nf] 与多数 [$majority] 不一致 -> 输出 fast 真值"
                echo "【math】$nf (fast 真值)"
                echo
                return
            fi
            # fast 有且与多数一致 (或 fast 无归一化核心) -> 输出多数
            if [ -n "$majority" ]; then
                echo "【校验】✅ fast 与多数一致 -> 输出多数答案"
                echo "【math】$majority"
                echo
                return
            fi
            echo "【校验】三副本无法多数, 采用 fast 答案"
            echo "【math】$nf"
            echo
            return
        fi
        # fast 没有结果 -> 多数一致输出; 无法多数 -> 走函数末尾搜索兜底
        if [ -n "$majority" ]; then
            echo "【校验】fast 无结果 -> 输出多数答案"
            echo "【math】$majority"
            echo
            return
        fi
    else
        echo "【路由】→ 数学副本未编译/未训练 (可先运行 bash split_train_math.sh)"
    fi
    # ---- ④ 兜底: 三副本无法多数且 fast 无结果 -> 提示联网搜索 ----
    echo "【提示】本地未命中 (三副本无法多数且 fast 无结果), 建议联网搜索: bash search.sh \"$input\""
    echo
}

ask() {
    local input="$1" ans expert reason t0 t1 fast_ms rc
    # ---- v3.2: 数学问题 -> fast优先 (先查 ai_brain_fast 快答层, 命中直接输出;
    #      未命中才三副本投票, 投票结果再与 fast 比对校验) ----
    expert="$(route_expert "$input")"
    if [ "$expert" = "math" ]; then
        ask_math "$input"
        return
    fi
    # ---- 第1层: FAST 快答层 (非数学: 知识问答秒答, 零训练零权重) ----
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
    # ---- 第2层: 专家模型 (快答未命中 -> 关键词路由 -> 生成/续写; math 已由 ask_math 处理) ----
    case "$expert" in
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
    echo "===== 客尘AI集群v3.2 · 路由交互模式 (混合架构) ====="
    echo "数学问题 → FAST 快答层优先(命中秒答) → 未命中才三副本投票; 其他 → FAST 快答层秒答 → 未命中才路由专家 (quit/exit 退出)"
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
