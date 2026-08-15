# 客尘AI集群 v2.0 混合架构 (Kechen AI Cluster v2.0 Hybrid)

## v2.0 混合架构（本次：快答层 + 专家层）

```
用户提问
   │
   ▼
┌─────────────────────────────┐
│ 第1层  FAST 快答层           │   ai_brain_fast (C++)
│  ai_brain_fast               │   零训练 / 零权重 / 纯查表
│  启动时加载7个语料文件建索引  │   字符+双字倒排索引
│  提问 → 语料相似度检索        │   相似度=0.6×字符覆盖率+0.4×双字覆盖率
│  ┌──────────────┐            │   命中阈值 0.60
│  │ 命中 → 直接秒答 │          │   ~毫秒级 (实测 30~40ms 含进程启动)
│  └──────────────┘            │
  提问前先查 ~/.fast_alias.txt 表达归一化 (变体→标准问法) 再检索
└──────────┬──────────────────┘
           │ 未命中 (语料外/生成类)
           ▼
┌─────────────────────────────┐
│ 第2层  专家模型              │   ai_brain_{poet,lunyu,chat,cosmos,
│  关键词路由 → ai_brain_*     │   math,history} (v0.8 迷你Transformer)
│  生成 / 续写                 │   训练后权重自动保存、启动秒加载
└─────────────────────────────┘
```

测试对比（全部通过）：

| 输入 | 语料内? | 路由结果 | 耗时 |
|---|---|---|---|
| 床前明月 | ✅ | FAST 快答层命中 → 全诗 | ~30ms 秒答 |
| 负一元素是什么 | ✅ | FAST 快答层命中 → 负一元素是奇点物质… | ~35ms 秒答 |
| 6×7等于几 | ✅ | FAST 快答层命中 → 42 | ~40ms 秒答 |
| 大禹治水 | ✅ | FAST 快答层命中 → 三过家门而不入 | ~39ms 秒答 |
| 给我讲讲哲学（没学过） | ❌ | FAST 未命中 → 关键词[学]→ lunyu 专家生成 | 专家模型 |

快答层用法：

```bash
./ai_brain_fast              # 交互模式
./ai_brain_fast -q "床前明月" # 单次问答: 命中输出答案并退出0; 未命中退出1
./ai_brain_fast --stats      # 索引统计 (7语料/647条/1412字符)
./ai_brain_fast --demo       # 内置演示 (4题秒答 + 1题未命中)
```

路由器（router.sh）：先走 `./ai_brain_fast -q` 查表，退出码 0 且非空 → 秒答；
否则关键词路由专家模型。编译：`clang++ -O2 -std=c++17 ai_brain_fast.cpp -o ai_brain_fast`

表达归一化（v2.0）：提问前查 `~/.fast_alias.txt`（格式：`变体1|变体2|...=标准问法`），
整句命中变体则映射为标准问法后再检索，如「在吗→你好」「3乘3→3×3」「光速多少→光速是多少」。

---

# 客尘AI集群 v1.2 (Kechen AI Cluster v1.2)
# 客尘AI集群 v1.2 (Kechen AI Cluster v1.2)

6 个专家 AI + 1 个路由器，基于 v0.8 架构
（迷你 Transformer：字符 Embedding → Elman RNN → 可学习位置编码
 → 2 层多头自注意力块(LayerNorm+因果注意力+FFN/GELU) → Softmax 预测下一字符，
 优化器 Adam + 梯度裁剪 + BPTT，训练后权重自动保存、再次启动秒加载）。

## v1.2 全面进化（本次）
| 专家 | 语料扩充 (+50%) | 旧段数 | 新段数 |
|---|---|---|---|
| poet   | +10 首新诗（题西林壁/泊船瓜洲/小池/晓出净慈寺/饮湖上初晴后雨/惠崇春江晚景/示儿/秋夜将晓出篱门迎凉有感/村居/所见） | 46 | 66 |
| lunyu  | +5 句（君子和而不同/欲速则不达/工欲善其事必先利其器/知之为知之不知为不知/逝者如斯夫不舍昼夜） | 22 | 27 |
| chat   | +10 条对话（你会背多少首诗/你觉得宇宙大吗/你喜欢什么/你累吗/教我背诗吧/你是男生女生/你怕什么/你最喜欢什么数字/你能陪我聊天吗/你相信梦想吗） | 37 | 47 |
| math   | +10 题（12+8/50-15/6×7/100÷4/1000-999/2×2×2/正方形周长/长方形面积/一小时多少秒/一周几天） | 18 | 28 |
| history| +10 条（炎黄子孙的由来/大禹治水/商鞅变法/汉武帝/贞观之治/岳飞/成吉思汗/郑成功/辛亥革命/万里长城有多长） | 20 | 30 |
| cosmos | +10 条（第1条能量底基律/第24条双层转化律/第25条暗影投影律/第26条开天观测律/第47项浑沌律/第49项三面能量律/归源衰变是什么/奇点相变是什么/引力元素是什么/观测者是谁） | 36 | 46 |

全部 6 专家**重训 300 轮**（Adam，lr 0.01 每 100 轮减半，梯度裁剪，loss 阈值 0.02 提前停止），
权重自动保存到 `ai_brain_<专家>_weights.bin`（再次启动秒加载）。

## 语料拆分（split_corpus.sh）
| 语料源 | 前缀规则 | 专家语料 | 段数 |
|---|---|---|---|
| kechen_corpus.txt | 无前缀行 = 古诗（双份：标题=诗句 + 纯诗句） | corpus_poet.txt | 66 |
| kechen_corpus.txt | 论语- / 名言- | corpus_lunyu.txt | 27 |
| kechen_corpus.txt | 对话- / 字典- | corpus_chat.txt | 47 |
| kechen_cosmos.txt | 宇宙-（疑问键 **或编号律条(含 律/项)** → Q=A 问答对） | corpus_cosmos.txt | 46 |
| kechen_math.txt   | 数学知识（每行一段：Q=A 问答对 / 纯陈述，原样拷贝） | corpus_math.txt | 28 |
| kechen_history.txt| 历史知识（每行一段：Q=A 问答对，原样拷贝） | corpus_history.txt | 30 |

> 古诗保留「标题=诗句」问答对（静夜思=床前明月光…），同时保留纯诗句行。
>
> 宇宙语料中的疑问键（如「负一是什么」「暗物质是什么」）以及编号律条
> （如「第49项三面能量律」「第24条双层转化律」）按 Q=A 问答对保留，
> 使宇宙专家能直接回答「什么/是谁/怎样」类问题与「第X条/第X项」律条。

## 专家（ai_brain_v08.cpp 同一源码，-D 宏注入身份；hidden=128，2 层 Transformer 块）
| 专家 | 语料 | 权重文件 | 演示前缀 (DEMOS) |
|---|---|---|---|
| ai_brain_poet   | corpus_poet.txt    | ai_brain_poet_weights.bin   | 床前明月\|题西林壁\|泊船瓜洲\|小池 |
| ai_brain_lunyu  | corpus_lunyu.txt   | ai_brain_lunyu_weights.bin  | 学而时习之\|君子和而不同\|欲速则不达\|逝者如斯夫 |
| ai_brain_chat   | corpus_chat.txt    | ai_brain_chat_weights.bin   | 你好\|你喜欢什么\|你累吗\|你怕什么 |
| ai_brain_cosmos | corpus_cosmos.txt  | ai_brain_cosmos_weights.bin | 负一元素是\|第1条能量底基律\|第49项三面能量律\|引力元素是什么 |
| ai_brain_math   | corpus_math.txt    | ai_brain_math_weights.bin   | 1+1等于几\|6×7等于几\|什么是加法\|正方形周长怎么算 |
| ai_brain_history| corpus_history.txt | ai_brain_history_weights.bin| 中国第一个朝代是什么\|大禹治水\|万里长城有多长\|历史是什么 |

用法：`./ai_brain_poet`（首次训练并保存权重，再次加载秒启动）；
`./ai_brain_poet -q 床前明月`（路由器专用单次问答）。

## 路由器（router.sh）
关键词路由（优先级从高到低）：
- 加 / 减 / 乘 / 除 / 数学 / 等于 / 多少 / 几 → math
- 宇宙 / 负一 / 能量 / 混沌 / 元素 / 循环 / 归源 → cosmos
- 诗 / 床 / 月 / 望 / 江 / **壁** → poet
- 学 / 君子 / 论语 / 名言 → lunyu
- 朝代 / 历史 / 皇帝 / 明朝 / 唐朝 / 宋朝 / 清朝 / 战争 / 公元 / 年 / **禹** / **治水** → history
- 其他 → chat

v1.2 新增路由词：**壁**（题西林壁→poet）、**禹/治水**（大禹治水→history）。

用法：`./router.sh "床前明月"` 单次问答；`./router.sh` 交互模式。

## 自主问答 + 记忆闭环（ask.sh）
`bash ask.sh "问题"`：快答层查表 → 未命中：数学问题本地直算（python3，精确秒答、无需联网）/
其他自主搜索(Bing) + 本地Qwen3B思考整理 → 回答并记忆。

- 学到的问答追加到 `corpus_learn.txt`（格式：`问题=答案`，单行、不含 `=`；`#` 开头为注释行，
  `ai_brain_fast` 自动跳过）
- `ai_brain_fast` 启动时自动扫描全部 `corpus_*.txt`（含 `corpus_learn.txt`），因此**下次提问直接
  快答层命中（秒答）**，无需重编译、无需重训练
- 重复提问自动去重（已记住的问题不再重复追加）
- 快答层带 key 相关性门槛：学了 `2的10次方=1024` 后，`2的16次方` 不会误命中 1024，而是正确直算并
  单独记住（防近相似问题串答案）

## 回归测试（run_demo.sh）
新知识：题西林壁→poet；君子和而不同→lunyu；6×7等于几→math；
大禹治水→history；第49项三面能量律→cosmos；你喜欢什么→chat
旧知识回归：床前明月→poet；你好→chat；负一元素→cosmos；1+1等于几→math

```bash
./run_demo.sh    # 一键跑全部路由回归
```

## 重新构建
```bash
./split_corpus.sh   # 拆分语料（6 份 corpus_*.txt）
rm -f ai_brain_*_weights.bin   # 若需强制重训
./build_all.sh      # 编译 6 个专家
./run_demo.sh       # 回归测试
./router.sh         # 开始路由
```
