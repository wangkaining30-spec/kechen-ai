# 客尘AI集群 v1.0 (Kechen AI Cluster v1.0)

5 个专家 AI + 1 个路由器，基于 v0.8 架构
（迷你 Transformer：字符 Embedding → Elman RNN → 可学习位置编码
 → 2 层多头自注意力块(LayerNorm+因果注意力+FFN/GELU) → Softmax 预测下一字符，
 优化器 Adam + 梯度裁剪 + BPTT，训练后权重自动保存、再次启动秒加载）。

## 语料拆分（split_corpus.sh）
| 语料源 | 前缀规则 | 专家语料 | 段数 |
|---|---|---|---|
| kechen_corpus.txt | 无前缀行 = 古诗（双份：标题=诗句 + 纯诗句） | corpus_poet.txt | 46 |
| kechen_corpus.txt | 论语- / 名言- | corpus_lunyu.txt | 22 |
| kechen_corpus.txt | 对话- / 字典- | corpus_chat.txt | 37 |
| kechen_cosmos.txt | 宇宙-（全部；疑问键→Q=A 问答对） | corpus_cosmos.txt | 36 |
| kechen_math.txt   | 数学知识（每行一段：Q=A 问答对 / 纯陈述，原样拷贝） | corpus_math.txt | 18 |

> 古诗保留「标题=诗句」问答对（静夜思=床前明月光…），同时保留纯诗句行。
>
> 宇宙语料中的疑问键（如「负一是什么」「暗物质是什么」）按 Q=A 问答对保留：
> 「负一是什么=负一元素是奇点物质，宇宙的永恒底料，一切物质终将归源回它。」
> 使宇宙专家能直接回答「什么/是谁/怎样」类问题。

## 专家（ai_brain_v08.cpp 同一源码，-D 宏注入身份；hidden=128，2 层 Transformer 块）
| 专家 | 语料 | 权重文件 | 演示前缀 (DEMOS) |
|---|---|---|---|
| ai_brain_poet   | corpus_poet.txt    | ai_brain_poet_weights.bin   | 床前明月\|白日依山\|秦时明月\|春眠不觉 |
| ai_brain_lunyu  | corpus_lunyu.txt   | ai_brain_lunyu_weights.bin  | 学而时习之\|温故而知\|君子坦荡荡\|见贤思齐 |
| ai_brain_chat   | corpus_chat.txt    | ai_brain_chat_weights.bin   | 你好\|你叫什么\|再见\|你会什么 |
| ai_brain_cosmos | corpus_cosmos.txt  | ai_brain_cosmos_weights.bin | 负一元素是\|宇宙的本原\|归源衰变\|引力元素 |
| ai_brain_math   | corpus_math.txt    | ai_brain_math_weights.bin   | 1+1等于几\|3×3等于几\|什么是加法\|圆周率 |

用法：`./ai_brain_poet`（首次训练并保存权重，再次加载秒启动）；
`./ai_brain_poet -q 床前明月`（路由器专用单次问答）。

## 路由器（router.sh）
关键词路由（优先级从高到低）：
- 加 / 减 / 乘 / 除 / 数学 / 等于 / 多少 / 几 → math
- 宇宙 / 负一 / 能量 / 混沌 / 元素 / 循环 / 归源 → cosmos
- 诗 / 床 / 月 / 望 / 江 → poet
- 学 / 君子 / 论语 / 名言 → lunyu
- 其他 → chat

用法：`./router.sh "床前明月"` 单次问答；`./router.sh` 交互模式。

## 重新构建
```bash
./split_corpus.sh   # 拆分语料（5 份 corpus_*.txt）
./build_all.sh      # 编译 5 个专家
./router.sh         # 开始路由
```
