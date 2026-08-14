# 客尘AI (Kechen AI)

> 从单细胞开始的AI进化项目 —— An AI evolution project starting from a single cell.

**作者：王铠宁（客尘）** — 14岁学生，梦想创造一个开源、便宜、人人都用得起的AI。

## 🌟 梦想

DeepSeek曾让AI普惠，但涨价让很多人用不起。我的梦想是：
- **用C++写AI**（推理引擎C++原生，快+省）
- **开源**（让全世界的开发者一起进化）
- **便宜**（回到"人人都能用得起"的初心）

## 🧬 进化路线（全部真实运行过）

| 版本 | 能力 | 状态 |
|---|---|---|
| **v0.1** | 多层感知机（MLP）学会 XOR 逻辑 | ✅ 100%正确 |
| **v0.2** | 手写数字识别（8x8点阵+噪声模拟） | ✅ 干净100% / 含噪78.3% |
| **v0.3** | 交互版：用户手绘数字实时识别 | ✅ 训练后手动输入识别 |

## 📁 文件

- `ai_brain_v0.cpp` — v0.1：MLP学习XOR（157行，2→2→1，反向传播）
- `ai_brain_v1.cpp` — v0.2：数字识别（453行，64→256→10，softmax+SGD+动量）
- `ai_brain_v1i.cpp` — v0.3：交互版（训练后手绘识别）

## 🔧 编译运行

```bash
# v0.1 - 学习XOR
g++ -O2 -std=c++17 -o ai_brain_v0 ai_brain_v0.cpp && ./ai_brain_v0

# v0.2 - 数字识别
g++ -O2 -std=c++17 -o ai_brain_v1 ai_brain_v1.cpp && ./ai_brain_v1

# v0.3 - 交互手绘识别
g++ -O2 -std=c++17 -o ai_brain_v1i ai_brain_v1i.cpp && ./ai_brain_v1i
```

## 📜 协议

MIT License — 随便用，随便改，开源万岁。

## 🚀 下一步

v1.0：客尘引擎（GGUF推理，llama.cpp路线）——真正的"大语言模型大脑"。

---

*"最纯粹的能量，不可逆地变成-1号元素——宇宙由此诞生。我的AI，也从一个神经元开始。"* —— 客尘
