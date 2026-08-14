// ============================================================
// ai_brain_v1.cpp —— 手写数字识别神经网络（MNIST 风格，8x8 简化版）
//
// 针对 Termux 环境的考虑：
//   * 不下载 MNIST 数据集（在手机上网络/存储不便），而是内置
//     0~9 的 8x8 点阵数字字形，并在训练/测试时随机叠加
//     像素翻转噪声 + 随机平移，模拟手写书写的自然变化。
//
// 网络结构：
//   输入层(64 像素) -> 隐藏层(32, ReLU) -> 输出层(10, Softmax)
//
// 训练：
//   损失 = 交叉熵(Cross Entropy) + L2 正则
//   优化 = 小批量随机梯度下降(SGD)，学习率按 epoch 衰减
//
// 编译（Termux）：
//   g++ -O2 -std=c++17 -o ai_brain_v1 ai_brain_v1.cpp
// 运行：
//   ./ai_brain_v1
// ============================================================

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ------------------------------------------------------------
// 1. 内置 8x8 点阵数字字形（'#' = 笔画，'.' = 空白）
// ------------------------------------------------------------
struct Glyph {
    int label;
    std::string rows[8];
};

static const std::vector<Glyph> kGlyphs = {
    {0, {".#####.",
         "#.....#",
         "#.....#",
         "#.....#",
         "#.....#",
         "#.....#",
         "#.....#",
         ".#####."}},
    {1, {"...#...",
         "..##...",
         "...#...",
         "...#...",
         "...#...",
         "...#...",
         "...#...",
         ".#####."}},
    {2, {".#####.",
         "#.....#",
         "......#",
         ".....#.",
         "....#..",
         "...#...",
         "..#....",
         "#######"}},
    {3, {".#####.",
         "#.....#",
         "......#",
         "..####.",
         "......#",
         "......#",
         "#.....#",
         ".#####."}},
    {4, {"....#..",
         "...##..",
         "..#.#..",
         ".#..#..",
         "#...#..",
         "#######",
         "....#..",
         "....#.."}},
    {5, {"#######",
         "#......",
         "#......",
         "######.",
         "......#",
         "......#",
         "#.....#",
         ".#####."}},
    {6, {"..####.",
         ".#.....",
         "#......",
         "#.####.",
         "#.....#",
         "#.....#",
         "#.....#",
         ".#####."}},
    {7, {"#######",
         "......#",
         ".....#.",
         "....#..",
         "...#...",
         "..#....",
         ".#.....",
         ".#....."}},
    {8, {".#####.",
         "#.....#",
         "#.....#",
         ".#####.",
         "#.....#",
         "#.....#",
         "#.....#",
         ".#####."}},
    {9, {".#####.",
         "#.....#",
         "#.....#",
         ".######",
         "......#",
         "......#",
         ".#....#",
         "..####."}},
};

// 取字形像素
inline bool pix(const Glyph& g, int r, int c) { return g.rows[r][c] == '#'; }

// ------------------------------------------------------------
// 2. 数据生成：字形 + 随机平移 + 随机像素翻转噪声
// ------------------------------------------------------------
struct Sample {
    std::vector<float> x;  // 64 维输入（0.0 / 1.0）
    int label;
};

// perDigit: 每个数字的样本数
// noiseMin/noiseMax: 每个样本的像素翻转概率在该区间内随机选取
// withShift: 是否随机平移 ±1 像素
std::vector<Sample> genData(const std::vector<Glyph>& glyphs, int perDigit,
                            float noiseMin, float noiseMax, bool withShift,
                            std::mt19937& rng) {
    std::vector<Sample> data;
    data.reserve(glyphs.size() * perDigit);

    std::uniform_int_distribution<int> dShift(-1, 1);
    std::uniform_real_distribution<float> dNoise(noiseMin, noiseMax);
    std::uniform_real_distribution<float> d01(0.0f, 1.0f);

    for (const Glyph& g : glyphs) {
        for (int n = 0; n < perDigit; ++n) {
            int sx = 0, sy = 0;
            if (withShift) { sx = dShift(rng); sy = dShift(rng); }
            float np = dNoise(rng);   // 本样本的噪声强度

            Sample s;
            s.x.resize(64);
            s.label = g.label;
            int k = 0;
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c, ++k) {
                    int sr = r - sy, sc = c - sx;   // 平移（裁剪出界）
                    float v = (sr >= 0 && sr < 8 && sc >= 0 && sc < 8 && pix(g, sr, sc))
                                  ? 1.0f
                                  : 0.0f;
                    if (d01(rng) < np) v = 1.0f - v;  // 翻转噪声
                    s.x[k] = v;
                }
            }
            data.push_back(std::move(s));
        }
    }
    return data;
}

// ------------------------------------------------------------
// 3. 三层全连接网络：64 -> 32(ReLU) -> 10(Softmax)
// ------------------------------------------------------------
class MLP {
public:
    int nIn, nHid, nOut;
    std::vector<float> W1, b1;   // 输入->隐藏：W1[nHid][nIn]
    std::vector<float> W2, b2;   // 隐藏->输出：W2[nOut][nHid]
    std::vector<float> vW1, vb1, vW2, vb2;  // 动量缓存

    MLP(int ni, int nh, int no, std::mt19937& rng)
        : nIn(ni), nHid(nh), nOut(no),
          W1(nh * ni), b1(nh, 0.0f),
          W2(no * nh), b2(no, 0.0f),
          vW1(nh * ni, 0.0f), vb1(nh, 0.0f),
          vW2(no * nh, 0.0f), vb2(no, 0.0f) {
        // He 初始化（适配 ReLU）
        std::normal_distribution<float> nd(0.0f, 1.0f);
        float s1 = std::sqrt(2.0f / ni);
        for (auto& w : W1) w = nd(rng) * s1;
        float s2 = std::sqrt(2.0f / nh);
        for (auto& w : W2) w = nd(rng) * s2;
    }

    // 前向传播，返回 softmax 概率向量；hPre/h 为隐藏层原始/激活值
    std::vector<float> forward(const std::vector<float>& x,
                               std::vector<float>& hPre,
                               std::vector<float>& h) const {
        hPre.assign(nHid, 0.0f);
        h.assign(nHid, 0.0f);
        for (int j = 0; j < nHid; ++j) {
            float s = b1[j];
            const float* wp = &W1[j * nIn];
            for (int i = 0; i < nIn; ++i) s += wp[i] * x[i];
            hPre[j] = s;
            h[j] = s > 0.0f ? s : 0.0f;   // ReLU
        }
        std::vector<float> z(nOut, 0.0f);
        for (int k = 0; k < nOut; ++k) {
            float s = b2[k];
            const float* wp = &W2[k * nHid];
            for (int j = 0; j < nHid; ++j) s += wp[j] * h[j];
            z[k] = s;
        }
        // softmax（数值稳定）
        float m = *std::max_element(z.begin(), z.end());
        float sum = 0.0f;
        for (float& v : z) { v = std::exp(v - m); sum += v; }
        for (float& v : z) v /= sum;
        return z;
    }

    int predict(const std::vector<float>& x) const {
        std::vector<float> hp, h;
        std::vector<float> p = forward(x, hp, h);
        return (int)(std::max_element(p.begin(), p.end()) - p.begin());
    }

    // 在一个小批量上累计梯度并更新参数
    void trainBatch(const std::vector<Sample>& batch, float lr, float lambda, float mu) {
        const int B = (int)batch.size();
        std::vector<float> gW1(nHid * nIn, 0.0f), gb1(nHid, 0.0f);
        std::vector<float> gW2(nOut * nHid, 0.0f), gb2(nOut, 0.0f);
        std::vector<float> hp(nHid), h(nHid);

        for (const Sample& s : batch) {
            std::vector<float> p = forward(s.x, hp, h);
            // dL/dz = p - onehot(label)（softmax + 交叉熵 的简洁形式）
            std::vector<float> dz(nOut);
            for (int k = 0; k < nOut; ++k)
                dz[k] = p[k] - (k == s.label ? 1.0f : 0.0f);

            // 输出层梯度
            for (int k = 0; k < nOut; ++k) {
                gb2[k] += dz[k];
                for (int j = 0; j < nHid; ++j)
                    gW2[k * nHid + j] += dz[k] * h[j];
            }
            // 隐藏层梯度（ReLU 导数：hPre > 0 时 1，否则 0）
            std::vector<float> dh(nHid, 0.0f);
            for (int j = 0; j < nHid; ++j)
                for (int k = 0; k < nOut; ++k)
                    dh[j] += W2[k * nHid + j] * dz[k];
            for (int j = 0; j < nHid; ++j) {
                float d = dh[j] * (hp[j] > 0.0f ? 1.0f : 0.0f);
                gb1[j] += d;
                for (int i = 0; i < nIn; ++i)
                    gW1[j * nIn + i] += d * s.x[i];
            }
        }

        // 参数更新：平均梯度 + L2 正则 + 动量
        // v = mu*v - lr*grad ; w += v
        float inv = 1.0f / B;
        for (int j = 0; j < nHid; ++j) {
            for (int i = 0; i < nIn; ++i) {
                float g = gW1[j * nIn + i] * inv + lambda * W1[j * nIn + i];
                vW1[j * nIn + i] = mu * vW1[j * nIn + i] - lr * g;
                W1[j * nIn + i] += vW1[j * nIn + i];
            }
            vb1[j] = mu * vb1[j] - lr * gb1[j] * inv;
            b1[j] += vb1[j];
        }
        for (int k = 0; k < nOut; ++k) {
            for (int j = 0; j < nHid; ++j) {
                float g = gW2[k * nHid + j] * inv + lambda * W2[k * nHid + j];
                vW2[k * nHid + j] = mu * vW2[k * nHid + j] - lr * g;
                W2[k * nHid + j] += vW2[k * nHid + j];
            }
            vb2[k] = mu * vb2[k] - lr * gb2[k] * inv;
            b2[k] += vb2[k];
        }
    }

    // 平均交叉熵损失
    float loss(const std::vector<Sample>& data) const {
        float L = 0.0f;
        std::vector<float> hp, h;
        for (const Sample& s : data) {
            std::vector<float> p = forward(s.x, hp, h);
            L += -std::log(p[s.label] + 1e-12f);
        }
        return L / data.size();
    }

    // 识别正确率（百分比）
    float accuracy(const std::vector<Sample>& data) const {
        int ok = 0;
        std::vector<float> hp, h;
        for (const Sample& s : data) {
            std::vector<float> p = forward(s.x, hp, h);
            int pred = (int)(std::max_element(p.begin(), p.end()) - p.begin());
            if (pred == s.label) ++ok;
        }
        return 100.0f * ok / (float)data.size();
    }
};

// 打印一个 8x8 样本
void printGlyph(const std::vector<float>& x) {
    for (int r = 0; r < 8; ++r) {
        std::cout << "  ";
        for (int c = 0; c < 8; ++c)
            std::cout << (x[r * 8 + c] > 0.5f ? '#' : '.');
        std::cout << '\n';
    }
}

// ------------------------------------------------------------
// 4. 主流程：生成数据 -> 训练 -> 输出 loss 曲线与测试结果
// ------------------------------------------------------------
int main() {
    std::mt19937 rng(20240520);   // 固定随机种子，结果可复现

    std::cout << "==========================================\n";
    std::cout << " ai_brain_v1 —— 手写数字识别神经网络\n";
    std::cout << " 结构: 64(输入) -> 256(ReLU) -> 10(Softmax)\n";
    std::cout << " 数据: 内置 8x8 点阵字形 + 随机噪声/平移\n";
    std::cout << "==========================================\n\n";

    // 展示内置字形
    std::cout << "内置 0-9 数字字形:\n";
    for (const Glyph& g : kGlyphs) {
        std::cout << "  [" << g.label << "]\n";
        for (int r = 0; r < 8; ++r) std::cout << "   " << g.rows[r] << '\n';
    }
    std::cout << '\n';

    // ---------------- 数据生成 ----------------
    const int perTrain = 1000;  // 每个数字 1000 个训练样本 -> 共 10000
    const int perTest  = 200;   // 每个数字 200 个测试样本 -> 共 2000
    auto train = genData(kGlyphs, perTrain, 0.05f, 0.30f, true, rng);    // 轻~高噪声 + 平移
    auto testClean = genData(kGlyphs, perTest, 0.0f, 0.0f, false, rng);  // 干净数字
    auto testNoisy = genData(kGlyphs, perTest, 0.15f, 0.20f, true, rng); // 中噪声 + 平移
    auto testHeavy = genData(kGlyphs, perTest, 0.30f, 0.30f, true, rng); // 高压噪声 + 平移

    std::cout << "训练集: " << train.size() << " 样本 (噪声 5%~30% + 平移)\n"
              << "测试集(干净): " << testClean.size() << " 样本\n"
              << "测试集(噪声): " << testNoisy.size() << " 样本 (噪声 15%~20% + 平移)\n"
              << "测试集(高压噪声): " << testHeavy.size() << " 样本 (噪声 30% + 平移)\n\n";

    // ---------------- 网络与训练超参 ----------------
    MLP net(64, 256, 10, rng);
    const int epochs = 60;
    const int batchSize = 32;
    const float lr0 = 0.15f;     // 初始学习率
    const float decay = 0.98f;   // 每 epoch 衰减系数
    const float lambda = 1e-4f;  // L2 正则强度
    const float momentum = 0.9f; // SGD 动量

    // ---------------- 训练 ----------------
    std::cout << "开始训练 (" << epochs << " epochs, batch=" << batchSize << ")\n";
    std::cout << "--------------------------------------------------\n";
    std::cout << "Epoch |   Loss   | TrainAcc | CleanAcc | NoisyAcc | HeavyAcc\n";
    std::cout << "--------------------------------------------------\n";

    std::vector<float> lossHistory;

    for (int ep = 1; ep <= epochs; ++ep) {
        // 洗牌
        std::shuffle(train.begin(), train.end(), rng);
        float lr = lr0 * std::pow(decay, ep - 1);

        for (size_t i = 0; i < train.size(); i += batchSize) {
            size_t end = std::min(i + batchSize, train.size());
            std::vector<Sample> batch(train.begin() + i, train.begin() + end);
            net.trainBatch(batch, lr, lambda, momentum);
        }

        float L = net.loss(train);
        float trAcc = net.accuracy(train);
        float clAcc = net.accuracy(testClean);
        float nsAcc = net.accuracy(testNoisy);
        float hvAcc = net.accuracy(testHeavy);
        lossHistory.push_back(L);

        std::cout << std::setw(5) << ep << " | " << std::fixed << std::setprecision(4)
                  << L << " | " << std::setw(7) << std::setprecision(1) << trAcc
                  << "% | " << std::setw(7) << clAcc << "% | "
                  << std::setw(7) << nsAcc << "% | "
                  << std::setw(7) << hvAcc << "%   (lr=" << lr << ")\n";
    }
    std::cout << "--------------------------------------------------\n\n";

    // ---------------- 训练 Loss 曲线（文本图） ----------------
    std::cout << "===== 训练 Loss 曲线 =====\n";
    float maxL = *std::max_element(lossHistory.begin(), lossHistory.end());
    float minL = *std::min_element(lossHistory.begin(), lossHistory.end());
    float span = std::max(maxL - minL, 1e-6f);
    for (int ep = 0; ep < epochs; ++ep) {
        int barLen = 2 + (int)((lossHistory[ep] - minL) / span * 55.0f);
        std::cout << std::setw(4) << (ep + 1) << " | "
                  << std::string(barLen, '#')
                  << std::fixed << std::setprecision(4) << "  " << lossHistory[ep] << '\n';
    }
    std::cout << "（# 长度 ∝ 该 epoch 的 loss，越短越好）\n\n";

    // ---------------- 最终测试结果 ----------------
    std::cout << "===== 最终测试结果 =====\n";
    std::cout << "干净数字识别正确率 : " << std::fixed << std::setprecision(2)
              << net.accuracy(testClean) << "%\n";
    std::cout << "含噪数字识别正确率 : " << std::fixed << std::setprecision(2)
              << net.accuracy(testNoisy) << "%\n";
    std::cout << "高压噪声(30%)识别率 : " << std::fixed << std::setprecision(2)
              << net.accuracy(testHeavy) << "%\n";

    // 逐数字识别率（含噪测试集）
    std::cout << "\n逐数字识别率（含噪测试集, 噪声 15%~20%）:\n  ";
    std::vector<float> hp2(net.nHid), h2(net.nHid);
    for (int d = 0; d < 10; ++d) {
        int ok = 0, total = 0;
        for (int n = 0; n < perTest; ++n) {
            const Sample& s = testNoisy[d * perTest + n];
            std::vector<float> p = net.forward(s.x, hp2, h2);
            int pred = (int)(std::max_element(p.begin(), p.end()) - p.begin());
            ++total;
            if (pred == d) ++ok;
        }
        std::cout << "[" << d << "] " << std::fixed << std::setprecision(1)
                  << 100.0f * ok / total << "%  ";
        if (d == 4 || d == 9) std::cout << "\n  ";
    }
    std::cout << "\n\n";

    // ---------------- 示例预测：随机抽几个含噪样本 ----------------
    std::cout << "===== 含噪数字示例预测（测试集抽样） =====\n";
    std::uniform_int_distribution<int> pick(0, perTest - 1);
    std::vector<float> hp(net.nHid), h(net.nHid);
    for (int d = 0; d < 10; ++d) {

        const Sample& s = testNoisy[d * perTest + pick(rng)];
        std::vector<float> p = net.forward(s.x, hp, h);
        int pred = (int)(std::max_element(p.begin(), p.end()) - p.begin());
        std::cout << "真实=" << s.label << " 预测=" << pred
                  << " 置信度=" << std::fixed << std::setprecision(3) << p[pred] << "\n";
        printGlyph(s.x);
    }

    std::cout << "\n训练完成。文件: ~/ai_brain_v1.cpp\n";
    return 0;
}
