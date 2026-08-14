// ai_brain_v0.cpp — 一个简单的多层感知机（MLP）神经网络，学习 XOR 逻辑运算
// 说明：单层感知机无法解决 XOR（线性不可分），因此这里实现带一个隐藏层的
// 多层感知机 + 反向传播（BP），这是能学习 XOR 的最简神经网络。
// 网络结构：2 个输入 -> 2 个隐藏神经元(sigmoid) -> 1 个输出神经元(sigmoid)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

using namespace std;

// ---- 工具函数 ----
static inline double frand() {          // [0,1) 均匀随机
    return rand() / (double)RAND_MAX;
}

static inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static inline double sigmoidDeriv(double y) {   // y 是 sigmoid 的输出
    return y * (1.0 - y);
}

// ---- 单层（全连接层 + sigmoid 激活）----
struct Layer {
    int nIn, nOut;
    vector<vector<double>> w;   // w[j][i]：第 j 个输出神经元连到第 i 个输入
    vector<double> b;           // 偏置
    vector<double> out;         // 前向输出（也用于反向传播）
    vector<double> in;          // 本层输入（也用于反向传播）

    Layer(int in_, int out_) : nIn(in_), nOut(out_) {
        w.assign(nOut, vector<double>(nIn, 0.0));
        b.assign(nOut, 0.0);
        out.assign(nOut, 0.0);
        in.assign(nIn, 0.0);
        for (int j = 0; j < nOut; ++j) {
            b[j] = frand() * 2 - 1;                     // [-1,1)
            for (int i = 0; i < nIn; ++i)
                w[j][i] = frand() * 2 - 1;              // 小随机初始化
        }
    }

    // 前向传播：输入 x，返回输出向量
    vector<double> forward(const vector<double>& x) {
        in = x;
        for (int j = 0; j < nOut; ++j) {
            double z = b[j];
            for (int i = 0; i < nIn; ++i) z += w[j][i] * x[i];
            out[j] = sigmoid(z);
        }
        return out;
    }
};

// ---- 神经网络 ----
struct NeuralNet {
    Layer h;   // 隐藏层
    Layer o;   // 输出层
    double lr; // 学习率

    NeuralNet(double lr_ = 0.5) : h(2, 2), o(2, 1), lr(lr_) {}

    // 前向：返回网络输出（1 个元素）
    double predict(double x1, double x2) {
        vector<double> x = {x1, x2};
        vector<double> hOut = h.forward(x);
        vector<double> y = o.forward(hOut);
        return y[0];
    }

    // 训练一个样本（反向传播）
    void trainSample(double x1, double x2, double target) {
        vector<double> x = {x1, x2};

        // 1. 前向
        vector<double> hOut = h.forward(x);
        vector<double> y = o.forward(hOut);   // y[0] 是预测

        // 2. 输出层误差
        double errOut = y[0] - target;
        double deltaOut = errOut * sigmoidDeriv(y[0]);

        // 3. 隐藏层误差（误差反向传播）
        vector<double> deltaH(h.nOut, 0.0);
        for (int j = 0; j < h.nOut; ++j) {
            double sum = deltaOut * o.w[0][j];
            deltaH[j] = sum * sigmoidDeriv(hOut[j]);
        }

        // 4. 更新输出层权重
        for (int j = 0; j < o.nOut; ++j) {
            o.b[j] -= lr * deltaOut;
            for (int i = 0; i < o.nIn; ++i)
                o.w[j][i] -= lr * deltaOut * hOut[i];
        }
        // 5. 更新隐藏层权重
        for (int j = 0; j < h.nOut; ++j) {
            h.b[j] -= lr * deltaH[j];
            for (int i = 0; i < h.nIn; ++i)
                h.w[j][i] -= lr * deltaH[j] * x[i];
        }
    }
};

// XOR 真值表
struct Sample { double x1, x2, t; };

int main() {
    srand((unsigned)time(nullptr));

    Sample data[4] = {
        {0, 0, 0},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0},
    };

    NeuralNet net(0.5);

    // 训练 10000 个 epoch（每个 epoch 遍历一遍 4 个样本）
    const int EPOCHS = 10000;
    for (int e = 1; e <= EPOCHS; ++e) {
        for (int s = 0; s < 4; ++s) {
            // 每次随机打乱样本顺序，训练更稳定
            int i = rand() % 4;
            net.trainSample(data[i].x1, data[i].x2, data[i].t);
        }
        // 打印每 1000 轮的损失，观察收敛
        if (e % 1000 == 0) {
            double loss = 0;
            for (int s = 0; s < 4; ++s) {
                double p = net.predict(data[s].x1, data[s].x2);
                loss += (p - data[s].t) * (p - data[s].t);
            }
            printf("epoch %5d  loss = %.6f\n", e, loss / 4.0);
        }
    }

    // 输出训练后的预测结果
    printf("\n训练完成！XOR 预测结果（输出 > 0.5 视为 1）：\n");
    printf("------------------------------------------\n");
    printf(" 输入   | 期望输出 |  预测输出 | 判定\n");
    printf("------------------------------------------\n");
    for (int s = 0; s < 4; ++s) {
        double p = net.predict(data[s].x1, data[s].x2);
        int label = p >= 0.5 ? 1 : 0;
        printf(" %d XOR %d |    %d     |  %.4f  |   %d\n",
               (int)data[s].x1, (int)data[s].x2,
               (int)data[s].t, p, label);
    }
    printf("------------------------------------------\n");
    return 0;
}
