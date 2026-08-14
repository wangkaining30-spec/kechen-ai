// ============================================================================
//  ai_brain_v08.cpp — 客尘AI集群v1.0 专家模型 (v0.8 架构)
//
//  架构沿袭 v7 (RNN + 2 层多头自注意力 迷你 Transformer), 升级为"专家版":
//   1. 语料从文件读取 (每行一段, 可含 "前缀:标题：" 会被剥离):
//        poet=古诗 / lunyu=论语+名言 / chat=对话+字典 / cosmos=宇宙模型
//   2. 四个专家共享同一份源码, 编译时用 -D 宏注入身份:
//        -DEXPERT_NAME="poet" -DCORPUS_FILE="corpus_poet.txt"
//        -DWEIGHTS_FILE="ai_brain_poet_weights.bin" -DDEMOS="床前明月|..."
//   3. 每个专家独立保存/加载权重 (ai_brain_<专家>_weights.bin):
//        首次运行 -> 训练并自动保存; 再次运行 -> 检测到权重直接加载, 秒启动。
//   4. -q 模式: 路由器 router.sh 调用, 只输出回答, 不打印横幅 (单次问答)。
//  模型结构: (同 v7)
//     字符 Embedding (V x H)
//        -> Elman RNN:     h_t = tanh(Wx·E[x_t] + Wh·h_{t-1} + bh)
//        -> 可学习位置编码: p_t = h_t + P[t]
//        -> [Transformer 块 × 2 层] 每层:
//             LayerNorm -> 多头自注意力 (NH 头, 因果掩码, 1/sqrt(dk) 缩放)
//             -> 残差 + LayerNorm -> FFN: GELU(W1·x + b1) -> W2 -> 残差
//        -> 线性输出 y_t = Wy·z_t + by -> Softmax 预测下一字符
//  训练: BPTT(RNN) + 2 层注意力/FFN/LayerNorm 反向传播 + Adam + 梯度裁剪
//  交互: 输入问题(如"你好") -> AI 直接回答 (自动隐藏 "Q=" 只显示回答);
//        输入中文开头 -> 模型续写; 空行 = 随机创作; quit = 退出
//
//  编译示例: clang++ -O2 -std=c++17 -DEXPERT_NAME="poet" \
//            -DCORPUS_FILE="corpus_poet.txt" \
//            -DWEIGHTS_FILE="ai_brain_poet_weights.bin" \
//            -DDEMOS="床前明月|白日依山|秦时明月" \
//            ai_brain_v08.cpp -o ai_brain_poet
//  运行: ./ai_brain_poet          (首次: 训练 + 自动保存权重)
//        ./ai_brain_poet -q 床前明月   (路由器: 只输出回答)
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace std;

// --------------------------- 专家身份宏 (编译期注入) ---------------------------
#ifndef EXPERT_NAME
#define EXPERT_NAME "expert"
#endif
#ifndef CORPUS_FILE
#define CORPUS_FILE "corpus.txt"
#endif
#ifndef WEIGHTS_FILE
#define WEIGHTS_FILE "expert_weights.bin"
#endif
#ifndef DEMOS
#define DEMOS ""
#endif

// --------------------------- UTF-8 工具 ---------------------------
static vector<string> utf8_split(const string& s) {
    vector<string> out;
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        int len = 1;
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        if (i + len > n) len = 1; // 防御: 非法/截断字节
        out.push_back(s.substr(i, (size_t)len));
        i += (size_t)len;
    }
    return out;
}

static string trim(const string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// --------------------------- 词表 ---------------------------
struct Tokenizer {
    vector<string> itos;          // id -> 字符
    map<string, int> stoi;        // 字符 -> id
    int bos = 0, eos = 0, unk = 0;
    int add(const string& c) {
        auto it = stoi.find(c);
        if (it != stoi.end()) return it->second;
        int id = (int)itos.size();
        stoi[c] = id;
        itos.push_back(c);
        return id;
    }
    int get(const string& c) const {
        auto it = stoi.find(c);
        return it == stoi.end() ? unk : it->second;
    }
};

// --------------------------- 参数 (值 + 梯度 + Adam 动量) ---------------------------
struct Param {
    int rows = 0, cols = 0;
    vector<double> val, grad, m, v;
    void init(int r, int c, mt19937& rng, double scale) {
        rows = r; cols = c;
        size_t n = (size_t)r * (size_t)c;
        val.assign(n, 0.0);
        grad.assign(n, 0.0);
        m.assign(n, 0.0);
        v.assign(n, 0.0);
        if (scale > 0.0) {
            normal_distribution<double> nd(0.0, scale);
            for (auto& x : val) x = nd(rng);
        }
    }
    double& at(int i, int j) { return val[(size_t)i * cols + j]; }
    double at(int i, int j) const { return val[(size_t)i * cols + j]; }
    void zero_grad() { fill(grad.begin(), grad.end(), 0.0); }
    void fill_all(double x) { fill(val.begin(), val.end(), x); }
};

// --------------------------- 激活函数 ---------------------------
static double gelu(double x) { // 精确 GELU: 0.5x(1+erf(x/√2))
    return 0.5 * x * (1.0 + erf(x * 0.7071067811865476));
}
static double gelu_d(double x) { // d/dx GELU
    return 0.5 * (1.0 + erf(x * 0.7071067811865476))
         + x * exp(-0.5 * x * x) * 0.3989422804014327; // 1/√(2π)
}

// LayerNorm 前向: y = gamma·(x-mean)/sqrt(var+eps) + beta
static void layernorm_fwd(const vector<double>& x, const vector<double>& g,
                          const vector<double>& b, vector<double>& y,
                          double& mean, double& inv) {
    int H = (int)x.size();
    double m = 0.0;
    for (double v : x) m += v;
    m /= H;
    double var = 0.0;
    for (double v : x) { double d = v - m; var += d * d; }
    var /= H;
    inv = 1.0 / sqrt(var + 1e-5);
    y.resize(H);
    for (int j = 0; j < H; ++j) y[j] = (x[j] - m) * inv * g[j] + b[j];
    mean = m;
}

// LayerNorm 反向: 累加 dgamma/dbeta, 输出 dx (梯度 wrt 输入)
static void layernorm_bwd(const vector<double>& x, double mean, double inv,
                          const vector<double>& g, const vector<double>& dy,
                          vector<double>& dx, vector<double>& dg, vector<double>& db) {
    int H = (int)x.size();
    vector<double> xhat(H);
    for (int j = 0; j < H; ++j) xhat[j] = (x[j] - mean) * inv;
    vector<double> dxhat(H);
    for (int j = 0; j < H; ++j) dxhat[j] = dy[j] * g[j];
    double sum_dxhat = 0.0, dnu = 0.0, sum_xm = 0.0;
    for (int j = 0; j < H; ++j) {
        sum_dxhat += dxhat[j];
        dnu += dxhat[j] * xhat[j];          // Σ dxhat·xhat·(-0.5)·inv²
        sum_xm += x[j] - mean;
    }
    dnu *= -0.5 * inv * inv;
    double dmu = sum_dxhat * (-inv) + dnu * (-2.0 / H) * sum_xm;
    dx.resize(H);
    for (int j = 0; j < H; ++j) {
        dx[j] = dxhat[j] * inv + dnu * 2.0 * (x[j] - mean) / H + dmu / H;
        dg[j] += dxhat[j] * xhat[j];
        db[j] += dy[j];
    }
}

// --------------------------- Transformer 块 (v6: 堆叠 2 层) ---------------------------
// 每层: LN1 -> 多头自注意力 -> 残差 -> LN2 -> FFN(GELU) -> 残差
struct Block {
    Param LN1g, LN1b, LN2g, LN2b;   // H x 1
    Param Wq, Wk, Wv, Wo;           // H x H
    Param W1, b1, W2, b2;           // H1 x H, H1, H x H1, H
    void init(int H, int H1, mt19937& rng) {
        LN1g.init(H, 1, rng, 0.0); LN1g.fill_all(1.0);
        LN1b.init(H, 1, rng, 0.0);
        LN2g.init(H, 1, rng, 0.0); LN2g.fill_all(1.0);
        LN2b.init(H, 1, rng, 0.0);
        double s = 1.0 / sqrt((double)H);
        Wq.init(H, H, rng, s);
        Wk.init(H, H, rng, s);
        Wv.init(H, H, rng, s);
        Wo.init(H, H, rng, s);
        W1.init(H1, H, rng, s);
        b1.init(H1, 1, rng, 0.0);
        W2.init(H, H1, rng, 1.0 / sqrt((double)H1));
        b2.init(H, 1, rng, 0.0);
    }
    size_t nparam() const {
        return LN1g.val.size() + LN1b.val.size() + LN2g.val.size() + LN2b.val.size()
             + Wq.val.size() + Wk.val.size() + Wv.val.size() + Wo.val.size()
             + W1.val.size() + b1.val.size() + W2.val.size() + b2.val.size();
    }
    void zero_grad() {
        vector<Param*> ps = {&LN1g, &LN1b, &LN2g, &LN2b, &Wq, &Wk, &Wv, &Wo,
                             &W1, &b1, &W2, &b2};
        for (Param* p : ps) p->zero_grad();
    }
    void clip_grads(double clip, double& n2) {
        vector<Param*> ps = {&LN1g, &LN1b, &LN2g, &LN2b, &Wq, &Wk, &Wv, &Wo,
                             &W1, &b1, &W2, &b2};
        for (Param* p : ps)
            for (double g : p->grad) n2 += g * g;
    }
    void scale_grads(double s) {
        vector<Param*> ps = {&LN1g, &LN1b, &LN2g, &LN2b, &Wq, &Wk, &Wv, &Wo,
                             &W1, &b1, &W2, &b2};
        for (Param* p : ps)
            for (auto& g : p->grad) g *= s;
    }
    void adam_update(double lr, double bc1, double bc2) {
        vector<Param*> ps = {&LN1g, &LN1b, &LN2g, &LN2b, &Wq, &Wk, &Wv, &Wo,
                             &W1, &b1, &W2, &b2};
        const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        for (Param* p : ps) {
            for (size_t i = 0; i < p->val.size(); ++i) {
                double g = p->grad[i];
                p->m[i] = beta1 * p->m[i] + (1 - beta1) * g;
                p->v[i] = beta2 * p->v[i] + (1 - beta2) * g * g;
                double mh = p->m[i] / bc1, vh = p->v[i] / bc2;
                p->val[i] -= lr * mh / (sqrt(vh) + eps);
            }
        }
    }
};

// 块的中间激活 (前向存储, 供反向使用)
struct BlockActs {
    int T = 0, H = 0, NH = 0, DK = 0, H1 = 0;
    vector<vector<double>> r;     // LN1 输出 (注意力输入)
    vector<vector<double>> att;   // 多头拼接输出 (Wo 前)
    vector<vector<double>> a;     // Wo 投影后
    vector<vector<double>> u;     // r + a
    vector<vector<double>> q;     // LN2 输出 (FFN 输入)
    vector<vector<double>> ff_in, ff_g;   // GELU 前后
    vector<vector<double>> f;     // FFN 输出
    vector<vector<double>> y;     // 块输出 u + f
    vector<double> ln1_mean, ln1_inv, ln2_mean, ln2_inv;
    vector<vector<vector<double>>> pH;    // 注意力权重 [NH][T][T]
    vector<vector<vector<double>>> qH, kH, vH;  // [NH][T][DK]
    void alloc(int T_, int H_, int NH_, int DK_, int H1_) {
        T = T_; H = H_; NH = NH_; DK = DK_; H1 = H1_;
        r.assign(T, vector<double>(H, 0.0));
        att.assign(T, vector<double>(H, 0.0));
        a.assign(T, vector<double>(H, 0.0));
        u.assign(T, vector<double>(H, 0.0));
        q.assign(T, vector<double>(H, 0.0));
        ff_in.assign(T, vector<double>(H1, 0.0));
        ff_g.assign(T, vector<double>(H1, 0.0));
        f.assign(T, vector<double>(H, 0.0));
        y.assign(T, vector<double>(H, 0.0));
        ln1_mean.assign(T, 0.0); ln1_inv.assign(T, 0.0);
        ln2_mean.assign(T, 0.0); ln2_inv.assign(T, 0.0);
        pH.assign(NH, vector<vector<double>>(T, vector<double>(T, 0.0)));
        qH.assign(NH, vector<vector<double>>(T, vector<double>(DK, 0.0)));
        kH.assign(NH, vector<vector<double>>(T, vector<double>(DK, 0.0)));
        vH.assign(NH, vector<vector<double>>(T, vector<double>(DK, 0.0)));
    }
};

// 块前向: x[T][H] -> y[T][H], 中间结果存入 s
static void block_fwd(const Block& b, BlockActs& s, const vector<vector<double>>& x) {
    const int T = s.T, H = s.H, NH = s.NH, DK = s.DK, H1 = s.H1;
    const double inv_dk = 1.0 / sqrt((double)DK);
    for (int t = 0; t < T; ++t)
        layernorm_fwd(x[t], b.LN1g.val, b.LN1b.val, s.r[t], s.ln1_mean[t], s.ln1_inv[t]);
    // 多头自注意力 (因果掩码: 位置 i 只能看到 j<=i)
    for (int hd = 0; hd < NH; ++hd) {
        for (int i = 0; i < T; ++i)
            for (int d = 0; d < DK; ++d) {
                double sq = 0.0, sk = 0.0, sv = 0.0;
                for (int j = 0; j < H; ++j) {
                    sq += b.Wq.at(hd * DK + d, j) * s.r[i][j];
                    sk += b.Wk.at(hd * DK + d, j) * s.r[i][j];
                    sv += b.Wv.at(hd * DK + d, j) * s.r[i][j];
                }
                s.qH[hd][i][d] = sq; s.kH[hd][i][d] = sk; s.vH[hd][i][d] = sv;
            }
        vector<double> sc(T);
        for (int i = 0; i < T; ++i) {
            double mx = -1e30;
            for (int j = 0; j <= i; ++j) {
                double dot = 0.0;
                for (int d = 0; d < DK; ++d) dot += s.qH[hd][i][d] * s.kH[hd][j][d];
                sc[j] = dot * inv_dk;
                mx = max(mx, sc[j]);
            }
            double sum = 0.0;
            for (int j = 0; j <= i; ++j) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
            for (int j = 0; j <= i; ++j) s.pH[hd][i][j] = sc[j] / sum;
            for (int d = 0; d < DK; ++d) {
                double svv = 0.0;
                for (int j = 0; j <= i; ++j) svv += s.pH[hd][i][j] * s.vH[hd][j][d];
                s.att[i][hd * DK + d] = svv;
            }
        }
    }
    for (int i = 0; i < T; ++i)
        for (int o = 0; o < H; ++o) {
            double svv = 0.0;
            for (int hh = 0; hh < H; ++hh) svv += b.Wo.at(o, hh) * s.att[i][hh];
            s.a[i][o] = svv;
        }
    // 残差 + LN2 + FFN
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < H; ++j) s.u[t][j] = s.r[t][j] + s.a[t][j];
        layernorm_fwd(s.u[t], b.LN2g.val, b.LN2b.val, s.q[t], s.ln2_mean[t], s.ln2_inv[t]);
        for (int m = 0; m < H1; ++m) {
            double svv = b.b1.val[m];
            for (int j = 0; j < H; ++j) svv += b.W1.at(m, j) * s.q[t][j];
            s.ff_in[t][m] = svv;
            s.ff_g[t][m] = gelu(svv);
        }
        for (int j = 0; j < H; ++j) {
            double svv = b.b2.val[j];
            for (int m = 0; m < H1; ++m) svv += b.W2.at(j, m) * s.ff_g[t][m];
            s.f[t][j] = svv;
        }
        for (int j = 0; j < H; ++j) s.y[t][j] = s.u[t][j] + s.f[t][j];
    }
}

// 块反向: dy = 块输出的梯度, 输出 dx = 块输入的梯度 (并累加本块所有参数梯度)
static void block_bwd(Block& b, const BlockActs& s, const vector<vector<double>>& x,
                      const vector<vector<double>>& dy, vector<vector<double>>& dx) {
    const int T = s.T, H = s.H, NH = s.NH, DK = s.DK, H1 = s.H1;
    const double inv_dk = 1.0 / sqrt((double)DK);
    vector<vector<double>> du(T, vector<double>(H, 0.0));
    vector<vector<double>> df(T, vector<double>(H, 0.0));
    vector<vector<double>> da(T, vector<double>(H, 0.0));
    vector<vector<double>> dr(T, vector<double>(H, 0.0));
    vector<vector<double>> datt(T, vector<double>(H, 0.0));
    vector<double> dq_ff(H, 0.0);

    // y = u + f
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < H; ++j) { du[t][j] += dy[t][j]; df[t][j] += dy[t][j]; }

    // FFN 反向 + LN2 反向
    for (int t = 0; t < T; ++t) {
        layernorm_bwd(s.u[t], s.ln2_mean[t], s.ln2_inv[t], b.LN2g.val, df[t],
                      dq_ff, b.LN2g.grad, b.LN2b.grad);
        vector<double> ds(H1, 0.0);
        for (int m = 0; m < H1; ++m) {
            double dgm = 0.0;
            for (int j = 0; j < H; ++j) dgm += b.W2.at(j, m) * df[t][j];
            ds[m] = dgm * gelu_d(s.ff_in[t][m]);
        }
        for (int m = 0; m < H1; ++m) {
            for (int j = 0; j < H; ++j) {
                b.W2.grad[(size_t)j * H1 + m] += df[t][j] * s.ff_g[t][m];
                b.W1.grad[(size_t)m * H + j] += ds[m] * s.q[t][j];
            }
            b.b1.grad[m] += ds[m];
        }
        for (int j = 0; j < H; ++j) {
            b.b2.grad[j] += df[t][j];
            double dqj = 0.0;
            for (int m = 0; m < H1; ++m) dqj += b.W1.at(m, j) * ds[m];
            dq_ff[j] += dqj;
        }
        for (int j = 0; j < H; ++j) du[t][j] += dq_ff[j];
    }
    // u = r + a
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < H; ++j) { da[t][j] += du[t][j]; dr[t][j] += du[t][j]; }

    // 注意力反向: Wo
    for (int i = 0; i < T; ++i)
        for (int o = 0; o < H; ++o)
            for (int hh = 0; hh < H; ++hh)
                b.Wo.grad[(size_t)o * H + hh] += da[i][o] * s.att[i][hh];
    for (int i = 0; i < T; ++i)
        for (int hh = 0; hh < H; ++hh) {
            double svv = 0.0;
            for (int o = 0; o < H; ++o) svv += b.Wo.at(o, hh) * da[i][o];
            datt[i][hh] = svv;
        }

    // 注意力反向: 每头 Q/K/V 与概率
    vector<vector<double>> dqH(T, vector<double>(DK, 0.0));
    vector<vector<double>> dkH(T, vector<double>(DK, 0.0));
    vector<vector<double>> dVH(T, vector<double>(DK, 0.0));
    for (int hd = 0; hd < NH; ++hd) {
        for (int i = 0; i < T; ++i)
            for (int d = 0; d < DK; ++d) dqH[i][d] = dkH[i][d] = dVH[i][d] = 0.0;
        for (int i = 0; i < T; ++i) {
            for (int d = 0; d < DK; ++d) {
                double g = datt[i][hd * DK + d];
                for (int j = 0; j <= i; ++j) dVH[j][d] += s.pH[hd][i][j] * g;
            }
            vector<double> dp_i(T, 0.0);
            double dotpdg = 0.0;
            for (int j = 0; j <= i; ++j) {
                double dp = 0.0;
                for (int d = 0; d < DK; ++d) dp += datt[i][hd * DK + d] * s.vH[hd][j][d];
                dp_i[j] = dp;
                dotpdg += s.pH[hd][i][j] * dp;
            }
            for (int j = 0; j <= i; ++j) {
                double ds = s.pH[hd][i][j] * (dp_i[j] - dotpdg);
                for (int d = 0; d < DK; ++d) {
                    dqH[i][d] += ds * s.kH[hd][j][d];
                    dkH[j][d] += ds * s.qH[hd][i][d];
                }
            }
        }
        for (int i = 0; i < T; ++i)
            for (int d = 0; d < DK; ++d) { dqH[i][d] *= inv_dk; dkH[i][d] *= inv_dk; }
        for (int i = 0; i < T; ++i) {
            for (int d = 0; d < DK; ++d)
                for (int j = 0; j < H; ++j) {
                    b.Wq.grad[(size_t)(hd * DK + d) * H + j] += dqH[i][d] * s.r[i][j];
                    b.Wk.grad[(size_t)(hd * DK + d) * H + j] += dkH[i][d] * s.r[i][j];
                    b.Wv.grad[(size_t)(hd * DK + d) * H + j] += dVH[i][d] * s.r[i][j];
                }
            for (int j = 0; j < H; ++j) {
                double svv = 0.0;
                for (int d = 0; d < DK; ++d)
                    svv += b.Wq.at(hd * DK + d, j) * dqH[i][d]
                         + b.Wk.at(hd * DK + d, j) * dkH[i][d]
                         + b.Wv.at(hd * DK + d, j) * dVH[i][d];
                dr[i][j] += svv;
            }
        }
    }

    // LN1 反向
    for (int t = 0; t < T; ++t)
        layernorm_bwd(x[t], s.ln1_mean[t], s.ln1_inv[t], b.LN1g.val, dr[t],
                      dx[t], b.LN1g.grad, b.LN1b.grad);
}

// --------------------------- 模型: RNN + 2 层多头自注意力 (迷你 Transformer) ---------------------------
class CharRNNTransformer {
public:
    int V, H, NH, DK, H1, T_MAX, NBLK;
    // Embedding + RNN
    Param E, Wx, Wh, bh;
    // 可学习位置编码 P (T_MAX x H)
    Param P;
    // Transformer 块 x NBLK (v6: 2 层)
    vector<Block> blocks;
    // 输出
    Param Wy, by;

    CharRNNTransformer(int vocab, int hidden, int heads, int ff, int tmax,
                       int nblk, mt19937& rng)
        : V(vocab), H(hidden), NH(heads), DK(hidden / heads), H1(ff),
          T_MAX(tmax), NBLK(nblk) {
        E.init(V, H, rng, 0.10);
        Wx.init(H, H, rng, 1.0 / sqrt((double)H));
        Wh.init(H, H, rng, 1.0 / sqrt((double)H));
        bh.init(H, 1, rng, 0.0);
        P.init(T_MAX, H, rng, 0.10);
        blocks.resize(NBLK);
        for (auto& blk : blocks) blk.init(H, H1, rng);
        Wy.init(V, H, rng, 0.10);
        by.init(V, 1, rng, 0.0);
    }

    size_t param_count() const {
        size_t n = E.val.size() + Wx.val.size() + Wh.val.size() + bh.val.size()
                 + P.val.size() + Wy.val.size() + by.val.size();
        for (auto& blk : blocks) n += blk.nparam();
        return n;
    }

    void zero_grad() {
        E.zero_grad(); Wx.zero_grad(); Wh.zero_grad(); bh.zero_grad();
        P.zero_grad(); Wy.zero_grad(); by.zero_grad();
        for (auto& blk : blocks) blk.zero_grad();
    }

    void clip_grads(double clip) {
        double n2 = 0.0;
        for (double g : E.grad) n2 += g * g;
        for (double g : Wx.grad) n2 += g * g;
        for (double g : Wh.grad) n2 += g * g;
        for (double g : bh.grad) n2 += g * g;
        for (double g : P.grad) n2 += g * g;
        for (double g : Wy.grad) n2 += g * g;
        for (double g : by.grad) n2 += g * g;
        for (auto& blk : blocks) blk.clip_grads(clip, n2);
        double n = sqrt(n2);
        if (n > clip) {
            double s = clip / n;
            for (auto& g : E.grad) g *= s;
            for (auto& g : Wx.grad) g *= s;
            for (auto& g : Wh.grad) g *= s;
            for (auto& g : bh.grad) g *= s;
            for (auto& g : P.grad) g *= s;
            for (auto& g : Wy.grad) g *= s;
            for (auto& g : by.grad) g *= s;
            for (auto& blk : blocks) blk.scale_grads(s);
        }
    }

    void adam_update(double lr, int t) {
        const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        double bc1 = 1.0 - pow(beta1, t), bc2 = 1.0 - pow(beta2, t);
        vector<Param*> ps = {&E, &Wx, &Wh, &bh, &P, &Wy, &by};
        for (Param* p : ps) {
            for (size_t i = 0; i < p->val.size(); ++i) {
                double g = p->grad[i];
                p->m[i] = beta1 * p->m[i] + (1 - beta1) * g;
                p->v[i] = beta2 * p->v[i] + (1 - beta2) * g * g;
                double mh = p->m[i] / bc1, vh = p->v[i] / bc2;
                p->val[i] -= lr * mh / (sqrt(vh) + eps);
            }
        }
        for (auto& blk : blocks) blk.adam_update(lr, bc1, bc2);
    }

    // 对一整行(含 <BOS> 与 <EOS>)前向 + 反向, 返回交叉熵损失
    double train_line(const vector<int>& ids) {
        const int T = (int)ids.size();
        if (T < 2) return 0.0;

        // ---- RNN 前向 + 位置编码 -> p[t] ----
        vector<vector<double>> emb(T, vector<double>(H, 0.0));
        vector<vector<double>> h(T, vector<double>(H, 0.0));
        vector<vector<double>> hprev(T, vector<double>(H, 0.0));
        vector<vector<double>> p(T, vector<double>(H, 0.0));
        vector<double> h_prev(H, 0.0), pre(H);
        for (int t = 0; t < T; ++t) {
            hprev[t] = h_prev;
            int id = ids[t];
            for (int k = 0; k < H; ++k) emb[t][k] = E.at(id, k);
            for (int j = 0; j < H; ++j) {
                double s = bh.val[j];
                for (int k = 0; k < H; ++k)
                    s += Wx.at(j, k) * emb[t][k] + Wh.at(j, k) * h_prev[k];
                pre[j] = s;
            }
            for (int j = 0; j < H; ++j) h[t][j] = tanh(pre[j]);
            int pt = min(t, T_MAX - 1);
            for (int j = 0; j < H; ++j) p[t][j] = h[t][j] + P.at(pt, j);
            h_prev = h[t];
        }

        // ---- 2 层 Transformer 块 ----
        BlockActs s0, s1;
        s0.alloc(T, H, NH, DK, H1);
        block_fwd(blocks[0], s0, p);
        s1.alloc(T, H, NH, DK, H1);
        block_fwd(blocks[1], s1, s0.y);

        // ---- 输出 + loss ----
        vector<vector<double>> soft(T, vector<double>(V, 0.0));
        vector<double> logit(V);
        double loss = 0.0;
        for (int t = 0; t < T; ++t) {
            for (int v = 0; v < V; ++v) {
                double s = by.val[v];
                for (int j = 0; j < H; ++j) s += Wy.at(v, j) * s1.y[t][j];
                logit[v] = s;
            }
            double mx = *max_element(logit.begin(), logit.end());
            double sum = 0.0;
            for (int v = 0; v < V; ++v) { soft[t][v] = exp(logit[v] - mx); sum += soft[t][v]; }
            for (int v = 0; v < V; ++v) soft[t][v] /= sum;
            if (t + 1 < T) loss -= log(max(soft[t][ids[t + 1]], 1e-12));
        }

        // ---- 反向 ----
        vector<vector<double>> dz(T, vector<double>(H, 0.0));
        vector<double> dlogit(V);
        for (int t = T - 1; t >= 0; --t) {
            if (t + 1 < T) {
                for (int v = 0; v < V; ++v) dlogit[v] = soft[t][v];
                dlogit[ids[t + 1]] -= 1.0;
                for (int v = 0; v < V; ++v) {
                    by.grad[v] += dlogit[v];
                    for (int j = 0; j < H; ++j)
                        Wy.grad[(size_t)v * H + j] += dlogit[v] * s1.y[t][j];
                }
                for (int j = 0; j < H; ++j) {
                    double svv = 0.0;
                    for (int v = 0; v < V; ++v) svv += Wy.at(v, j) * dlogit[v];
                    dz[t][j] += svv;
                }
            }
        }
        // 块 1 反向 -> dz1; 块 0 反向 -> dp (梯度 wrt p = h + 位置编码)
        vector<vector<double>> dz1(T, vector<double>(H, 0.0));
        block_bwd(blocks[1], s1, s0.y, dz, dz1);
        vector<vector<double>> dp(T, vector<double>(H, 0.0));
        block_bwd(blocks[0], s0, p, dz1, dp);

        // P 梯度 + RNN BPTT
        for (int t = 0; t < T; ++t) {
            int pt = min(t, T_MAX - 1);
            for (int j = 0; j < H; ++j) P.grad[(size_t)pt * H + j] += dp[t][j];
        }
        vector<double> dh_next(H, 0.0), dh(H, 0.0), dh_raw(H, 0.0);
        for (int t = T - 1; t >= 0; --t) {
            for (int j = 0; j < H; ++j) dh[j] = dh_next[j] + dp[t][j];
            for (int j = 0; j < H; ++j) dh_raw[j] = dh[j] * (1.0 - h[t][j] * h[t][j]);
            for (int j = 0; j < H; ++j) {
                bh.grad[j] += dh_raw[j];
                for (int k = 0; k < H; ++k) {
                    Wx.grad[(size_t)j * H + k] += dh_raw[j] * emb[t][k];
                    Wh.grad[(size_t)j * H + k] += dh_raw[j] * hprev[t][k];
                }
            }
            for (int k = 0; k < H; ++k) {
                double s1 = 0.0, s2 = 0.0;
                for (int j = 0; j < H; ++j) {
                    s1 += Wx.at(j, k) * dh_raw[j];
                    s2 += Wh.at(j, k) * dh_raw[j];
                }
                E.grad[(size_t)ids[t] * H + k] += s1;
                dh_next[k] = s2;
            }
        }
        return loss;
    }

    // 前向只算最后一位置的 logits (用于续写采样)
    vector<double> forward_last_logits(const vector<int>& ids) {
        const int T = (int)ids.size();
        const double inv_dk = 1.0 / sqrt((double)DK);
        vector<vector<double>> h(T, vector<double>(H, 0.0));
        vector<vector<double>> p(T, vector<double>(H, 0.0));
        vector<vector<double>> cur(T, vector<double>(H, 0.0)); // 当前块的输入
        vector<vector<double>> r(T, vector<double>(H, 0.0));   // LN1 输出
        double mn, iv;
        vector<double> h_prev(H, 0.0), pre(H);
        for (int t = 0; t < T; ++t) {
            int id = ids[t];
            for (int j = 0; j < H; ++j) {
                double s = bh.val[j];
                for (int k = 0; k < H; ++k)
                    s += Wx.at(j, k) * E.at(id, k) + Wh.at(j, k) * h_prev[k];
                pre[j] = s;
            }
            for (int j = 0; j < H; ++j) h[t][j] = tanh(pre[j]);
            int pt = min(t, T_MAX - 1);
            for (int j = 0; j < H; ++j) p[t][j] = h[t][j] + P.at(pt, j);
            h_prev = h[t];
        }
        cur = p;
        for (int bl = 0; bl < NBLK; ++bl) {
            const Block& b = blocks[bl];
            for (int t = 0; t < T; ++t)
                layernorm_fwd(cur[t], b.LN1g.val, b.LN1b.val, r[t], mn, iv);
            vector<vector<double>> att(T, vector<double>(H, 0.0));
            for (int hd = 0; hd < NH; ++hd) {
                vector<vector<double>> qH(T, vector<double>(DK, 0.0));
                vector<vector<double>> kH(T, vector<double>(DK, 0.0));
                vector<vector<double>> vH(T, vector<double>(DK, 0.0));
                for (int i = 0; i < T; ++i)
                    for (int d = 0; d < DK; ++d) {
                        double sq = 0.0, sk = 0.0, sv = 0.0;
                        for (int j = 0; j < H; ++j) {
                            sq += b.Wq.at(hd * DK + d, j) * r[i][j];
                            sk += b.Wk.at(hd * DK + d, j) * r[i][j];
                            sv += b.Wv.at(hd * DK + d, j) * r[i][j];
                        }
                        qH[i][d] = sq; kH[i][d] = sk; vH[i][d] = sv;
                    }
                vector<double> sc(T);
                for (int i = 0; i < T; ++i) {
                    double mx = -1e30;
                    for (int j = 0; j <= i; ++j) {
                        double dot = 0.0;
                        for (int d = 0; d < DK; ++d) dot += qH[i][d] * kH[j][d];
                        sc[j] = dot * inv_dk;
                        mx = max(mx, sc[j]);
                    }
                    double sum = 0.0;
                    for (int j = 0; j <= i; ++j) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
                    for (int d = 0; d < DK; ++d) {
                        double s = 0.0;
                        for (int j = 0; j <= i; ++j) s += (sc[j] / sum) * vH[j][d];
                        att[i][hd * DK + d] = s;
                    }
                }
            }
            vector<vector<double>> u(T, vector<double>(H, 0.0));
            vector<vector<double>> qq(T, vector<double>(H, 0.0));
            vector<vector<double>> ff_g(T, vector<double>(H1, 0.0));
            for (int i = 0; i < T; ++i) {
                vector<double> a(H, 0.0);
                for (int o = 0; o < H; ++o) {
                    double s = 0.0;
                    for (int hh = 0; hh < H; ++hh) s += b.Wo.at(o, hh) * att[i][hh];
                    a[o] = s;
                }
                for (int j = 0; j < H; ++j) u[i][j] = r[i][j] + a[j];
                layernorm_fwd(u[i], b.LN2g.val, b.LN2b.val, qq[i], mn, iv);
                for (int m = 0; m < H1; ++m) {
                    double s = b.b1.val[m];
                    for (int j = 0; j < H; ++j) s += b.W1.at(m, j) * qq[i][j];
                    ff_g[i][m] = gelu(s);
                }
                for (int j = 0; j < H; ++j) {
                    double s = b.b2.val[j];
                    for (int m = 0; m < H1; ++m) s += b.W2.at(j, m) * ff_g[i][m];
                    cur[i][j] = u[i][j] + s;
                }
            }
        }
        vector<double> logit(V);
        for (int v = 0; v < V; ++v) {
            double s = by.val[v];
            for (int j = 0; j < H; ++j) s += Wy.at(v, j) * cur[T - 1][j];
            logit[v] = s;
        }
        return logit;
    }

    // 自检: 用递增前缀(与 generate 相同)检查各位置 argmax 预测, 返回错误数
    int selftest(const vector<int>& ids, const vector<string>& itos) {
        int T = (int)ids.size(), bad = 0;
        for (int t = 0; t + 1 < T; ++t) {
            vector<int> pre(ids.begin(), ids.begin() + t + 1);
            vector<double> logit = forward_last_logits(pre);
            int am = (int)(max_element(logit.begin(), logit.end()) - logit.begin());
            if (am != ids[t + 1]) ++bad;
        }
        return bad;
    }

    // 续写: 温度采样 (屏蔽 <BOS>/<UNK>)
    string generate(const vector<int>& prefix, double temperature, int maxlen,
                    int bos_id, int eos_id, int unk_id,
                    const vector<string>& itos, mt19937& rng) {
        vector<int> seq;
        if (prefix.empty() || prefix[0] != bos_id) seq.push_back(bos_id);
        seq.insert(seq.end(), prefix.begin(), prefix.end());
        uniform_real_distribution<double> uni(0.0, 1.0);
        string out;
        for (int n = 0; n < maxlen; ++n) {
            vector<double> logit = forward_last_logits(seq);
            for (double& l : logit) l /= max(temperature, 1e-3);
            double mx = *max_element(logit.begin(), logit.end());
            vector<double> p(V);
            double sum = 0.0;
            for (int v = 0; v < V; ++v) { p[v] = exp(logit[v] - mx); sum += p[v]; }
            double m_bos = p[bos_id], m_unk = p[unk_id];
            p[bos_id] = 0.0; p[unk_id] = 0.0;
            sum -= m_bos + m_unk;

            int pick = -1;
            if (sum <= 0.0) {
                pick = (int)(max_element(logit.begin(), logit.end()) - logit.begin());
            } else {
                double r = uni(rng) * sum;
                double cum = 0.0;
                for (int v = 0; v < V; ++v) {
                    cum += p[v];
                    if (r <= cum) { pick = v; break; }
                }
                if (pick < 0) pick = V - 1;
            }
            if (pick == eos_id) break;
            out += itos[pick];
            seq.push_back(pick);
            if ((int)seq.size() >= T_MAX) break;
        }
        return out;
    }

    // ===================== v7: 权重文件保存 / 加载 =====================
    // 二进制格式: magic(u32) | version(u32) | 6 个结构维度(int)
    //           | 词表大小(int) + 每个字符(长度int + 字节)
    //           | 全部参数: 每个 Param = rows(int) + cols(int) + rows*cols 个 double
    // 只保存 val (权重), 不保存 Adam 动量 (推理不需要)。
    static constexpr uint32_t W_MAGIC = 0x41564238;   // "AVB8" (v0.8 专家)
    static constexpr uint32_t W_VERSION = 1;

    static bool write_param(FILE* f, const Param& p) {
        if (fwrite(&p.rows, sizeof(int), 1, f) != 1) return false;
        if (fwrite(&p.cols, sizeof(int), 1, f) != 1) return false;
        return fwrite(p.val.data(), sizeof(double), p.val.size(), f) == p.val.size();
    }
    // 读取一个 Param, 并校验形状必须为 er x ec
    static bool read_param(FILE* f, Param& p, int er, int ec) {
        int r = 0, c = 0;
        if (fread(&r, sizeof(int), 1, f) != 1) return false;
        if (fread(&c, sizeof(int), 1, f) != 1) return false;
        if (r != er || c != ec) return false;
        size_t n = (size_t)r * (size_t)c;
        p.val.assign(n, 0.0);
        p.grad.assign(n, 0.0);
        p.m.assign(n, 0.0);
        p.v.assign(n, 0.0);
        return fread(p.val.data(), sizeof(double), n, f) == n;
    }

    // 保存全部权重 + 词表到 path
    bool save_weights(const string& path, const Tokenizer& tok) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) return false;
        bool ok = true;
        auto w32 = [&](uint32_t v) { if (ok && fwrite(&v, 4, 1, f) != 1) ok = false; };
        auto wint = [&](int v) { if (ok && fwrite(&v, sizeof(int), 1, f) != 1) ok = false; };
        auto wp = [&](const Param& p) { if (ok && !write_param(f, p)) ok = false; };
        w32(W_MAGIC);
        w32(W_VERSION);
        wint(V); wint(H); wint(NH); wint(H1); wint(T_MAX); wint(NBLK);
        int vs = (int)tok.itos.size();
        wint(vs);
        if (ok) for (const string& s : tok.itos) {
            int len = (int)s.size();
            wint(len);
            if (ok && fwrite(s.data(), 1, (size_t)len, f) != (size_t)len) ok = false;
        }
        if (ok) { wp(E); wp(Wx); wp(Wh); wp(bh); wp(P); wp(Wy); wp(by); }
        if (ok) for (const Block& b : blocks) {
            wp(b.LN1g); wp(b.LN1b); wp(b.LN2g); wp(b.LN2b);
            wp(b.Wq); wp(b.Wk); wp(b.Wv); wp(b.Wo);
            wp(b.W1); wp(b.b1); wp(b.W2); wp(b.b2);
        }
        fclose(f);
        return ok;
    }

    // 从 path 加载权重 + 词表; 校验 magic/版本/结构维度/词表一致, 失败返回 false
    bool load_weights(const string& path, Tokenizer& tok) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        bool ok = true;
        auto r32 = [&](uint32_t& v) { if (ok && fread(&v, 4, 1, f) != 1) ok = false; };
        auto rint = [&](int& v) { if (ok && fread(&v, sizeof(int), 1, f) != 1) ok = false; };
        auto rp = [&](Param& p, int er, int ec) { if (ok && !read_param(f, p, er, ec)) ok = false; };
        uint32_t magic = 0, ver = 0;
        r32(magic); r32(ver);
        if (ok && magic != W_MAGIC) ok = false;
        if (ok && ver != W_VERSION) ok = false;
        int dV = 0, dH = 0, dNH = 0, dH1 = 0, dT = 0, dNB = 0;
        rint(dV); rint(dH); rint(dNH); rint(dH1); rint(dT); rint(dNB);
        if (ok && (dV != V || dH != H || dNH != NH || dH1 != H1 ||
                   dT != T_MAX || dNB != NBLK)) ok = false;
        int vs = 0;
        rint(vs);
        if (ok && (vs <= 0 || vs > 100000)) ok = false;
        if (ok) {
            Tokenizer t2;
            for (int i = 0; i < vs; ++i) {
                int len = 0;
                rint(len);
                if (!ok || len <= 0 || len > 64) { ok = false; break; }
                string s((size_t)len, '\0');
                if (ok && fread(&s[0], 1, (size_t)len, f) != (size_t)len) { ok = false; break; }
                t2.add(s);
            }
            if (ok) {
                if (t2.itos.size() != tok.itos.size()) ok = false;
                if (ok) for (size_t i = 0; i < tok.itos.size(); ++i)
                    if (t2.itos[i] != tok.itos[i]) { ok = false; break; }
            }
        }
        if (ok) { rp(E, V, H); rp(Wx, H, H); rp(Wh, H, H); rp(bh, H, 1);
                  rp(P, T_MAX, H); rp(Wy, V, H); rp(by, V, 1); }
        if (ok) for (Block& b : blocks) {
            rp(b.LN1g, H, 1); rp(b.LN1b, H, 1); rp(b.LN2g, H, 1); rp(b.LN2b, H, 1);
            rp(b.Wq, H, H); rp(b.Wk, H, H); rp(b.Wv, H, H); rp(b.Wo, H, H);
            rp(b.W1, H1, H); rp(b.b1, H1, 1); rp(b.W2, H, H1); rp(b.b2, H, 1);
        }
        fclose(f);
        return ok;
    }
};

// --------------------------- 主程序 (v0.8 专家版) ---------------------------
// 从语料文件读取 -> 训练/加载权重 -> 自检+演示 -> 交互
// 支持 -q <输入>: 路由器专用单次问答模式 (只输出回答)
// 问答检索: 两段 UTF-8 字符序列的字符重叠相似度 |a∩b| / max(|a|,|b|)
static double overlap_score(const vector<string>& a, const vector<string>& b) {
    if (a.empty() || b.empty()) return 0.0;
    double inter = 0.0;
    for (auto& x : a)
        if (find(b.begin(), b.end(), x) != b.end()) ++inter;
    return inter / max(a.size(), b.size());
}

static vector<string> load_corpus_lines(const string& path) {
    vector<string> out;
    FILE* f = fopen(path.c_str(), "r");
    if (!f) {
        printf("[错误] 无法打开语料文件: %s\n", path.c_str());
        exit(1);
    }
    char buf[8192];
    while (fgets(buf, sizeof(buf), f)) {
        string s = trim(buf);
        if (s.empty()) continue;
        size_t pos = s.find("：");          // 防御: 若还带 "前缀:标题：" 则剥离
        if (pos != string::npos) s = s.substr(pos + 1);
        s = trim(s);
        if (s.empty()) continue;
        out.push_back(s);
    }
    fclose(f);
    return out;
}

int main(int argc, char** argv) {
    bool quiet = false;
    string query;
    if (argc >= 3 && string(argv[1]) == "-q") { quiet = true; query = argv[2]; }

    mt19937 rng(20240417);

    Tokenizer tok;
    tok.bos = tok.add("<BOS>");
    tok.eos = tok.add("<EOS>");
    tok.unk = tok.add("<UNK>");

    // 读取专家语料 (每行一段)
    vector<string> corpus = load_corpus_lines(CORPUS_FILE);
    if (corpus.empty()) { printf("[错误] 语料为空: %s\n", CORPUS_FILE); return 1; }

    // 构建训练序列: <BOS> + 字符 + <EOS>
    vector<vector<int>> lines;
    int total_chars = 0;
    for (auto& s : corpus) {
        vector<string> cs = utf8_split(s);
        vector<int> ids;
        ids.push_back(tok.bos);
        for (auto& c : cs) { tok.add(c); ids.push_back(tok.get(c)); }
        ids.push_back(tok.eos);
        lines.push_back(ids);
        total_chars += (int)cs.size();
    }

    // v0.8 结构同 v7: hidden 128, FFN 256, Transformer 块 2 层
    const int H = 128, NH = 4, FF = 256, T_MAX = 48, NBLK = 2;
    const string WFILE = WEIGHTS_FILE;
    CharRNNTransformer model((int)tok.itos.size(), H, NH, FF, T_MAX, NBLK, rng);

    if (!quiet) {
        cout << "========== 客尘AI集群v1.0 · 专家【" << EXPERT_NAME << "】 (v0.8 架构) ==========\n";
        cout << "迷你 Transformer (RNN + 2 层多头自注意力) 字符级中文语言模型\n";
        cout << "语料 : " << (int)corpus.size() << " 段, 共 " << total_chars << " 字符 (" << CORPUS_FILE << ")\n";
        cout << "词表 : " << (int)tok.itos.size() << " 字符 (含 <BOS>/<EOS>/<UNK>)\n";
        cout << "结构 : Embedding(" << (int)tok.itos.size() << "x" << H << ")\n";
        cout << "      -> RNN(hidden=" << H << ") + 可学习位置编码\n";
        for (int bl = 0; bl < NBLK; ++bl)
            cout << "      -> 块" << (bl + 1) << ": LayerNorm -> 多头自注意力(" << NH << "头, 因果掩码, dk="
                 << H / NH << ") -> 残差+LayerNorm -> FFN(" << H << "->" << FF << "->" << H << ", GELU) -> 残差\n";
        cout << "      -> Softmax 预测下一字符\n";
        cout << "参数 : " << model.param_count() << "  (优化器 Adam, 梯度裁剪, BPTT + 2 层注意力反向)\n";
    }

    // ---- 权重自动加载 (秒启动) ----
    auto t_start = chrono::steady_clock::now();
    bool loaded = false;
    {
        FILE* probe = fopen(WFILE.c_str(), "rb");
        bool exists = (probe != nullptr);
        if (probe) fclose(probe);
        if (exists && model.load_weights(WFILE, tok)) {
            loaded = true;
            auto ms = chrono::duration_cast<chrono::milliseconds>(
                          chrono::steady_clock::now() - t_start).count();
            if (quiet)
                fprintf(stderr, "[权重] 已加载 %s (专家 %s)\n", WFILE.c_str(), EXPERT_NAME);
            else
                printf("[权重] 检测到 %s，已加载专家【%s】 -> 跳过训练 (加载耗时 %lld ms)\n\n",
                       WFILE.c_str(), EXPERT_NAME, (long long)ms);
        } else if (exists) {
            printf("[权重] 检测到 %s 但版本/词表不匹配，将重新训练并覆盖...\n", WFILE.c_str());
        }
    }

    if (!loaded) {
        // ---- 首次运行: 训练 ----
        if (!quiet) cout << "训练中...\n\n";
        else fprintf(stderr, "[训练] 专家【%s】首次运行, 训练 %zu 段语料...\n", EXPERT_NAME, corpus.size());

        const int MAX_EPOCHS = 300;
        const double STOP_LOSS = 0.02;
        const double LR0 = 0.01;
        vector<pair<int, double>> losses;

        int epoch = 0;
        for (epoch = 1; epoch <= MAX_EPOCHS; ++epoch) {
            model.zero_grad();
            double total = 0.0;
            int cnt = 0;
            for (auto& ids : lines) {
                total += model.train_line(ids);
                cnt += (int)ids.size() - 1;
            }
            model.clip_grads(5.0);
            double lr = LR0 * pow(0.5, (epoch - 1) / 100);
            model.adam_update(lr, epoch);
            double avg = total / cnt;
            if (epoch == 1 || epoch % 20 == 0 || epoch == MAX_EPOCHS) {
                losses.push_back({epoch, avg});
                if (!quiet) printf("[train] epoch %5d   loss %.4f\n", epoch, avg);
            }
            if (avg < STOP_LOSS) break;
        }

        if (!quiet) {
            printf("\n[loss 曲线] ");
            for (auto& pr : losses) printf("%d:%.3f  ", pr.first, pr.second);
            printf("\n训练完成: 共 %d epochs, 最终平均 loss = %.4f\n", min(epoch, MAX_EPOCHS), losses.back().second);
            printf("[说明] 语料 = %zu 段 (%s), 共 %d 字符, 全部共享同一词表与同一套参数记忆。\n",
                   corpus.size(), CORPUS_FILE, total_chars);
            printf("       <BOS> 后首字符存在固有歧义 (一段一开头, 概率均摊);\n");
            printf("       若多段共享相同前缀, 该前缀之后的字符也存歧义, 为理论下界。\n\n");
        }

        // ---- 训练完成后自动保存权重 ----
        if (model.save_weights(WFILE, tok)) {
            if (!quiet) {
                printf("[权重] 训练完成，专家【%s】权重已自动保存到 %s\n", EXPERT_NAME, WFILE.c_str());
                printf("       (共 %zu 个参数, 文件大小 %.2f MB, 下次启动将直接加载跳过训练)\n\n",
                       model.param_count(), model.param_count() * sizeof(double) / 1048576.0);
            } else {
                fprintf(stderr, "[权重] 已保存到 %s\n", WFILE.c_str());
            }
        } else {
            printf("[权重] 权重保存失败!\n\n");
        }
    }

    // ---- -q 模式: 路由器调用, 只输出回答 ----
    if (quiet) {
        // 第一步: 语料问答检索 (Q=A 对, 字符重叠相似度 >= 0.5 直接作答)
        vector<string> qcs = utf8_split(query);
        string best_a;
        double best = 0.0;
        for (auto& s : corpus) {
            size_t eq = s.find('=');
            if (eq == string::npos || eq == 0 || eq + 1 >= s.size()) continue;
            string q = s.substr(0, eq), a = s.substr(eq + 1);
            double sc = overlap_score(qcs, utf8_split(q));
            if (sc > best) { best = sc; best_a = a; }
        }
        if (best >= 0.5) {
            cout << best_a << "\n";
            return 0;
        }
        // 第二步: 模型续写生成
        vector<string> cs = utf8_split(query);
        vector<int> ids;
        for (auto& c : cs) ids.push_back(tok.get(c));
        string gen = model.generate(ids, 0.2, 40, tok.bos, tok.eos, tok.unk, tok.itos, rng);
        string reply;
        if (!gen.empty() && gen[0] == '=') {
            reply = gen.substr(1);                 // 问答对: 只显示回答
        } else {
            reply = query + gen;                   // 续写: 显示完整内容
        }
        cout << reply << "\n";
        return 0;
    }

    // ---- 自检: 逐位置核对 ----
    printf("\n[自检] 逐位置核对 (应与语料完全一致):\n");
    {
        int bad = 0, total = 0;
        for (auto& ids : lines) {
            bad += model.selftest(ids, tok.itos);
            total += (int)ids.size() - 1;
        }
        printf("  命中 %d/%d 位置 (未命中为固有歧义: 多段共享 <BOS> 后首字符或相同前缀)\n",
               total - bad, total);
    }

    // ---- 自动演示续写/问答 (来自 DEMOS 宏, '|' 分隔) ----
    cout << "\n[演示] 专家【" << EXPERT_NAME << "】续写/问答示例:\n";
    {
        string demos = DEMOS;
        size_t start = 0;
        int n = 0;
        while (start < demos.size() && n < 6) {
            size_t bar = demos.find('|', start);
            string p = demos.substr(start, bar == string::npos ? string::npos : bar - start);
            p = trim(p);
            if (!p.empty()) {
                vector<string> cs = utf8_split(p);
                vector<int> ids;
                for (auto& c : cs) ids.push_back(tok.get(c));
                string gen = model.generate(ids, 0.6, 40, tok.bos, tok.eos, tok.unk, tok.itos, rng);
                if (!gen.empty() && gen[0] == '=') {          // 问答对: 只显示回答
                    cout << "  [问答] 你: " << p << "  ->  AI: " << gen.substr(1) << "\n";
                } else {                                      // 续写
                    cout << "  [续写] \"" << p << gen << "\"\n";
                }
                ++n;
            }
            if (bar == string::npos) break;
            start = bar + 1;
        }
    }
    cout << "\n";

    // ---- 交互模式 ----
    cout << "========== 交互模式 ==========\n";
    cout << "输入问题(如\"你好\"/\"你叫什么\") -> AI 直接回答;\n";
    cout << "输入中文开头(几个字) -> 模型续写; 直接回车 -> 随机创作; quit/exit -> 退出\n";
    while (true) {
        cout << "\n> " << flush;
        string line;
        if (!getline(cin, line)) { cout << "再见!\n"; break; }
        line = trim(line);
        if (line.empty()) {
            vector<int> ids = {tok.bos};
            string gen = model.generate(ids, 0.7, 30, tok.bos, tok.eos, tok.unk, tok.itos, rng);
            cout << "  " << gen << "\n";
            continue;
        }
        if (line == "quit" || line == "exit" || line == "q" || line == "退出") {
            cout << "再见!\n";
            break;
        }
        vector<string> cs = utf8_split(line);
        vector<int> ids;
        for (auto& c : cs) ids.push_back(tok.get(c));
        string gen = model.generate(ids, 0.6, 30, tok.bos, tok.eos, tok.unk, tok.itos, rng);
        if (!gen.empty() && gen[0] == '=') {          // 问答对: 自动隐藏 "Q=", 只显示回答
            cout << "  AI: " << gen.substr(1) << "\n";
        } else {                                      // 诗词/名言/宇宙 续写
            cout << "  " << line << gen << "\n";
        }
    }
    return 0;
}
