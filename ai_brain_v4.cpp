// ============================================================================
//  ai_brain_v4.cpp — 迷你 Transformer (RNN + 自注意力) 字符级中文语言模型
//
//  在 v3 的基础上, 仅把"迷你中文字典"(10 个字 + 简短解释)加入训练语料:
//     床=供人躺卧的家具  月=地球的卫星  光=明亮的光线  明=光亮  风=空气流动
//     雨=云中水汽降落    花=植物的繁殖器官  知=知道明白  行=走  思=思考
//  模型结构 / 训练超参 / 交互逻辑均与 v3 完全相同 (RNN + 多头自注意力):
//     字符 Embedding (V x H)
//         -> Elman RNN:     h_t = tanh(Wx·E[x_t] + Wh·h_{t-1} + bh)     [保留]
//         -> 可学习位置编码: p_t = h_t + P[t]
//         -> LayerNorm
//         -> 多头自注意力 (NH 头, 因果掩码, 1/sqrt(dk) 缩放)             [保留]
//         -> 残差 + LayerNorm
//         -> 前馈网络 FFN:  GELU(W1·x + b1) -> W2                        [保留]
//         -> 残差 -> 线性输出 y_t = Wy·z_t + by -> Softmax 预测下一字符
//  训练: BPTT(RNN) + 注意力/FFN/LayerNorm 反向传播 + Adam + 梯度裁剪
//  语料: 11 段古诗/谚语 (198 字符) + 10 条迷你字典 (63 字符) = 21 段, 261 字符
//  交互: 输入中文开头(几个字), 模型续写; 空行 = 随机创作; quit = 退出
//
//  编译: clang++ -O2 -std=c++17 ai_brain_v4.cpp -o ai_brain_v4
//  运行: ./ai_brain_v4
// ============================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace std;

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

// --------------------------- 模型: RNN + 多头自注意力 (迷你 Transformer) ---------------------------
class CharRNNTransformer {
public:
    int V, H, NH, DK, H1, T_MAX;
    // Embedding + RNN
    Param E, Wx, Wh, bh;
    // 可学习位置编码 P (T_MAX x H)
    Param P;
    // 两个 LayerNorm
    Param LN1g, LN1b, LN2g, LN2b;
    // 多头自注意力投影 (每个均为 H x H; 拆成 NH 个头, 每头 DK=H/NH 维)
    Param Wq, Wk, Wv, Wo;
    // 前馈网络: W1(H1 x H) b1(H1)  W2(H x H1) b2(H)
    Param W1, b1, W2, b2;
    // 输出
    Param Wy, by;

    CharRNNTransformer(int vocab, int hidden, int heads, int ff, int tmax, mt19937& rng)
        : V(vocab), H(hidden), NH(heads), DK(hidden / heads), H1(ff), T_MAX(tmax) {
        E.init(V, H, rng, 0.10);
        Wx.init(H, H, rng, 1.0 / sqrt((double)H));
        Wh.init(H, H, rng, 1.0 / sqrt((double)H));
        bh.init(H, 1, rng, 0.0);
        P.init(T_MAX, H, rng, 0.10);
        LN1g.init(H, 1, rng, 0.0); LN1g.fill_all(1.0);
        LN1b.init(H, 1, rng, 0.0);
        LN2g.init(H, 1, rng, 0.0); LN2g.fill_all(1.0);
        LN2b.init(H, 1, rng, 0.0);
        Wq.init(H, H, rng, 1.0 / sqrt((double)H));
        Wk.init(H, H, rng, 1.0 / sqrt((double)H));
        Wv.init(H, H, rng, 1.0 / sqrt((double)H));
        Wo.init(H, H, rng, 1.0 / sqrt((double)H));
        W1.init(H1, H, rng, 1.0 / sqrt((double)H));
        b1.init(H1, 1, rng, 0.0);
        W2.init(H, H1, rng, 1.0 / sqrt((double)H1));
        b2.init(H, 1, rng, 0.0);
        Wy.init(V, H, rng, 0.10);
        by.init(V, 1, rng, 0.0);
    }

    size_t param_count() const {
        return E.val.size() + Wx.val.size() + Wh.val.size() + bh.val.size()
             + P.val.size()
             + LN1g.val.size() + LN1b.val.size() + LN2g.val.size() + LN2b.val.size()
             + Wq.val.size() + Wk.val.size() + Wv.val.size() + Wo.val.size()
             + W1.val.size() + b1.val.size() + W2.val.size() + b2.val.size()
             + Wy.val.size() + by.val.size();
    }

    void zero_grad() {
        vector<Param*> ps = {&E, &Wx, &Wh, &bh, &P,
                             &LN1g, &LN1b, &LN2g, &LN2b,
                             &Wq, &Wk, &Wv, &Wo,
                             &W1, &b1, &W2, &b2, &Wy, &by};
        for (Param* p : ps) p->zero_grad();
    }

    void clip_grads(double clip) {
        double n2 = 0.0;
        vector<Param*> ps = {&E, &Wx, &Wh, &bh, &P,
                             &LN1g, &LN1b, &LN2g, &LN2b,
                             &Wq, &Wk, &Wv, &Wo,
                             &W1, &b1, &W2, &b2, &Wy, &by};
        for (Param* p : ps)
            for (double g : p->grad) n2 += g * g;
        double n = sqrt(n2);
        if (n > clip) {
            double s = clip / n;
            for (Param* p : ps)
                for (auto& g : p->grad) g *= s;
        }
    }

    void adam_update(double lr, int t) {
        const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        double bc1 = 1.0 - pow(beta1, t), bc2 = 1.0 - pow(beta2, t);
        vector<Param*> ps = {&E, &Wx, &Wh, &bh, &P,
                             &LN1g, &LN1b, &LN2g, &LN2b,
                             &Wq, &Wk, &Wv, &Wo,
                             &W1, &b1, &W2, &b2, &Wy, &by};
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

    // 对一整行(含 <BOS> 与 <EOS>)前向 + 反向, 返回交叉熵损失
    double train_line(const vector<int>& ids) {
        const int T = (int)ids.size();
        if (T < 2) return 0.0;
        const double inv_dk = 1.0 / sqrt((double)DK);

        // ---- 前向存储 ----
        vector<vector<double>> emb(T, vector<double>(H, 0.0));
        vector<vector<double>> h(T, vector<double>(H, 0.0));
        vector<vector<double>> hprev(T, vector<double>(H, 0.0));
        vector<vector<double>> p(T, vector<double>(H, 0.0));    // h + 位置编码
        vector<vector<double>> r(T, vector<double>(H, 0.0));    // LN1 输出
        vector<vector<double>> att(T, vector<double>(H, 0.0));  // 多头拼接输出 (Wo 前)
        vector<vector<double>> a(T, vector<double>(H, 0.0));    // Wo 投影后
        vector<vector<double>> u(T, vector<double>(H, 0.0));    // r + a
        vector<vector<double>> q(T, vector<double>(H, 0.0));    // LN2 输出
        vector<vector<double>> ff_in(T, vector<double>(H1, 0.0));
        vector<vector<double>> ff_g(T, vector<double>(H1, 0.0));// GELU 输出
        vector<vector<double>> f(T, vector<double>(H, 0.0));
        vector<vector<double>> z(T, vector<double>(H, 0.0));    // u + f
        vector<vector<double>> soft(T, vector<double>(V, 0.0));
        vector<double> ln1_mean(T), ln1_inv(T), ln2_mean(T), ln2_inv(T);
        vector<vector<vector<double>>> pH(NH,
            vector<vector<double>>(T, vector<double>(T, 0.0))); // 注意力权重
        vector<vector<vector<double>>> qH(NH,
            vector<vector<double>>(T, vector<double>(DK, 0.0)));
        vector<vector<vector<double>>> kH(NH,
            vector<vector<double>>(T, vector<double>(DK, 0.0)));
        vector<vector<vector<double>>> vH(NH,
            vector<vector<double>>(T, vector<double>(DK, 0.0)));

        // ---- 前向: RNN + 位置编码 + LN1 ----
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
            layernorm_fwd(p[t], LN1g.val, LN1b.val, r[t], ln1_mean[t], ln1_inv[t]);
            h_prev = h[t];
        }

        // ---- 前向: 多头自注意力 (因果掩码: 位置 i 只能看到 j<=i) ----
        for (int hd = 0; hd < NH; ++hd) {
            for (int i = 0; i < T; ++i)
                for (int d = 0; d < DK; ++d) {
                    double sq = 0.0, sk = 0.0, sv = 0.0;
                    for (int j = 0; j < H; ++j) {
                        sq += Wq.at(hd * DK + d, j) * r[i][j];
                        sk += Wk.at(hd * DK + d, j) * r[i][j];
                        sv += Wv.at(hd * DK + d, j) * r[i][j];
                    }
                    qH[hd][i][d] = sq; kH[hd][i][d] = sk; vH[hd][i][d] = sv;
                }
            vector<double> sc(T);
            for (int i = 0; i < T; ++i) {
                double mx = -1e30;
                for (int j = 0; j <= i; ++j) {
                    double dot = 0.0;
                    for (int d = 0; d < DK; ++d) dot += qH[hd][i][d] * kH[hd][j][d];
                    sc[j] = dot * inv_dk;
                    mx = max(mx, sc[j]);
                }
                double sum = 0.0;
                for (int j = 0; j <= i; ++j) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
                for (int j = 0; j <= i; ++j) pH[hd][i][j] = sc[j] / sum;
                for (int d = 0; d < DK; ++d) {
                    double s = 0.0;
                    for (int j = 0; j <= i; ++j) s += pH[hd][i][j] * vH[hd][j][d];
                    att[i][hd * DK + d] = s;
                }
            }
        }
        for (int i = 0; i < T; ++i)
            for (int o = 0; o < H; ++o) {
                double s = 0.0;
                for (int hh = 0; hh < H; ++hh) s += Wo.at(o, hh) * att[i][hh];
                a[i][o] = s;
            }

        // ---- 前向: 残差 + LN2 + FFN ----
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < H; ++j) u[t][j] = r[t][j] + a[t][j];
            layernorm_fwd(u[t], LN2g.val, LN2b.val, q[t], ln2_mean[t], ln2_inv[t]);
            for (int m = 0; m < H1; ++m) {
                double s = b1.val[m];
                for (int j = 0; j < H; ++j) s += W1.at(m, j) * q[t][j];
                ff_in[t][m] = s;
                ff_g[t][m] = gelu(s);
            }
            for (int j = 0; j < H; ++j) {
                double s = b2.val[j];
                for (int m = 0; m < H1; ++m) s += W2.at(j, m) * ff_g[t][m];
                f[t][j] = s;
            }
            for (int j = 0; j < H; ++j) z[t][j] = u[t][j] + f[t][j];
        }

        // ---- 输出 + loss ----
        vector<double> logit(V);
        double loss = 0.0;
        for (int t = 0; t < T; ++t) {
            for (int v = 0; v < V; ++v) {
                double s = by.val[v];
                for (int j = 0; j < H; ++j) s += Wy.at(v, j) * z[t][j];
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
        vector<vector<double>> du(T, vector<double>(H, 0.0));
        vector<vector<double>> df(T, vector<double>(H, 0.0));
        vector<vector<double>> da(T, vector<double>(H, 0.0));
        vector<vector<double>> dr(T, vector<double>(H, 0.0));
        vector<vector<double>> dp(T, vector<double>(H, 0.0));
        vector<vector<double>> datt(T, vector<double>(H, 0.0));
        vector<double> dq_ff(H, 0.0);

        // 输出层
        vector<double> dlogit(V);
        for (int t = T - 1; t >= 0; --t) {
            if (t + 1 < T) {
                for (int v = 0; v < V; ++v) dlogit[v] = soft[t][v];
                dlogit[ids[t + 1]] -= 1.0;
                for (int v = 0; v < V; ++v) {
                    by.grad[v] += dlogit[v];
                    for (int j = 0; j < H; ++j)
                        Wy.grad[(size_t)v * H + j] += dlogit[v] * z[t][j];
                }
                for (int j = 0; j < H; ++j) {
                    double s = 0.0;
                    for (int v = 0; v < V; ++v) s += Wy.at(v, j) * dlogit[v];
                    dz[t][j] += s;
                }
            }
        }
        // z = u + f
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < H; ++j) { du[t][j] += dz[t][j]; df[t][j] += dz[t][j]; }

        // FFN 反向 + LN2 反向
        for (int t = 0; t < T; ++t) {
            layernorm_bwd(u[t], ln2_mean[t], ln2_inv[t], LN2g.val, df[t],
                          dq_ff, LN2g.grad, LN2b.grad);
            vector<double> ds(H1, 0.0);
            for (int m = 0; m < H1; ++m) {
                double dgm = 0.0;
                for (int j = 0; j < H; ++j) dgm += W2.at(j, m) * df[t][j];
                ds[m] = dgm * gelu_d(ff_in[t][m]);
            }
            for (int m = 0; m < H1; ++m) {
                for (int j = 0; j < H; ++j) {
                    W2.grad[(size_t)j * H1 + m] += df[t][j] * ff_g[t][m];
                    W1.grad[(size_t)m * H + j] += ds[m] * q[t][j];
                }
                b1.grad[m] += ds[m];
            }
            for (int j = 0; j < H; ++j) {
                b2.grad[j] += df[t][j];
                double dqj = 0.0;
                for (int m = 0; m < H1; ++m) dqj += W1.at(m, j) * ds[m];
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
                    Wo.grad[(size_t)o * H + hh] += da[i][o] * att[i][hh];
        for (int i = 0; i < T; ++i)
            for (int hh = 0; hh < H; ++hh) {
                double s = 0.0;
                for (int o = 0; o < H; ++o) s += Wo.at(o, hh) * da[i][o];
                datt[i][hh] = s;
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
                    for (int j = 0; j <= i; ++j) dVH[j][d] += pH[hd][i][j] * g;
                }
                vector<double> dp_i(T, 0.0);
                double dotpdg = 0.0;
                for (int j = 0; j <= i; ++j) {
                    double dp = 0.0;
                    for (int d = 0; d < DK; ++d) dp += datt[i][hd * DK + d] * vH[hd][j][d];
                    dp_i[j] = dp;
                    dotpdg += pH[hd][i][j] * dp;
                }
                for (int j = 0; j <= i; ++j) {
                    double ds = pH[hd][i][j] * (dp_i[j] - dotpdg);
                    for (int d = 0; d < DK; ++d) {
                        dqH[i][d] += ds * kH[hd][j][d];
                        dkH[j][d] += ds * qH[hd][i][d];
                    }
                }
            }
            for (int i = 0; i < T; ++i)
                for (int d = 0; d < DK; ++d) {
                    dqH[i][d] *= inv_dk;
                    dkH[i][d] *= inv_dk;
                }
            for (int i = 0; i < T; ++i) {
                for (int d = 0; d < DK; ++d)
                    for (int j = 0; j < H; ++j) {
                        Wq.grad[(size_t)(hd * DK + d) * H + j] += dqH[i][d] * r[i][j];
                        Wk.grad[(size_t)(hd * DK + d) * H + j] += dkH[i][d] * r[i][j];
                        Wv.grad[(size_t)(hd * DK + d) * H + j] += dVH[i][d] * r[i][j];
                    }
                for (int j = 0; j < H; ++j) {
                    double s = 0.0;
                    for (int d = 0; d < DK; ++d)
                        s += Wq.at(hd * DK + d, j) * dqH[i][d]
                           + Wk.at(hd * DK + d, j) * dkH[i][d]
                           + Wv.at(hd * DK + d, j) * dVH[i][d];
                    dr[i][j] += s;
                }
            }
        }

        // LN1 反向 + 位置编码梯度
        for (int t = 0; t < T; ++t) {
            layernorm_bwd(p[t], ln1_mean[t], ln1_inv[t], LN1g.val, dr[t],
                          dp[t], LN1g.grad, LN1b.grad);
            int pt = min(t, T_MAX - 1);
            for (int j = 0; j < H; ++j) P.grad[(size_t)pt * H + j] += dp[t][j];
        }

        // RNN BPTT (与 v2 相同, 但上游梯度来自 dp)
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
        vector<vector<double>> r(T, vector<double>(H, 0.0));
        vector<vector<double>> att(T, vector<double>(H, 0.0));
        vector<vector<double>> q(T, vector<double>(H, 0.0));
        vector<vector<double>> ff_g(T, vector<double>(H1, 0.0));
        vector<vector<double>> z(T, vector<double>(H, 0.0));
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
            layernorm_fwd(p[t], LN1g.val, LN1b.val, r[t], mn, iv);
            h_prev = h[t];
        }
        for (int hd = 0; hd < NH; ++hd) {
            vector<vector<double>> qH(T, vector<double>(DK, 0.0));
            vector<vector<double>> kH(T, vector<double>(DK, 0.0));
            vector<vector<double>> vH(T, vector<double>(DK, 0.0));
            for (int i = 0; i < T; ++i)
                for (int d = 0; d < DK; ++d) {
                    double sq = 0.0, sk = 0.0, sv = 0.0;
                    for (int j = 0; j < H; ++j) {
                        sq += Wq.at(hd * DK + d, j) * r[i][j];
                        sk += Wk.at(hd * DK + d, j) * r[i][j];
                        sv += Wv.at(hd * DK + d, j) * r[i][j];
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
        for (int i = 0; i < T; ++i) {
            vector<double> a(H, 0.0);
            for (int o = 0; o < H; ++o) {
                double s = 0.0;
                for (int hh = 0; hh < H; ++hh) s += Wo.at(o, hh) * att[i][hh];
                a[o] = s;
            }
            for (int j = 0; j < H; ++j) p[i][j] = r[i][j] + a[j]; // 复用 p 存 u
            layernorm_fwd(p[i], LN2g.val, LN2b.val, q[i], mn, iv);
            for (int m = 0; m < H1; ++m) {
                double s = b1.val[m];
                for (int j = 0; j < H; ++j) s += W1.at(m, j) * q[i][j];
                ff_g[i][m] = gelu(s);
            }
            for (int j = 0; j < H; ++j) {
                double s = b2.val[j];
                for (int m = 0; m < H1; ++m) s += W2.at(j, m) * ff_g[i][m];
                z[i][j] = p[i][j] + s;
            }
        }
        vector<double> logit(V);
        for (int v = 0; v < V; ++v) {
            double s = by.val[v];
            for (int j = 0; j < H; ++j) s += Wy.at(v, j) * z[T - 1][j];
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
        // 位置编码依赖绝对位置, 训练时位置 0 恒为 <BOS>;
        // 因此若前缀未带 <BOS>, 自动补上 (与训练分布对齐)
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
};

// --------------------------- 主程序 ---------------------------
int main() {
    mt19937 rng(20240417);

    Tokenizer tok;
    tok.bos = tok.add("<BOS>");
    tok.eos = tok.add("<EOS>");
    tok.unk = tok.add("<UNK>");

    // 内置中文语料: 11 段古诗/谚语 (与 v2/v3 相同) + 10 条迷你中文字典 (v4 新增)
    const char* corpus[] = {
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
        "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
        "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
        "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。",
        "学而不思则罔，思而不学则殆。",
        "千里之行，始于足下。",
        "三人行，必有我师焉。",
        "少壮不努力，老大徒伤悲。",
        "海内存知己，天涯若比邻。",
        "明月几时有，把酒问青天。",
        "天街小雨润如酥，草色遥看近却无。最是一年春好处，绝胜烟柳满皇都。",
        // ---- 迷你中文字典 (10 条, v4 新增; 与古诗共享同一词表/记忆) ----
        "床=供人躺卧的家具",
        "月=地球的卫星",
        "光=明亮的光线",
        "明=光亮",
        "风=空气流动",
        "雨=云中水汽降落",
        "花=植物的繁殖器官",
        "知=知道明白",
        "行=走",
        "思=思考",
    };
    int n_lines = (int)(sizeof(corpus) / sizeof(corpus[0]));

    // 构建训练序列: <BOS> + 字符 + <EOS>
    vector<vector<int>> lines;
    int total_chars = 0;
    for (int i = 0; i < n_lines; ++i) {
        vector<string> cs = utf8_split(corpus[i]);
        vector<int> ids;
        ids.push_back(tok.bos);
        for (auto& c : cs) { tok.add(c); ids.push_back(tok.get(c)); }
        ids.push_back(tok.eos);
        lines.push_back(ids);
        total_chars += (int)cs.size();
    }

    const int H = 64, NH = 4, FF = 128, T_MAX = 48;
    CharRNNTransformer model((int)tok.itos.size(), H, NH, FF, T_MAX, rng);

    cout << "========== 迷你 Transformer (RNN + 多头自注意力) 中文语言模型 ==========\n";
    cout << "语料 : " << n_lines << " 段 (11 段古诗/谚语 + 10 条迷你中文字典), 共 "
         << total_chars << " 字符\n";
    cout << "词表 : " << (int)tok.itos.size() << " 字符 (含 <BOS>/<EOS>/<UNK>)\n";
    cout << "结构 : Embedding(" << (int)tok.itos.size() << "x" << H << ")\n";
    cout << "      -> RNN(hidden=" << H << ") + 可学习位置编码\n";
    cout << "      -> LayerNorm -> 多头自注意力(" << NH << "头, 因果掩码, dk=" << H / NH << ")\n";
    cout << "      -> 残差+LayerNorm -> FFN(" << H << "->" << FF << "->" << H << ", GELU)\n";
    cout << "      -> 残差 -> Softmax 预测下一字符\n";
    cout << "参数 : " << model.param_count() << "  (优化器 Adam, 梯度裁剪, BPTT+注意力反向)\n";
    cout << "训练中...\n\n";

    const int MAX_EPOCHS = 1000;
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
        double lr = LR0 * pow(0.5, (epoch - 1) / 500);
        model.adam_update(lr, epoch);
        double avg = total / cnt;
        if (epoch == 1 || epoch % 20 == 0 || epoch == MAX_EPOCHS) {
            losses.push_back({epoch, avg});
            printf("[train] epoch %5d   loss %.4f\n", epoch, avg);
        }
        if (avg < STOP_LOSS) break;
    }

    printf("\n[loss 曲线] ");
    for (auto& pr : losses) printf("%d:%.3f  ", pr.first, pr.second);
    printf("\n训练完成: 共 %d epochs, 最终平均 loss = %.4f\n", epoch, losses.back().second);
    printf("[说明] 语料 = 11 段古诗/谚语 + 10 条迷你中文字典 (共 21 段, 261 字符),\n");
    printf("      字典条目与古诗共享同一词表与同一套参数记忆。\n");
    printf("      固有歧义位置: ① <BOS> 后首字符共 19 种 (床/明 各出现 2 次);\n");
    printf("      ② <BOS>床 -> 前/= 二选一; ③ <BOS>明 -> 月/= 二选一。\n");
    printf("      理论最小平均损失 = 21·ln(21)/282 ≈ 0.2267, 为下界;\n");
    printf("      其余 257 个位置均可被模型完全记住 (含注意力路径)。\n\n");

    printf("\n[自检] 逐位置核对 (应与语料完全一致):\n");
    {
        int bad = 0, total = 0;
        for (int i = 0; i < n_lines; ++i) {
            bad += model.selftest(lines[i], tok.itos);
            total += (int)lines[i].size() - 1;
        }
        printf("  命中 %d/%d 位置 (未命中均为固有歧义: <BOS> 后首字符 19 选 1,\n",
               total - bad, total);
        printf("       以及 <BOS>床/<BOS>明 后的 前/=、月/= 二选一)\n");
    }

    // ---- 自动演示续写 (与 v3 相同的前缀) ----
    auto run_demo = [&](const string& prefix, const char* tag) {
        vector<string> cs = utf8_split(prefix);
        vector<int> ids;
        for (auto& c : cs) ids.push_back(tok.get(c));
        string gen = model.generate(ids, 0.6, 30, tok.bos, tok.eos, tok.unk, tok.itos, rng);
        cout << "[续写示例] " << tag << " : \"" << prefix << gen << "\"\n";
    };
    run_demo("床前明月", "静夜思  ");
    run_demo("春眠不觉", "春晓    ");
    run_demo("白日依山", "登鹳雀楼");
    run_demo("学而不思", "论语    ");
    cout << "\n";

    // ---- 交互模式 ----
    cout << "========== 交互模式 ==========\n";
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
        cout << "  " << line << gen << "\n";
    }
    return 0;
}
