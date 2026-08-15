// ============================================================================
//  ai_brain_fast.cpp — 客尘AI集群v2.0 · 轻量问答程序 (Fast 快答层)
//
//  混合架构第一层: 知识问答秒答 (零训练 / 零权重 / 纯查表)
//    1. 启动时加载 6 个语料文件 (corpus_{poet,lunyu,chat,cosmos,math,history}.txt)
//    2. 每行解析为 (key, value) 问答对, 支持 "前缀-X=Q=A" 三段格式
//    3. 建立倒排索引 (字符 + 双字n-gram -> 条目), 提问时只检索相关条目
//    4. 相似度 = 0.6 × 字符覆盖率 + 0.4 × 双字覆盖率, 命中阈值 0.60
//    5. 命中直接输出答案 (毫秒级, 秒答); 未命中退出码 1, 交给 router.sh
//       转专家模型 (生成/续写)
//
//  用法:
//    ./ai_brain_fast                交互模式 (输入 quit 退出)
//    ./ai_brain_fast -q "问题"      单次问答: 命中打印答案并退出0; 未命中退出1
//    ./ai_brain_fast --stats        打印索引统计
//    ./ai_brain_fast --demo         演示: 4 个知识问答(秒答) + 1 个未命中
//
//  编译: clang++ -O2 -std=c++17 ai_brain_fast.cpp -o ai_brain_fast
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;

// --------------------------- 可调参数 ---------------------------
static const double kCharWeight   = 0.6;   // 字符覆盖率权重
static const double kBigramWeight = 0.4;   // 双字覆盖率权重
static const double kThreshold    = 0.60;  // 命中阈值 (语料内问答 >0.9, 未学输入 <0.55)
static const char*  kCorpusFiles[] = {
    "corpus_poet.txt", "corpus_lunyu.txt", "corpus_chat.txt",
    "corpus_cosmos.txt", "corpus_math.txt", "corpus_history.txt"
};
static const int kCorpusCount = 6;

// --------------------------- UTF-8 工具 ---------------------------
// 把 UTF-8 字符串按字符切分 (字节方式, 与 ai_brain_v08.cpp 一致)
static vector<string> utf8_chars(const string& s) {
    vector<string> out;
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        int len = 1;
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        if (i + len > n) len = 1;               // 防御: 非法/截断字节
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

// 读取整个文件 (UTF-8 文本); 失败返回空串
static string load_file(const string& path) {
    ifstream f(path, ios::binary);
    if (!f) return "";
    ostringstream ss;
    ss << f.rdbuf();
    string s = ss.str();
    // 去 BOM
    if (s.size() >= 3 && (unsigned char)s[0] == 0xEF &&
        (unsigned char)s[1] == 0xBB && (unsigned char)s[2] == 0xBF)
        s = s.substr(3);
    return s;
}

// 按 '=' 切分, 最多 limit 段 (第 limit 段起合并)
static vector<string> split_eq(const string& s, int limit) {
    vector<string> out;
    size_t start = 0;
    while (true) {
        size_t p = s.find('=', start);
        if (p == string::npos || (int)out.size() == limit - 1) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, p - start));
        start = p + 1;
    }
    return out;
}

// --------------------------- 索引结构 ---------------------------
struct Entry {
    string key;                          // 问题 / 标题 / 纯文本
    string value;                        // 答案 / 内容
    bool   is_qa;                        // key != value (问答对)
    string text;                         // key+value 缓存 (用于检索)
    unordered_set<string> chars;         // 文本字符集
    unordered_set<string> bigrams;       // 文本双字集
};

struct Index {
    vector<Entry> entries;               // 全部条目 (文件顺序)
    unordered_map<string, vector<int>> char_inv;   // 字符 -> 条目下标
    unordered_map<string, vector<int>> bigram_inv; // 双字 -> 条目下标
    int file_count = 0;
};

// 从 key+value 文本生成 字符集 / 双字集, 并加入倒排索引
static void add_to_index(Index& idx, Entry& e) {
    vector<string> cs = utf8_chars(e.text);
    for (size_t i = 0; i < cs.size(); ++i) {
        e.chars.insert(cs[i]);
        idx.char_inv[cs[i]].push_back((int)idx.entries.size());
        if (i + 1 < cs.size()) {
            string bg = cs[i] + cs[i + 1];
            e.bigrams.insert(bg);
            idx.bigram_inv[bg].push_back((int)idx.entries.size());
        }
    }
}

// 解析一行语料: "Q=A" / "前缀-X=Q=A" / 纯文本
static void parse_line(Index& idx, const string& raw) {
    string line = trim(raw);
    if (line.empty()) return;
    vector<string> parts = split_eq(line, 3);
    Entry e;
    if (parts.size() == 1) {
        e.key = e.value = parts[0];      // 纯文本 (古诗/名言/宇宙陈述)
        e.is_qa = false;
    } else if (parts.size() == 2) {
        e.key = parts[0];                 // Q= A
        e.value = parts[1];
        e.is_qa = (e.key != e.value);
    } else {
        e.key = parts[1];                 // 前缀-X=Q=A -> Q= A
        e.value = parts[2];
        e.is_qa = (e.key != e.value);
    }
    e.key   = trim(e.key);
    e.value = trim(e.value);
    if (e.key.empty() && e.value.empty()) return;
    e.text = e.key + e.value;
    add_to_index(idx, e);
    idx.entries.push_back(e);
}

// 加载全部 6 个语料文件, 建立索引; file_counts 记录每个文件的条目数
static size_t build_index(Index& idx, vector<int>& file_counts) {
    for (int i = 0; i < kCorpusCount; ++i) {
        string path = kCorpusFiles[i];
        string content = load_file(path);
        if (content.empty()) {
            cerr << "[警告] 找不到语料文件: " << path << endl;
            continue;
        }
        ++idx.file_count;
        size_t before = idx.entries.size();
        istringstream in(content);
        string line;
        while (getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            parse_line(idx, line);
        }
        file_counts.push_back((int)(idx.entries.size() - before));
    }
    return idx.entries.size();
}

// --------------------------- 相似度检索 ---------------------------
struct Query {
    vector<string> chars;
    vector<string> bigrams;
};

static Query build_query(const string& q) {
    Query qr;
    qr.chars = utf8_chars(q);
    for (size_t i = 0; i + 1 < qr.chars.size(); ++i)
        qr.bigrams.push_back(qr.chars[i] + qr.chars[i + 1]);
    return qr;
}

// 得分: 0.6 × 字符覆盖率 + 0.4 × 双字覆盖率 (基于 key+value 全文)
static double score_entry(const Query& q, const Entry& e) {
    if (q.chars.empty()) return 0.0;
    int hit_c = 0;
    for (const auto& c : q.chars) if (e.chars.count(c)) ++hit_c;
    double char_cov = (double)hit_c / (double)q.chars.size();
    double bigram_cov = 0.0;
    if (!q.bigrams.empty()) {
        int hit_b = 0;
        for (const auto& b : q.bigrams) if (e.bigrams.count(b)) ++hit_b;
        bigram_cov = (double)hit_b / (double)q.bigrams.size();
    }
    return kCharWeight * char_cov + kBigramWeight * bigram_cov;
}

// key 与问题的匹配度 (用于同分决胜):
//   3=key完全等于问题  2=key是问题前缀/问题是key的子串(主题词)  1=key含问题(诗句片段)  0=无
static int key_match(const Entry& e, const Query& q) {
    string qs;
    for (const auto& c : q.chars) qs += c;
    if (e.key == qs) return 3;
    if (qs.size() > e.key.size() && qs.compare(0, e.key.size(), e.key) == 0) return 2; // key是问题前缀
    if (qs.find(e.key) != string::npos && !e.key.empty()) return 2;                   // 问题含key (主题词)
    if (e.key.find(qs) != string::npos) return 1;                                     // key含问题 (如整句诗)
    return 0;
}

// 决胜排序: 得分高 > key匹配度高 > 问答对优先 > 答案短 > 文件顺序靠前
static bool better(const Entry& a, const Entry& b, const Query& q,
                   double sa, double sb) {
    if (sa != sb) return sa > sb;
    int ka = key_match(a, q), kb = key_match(b, q);
    if (ka != kb) return ka > kb;
    if (a.is_qa != b.is_qa) return a.is_qa;
    if (a.value.size() != b.value.size()) return a.value.size() < b.value.size();
    return false; // 保持原顺序 (先出现的优先)
}

struct Hit {
    bool found = false;
    int  index = -1;
    double score = 0.0;
};

static Hit retrieve(const Index& idx, const string& question) {
    Query q = build_query(question);
    Hit best;
    if (q.chars.empty()) return best;
    // 倒排索引召回候选: 问题字符/双字命中的条目并集
    unordered_set<int> cands;
    for (const auto& c : q.chars) {
        auto it = idx.char_inv.find(c);
        if (it != idx.char_inv.end())
            for (int ix : it->second) cands.insert(ix);
    }
    for (const auto& b : q.bigrams) {
        auto it = idx.bigram_inv.find(b);
        if (it != idx.bigram_inv.end())
            for (int ix : it->second) cands.insert(ix);
    }
    // 只对候选打分
    for (int ix : cands) {
        double s = score_entry(q, idx.entries[ix]);
        if (!best.found || better(idx.entries[ix], idx.entries[best.index], q, s, best.score)) {
            best.found = true;
            best.index = ix;
            best.score = s;
        }
    }
    if (best.found && best.score < kThreshold) best.found = false;
    return best;
}

// --------------------------- 演示/统计 ---------------------------
static double ms_since(Clock::time_point t0) {
    return chrono::duration<double, milli>(Clock::now() - t0).count();
}

static void show_stats(const Index& idx, double build_ms) {
    int qa = 0, plain = 0;
    for (const auto& e : idx.entries) { if (e.is_qa) ++qa; else ++plain; }
    cout << "===== ai_brain_fast 索引统计 =====" << endl;
    cout << "语料文件 : " << idx.file_count << " 个" << endl;
    cout << "条目总数 : " << idx.entries.size() << " 条" << endl;
    cout << "  问答对 : " << qa << " 条 (key=问题, value=答案)" << endl;
    cout << "  纯文本 : " << plain << " 条 (古诗/名言/陈述, 可续写匹配)" << endl;
    cout << "倒排索引 : 字符 " << idx.char_inv.size() << " 个, 双字 " << idx.bigram_inv.size() << " 个" << endl;
    cout << "建索引耗时: " << build_ms << " ms (零训练/零权重)" << endl;
}

static int demo_mode() {
    Index idx;
    vector<int> counts;
    Clock::time_point t0 = Clock::now();
    size_t n = build_index(idx, counts);
    double build_ms = ms_since(t0);
    cout << "========== ai_brain_fast --demo · 快答层演示 ==========" << endl;
    cout << "[启动] 加载 6 个语料文件, 建立倒排索引 ..." << endl;
    cout << "[启动] ";
    for (int i = 0; i < kCorpusCount; ++i) {
        if (i) cout << " | ";
        string fname(kCorpusFiles[i]);
        string label = fname.substr(7, fname.size() - 11);   // corpus_xxx.txt -> xxx
        cout << label << "=" << (i < (int)counts.size() ? counts[i] : 0) << "条";
    }
    cout << endl;
    cout << "[启动] 共 " << n << " 条, 相似度阈值 " << kThreshold
         << " (字符" << kCharWeight << " + 双字" << kBigramWeight << "), 建索引 "
         << build_ms << " ms" << endl;
    cout << endl;

    const char* known[] = { "床前明月", "负一元素是什么", "6×7等于几", "大禹治水" };
    cout << "---- 一、知识问答 (语料内, 应命中快答层 → 秒答) ----" << endl;
    for (const char* q : known) {
        Clock::time_point q0 = Clock::now();
        Hit h = retrieve(idx, q);
        double qms = ms_since(q0);
        if (h.found)
            cout << "Q: " << q << "\n   ✅ 命中 (相似度 " << h.score << ") "
                 << qms << " ms → " << idx.entries[h.index].value << endl;
        else
            cout << "Q: " << q << "\n   ❌ 未命中?! (" << qms << " ms)" << endl;
    }
    cout << endl;

    cout << "---- 二、未学过的输入 (语料外, 应未命中 → 转专家模型) ----" << endl;
    const char* unknown = "给我讲讲哲学";
    Clock::time_point q0 = Clock::now();
    Hit h = retrieve(idx, unknown);
    double qms = ms_since(q0);
    if (h.found)
        cout << "Q: " << unknown << "\n   ✅ 命中 (相似度 " << h.score << ") → "
             << idx.entries[h.index].value << "   <-- 异常, 应未命中!" << endl;
    else
        cout << "Q: " << unknown << "\n   ❌ 未命中 (最高相似度 "
             << (h.index >= 0 ? h.score : 0.0) << " < 阈值 " << kThreshold
             << ") " << qms << " ms → 交给 router.sh 转专家模型 (生成/续写)" << endl;
    cout << "=====================================================" << endl;
    return 0;
}

static int query_mode(const string& question) {
    static Index idx;
    static bool built = false;
    if (!built) {
        vector<int> counts;
        build_index(idx, counts);
        built = true;
    }
    Hit h = retrieve(idx, question);
    if (!h.found) return 1;                    // 未命中: 无输出, 退出码 1
    cout << idx.entries[h.index].value << endl; // 命中: 只输出答案 (router 用)
    return 0;
}

static int interactive_mode() {
    Index idx;
    vector<int> counts;
    Clock::time_point t0 = Clock::now();
    size_t n = build_index(idx, counts);
    double build_ms = ms_since(t0);
    cout << "===== 客尘AI集群v2.0 · ai_brain_fast 快答层 (零训练/零权重/纯查表) =====" << endl;
    cout << "已加载 " << idx.file_count << " 份语料, 共 " << n << " 条问答/文本, 建索引 "
         << build_ms << " ms" << endl;
    cout << "输入问题 → 知识秒答; 未命中提示转专家模型; quit/exit 退出" << endl;
    cout << endl;
    while (true) {
        printf("你> ");
        string line;
        if (!getline(cin, line)) break;
        line = trim(line);
        if (line.empty()) continue;
        if (line == "quit" || line == "exit" || line == "q" || line == "退出") {
            cout << "再见！" << endl;
            break;
        }
        Clock::time_point q0 = Clock::now();
        Hit h = retrieve(idx, line);
        double qms = ms_since(q0);
        if (h.found)
            cout << "【命中·" << h.score << "·" << qms << "ms】"
                 << idx.entries[h.index].value << endl;
        else
            cout << "【未命中·" << qms << "ms】知识库没有这个, 将转专家模型生成/续写" << endl;
    }
    return 0;
}

// --------------------------- 入口 ---------------------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    if (argc >= 2 && string(argv[1]) == "--demo") return demo_mode();
    if (argc >= 2 && string(argv[1]) == "--stats") {
        Index idx;
        vector<int> counts;
        Clock::time_point t0 = Clock::now();
        build_index(idx, counts);
        show_stats(idx, ms_since(t0));
        return 0;
    }
    string question;
    if (argc >= 2 && string(argv[1]) == "-q") {
        for (int i = 2; i < argc; ++i) {
            if (i > 2) question += " ";
            question += argv[i];
        }
    } else if (argc >= 2) {
        for (int i = 1; i < argc; ++i) {
            if (i > 1) question += " ";
            question += argv[i];
        }
    }
    if (!question.empty()) return query_mode(question);
    return interactive_mode();
}
