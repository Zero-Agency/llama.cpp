#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <json.hpp>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../../src/llama-context.h"
#include "../../src/llama-vocab.h"
#include "arg.h"
#include "chat.h"
#include "common.h"
#include "json.hpp"
#include "llama.h"
#include "log.h"
#include "minja/chat-template.hpp"

using json = nlohmann::ordered_json;

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s \\\n"
            "       -m model.gguf -f some-text.txt [-o imatrix.dat] [--process-output] \\\n"
            "       [--no-ppl] [--chunk 123] [--output-frequency 10] [--save-frequency 0] \\\n"
            "       [--in-file imatrix-prev-0.dat --in-file imatrix-prev-1.dat ...] \\\n"
            "       [--parse-special]\n" , argv[0]);
    LOG("\n");
}

struct Stats {
    std::vector<float> values;
    std::vector<int> counts;
    int ncall = 0;
};

class IMatrixCollector {
public:
    IMatrixCollector() = default;
    void set_params(common_params params) { m_params = std::move(params); }
    bool collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data);
    void save_imatrix(int ncall = -1) const;
    bool load_imatrix(const char * fname);
private:
    std::unordered_map<std::string, Stats> m_stats;
    common_params                          m_params;
    std::mutex                             m_mutex;
    int                                    m_last_call = 0;
    std::vector<char>                      m_src1_data;
    std::vector<char>                      m_ids; // the expert ids from ggml_mul_mat_id
};

// remove any prefix and suffixes from the name
// CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight
static std::string filter_tensor_name(const char * name) {
    std::string wname;
    const char * p = strchr(name, '#');
    if (p != NULL) {
        p = p + 1;
        const char * q = strchr(p, '#');
        if (q != NULL) {
            wname = std::string(p, q - p);
        } else {
            wname = p;
        }
    } else {
        wname = name;
    }
    return wname;
}

bool IMatrixCollector::collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];
    std::string wname = filter_tensor_name(src0->name);

    // when ask is true, the scheduler wants to know if we are interested in data from this tensor
    // if we return true, a follow-up call will be made with ask=false in which we can do the actual collection
    if (ask) {
        if (t->op == GGML_OP_MUL_MAT_ID) return true; // collect all indirect matrix multiplications
        if (t->op != GGML_OP_MUL_MAT) return false;
        // why are small batches ignored (<16 tokens)?
        if (src1->ne[1] < 16 || src1->type != GGML_TYPE_F32) return false;
        if (!(wname.substr(0, 4) == "blk." || (m_params.process_output && wname == "output.weight"))) return false;
        return true;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(src1->buffer);

    if (!is_host) {
        const size_t src1_nbytes = ggml_nbytes(src1);
        m_src1_data.resize(src1_nbytes);
        ggml_backend_tensor_get(src1, m_src1_data.data(), 0, src1_nbytes);
    }

    const char * data = is_host ? (const char *) src1->data : m_src1_data.data();
    GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));

    // this has been adapted to the new format of storing merged experts in a single 3d tensor
    // ref: https://github.com/ggml-org/llama.cpp/pull/6387
    if (t->op == GGML_OP_MUL_MAT_ID) {
        //   ids  -> [n_experts_used, n_tokens]
        //   src1 -> [cols, n_expert_used, n_tokens]
        const ggml_tensor * ids = t->src[2];
        const int n_as = src0->ne[2];
        const int n_ids = ids->ne[0];

        // the top-k selected expert ids are stored in the ids tensor
        // for simplicity, always copy ids to host, because it is small
        // take into account that ids is not contiguous!

        GGML_ASSERT(ids->ne[1] == src1->ne[2]);

        m_ids.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, m_ids.data(), 0, ggml_nbytes(ids));

        auto & e = m_stats[wname];

        ++e.ncall;

        if (e.values.empty()) {
            e.values.resize(src1->ne[0]*n_as, 0);
            e.counts.resize(src1->ne[0]*n_as, 0);
        }
        else if (e.values.size() != (size_t)src1->ne[0]*n_as) {
            LOG_ERR("%s: inconsistent size for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.values.size(), (int)src1->ne[0]*n_as);
            exit(1); //GGML_ABORT("fatal error");
        }
        LOG_DBGV(2, "%s[%d]: %32s, %s, %5d x %5d, %d\n", __func__, m_last_call, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[2], (int)src1->type);
        // loop over all possible experts, regardless if they are used or not in the batch
        for (int ex = 0; ex < n_as; ++ex) {
            size_t e_start = ex*src1->ne[0];

            for (int idx = 0; idx < n_ids; ++idx) {
                for (int row = 0; row < (int)src1->ne[2]; ++row) {
                    const int excur = *(const int32_t *) (m_ids.data() + row*ids->nb[1] + idx*ids->nb[0]);

                    GGML_ASSERT(excur >= 0 && excur < n_as); // sanity check

                    if (excur != ex) continue;

                    const int64_t i11 = idx % src1->ne[1];
                    const int64_t i12 = row;
                    const float * x = (const float *)(data + i11*src1->nb[1] + i12*src1->nb[2]);

                    for (int j = 0; j < (int)src1->ne[0]; ++j) {
                        e.values[e_start + j] += x[j]*x[j];
                        e.counts[e_start + j]++;
                        if (!std::isfinite(e.values[e_start + j])) {
                            LOG("\n");
                            LOG_ERR("%f detected in %s\n", e.values[e_start + j], wname.c_str());
                            exit(1);
                        }
                    }
                }
            }
            if (e.ncall > m_last_call) {
                m_last_call = e.ncall;
                if (m_last_call % m_params.n_out_freq == 0) {
                    save_imatrix();
                }
                if (m_params.n_save_freq > 0 && m_last_call%m_params.n_save_freq == 0) {
                    save_imatrix(m_last_call);
                }
            }
        }
    } else {
        auto & e = m_stats[wname];
        if (e.values.empty()) {
            e.values.resize(src1->ne[0], 0);
            e.counts.resize(src1->ne[0], 0);
        }
        else if (e.values.size() != (size_t)src1->ne[0]) {
            LOG_ERR("%s: inconsistent size for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.values.size(), (int)src1->ne[0]);
            exit(1); //GGML_ABORT("fatal error");
        }
        ++e.ncall;
        LOG_DBGV(2, "%s[%d]: %32s, %s, %5d x %5d, %d\n", __func__, m_last_call, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[1], (int)src1->type);
        for (int row = 0; row < (int)src1->ne[1]; ++row) {
            const float * x = (const float *) (data + row * src1->nb[1]);
            for (int j = 0; j < (int)src1->ne[0]; ++j) {
                e.values[j] += x[j]*x[j];
                e.counts[j]++;
                if (!std::isfinite(e.values[j])) {
                    LOG_ERR("%f detected in %s\n", e.values[j], wname.c_str());
                    exit(1);
                }
            }
        }
        if (e.ncall > m_last_call) {
            m_last_call = e.ncall;
            if (m_last_call % m_params.n_out_freq == 0) {
                save_imatrix();
            }
            if (m_params.n_save_freq > 0 && m_last_call%m_params.n_save_freq == 0) {
                save_imatrix(m_last_call);
            }
        }
    }

    return true;
}

void IMatrixCollector::save_imatrix(int ncall) const {
    auto fname = m_params.out_file;

    if (ncall > 0) {
        fname += ".at_";
        fname += std::to_string(ncall);
    }

    // avoid writing imatrix entries that do not have full data
    // this can happen with MoE models where some of the experts end up not being exercised by the provided training data

    int n_entries = 0;
    std::vector<std::string> to_store;

    bool is_first = true; // for printing
    for (const auto & kv : m_stats) {
        const int n_all = kv.second.counts.size();

        if (n_all == 0) {
            continue;
        }

        int n_zeros = 0;
        for (const int c : kv.second.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros != 0 && is_first) {
            LOG_INF("\n");
            is_first = false;
        }

        if (n_zeros == n_all) {
            LOG_WRN("%s: entry '%40s' has no data - skipping\n", __func__, kv.first.c_str());
            continue;
        }

        if (n_zeros > 0) {
            LOG_WRN("%s: entry '%40s' has partial data (%.2f%%) - skipping\n", __func__, kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
            continue;
        }

        n_entries++;
        to_store.push_back(kv.first);
    }

    if (to_store.size() < m_stats.size()) {
        LOG_WRN("%s: storing only %zu out of %zu entries\n", __func__, to_store.size(), m_stats.size());
    }

    std::ofstream out(fname, std::ios::binary);
    out.write((const char *) &n_entries, sizeof(n_entries));
    for (const auto & name : to_store) {
        const auto & stat = m_stats.at(name);
        int len = name.size();
        out.write((const char *) &len, sizeof(len));
        out.write(name.c_str(), len);
        out.write((const char *) &stat.ncall, sizeof(stat.ncall));
        int nval = stat.values.size();
        out.write((const char *) &nval, sizeof(nval));
        if (nval > 0) {
            std::vector<float> tmp(nval);
            for (int i = 0; i < nval; i++) {
                tmp[i] = (stat.values[i] / static_cast<float>(stat.counts[i])) * static_cast<float>(stat.ncall);
            }
            out.write((const char*)tmp.data(), nval*sizeof(float));
        }
    }

    // Write the number of call the matrix was computed with
    out.write((const char *) &m_last_call, sizeof(m_last_call));

    // Write the input filename at the end of the file to later on specify it in quantize
    {
        int len = m_params.prompt_file.size();
        out.write((const char *) &len, sizeof(len));
        out.write(m_params.prompt_file.c_str(), len);
    }

    LOGV(1, "\n");
    LOG_DBGV(1, "%s: stored collected data after %d chunks in %s\n", __func__, m_last_call, fname.c_str());
}

bool IMatrixCollector::load_imatrix(const char * fname) {
    std::ifstream in(fname, std::ios::binary);
    if (!in) {
        LOG_ERR("%s: failed to open %s\n",__func__, fname);
        return false;
    }
    int n_entries;
    in.read((char*)&n_entries, sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        LOG_ERR("%s: no data in file %s\n", __func__, fname);
        return false;
    }
    for (int i = 0; i < n_entries; ++i) {
        int len; in.read((char *)&len, sizeof(len));
        std::vector<char> name_as_vec(len+1);
        in.read((char *)name_as_vec.data(), len);
        if (in.fail()) {
            LOG_ERR("%s: failed reading name for entry %d from %s\n",__func__,i+1, fname);
            return false;
        }
        name_as_vec[len] = 0;
        std::string name{name_as_vec.data()};
        auto & e = m_stats[std::move(name)];
        int ncall;
        in.read((char*)&ncall, sizeof(ncall));
        int nval;
        in.read((char *)&nval, sizeof(nval));
        if (in.fail() || nval < 1) {
            LOG_ERR("%s: failed reading number of values for entry %d\n",__func__,i);
            m_stats = {};
            return false;
        }

        if (e.values.empty()) {
            e.values.resize(nval, 0);
            e.counts.resize(nval, 0);
        }

        std::vector<float> tmp(nval);
        in.read((char*)tmp.data(), nval*sizeof(float));
        if (in.fail()) {
            LOG_ERR("%s: failed reading data for entry %d\n",__func__,i);
            m_stats = {};
            return false;
        }

        // Recreate the state as expected by save_imatrix(), and corerct for weighted sum.
        for (int i = 0; i < nval; i++) {
            e.values[i] += tmp[i];
            e.counts[i] += ncall;
        }
        e.ncall += ncall;

    }
    return true;
}

static IMatrixCollector g_collector;

static bool ik_collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    return g_collector.collect_imatrix(t, ask, user_data);
}


struct results_log_softmax {
    double log_softmax;
    float  logit;
    float  prob;
};

static std::vector<float> softmax(const std::vector<float> & logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) {
        max_logit = std::max(max_logit, v);
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum_exp;
    }
    return probs;
}

static results_log_softmax log_softmax(int n_vocab, const float * logits, int tok) {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    return {logits[tok] - max_logit - log(sum_exp), logits[tok], expf(logits[tok] - max_logit) / (float) sum_exp};
}

static void process_logits(
    int n_vocab, const float * logits, const int * tokens, int n_token, std::vector<std::thread> & workers,
    double & nll, double & nll2, float * logit_history, float * prob_history) {
    std::mutex mutex;
    int counter = 0;
    auto compute = [&mutex, &counter, &nll, &nll2, logit_history, prob_history, n_vocab, logits, tokens, n_token] () {
        double local_nll  = 0;
        double local_nll2 = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int i = counter++;
            if (i >= n_token) {
                nll += local_nll; nll2 += local_nll2;
                break;
            }
            lock.unlock();
            const results_log_softmax results = log_softmax(n_vocab, logits + i*n_vocab, tokens[i+1]);
            const double v = -results.log_softmax;
            local_nll += v;
            local_nll2 += v*v;

            logit_history[i] = results.logit;
            prob_history[i]  = results.prob;
        }
    };
    for (auto & w : workers) {
        w = std::thread(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
}

typedef minja::chat_template common_chat_template;
struct common_chat_templates {
    bool has_explicit_template; // Model had builtin template or template overridde was specified.
    std::unique_ptr<common_chat_template> template_default; // always set (defaults to chatml)
    std::unique_ptr<common_chat_template> template_tool_use;
};

static bool compute_imatrix(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const int n_ctx = llama_n_ctx(ctx);

    GGML_ASSERT(!llama_vocab_get_add_eos(vocab));

    auto tim1 = std::chrono::high_resolution_clock::now();

    LOG_INF("%s: reading jsonl inputs...\n", __func__);

    std::string default_template_src = llama_model_chat_template(model, /* name */ nullptr);
    const auto get_token = [&](llama_token token, const char * name, const char * jinja_variable_name) {
        if (token == LLAMA_TOKEN_NULL) {
            if (default_template_src.find(jinja_variable_name) != std::string::npos) {
                LOG_WRN("common_chat_templates_init: warning: vocab does not have a %s token, jinja template won't work as intended.\n", name);
            }
            return std::string();
        }
        return common_token_to_piece(vocab, token, true);
    };
    std::string token_bos = get_token(llama_vocab_bos(vocab), "BOS", "bos_token");
    std::string token_eos = get_token(llama_vocab_eos(vocab), "EOS", "eos_token");
    common_chat_templates_ptr tmpls(new common_chat_templates());
    tmpls->has_explicit_template = false;
    try {
        tmpls->template_default = std::make_unique<minja::chat_template>(default_template_src, token_bos, token_eos);
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to parse chat template (defaulting to chatml): %s \n", __func__, e.what());
        return false;
    }


    std::stringstream ss(params.prompt);
    std::string       line;
    std::vector<std::vector<llama_token>> data;
    std::string                           legacy_imatrix_input;

    json body;

    while (std::getline(ss, line)) {
        if (!line.empty()) {
            body = json::parse(line);
            common_chat_templates_inputs inputs;
            inputs.add_generation_prompt = false;
            inputs.use_jinja = true;
            inputs.messages = common_chat_msgs_parse_oaicompat(body.at("messages"));

            // leave only questions, not assistant answers
            if (inputs.messages.back().role == "assistant") {
                inputs.messages.pop_back();
            }

            common_chat_params chat_params = common_chat_templates_apply(tmpls.get(), inputs);
            legacy_imatrix_input += chat_params.prompt + "\n";
            std::vector<llama_token> tokenized = common_tokenize(ctx, chat_params.prompt, true, params.parse_special);
            
            data.push_back(tokenized);
        }
    }

    std::ofstream                         legacy_file("legacy_imatrix_input.txt");
    legacy_file << legacy_imatrix_input;
    legacy_file.close();
    

    LOG_INF("%s: tokenizing the input ..\n", __func__);


    std::vector<float> logit_history;
    std::vector<float> prob_history;
    const int n_chunk = 0;
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_batch = params.n_batch;

    int count = 0;
    double nll = 0.0;
    double nll2 = 0.0;

    LOG_INF("%s: computing over %d chunks with batch_size %d\n", __func__, n_chunk, n_batch);

    std::vector<std::thread> workers(std::thread::hardware_concurrency() - 1);

    const int num_batches = sizeof(data) / n_batch;//(n_ctx + n_batch - 1) / n_batch;

    std::vector<float> logits;
    if (params.compute_ppl && num_batches > 1) {
        logits.reserve((size_t)n_ctx * n_vocab);
    }

    const int batch_size = 16;

    for (int i = 0; i < data.size()/batch_size; ++i) {
        const int start = i * batch_size;
        const int end = std::min(start + batch_size, (int)data.size());

        llama_batch batch = llama_batch_init(n_batch, 0, n_batch);
        
        const auto t_start = std::chrono::high_resolution_clock::now();
        // clear the KV cache
        llama_kv_self_clear(ctx);

        
        common_batch_clear(batch);
        
        for (int j=0; j<batch_size; j++) {
            const int index = i*batch_size + j;
            if (index > end) {
                break;
            }
            for (int k : data[index]) {
                common_batch_add(batch, k, NULL, { j }, true);
            }
        }

        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            llama_batch_free(batch);
            return false;
        }

        llama_batch_free(batch);
        const auto t_end = std::chrono::high_resolution_clock::now();
        const float t_total = std::chrono::duration<float>(t_end - t_start).count();

        if (i == 0) {
            LOG("%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * data.size()/batch_size);
            if (total_seconds >= 60*60) {
                LOG("%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            LOG("%.2f minutes\n", total_seconds / 60.0);
        } else {
            LOG("%s: %.2f seconds for iteration %d of %ld\n", __func__, t_total, i, data.size()/batch_size);
        }
    }

    LOG("\n");

    if (params.compute_ppl) {
        nll2 /= count;
        nll /= count;
        const double ppl = exp(nll);
        nll2 -= nll * nll;
        if (nll2 > 0) {
            nll2 = sqrt(nll2/(count-1));
            LOG("Final estimate: PPL = %.4lf +/- %.5lf\n", ppl, nll2*ppl);
        } else {
            LOG("Unexpected negative standard deviation of log(prob)\n");
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    params.out_file = "imatrix.dat" ;

    params.n_ctx = 512;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    common_init();

    params.n_batch = std::min(params.n_batch, params.n_ctx);

    g_collector.set_params(params);

    for (const auto & in_file : params.in_files) {
        LOG_INF("%s : loading imatrix from '%s'\n", __func__, in_file.c_str());
        if (!g_collector.load_imatrix(in_file.c_str())) {
            LOG_ERR("%s : failed to load %s\n", __func__, in_file.c_str());
            return 1;
        }
    }

    if (params.in_files.size() > 1) {
        LOG_INF("%s : saving combined imatrix to '%s'\n", __func__, params.out_file.c_str());
        g_collector.save_imatrix();
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ik_collect_imatrix;
    params.cb_eval_user_data = NULL;
    params.warmup = false;

    // init
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_model_n_ctx_train(model);
    if (params.n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    if (params.prompt.empty()) {
        if (params.in_files.empty()) {
            LOG_ERR("Error: No prompt provided and no precomputed matrices (--in-file) to combine.\n");
            return 1;
        }
        LOG_INF("No prompt provided; combining precomputed matrices only.\n");
    } else {
        if (!compute_imatrix(ctx, params)) {
            return 1;
        }
    }


    g_collector.save_imatrix();

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
