#pragma once

#include "common.h"

#include <vector>
#include <unordered_map>
#include <mutex>

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
