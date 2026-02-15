#pragma once

#include "keyit/keyit.h"

#include <cstddef>
#include <string>
#include <vector>

namespace keyit {
namespace analysis_internal {

float clampf(float v, float lo, float hi);

std::vector<float> apply_harmonic_time_median_enhancement(const std::vector<float>& matrix,
                                                          std::size_t bins,
                                                          std::size_t frames,
                                                          const KeyitConfig& config);

std::vector<float> softmax(const std::vector<float>& logits);

std::vector<ClassScore> build_topk(const std::vector<float>& probs, std::size_t k);

bool aggregate_logits_over_windows(const std::vector<float>& cqt,
                                   std::size_t bins,
                                   std::size_t frames,
                                   const KeyitConfig& config,
                                   std::vector<float>* aggregated_logits,
                                   std::size_t* out_windows,
                                   std::string* error);

bool compute_section_voted_probs(const std::vector<float>& cqt,
                                 std::size_t bins,
                                 std::size_t frames,
                                 const KeyitConfig& config,
                                 std::vector<float>* out_probs,
                                 std::string* error);

} // namespace analysis_internal
} // namespace keyit
