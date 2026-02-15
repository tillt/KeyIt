//
//  analysis_features_internal.h
//  KeyIt
//
//  Created by Till Toenshoff on 10.02.26.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//
#pragma once

#include "keyit/keyit.h"

#include <cstddef>
#include <vector>

namespace keyit {
namespace analysis_internal {

std::vector<float> resample_linear(const std::vector<float>& input,
                                   double input_rate,
                                   std::size_t target_rate);

std::vector<float> compute_log_cqt_matrix(const std::vector<float>& samples,
                                          const KeyitConfig& config,
                                          std::size_t* out_frames);

std::vector<float> trim_trailing_frames(const std::vector<float>& matrix,
                                        std::size_t bins,
                                        std::size_t in_frames,
                                        std::size_t trim,
                                        std::size_t* out_frames);

} // namespace analysis_internal
} // namespace keyit
