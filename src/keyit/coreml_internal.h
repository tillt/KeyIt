#pragma once

#include "keyit/keyit.h"

#include <vector>

namespace keyit {

bool run_keynet_coreml(const std::vector<float>& input_nchw,
                       const KeyitConfig& config,
                       std::vector<float>* logits,
                       std::string* error);

} // namespace keyit
