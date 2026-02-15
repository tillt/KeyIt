//
//  coreml_internal.h
//  KeyIt
//
//  Created by Till Toenshoff on 10.02.26.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//
#pragma once

#include "keyit/keyit.h"

#include <vector>

namespace keyit {

bool run_keynet_coreml(const std::vector<float>& input_nchw,
                       const KeyitConfig& config,
                       std::vector<float>* logits,
                       std::string* error);

} // namespace keyit
