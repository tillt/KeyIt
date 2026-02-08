#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace keyit {

struct KeyitConfig {
    std::string model_path = "models/keynet.mlmodelc";
    std::string input_name = "spec";
    std::string output_name = "logits";
    std::size_t target_sample_rate = 44100;
    std::size_t bins_per_octave = 24;
    std::size_t cqt_bins = 105;
    std::size_t hop_length = 8820;
    float cqt_fmin_hz = 65.0f;
    std::size_t trim_trailing_frames = 2;
    std::size_t inference_window_frames = 100;
    std::size_t inference_hop_frames = 100;
    bool coreml_cpu_only = false;
    bool verbose = false;
};

struct ClassScore {
    int class_id = -1;
    float probability = 0.0f;
    std::string camelot;
    std::string key_name;
};

struct PipelineTiming {
    double feature_ms = 0.0;
    double inference_ms = 0.0;
    double total_ms = 0.0;
    std::size_t feature_frames = 0;
    std::size_t inference_windows = 0;
};

struct KeyEstimate {
    bool ok = false;
    std::string error;
    int class_id = -1;
    float confidence = 0.0f;
    std::string camelot;
    std::string key_name;
    std::vector<float> probabilities;
    std::vector<ClassScore> topk;
    PipelineTiming timing;
};

KeyEstimate estimate_key_from_samples(const std::vector<float>& samples,
                                      double sample_rate,
                                      const KeyitConfig& config = {});

std::vector<float> compute_log_cqt_features_from_samples(const std::vector<float>& samples,
                                                         double sample_rate,
                                                         const KeyitConfig& config,
                                                         std::size_t* out_frames,
                                                         std::string* error);

std::string camelot_label(int class_id);
std::string key_name_label(int class_id);

} // namespace keyit
