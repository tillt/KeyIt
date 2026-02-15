//
//  keyit.h
//  KeyIt
//
//  Public C++ API for musical key estimation.
//
//  Created by Till Toenshoff on 10.02.26.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace keyit {

/// @defgroup api keyit Public API
/// Public, supported C++ interfaces for key estimation and feature extraction.
/// @{

/**
 * @brief Configuration for feature extraction and key estimation.
 *
 * Default values are tuned for the bundled KeyNet CoreML model.
 */
struct KeyitConfig {
    std::string model_path = "models/keynet.mlmodelc";
    std::string input_name = "spec";
    std::string output_name = "logits";
    std::size_t target_sample_rate = 44100;
    std::size_t bins_per_octave = 24;
    std::size_t cqt_bins = 105;
    std::size_t hop_length = 8820;
    float cqt_fmin_hz = 65.0f;
    float cqt_magnitude_gain = 1.0f;
    std::vector<float> cqt_log_bias_calibration;
    std::size_t trim_trailing_frames = 2;
    std::size_t inference_window_frames = 100;
    std::size_t inference_hop_frames = 100;
    bool harmonic_time_median_enhance = true;
    std::size_t harmonic_time_median_window = 9;
    float harmonic_time_median_blend = 0.5f;
    float frame_energy_gate_quantile = 0.15f;
    float frame_energy_gate_softness = 0.10f;
    bool use_section_voting = true;
    std::size_t section_vote_count = 3;
    float section_vote_head_skip_ratio = 0.10f;
    float section_vote_tail_skip_ratio = 0.10f;
    float section_vote_weight = 0.40f;
    float ambiguity_margin_threshold = 0.08f;
    bool coreml_cpu_only = false;
    bool verbose = false;
};

/// @ingroup api
/// Probability score for a single key class.
struct ClassScore {
    int class_id = -1;
    float probability = 0.0f;
    std::string camelot;
    std::string key_name;
};

/// @ingroup api
/// Timing breakdown for the analysis pipeline.
struct PipelineTiming {
    double feature_ms = 0.0;
    double inference_ms = 0.0;
    double total_ms = 0.0;
    std::size_t feature_frames = 0;
    std::size_t inference_windows = 0;
};

/**
 * @brief Result of key estimation.
 *
 * When `ok == true`, `class_id`, `camelot`, `key_name`, `confidence`, and `probabilities`
 * contain the inferred result. When `ok == false`, `error` contains a short message.
 */
struct KeyEstimate {
    bool ok = false;
    std::string error;
    int class_id = -1;
    float confidence = 0.0f;
    std::string camelot;
    std::string key_name;
    bool ambiguous = false;
    float ambiguity_margin = 0.0f;
    int alternate_class_id = -1;
    std::string alternate_camelot;
    std::string alternate_key_name;
    std::vector<float> probabilities;
    std::vector<ClassScore> topk;
    PipelineTiming timing;
};

/**
 * @brief Estimate musical key from mono PCM samples.
 *
 * @param samples Input mono audio samples (`[-1, 1]` float32-style values recommended).
 * @param sample_rate Sample rate of `samples` in Hz.
 * @param config Runtime configuration (defaults are suitable for bundled model).
 * @return Key estimation result including top prediction, probabilities, and timing.
 */
KeyEstimate estimate_key_from_samples(const std::vector<float>& samples,
                                      double sample_rate,
                                      const KeyitConfig& config = {});  ///< @ingroup api

/**
 * @brief Compute log-CQT-like features from mono PCM samples.
 *
 * This exposes the frontend used by `estimate_key_from_samples(...)`.
 *
 * @param samples Input mono audio samples.
 * @param sample_rate Sample rate of `samples` in Hz.
 * @param config Frontend configuration.
 * @param out_frames Optional output frame count.
 * @param error Optional error message on failure.
 * @return Feature matrix in row-major `(bins x frames)` flattened storage.
 */
std::vector<float> compute_log_cqt_features_from_samples(const std::vector<float>& samples,
                                                         double sample_rate,
                                                         const KeyitConfig& config,
                                                         std::size_t* out_frames,
                                                         std::string* error);  ///< @ingroup api

/// @brief Convert a class id (`0..23`) to Camelot notation (`1A..12A`, `1B..12B`).
/// @ingroup api
std::string camelot_label(int class_id);

/// @brief Convert a class id (`0..23`) to a human-readable key name.
/// @ingroup api
std::string key_name_label(int class_id);

/// @}

}  // namespace keyit
