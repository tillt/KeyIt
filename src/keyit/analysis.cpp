//
//  analysis.cpp
//  KeyIt
//
//  Created by Till Toenshoff on 10.02.26.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//
#include "keyit/keyit.h"
#include "analysis_features_internal.h"
#include "analysis_inference_internal.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace keyit {
namespace {

const char* kCamelot[24] = {
    "1A",
	"2A",
	"3A",
	"4A",
	"5A",
	"6A",
	"7A",
	"8A",
	"9A",
	"10A",
	"11A",
	"12A",
    "1B",
	"2B",
	"3B",
	"4B",
	"5B",
	"6B",
	"7B",
	"8B",
	"9B",
	"10B",
	"11B",
	"12B"
};

const char* kKeyNames[24] = {
    "G# minor/Ab minor",
	"D# minor/Eb minor",
	"A# minor/Bb minor",
	"F minor",
    "C minor",
	"G minor",
	"D minor",
	"A minor",
	"E minor",
	"B minor",
    "F# minor/Gb minor",
	"C# minor/Db minor",
	"B major",
	"F# major/Gb major",
    "C# major/Db major",
	"G# major/Ab major",
	"D# major/Eb major",
	"A# major/Bb major",
    "F major",
	"C major",
	"G major",
	"D major",
	"A major",
	"E major"
};

const int kPitchClasses[24] = {
    8, 3, 10, 5, 0, 7, 2, 9, 4, 11, 6, 1,
    11, 6, 1, 8, 3, 10, 5, 0, 7, 2, 9, 4
};

const float kMajorProfile[12] = {
    6.35f, 2.23f, 3.48f, 2.33f, 4.38f, 4.09f,
    2.52f, 5.19f, 2.39f, 3.66f, 2.29f, 2.88f
};

const float kMinorProfile[12] = {
    6.33f, 2.68f, 3.52f, 5.38f, 2.60f, 3.53f,
    2.54f, 4.75f, 3.98f, 2.69f, 3.34f, 3.17f
};

struct ModeEvidence {
    float major = 0.0f;
    float minor = 0.0f;
};

} // namespace

std::string camelot_label(int class_id) {
    if (class_id < 0 || class_id >= 24) {
        return "Unknown";
    }
    return kCamelot[class_id];
}

std::string key_name_label(int class_id) {
    if (class_id < 0 || class_id >= 24) {
        return "Unknown";
    }
    return kKeyNames[class_id];
}

int pitch_class_for_class(int class_id) {
    if (class_id < 0 || class_id >= 24) {
        return -1;
    }
    return kPitchClasses[class_id];
}

int same_tonic_opposite_mode_class(int class_id) {
    if (class_id < 0 || class_id >= 24) {
        return -1;
    }
    const int pitch_class = pitch_class_for_class(class_id);
    const bool want_major = class_id < 12;
    for (int candidate = 0; candidate < 24; ++candidate) {
        if ((candidate >= 12) != want_major) {
            continue;
        }
        if (pitch_class_for_class(candidate) == pitch_class) {
            return candidate;
        }
    }
    return -1;
}

int cqt_start_pitch_class(const KeyitConfig& config) {
    if (config.cqt_fmin_hz <= 0.0f) {
        return 0;
    }
    const double midi = 69.0 + 12.0 * std::log2(static_cast<double>(config.cqt_fmin_hz) / 440.0);
    const int rounded_midi = static_cast<int>(std::lround(midi));
    const int pitch_class = rounded_midi % 12;
    return (pitch_class < 0) ? (pitch_class + 12) : pitch_class;
}

ModeEvidence compute_same_tonic_mode_evidence(const std::vector<float>& cqt,
                                              std::size_t bins,
                                              std::size_t frames,
                                              const KeyitConfig& config,
                                              int tonic_pitch_class) {
    ModeEvidence evidence;
    if (bins == 0 || frames == 0 || cqt.size() != bins * frames || tonic_pitch_class < 0) {
        return evidence;
    }

    std::vector<float> chroma(12, 0.0f);
    const int start_pitch_class = cqt_start_pitch_class(config);
    const std::size_t bins_per_semitone = std::max<std::size_t>(1, config.bins_per_octave / 12);
    const float inv_frames = 1.0f / static_cast<float>(frames);

    for (std::size_t b = 0; b < bins; ++b) {
        const int semitone = static_cast<int>(b / bins_per_semitone);
        const int pitch_class = (start_pitch_class + semitone) % 12;
        const float* row = cqt.data() + (b * frames);
        float energy = 0.0f;
        for (std::size_t t = 0; t < frames; ++t) {
            energy += row[t];
        }
        chroma[static_cast<std::size_t>(pitch_class)] += energy * inv_frames;
    }

    float mean = 0.0f;
    for (float value : chroma) {
        mean += value;
    }
    mean /= 12.0f;

    float variance = 0.0f;
    for (float value : chroma) {
        const float delta = value - mean;
        variance += delta * delta;
    }
    variance /= 12.0f;
    const float stddev = std::sqrt(std::max(variance, 1e-6f));

    for (int pitch_class = 0; pitch_class < 12; ++pitch_class) {
        const float z = (chroma[static_cast<std::size_t>(pitch_class)] - mean) / stddev;
        const int profile_index = (pitch_class - tonic_pitch_class + 12) % 12;
        evidence.major += kMajorProfile[profile_index] * z;
        evidence.minor += kMinorProfile[profile_index] * z;
    }

    return evidence;
}

std::vector<float> compute_log_cqt_features_from_samples(const std::vector<float>& samples,
                                                         double sample_rate,
                                                         const KeyitConfig& config,
                                                         std::size_t* out_frames,
                                                         std::string* error) {
    if (out_frames) {
        *out_frames = 0;
    }
    if (samples.empty() || sample_rate <= 0.0) {
        if (error) {
            *error = "No audio samples provided";
        }
        return {};
    }
    if (!config.cqt_log_bias_calibration.empty() &&
        config.cqt_log_bias_calibration.size() != config.cqt_bins) {
        if (error) {
            *error = "cqt_log_bias_calibration size must equal cqt_bins";
        }
        return {};
    }

    const std::vector<float> resampled = analysis_internal::resample_linear(samples, sample_rate, config.target_sample_rate);
    if (resampled.empty()) {
        if (error) {
            *error = "Failed to resample audio";
        }
        return {};
    }

    std::size_t raw_frames = 0;
    const std::vector<float> raw = analysis_internal::compute_log_cqt_matrix(resampled, config, &raw_frames);
    if (raw.empty() || raw_frames == 0) {
        if (error) {
            *error = "Failed to compute CQT features";
        }
        return {};
    }

    std::size_t trimmed_frames = 0;
    std::vector<float> trimmed = analysis_internal::trim_trailing_frames(raw,
                                                                         config.cqt_bins,
                                                                         raw_frames,
                                                                         config.trim_trailing_frames,
                                                                         &trimmed_frames);
    if (trimmed.empty() || trimmed_frames == 0) {
        if (error) {
            *error = "CQT result too short after trailing-frame trim";
        }
        return {};
    }

    if (out_frames) {
        *out_frames = trimmed_frames;
    }
    return trimmed;
}

KeyEstimate estimate_key_from_samples(const std::vector<float>& samples,
                                      double sample_rate,
                                      const KeyitConfig& config) {
    KeyEstimate result;
    const auto t0 = std::chrono::steady_clock::now();

    std::size_t frames = 0;
    std::string feat_error;
    if (config.verbose) {
        std::cerr << "keyit: computing CQT features...\n";
    }
    const std::vector<float> cqt_raw =
        compute_log_cqt_features_from_samples(samples, sample_rate, config, &frames, &feat_error);
    const auto t_feat = std::chrono::steady_clock::now();
    if (cqt_raw.empty() || frames == 0) {
        result.error = feat_error.empty() ? "Failed to compute CQT features" : feat_error;
        return result;
    }

    const std::vector<float> cqt = analysis_internal::apply_harmonic_time_median_enhancement(
        cqt_raw, config.cqt_bins, frames, config);
    result.timing.feature_frames = frames;
    result.timing.feature_ms = std::chrono::duration<double, std::milli>(t_feat - t0).count();
    if (config.verbose) {
        std::cerr << "keyit: feature shape=[1,1," << config.cqt_bins << "," << frames << "]\n";
        std::cerr << "keyit: windowed inference with window="
                  << std::max<std::size_t>(1, config.inference_window_frames)
                  << " hop=" << std::max<std::size_t>(1, config.inference_hop_frames)
                  << " frames\n";
    }

    std::vector<float> logits;
    std::size_t windows = 0;
    std::string infer_error;
    if (config.verbose) {
        std::cerr << "keyit: running CoreML inference...\n";
    }
    if (!analysis_internal::aggregate_logits_over_windows(cqt,
                                                          config.cqt_bins,
                                                          frames,
                                                          config,
                                                          &logits,
                                                          &windows,
                                                          &infer_error)) {
        result.error = infer_error;
        return result;
    }
    const auto t_infer = std::chrono::steady_clock::now();
    result.timing.inference_windows = windows;
    result.timing.inference_ms = std::chrono::duration<double, std::milli>(t_infer - t_feat).count();
    if (logits.empty()) {
        result.error = "Model returned empty logits";
        return result;
    }

    std::vector<float> probs = analysis_internal::softmax(logits);
    if (probs.empty()) {
        result.error = "Softmax failed";
        return result;
    }

    std::string section_error;
    std::vector<float> section_probs;
    if (analysis_internal::compute_section_voted_probs(cqt, config.cqt_bins, frames, config, &section_probs, &section_error) &&
        section_probs.size() == probs.size()) {
        const float section_weight = analysis_internal::clampf(config.section_vote_weight, 0.0f, 0.8f);
        const float global_weight = 1.0f - section_weight;
        for (std::size_t i = 0; i < probs.size(); ++i) {
            probs[i] = global_weight * probs[i] + section_weight * section_probs[i];
        }
        float sum = 0.0f;
        for (float p : probs) {
            sum += p;
        }
        if (sum > std::numeric_limits<float>::min()) {
            const float inv = 1.0f / sum;
            for (float& p : probs) {
                p *= inv;
            }
        }
    } else if (config.verbose && !section_error.empty()) {
        std::cerr << "keyit: section voting skipped: " << section_error << "\n";
    }

    auto best_it = std::max_element(probs.begin(), probs.end());
    int best_class = static_cast<int>(std::distance(probs.begin(), best_it));
    int corrected_alternate_class = -1;

    const int opposite_mode_class = same_tonic_opposite_mode_class(best_class);
    if (opposite_mode_class >= 0) {
        const ModeEvidence mode_evidence =
            compute_same_tonic_mode_evidence(cqt, config.cqt_bins, frames, config, pitch_class_for_class(best_class));
        const bool model_prefers_major = best_class >= 12;
        const bool chroma_prefers_major = mode_evidence.major > mode_evidence.minor;
        const float mode_margin = std::fabs(mode_evidence.major - mode_evidence.minor);
        const float best_prob = probs[static_cast<std::size_t>(best_class)];
        const float opposite_prob = probs[static_cast<std::size_t>(opposite_mode_class)];
        const bool opposite_is_competitive =
            opposite_prob >= (best_prob * 0.45f) || (best_prob - opposite_prob) <= 0.08f;
        const bool should_flip_mode =
            (model_prefers_major != chroma_prefers_major) && opposite_is_competitive && mode_margin >= 1.0f;
        if (should_flip_mode) {
            corrected_alternate_class = best_class;
            best_class = opposite_mode_class;
            best_it = probs.begin() + best_class;
            if (config.verbose) {
                std::cerr << "keyit: corrected mode from " << camelot_label(corrected_alternate_class)
                          << " to " << camelot_label(best_class)
                          << " using same-tonic chroma evidence"
                          << " (major=" << mode_evidence.major
                          << ", minor=" << mode_evidence.minor
                          << ", best_p=" << best_prob
                          << ", opp_p=" << opposite_prob
                          << ")\n";
            }
        }
    }

    result.ok = true;
    result.class_id = best_class;
    result.confidence = *best_it;
    result.camelot = camelot_label(best_class);
    result.key_name = key_name_label(best_class);
    std::vector<float> sorted_probs = probs;
    std::sort(sorted_probs.begin(), sorted_probs.end(), std::greater<float>());
    if (sorted_probs.size() >= 2) {
        result.ambiguity_margin = sorted_probs[0] - sorted_probs[1];
    }
    const float ambiguity_threshold = analysis_internal::clampf(config.ambiguity_margin_threshold, 0.0f, 0.5f);
    result.ambiguous = result.ambiguity_margin < ambiguity_threshold;
    if (result.ambiguous) {
        std::vector<int> idx(probs.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + 2, idx.end(),
            [&](int a, int b) { return probs[static_cast<std::size_t>(a)] > probs[static_cast<std::size_t>(b)]; });
        if (idx.size() > 1) {
            result.alternate_class_id = idx[1];
            result.alternate_camelot = camelot_label(idx[1]);
            result.alternate_key_name = key_name_label(idx[1]);
        }
    }
    if (corrected_alternate_class >= 0) {
        result.ambiguous = true;
        result.alternate_class_id = corrected_alternate_class;
        result.alternate_camelot = camelot_label(corrected_alternate_class);
        result.alternate_key_name = key_name_label(corrected_alternate_class);
    }
    result.probabilities = probs;
    result.topk = analysis_internal::build_topk(probs, 5);
    result.timing.total_ms = std::chrono::duration<double, std::milli>(t_infer - t0).count();
    if (config.verbose) {
        std::cerr << "keyit: analysis completed in " << result.timing.total_ms << " ms\n";
    }
    return result;
}

} // namespace keyit
