//
//  analysis_inference_internal.cpp
//  KeyIt
//
//  Created by Till Toenshoff on 10.02.26.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//
#include "analysis_inference_internal.h"
#include "coreml_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace keyit {
namespace analysis_internal {
namespace {

std::vector<float> compute_frame_energy_weights(const std::vector<float>& matrix,
                                                std::size_t bins,
                                                std::size_t frames,
                                                const KeyitConfig& config) {
    if (bins == 0 || frames == 0 || matrix.size() != bins * frames) {
        return {};
    }
    std::vector<float> weights(frames, 1.0f);
    const float q = clampf(config.frame_energy_gate_quantile, 0.0f, 0.95f);
    if (q <= 0.0f) {
        return weights;
    }

    std::vector<float> energy(frames, 0.0f);
    for (std::size_t b = 0; b < bins; ++b) {
        const float* row = matrix.data() + b * frames;
        for (std::size_t t = 0; t < frames; ++t) {
            energy[t] += row[t];
        }
    }
    const float inv_bins = 1.0f / static_cast<float>(bins);
    for (float& e : energy) {
        e *= inv_bins;
    }

    std::vector<float> sorted = energy;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t qidx = static_cast<std::size_t>(q * static_cast<float>(sorted.size() - 1));
    const float threshold = sorted[qidx];
    const float emin = sorted.front();
    const float emax = sorted.back();
    const float spread = std::max(1e-6f, (emax - emin) * clampf(config.frame_energy_gate_softness, 0.02f, 1.0f));

    float mean_w = 0.0f;
    for (std::size_t t = 0; t < frames; ++t) {
        const float z = (energy[t] - threshold) / spread;
        const float w = 1.0f / (1.0f + std::exp(-z));
        weights[t] = w;
        mean_w += w;
    }
    mean_w /= static_cast<float>(frames);
    if (mean_w > 1e-6f) {
        const float inv_mean = 1.0f / mean_w;
        for (float& w : weights) {
            w = clampf(w * inv_mean, 0.2f, 2.0f);
        }
    }
    return weights;
}

std::vector<float> slice_cqt_frames(const std::vector<float>& cqt,
                                    std::size_t bins,
                                    std::size_t frames,
                                    std::size_t start,
                                    std::size_t count) {
    if (bins == 0 || frames == 0 || count == 0 || start >= frames || cqt.size() != bins * frames) {
        return {};
    }
    const std::size_t n = std::min(count, frames - start);
    std::vector<float> out(bins * n, 0.0f);
    for (std::size_t b = 0; b < bins; ++b) {
        const float* src = cqt.data() + b * frames + start;
        float* dst = out.data() + b * n;
        std::copy(src, src + n, dst);
    }
    return out;
}

} // namespace

float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

std::vector<float> apply_harmonic_time_median_enhancement(const std::vector<float>& matrix,
                                                          std::size_t bins,
                                                          std::size_t frames,
                                                          const KeyitConfig& config) {
    if (!config.harmonic_time_median_enhance ||
        config.harmonic_time_median_window < 3 ||
        bins == 0 || frames == 0 ||
        matrix.size() != bins * frames) {
        return matrix;
    }

    const std::size_t window = std::max<std::size_t>(3, config.harmonic_time_median_window | 1ull);
    const std::size_t half = window / 2;
    const float blend = clampf(config.harmonic_time_median_blend, 0.0f, 1.0f);
    if (blend <= 0.0f) {
        return matrix;
    }

    std::vector<float> out(matrix.size(), 0.0f);
    std::vector<float> tmp;
    tmp.reserve(window);

    for (std::size_t b = 0; b < bins; ++b) {
        const float* row = matrix.data() + b * frames;
        float* dst = out.data() + b * frames;
        for (std::size_t t = 0; t < frames; ++t) {
            const std::size_t s = (t > half) ? (t - half) : 0;
            const std::size_t e = std::min(frames, t + half + 1);
            tmp.assign(row + s, row + e);
            auto mid = tmp.begin() + static_cast<std::ptrdiff_t>(tmp.size() / 2);
            std::nth_element(tmp.begin(), mid, tmp.end());
            const float median = *mid;
            dst[t] = (1.0f - blend) * row[t] + blend * median;
        }
    }
    return out;
}

std::vector<float> softmax(const std::vector<float>& logits) {
    if (logits.empty()) {
        return {};
    }

    float max_logit = logits[0];
    for (float v : logits) {
        max_logit = std::max(max_logit, v);
    }

    std::vector<float> probs(logits.size(), 0.0f);
    float sum = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }

    if (sum <= std::numeric_limits<float>::min()) {
        return probs;
    }

    const float inv = 1.0f / sum;
    for (float& v : probs) {
        v *= inv;
    }
    return probs;
}

std::vector<ClassScore> build_topk(const std::vector<float>& probs, std::size_t k) {
    std::vector<int> idx(probs.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + std::min(k, idx.size()), idx.end(),
        [&](int a, int b) { return probs[static_cast<std::size_t>(a)] > probs[static_cast<std::size_t>(b)]; });

    std::vector<ClassScore> out;
    const std::size_t count = std::min(k, idx.size());
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const int cls = idx[i];
        out.push_back(ClassScore{cls, probs[static_cast<std::size_t>(cls)], camelot_label(cls), key_name_label(cls)});
    }
    return out;
}

bool aggregate_logits_over_windows(const std::vector<float>& cqt,
                                   std::size_t bins,
                                   std::size_t frames,
                                   const KeyitConfig& config,
                                   std::vector<float>* aggregated_logits,
                                   std::size_t* out_windows,
                                   std::string* error) {
    if (!aggregated_logits) {
        if (error) {
            *error = "Internal error: aggregated logits pointer is null";
        }
        return false;
    }
    aggregated_logits->clear();
    if (out_windows) {
        *out_windows = 0;
    }

    const std::size_t window_frames = std::max<std::size_t>(1, config.inference_window_frames);
    const std::size_t hop_frames = std::max<std::size_t>(1, config.inference_hop_frames);
    if (bins == 0 || frames == 0 || cqt.size() != bins * frames) {
        if (error) {
            *error = "Invalid CQT shape for windowed inference";
        }
        return false;
    }

    std::vector<std::size_t> starts;
    if (frames <= window_frames) {
        starts.push_back(0);
    } else {
        for (std::size_t s = 0; s + window_frames <= frames; s += hop_frames) {
            starts.push_back(s);
        }
        const std::size_t tail_start = frames - window_frames;
        if (starts.empty() || starts.back() != tail_start) {
            starts.push_back(tail_start);
        }
    }

    const std::vector<float> frame_weights = compute_frame_energy_weights(cqt, bins, frames, config);
    if (frame_weights.empty()) {
        if (error) {
            *error = "Failed to compute frame energy weights";
        }
        return false;
    }

    std::vector<float> sum_probs(24, 0.0f);
    float total_weight = 0.0f;
    std::size_t windows = 0;
    std::vector<float> window_input(1 * 1 * bins * window_frames, 0.0f);

    for (std::size_t start : starts) {
        std::fill(window_input.begin(), window_input.end(), 0.0f);

        const std::size_t available = (start < frames) ? (frames - start) : 0;
        const std::size_t copy_frames = std::min(window_frames, available);
        float window_quality = 0.0f;
        for (std::size_t t = 0; t < copy_frames; ++t) {
            window_quality += frame_weights[start + t];
        }
        window_quality /= std::max<std::size_t>(1, copy_frames);
        for (std::size_t b = 0; b < bins; ++b) {
            const float* src = cqt.data() + b * frames + start;
            float* dst = window_input.data() + b * window_frames;
            for (std::size_t t = 0; t < copy_frames; ++t) {
                dst[t] = src[t] * frame_weights[start + t];
            }
        }

        std::vector<float> logits;
        std::string infer_error;
        if (!run_keynet_coreml(window_input, config, &logits, &infer_error)) {
            if (error) {
                *error = "Windowed CoreML inference failed: " + infer_error;
            }
            return false;
        }
        if (logits.size() < 24) {
            if (error) {
                *error = "CoreML logits size is smaller than expected 24 classes";
            }
            return false;
        }

        std::vector<float> probs = softmax(std::vector<float>(logits.begin(), logits.begin() + 24));
        if (probs.empty()) {
            if (error) {
                *error = "Window softmax failed";
            }
            return false;
        }
        float max_prob = 0.0f;
        for (float p : probs) {
            max_prob = std::max(max_prob, p);
        }
        const float conf = clampf((max_prob - (1.0f / 24.0f)) / (1.0f - (1.0f / 24.0f)), 0.0f, 1.0f);
        const float window_weight = std::max(0.1f, window_quality * (0.5f + conf));
        for (std::size_t i = 0; i < 24; ++i) {
            sum_probs[i] += probs[i] * window_weight;
        }
        total_weight += window_weight;
        ++windows;
    }

    if (windows == 0 || total_weight <= std::numeric_limits<float>::min()) {
        if (error) {
            *error = "No inference windows were produced";
        }
        return false;
    }

    const float inv = 1.0f / total_weight;
    std::vector<float> merged_logits(24, 0.0f);
    for (std::size_t i = 0; i < 24; ++i) {
        const float p = std::max(sum_probs[i] * inv, 1e-9f);
        merged_logits[i] = std::log(p);
    }
    *aggregated_logits = std::move(merged_logits);
    if (out_windows) {
        *out_windows = windows;
    }
    return true;
}

bool compute_section_voted_probs(const std::vector<float>& cqt,
                                 std::size_t bins,
                                 std::size_t frames,
                                 const KeyitConfig& config,
                                 std::vector<float>* out_probs,
                                 std::string* error) {
    if (!out_probs) {
        return false;
    }
    out_probs->clear();
    if (!config.use_section_voting || config.section_vote_count == 0) {
        return false;
    }

    const float head_skip = clampf(config.section_vote_head_skip_ratio, 0.0f, 0.45f);
    const float tail_skip = clampf(config.section_vote_tail_skip_ratio, 0.0f, 0.45f);
    const std::size_t head_frames = static_cast<std::size_t>(static_cast<float>(frames) * head_skip);
    const std::size_t tail_frames = static_cast<std::size_t>(static_cast<float>(frames) * tail_skip);
    if (head_frames + tail_frames >= frames) {
        return false;
    }
    const std::size_t usable_start = head_frames;
    const std::size_t usable_frames = frames - head_frames - tail_frames;
    const std::size_t min_frames_for_sections = std::max<std::size_t>(
        config.inference_window_frames * 2,
        config.section_vote_count * config.inference_window_frames);
    if (usable_frames < min_frames_for_sections) {
        return false;
    }

    const std::size_t section_count = std::max<std::size_t>(1, config.section_vote_count);
    const std::size_t section_len = std::max<std::size_t>(config.inference_window_frames, usable_frames / section_count);
    std::vector<float> sum_probs(24, 0.0f);
    std::size_t used = 0;

    for (std::size_t i = 0; i < section_count; ++i) {
        const float center_pos = (static_cast<float>(i) + 0.5f) / static_cast<float>(section_count);
        const std::size_t center = usable_start + static_cast<std::size_t>(center_pos * static_cast<float>(usable_frames));
        std::size_t start = (center > section_len / 2) ? (center - section_len / 2) : usable_start;
        if (start < usable_start) {
            start = usable_start;
        }
        if (start + section_len > usable_start + usable_frames) {
            start = usable_start + usable_frames - section_len;
        }

        const std::vector<float> sec = slice_cqt_frames(cqt, bins, frames, start, section_len);
        if (sec.empty()) {
            continue;
        }
        std::vector<float> sec_logits;
        std::size_t sec_windows = 0;
        std::string sec_err;
        if (!aggregate_logits_over_windows(sec, bins, section_len, config, &sec_logits, &sec_windows, &sec_err)) {
            continue;
        }
        const std::vector<float> sec_probs = softmax(sec_logits);
        if (sec_probs.size() != 24) {
            continue;
        }
        for (std::size_t c = 0; c < 24; ++c) {
            sum_probs[c] += sec_probs[c];
        }
        ++used;
    }

    if (used == 0) {
        if (error) {
            *error = "Section voting produced no valid sections";
        }
        return false;
    }
    const float inv = 1.0f / static_cast<float>(used);
    for (float& p : sum_probs) {
        p *= inv;
    }
    *out_probs = std::move(sum_probs);
    return true;
}

} // namespace analysis_internal
} // namespace keyit
