//
//  analysis_features_internal.cpp
//  KeyIt
//
//  Created by Till Toenshoff on 10.02.26.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//
#include "analysis_features_internal.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace keyit {
namespace analysis_internal {
namespace {

constexpr float kPi = 3.14159265358979323846f;

struct CQTKernel {
    int first_offset = 0;
    std::vector<float> real;
    std::vector<float> imag;
    float sqrt_length = 1.0f;
    float inv_sqrt_length = 1.0f;
};

struct KernelCacheKey {
    std::size_t target_sample_rate = 0;
    std::size_t bins_per_octave = 0;
    std::size_t cqt_bins = 0;
    std::uint32_t fmin_bits = 0;

    bool operator==(const KernelCacheKey& other) const {
        return target_sample_rate == other.target_sample_rate &&
               bins_per_octave == other.bins_per_octave &&
               cqt_bins == other.cqt_bins &&
               fmin_bits == other.fmin_bits;
    }
};

struct KernelCacheKeyHash {
    std::size_t operator()(const KernelCacheKey& k) const noexcept {
        std::size_t h = 1469598103934665603ull;
        auto mix = [&](std::size_t v) {
            h ^= v;
            h *= 1099511628211ull;
        };
        mix(k.target_sample_rate);
        mix(k.bins_per_octave);
        mix(k.cqt_bins);
        mix(k.fmin_bits);
        return h;
    }
};

struct KernelPack {
    std::size_t bins = 0;
    int min_offset = 0;
    int support = 0;
    std::vector<CQTKernel> kernels;
    std::vector<float> kernel_real_matrix;
    std::vector<float> kernel_imag_matrix;
    std::vector<float> inv_sqrt;
};

struct CQTWorkspace {
    std::vector<float> padded;
    std::vector<float> frame_real;
    std::vector<float> frame_imag;
    std::vector<float> frame_mag;
    std::vector<float> frame_scaled;
    std::vector<float> frame_log;
};

KernelCacheKey make_kernel_cache_key(const KeyitConfig& config) {
    union {
        float f;
        std::uint32_t u;
    } u{};
    u.f = config.cqt_fmin_hz;
    return KernelCacheKey{
        config.target_sample_rate,
        config.bins_per_octave,
        config.cqt_bins,
        u.u,
    };
}

std::vector<CQTKernel> build_cqt_kernels(const KeyitConfig& config) {
    std::vector<CQTKernel> kernels;
    kernels.reserve(config.cqt_bins);

    const float bpo = static_cast<float>(config.bins_per_octave);
    const float q = 1.0f / (std::pow(2.0f, 1.0f / bpo) - 1.0f);

    for (std::size_t k = 0; k < config.cqt_bins; ++k) {
        const float freq = config.cqt_fmin_hz * std::pow(2.0f, static_cast<float>(k) / bpo);
        const float ilen = (q * static_cast<float>(config.target_sample_rate)) / freq;

        const int start = static_cast<int>(std::floor(-ilen / 2.0f));
        const int end = static_cast<int>(std::floor(ilen / 2.0f));
        const int n = std::max(1, end - start);

        CQTKernel kernel;
        kernel.first_offset = start;
        kernel.real.resize(static_cast<std::size_t>(n));
        kernel.imag.resize(static_cast<std::size_t>(n));
        kernel.sqrt_length = std::sqrt(static_cast<float>(n));
        kernel.inv_sqrt_length = (kernel.sqrt_length > 0.0f) ? (1.0f / kernel.sqrt_length) : 1.0f;

        for (int i = 0; i < n; ++i) {
            const int t = start + i;
            const float phase = 2.0f * kPi * freq * (static_cast<float>(t) / static_cast<float>(config.target_sample_rate));
            const float hann = 0.5f - 0.5f * std::cos(2.0f * kPi * static_cast<float>(i) / static_cast<float>(n));
            const float re = hann * std::cos(phase);
            const float im = -hann * std::sin(phase);

            kernel.real[static_cast<std::size_t>(i)] = re;
            kernel.imag[static_cast<std::size_t>(i)] = im;
        }

        std::vector<float> mags(static_cast<std::size_t>(n), 0.0f);
        DSPSplitComplex split{kernel.real.data(), kernel.imag.data()};
        vDSP_zvabs(&split, 1, mags.data(), 1, static_cast<vDSP_Length>(n));
        float l1 = 0.0f;
        vDSP_sve(mags.data(), 1, &l1, static_cast<vDSP_Length>(n));
        if (l1 > std::numeric_limits<float>::min()) {
            const float inv = 1.0f / l1;
            vDSP_vsmul(kernel.real.data(), 1, &inv, kernel.real.data(), 1, static_cast<vDSP_Length>(n));
            vDSP_vsmul(kernel.imag.data(), 1, &inv, kernel.imag.data(), 1, static_cast<vDSP_Length>(n));
        }

        kernels.push_back(std::move(kernel));
    }

    return kernels;
}

const KernelPack& get_cached_kernel_pack(const KeyitConfig& config) {
    static thread_local std::unordered_map<KernelCacheKey,
                                           std::shared_ptr<KernelPack>,
                                           KernelCacheKeyHash> cache;
    const KernelCacheKey key = make_kernel_cache_key(config);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return *(it->second);
    }

    auto pack = std::make_shared<KernelPack>();
    pack->bins = config.cqt_bins;
    pack->kernels = build_cqt_kernels(config);

    int min_offset = pack->kernels.front().first_offset;
    int max_offset_exclusive = pack->kernels.front().first_offset +
                               static_cast<int>(pack->kernels.front().real.size());
    for (const auto& k : pack->kernels) {
        min_offset = std::min(min_offset, k.first_offset);
        max_offset_exclusive =
            std::max(max_offset_exclusive, k.first_offset + static_cast<int>(k.real.size()));
    }
    pack->min_offset = min_offset;
    pack->support = max_offset_exclusive - min_offset;

    pack->kernel_real_matrix.assign(pack->bins * static_cast<std::size_t>(pack->support), 0.0f);
    pack->kernel_imag_matrix.assign(pack->bins * static_cast<std::size_t>(pack->support), 0.0f);
    pack->inv_sqrt.assign(pack->bins, 1.0f);

    for (std::size_t b = 0; b < pack->bins; ++b) {
        const CQTKernel& k = pack->kernels[b];
        const int row_offset = k.first_offset - min_offset;
        const std::size_t row_start = b * static_cast<std::size_t>(pack->support) +
                                      static_cast<std::size_t>(row_offset);
        std::copy(k.real.begin(), k.real.end(), pack->kernel_real_matrix.begin() + row_start);
        std::copy(k.imag.begin(), k.imag.end(), pack->kernel_imag_matrix.begin() + row_start);
        pack->inv_sqrt[b] = k.inv_sqrt_length;
    }

    auto inserted = cache.emplace(key, pack);
    return *(inserted.first->second);
}

CQTWorkspace& get_workspace() {
    static thread_local CQTWorkspace ws;
    return ws;
}

const float* resolve_log_bias_calibration(const KeyitConfig& config, std::size_t bins) {
    if (!config.cqt_log_bias_calibration.empty()) {
        if (config.cqt_log_bias_calibration.size() == bins) {
            return config.cqt_log_bias_calibration.data();
        }
        return nullptr;
    }
    return nullptr;
}

} // namespace

std::vector<float> resample_linear(const std::vector<float>& input,
                                   double input_rate,
                                   std::size_t target_rate) {
    if (input.empty() || input_rate <= 0.0 || target_rate == 0) {
        return {};
    }

    const std::size_t rounded_input = static_cast<std::size_t>(std::llround(input_rate));
    if (rounded_input == target_rate) {
        return input;
    }

    const double ratio = static_cast<double>(target_rate) / input_rate;
    const std::size_t out_size = static_cast<std::size_t>(std::llround(input.size() * ratio));
    std::vector<float> out(out_size, 0.0f);

    if (input.size() < 2 || out_size == 0) {
        if (!input.empty() && !out.empty()) {
            std::fill(out.begin(), out.end(), input.front());
        }
        return out;
    }

    std::vector<float> indices(out_size, 0.0f);
    float start = 0.0f;
    float step = static_cast<float>(1.0 / ratio);
    vDSP_vramp(&start, &step, indices.data(), 1, static_cast<vDSP_Length>(out_size));

    // Keep interpolation indices within [0, input_size - 1].
    const float lo = 0.0f;
    const float hi = static_cast<float>(input.size() - 1);
    vDSP_vclip(indices.data(), 1, &lo, &hi, indices.data(), 1, static_cast<vDSP_Length>(out_size));

    vDSP_vlint(input.data(),
               indices.data(),
               1,
               out.data(),
               1,
               static_cast<vDSP_Length>(out_size),
               static_cast<vDSP_Length>(input.size()));

    return out;
}

std::vector<float> compute_log_cqt_matrix(const std::vector<float>& samples,
                                          const KeyitConfig& config,
                                          std::size_t* out_frames) {
    if (out_frames) {
        *out_frames = 0;
    }
    if (samples.empty() || config.hop_length == 0 || config.cqt_bins == 0) {
        return {};
    }

    const std::size_t frames = (samples.size() / config.hop_length) + 1;
    if (frames == 0) {
        return {};
    }

    const KernelPack& pack = get_cached_kernel_pack(config);
    const std::size_t bins = pack.bins;
    const int min_offset = pack.min_offset;
    const int support = pack.support;
    if (support <= 0) {
        return {};
    }

    const std::size_t pad = static_cast<std::size_t>(support);
    CQTWorkspace& ws = get_workspace();
    ws.padded.assign(samples.size() + pad * 2, 0.0f);
    std::copy(samples.begin(), samples.end(), ws.padded.begin() + static_cast<std::ptrdiff_t>(pad));

    std::vector<float> matrix(config.cqt_bins * frames, 0.0f);
    ws.frame_real.assign(bins, 0.0f);
    ws.frame_imag.assign(bins, 0.0f);
    ws.frame_mag.assign(config.cqt_bins, 0.0f);
    ws.frame_scaled.assign(config.cqt_bins, 0.0f);
    ws.frame_log.assign(config.cqt_bins, 0.0f);
    const float* log_bias = resolve_log_bias_calibration(config, bins);

    for (std::size_t t = 0; t < frames; ++t) {
        const std::size_t center = t * config.hop_length;
        if (center > static_cast<std::size_t>(std::numeric_limits<long long>::max())) {
            continue;
        }
        const long long segment_start_signed =
            static_cast<long long>(center) +
            static_cast<long long>(min_offset) +
            static_cast<long long>(pad);
        if (segment_start_signed < 0) {
            continue;
        }
        const std::size_t segment_start = static_cast<std::size_t>(segment_start_signed);
        if (segment_start + static_cast<std::size_t>(support) > ws.padded.size()) {
            continue;
        }
        const float* segment = ws.padded.data() + static_cast<std::ptrdiff_t>(segment_start);

        // (bins x support) * (support x 1) => (bins x 1)
        vDSP_mmul(pack.kernel_real_matrix.data(),
                  1,
                  segment,
                  1,
                  ws.frame_real.data(),
                  1,
                  static_cast<vDSP_Length>(bins),
                  1,
                  static_cast<vDSP_Length>(support));
        vDSP_mmul(pack.kernel_imag_matrix.data(),
                  1,
                  segment,
                  1,
                  ws.frame_imag.data(),
                  1,
                  static_cast<vDSP_Length>(bins),
                  1,
                  static_cast<vDSP_Length>(support));

        DSPSplitComplex split{ws.frame_real.data(), ws.frame_imag.data()};
        vDSP_zvabs(&split, 1, ws.frame_mag.data(), 1, static_cast<vDSP_Length>(bins));
        vDSP_vsmul(ws.frame_mag.data(),
                   1,
                   &config.cqt_magnitude_gain,
                   ws.frame_scaled.data(),
                   1,
                   static_cast<vDSP_Length>(bins));
        int log_n = static_cast<int>(bins);
        vvlog1pf(ws.frame_log.data(), ws.frame_scaled.data(), &log_n);
        if (log_bias) {
            vDSP_vadd(ws.frame_log.data(), 1, log_bias, 1, ws.frame_log.data(), 1, static_cast<vDSP_Length>(bins));
        }

        for (std::size_t b = 0; b < bins; ++b) {
            matrix[b * frames + t] = ws.frame_log[b];
        }
    }

    if (out_frames) {
        *out_frames = frames;
    }
    return matrix;
}

std::vector<float> trim_trailing_frames(const std::vector<float>& matrix,
                                        std::size_t bins,
                                        std::size_t in_frames,
                                        std::size_t trim,
                                        std::size_t* out_frames) {
    if (out_frames) {
        *out_frames = 0;
    }
    if (bins == 0 || in_frames == 0) {
        return {};
    }
    if (trim >= in_frames) {
        return {};
    }

    const std::size_t frames = in_frames - trim;
    std::vector<float> out(bins * frames, 0.0f);
    for (std::size_t b = 0; b < bins; ++b) {
        const float* src = matrix.data() + b * in_frames;
        float* dst = out.data() + b * frames;
        std::copy(src, src + frames, dst);
    }

    if (out_frames) {
        *out_frames = frames;
    }
    return out;
}

} // namespace analysis_internal
} // namespace keyit
