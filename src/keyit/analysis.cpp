#include "keyit/keyit.h"
#include "coreml_internal.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace keyit {
namespace {

constexpr float kPi = 3.14159265358979323846f;

constexpr std::array<float, 105> kDefaultCqtLogBiasCalibration = {
    0.816209280f, 0.691344393f, 0.678358230f, 0.753089163f, 0.751547973f,
    0.711355049f, 0.716198483f, 0.759912857f, 0.812856443f, 0.774768212f,
    0.711787168f, 0.619642502f, 0.588299708f, 0.652690464f, 0.597054439f,
    0.584151417f, 0.556521805f, 0.430352961f, 0.482169870f, 0.447112485f,
    0.452582219f, 0.433626678f, 0.493246177f, 0.448857517f, 0.388343368f,
    0.420318738f, 0.381577386f, 0.390134552f, 0.402957371f, 0.370300336f,
    0.360008146f, 0.363319969f, 0.455445864f, 0.436736131f, 0.372235103f,
    0.337314140f, 0.366456869f, 0.397487394f, 0.422997965f, 0.361849434f,
    0.344424956f, 0.351010119f, 0.297234622f, 0.228245696f, 0.204559109f,
    0.263926589f, 0.251728098f, 0.193496538f, 0.239621082f, 0.182138216f,
    0.201962362f, 0.219739411f, 0.243641240f, 0.231498873f, 0.314374969f,
    0.315068730f, 0.080637425f, 0.121943142f, 0.052291274f, 0.128138334f,
    0.234914445f, 0.178965481f, 0.216092705f, 0.138084346f, 0.178549141f,
    0.185745757f, 0.135154533f, 0.164993032f, 0.092262185f, 0.168976764f,
    0.169595197f, 0.100837406f, 0.184808034f, 0.157645607f, 0.168720451f,
    0.101423349f, 0.092653439f, 0.203173753f, 0.112284406f, 0.232589637f,
    0.208997425f, 0.142273173f, 0.202643907f, 0.139134924f, 0.151037658f,
    0.081096587f, 0.085256607f, 0.082672996f, 0.064680264f, 0.054788602f,
    0.080078784f, 0.154635945f, 0.119011239f, 0.190702917f, 0.160737616f,
    0.087745315f, 0.064917036f, 0.089128870f, 0.059553782f, 0.093965025f,
    0.119848688f, 0.118956238f, 0.014480743f, 0.144949744f, -0.117349230f,
};

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
    if (config.use_default_cqt_log_bias_calibration &&
        bins == kDefaultCqtLogBiasCalibration.size()) {
        return kDefaultCqtLogBiasCalibration.data();
    }
    return nullptr;
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
        const int center = static_cast<int>(t * config.hop_length);
        const std::size_t segment_start =
            static_cast<std::size_t>(center + min_offset) + pad;
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

    std::vector<float> sum_logits;
    std::size_t windows = 0;
    std::vector<float> window_input(1 * 1 * bins * window_frames, 0.0f);

    for (std::size_t start : starts) {
        std::fill(window_input.begin(), window_input.end(), 0.0f);

        const std::size_t available = (start < frames) ? (frames - start) : 0;
        const std::size_t copy_frames = std::min(window_frames, available);
        for (std::size_t b = 0; b < bins; ++b) {
            const float* src = cqt.data() + b * frames + start;
            float* dst = window_input.data() + b * window_frames;
            std::copy(src, src + copy_frames, dst);
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

        if (sum_logits.empty()) {
            sum_logits.assign(logits.begin(), logits.begin() + 24);
        } else {
            for (std::size_t i = 0; i < 24; ++i) {
                sum_logits[i] += logits[i];
            }
        }
        ++windows;
    }

    if (windows == 0 || sum_logits.empty()) {
        if (error) {
            *error = "No inference windows were produced";
        }
        return false;
    }

    const float inv = 1.0f / static_cast<float>(windows);
    for (float& v : sum_logits) {
        v *= inv;
    }
    *aggregated_logits = std::move(sum_logits);
    if (out_windows) {
        *out_windows = windows;
    }
    return true;
}

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

    const std::vector<float> resampled = resample_linear(samples, sample_rate, config.target_sample_rate);
    if (resampled.empty()) {
        if (error) {
            *error = "Failed to resample audio";
        }
        return {};
    }

    std::size_t raw_frames = 0;
    const std::vector<float> raw = compute_log_cqt_matrix(resampled, config, &raw_frames);
    if (raw.empty() || raw_frames == 0) {
        if (error) {
            *error = "Failed to compute CQT features";
        }
        return {};
    }

    std::size_t trimmed_frames = 0;
    std::vector<float> trimmed = trim_trailing_frames(raw,
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
    const std::vector<float> cqt =
        compute_log_cqt_features_from_samples(samples, sample_rate, config, &frames, &feat_error);
    const auto t_feat = std::chrono::steady_clock::now();
    if (cqt.empty() || frames == 0) {
        result.error = feat_error.empty() ? "Failed to compute CQT features" : feat_error;
        return result;
    }
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
    if (!aggregate_logits_over_windows(cqt,
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

    const std::vector<float> probs = softmax(logits);
    if (probs.empty()) {
        result.error = "Softmax failed";
        return result;
    }

    auto best_it = std::max_element(probs.begin(), probs.end());
    const int best_class = static_cast<int>(std::distance(probs.begin(), best_it));

    result.ok = true;
    result.class_id = best_class;
    result.confidence = *best_it;
    result.camelot = camelot_label(best_class);
    result.key_name = key_name_label(best_class);
    result.probabilities = probs;
    result.topk = build_topk(probs, 5);
    result.timing.total_ms = std::chrono::duration<double, std::milli>(t_infer - t0).count();
    if (config.verbose) {
        std::cerr << "keyit: analysis completed in " << result.timing.total_ms << " ms\n";
    }
    return result;
}

} // namespace keyit
