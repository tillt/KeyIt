#include "keyit/keyit.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct WavData {
    int sample_rate = 0;
    std::vector<float> samples;
};

bool expect_true(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        return false;
    }
    return true;
}

bool read_u16(std::ifstream& in, std::uint16_t* value) {
    unsigned char b[2]{};
    in.read(reinterpret_cast<char*>(b), 2);
    if (!in) return false;
    *value = static_cast<std::uint16_t>(b[0] | (b[1] << 8));
    return true;
}

bool read_u32(std::ifstream& in, std::uint32_t* value) {
    unsigned char b[4]{};
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) return false;
    *value = static_cast<std::uint32_t>(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
    return true;
}

bool load_wav_pcm16_mono(const fs::path& path, WavData* out, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error) *error = "Cannot open WAV: " + path.string();
        return false;
    }

    char riff[4]{};
    in.read(riff, 4);
    std::uint32_t riff_size = 0;
    if (!read_u32(in, &riff_size)) {
        if (error) *error = "Invalid WAV header (size)";
        return false;
    }
    char wave[4]{};
    in.read(wave, 4);
    if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
        if (error) *error = "Not a RIFF/WAVE file";
        return false;
    }

    bool have_fmt = false;
    bool have_data = false;
    std::uint16_t audio_format = 0;
    std::uint16_t channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint16_t bits_per_sample = 0;
    std::vector<std::int16_t> pcm;

    while (in && !(have_fmt && have_data)) {
        char chunk_id[4]{};
        in.read(chunk_id, 4);
        if (!in) break;
        std::uint32_t chunk_size = 0;
        if (!read_u32(in, &chunk_size)) break;

        const std::string id(chunk_id, 4);
        if (id == "fmt ") {
            have_fmt = true;
            if (!read_u16(in, &audio_format) || !read_u16(in, &channels) || !read_u32(in, &sample_rate)) {
                if (error) *error = "Invalid fmt chunk";
                return false;
            }
            std::uint32_t byte_rate = 0;
            std::uint16_t block_align = 0;
            if (!read_u32(in, &byte_rate) || !read_u16(in, &block_align) || !read_u16(in, &bits_per_sample)) {
                (void)byte_rate;
                (void)block_align;
                if (error) *error = "Invalid fmt payload";
                return false;
            }
            const std::size_t consumed = 16;
            if (chunk_size > consumed) {
                in.seekg(static_cast<std::streamoff>(chunk_size - consumed), std::ios::cur);
            }
        } else if (id == "data") {
            have_data = true;
            pcm.resize(chunk_size / sizeof(std::int16_t));
            in.read(reinterpret_cast<char*>(pcm.data()), static_cast<std::streamsize>(chunk_size));
            if (!in) {
                if (error) *error = "Invalid data chunk";
                return false;
            }
        } else {
            in.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
        }

        if (chunk_size % 2 == 1) {
            in.seekg(1, std::ios::cur);
        }
    }

    if (!have_fmt || !have_data) {
        if (error) *error = "Missing fmt or data chunk";
        return false;
    }
    if (audio_format != 1 || channels != 1 || bits_per_sample != 16) {
        if (error) *error = "Expected PCM16 mono WAV";
        return false;
    }

    out->sample_rate = static_cast<int>(sample_rate);
    out->samples.resize(pcm.size());
    for (std::size_t i = 0; i < pcm.size(); ++i) {
        out->samples[i] = static_cast<float>(pcm[i] / 32768.0f);
    }
    return true;
}

std::size_t csv_column_count(const fs::path& path) {
    std::ifstream in(path);
    std::string line;
    if (!std::getline(in, line) || line.empty()) {
        return 0;
    }
    std::size_t cols = 1;
    for (char c : line) {
        if (c == ',') {
            ++cols;
        }
    }
    return cols;
}

int run_cli_and_dump_features_once(const fs::path& cli,
                                   const fs::path& audio,
                                   const fs::path& model,
                                   double max_seconds,
                                   const fs::path& dump_csv,
                                   bool cpu_only) {
    const fs::path log_path = dump_csv.parent_path() / (dump_csv.filename().string() + ".log");
    std::ostringstream cmd;
    cmd << '"' << cli.string() << '"'
        << " --input " << '"' << audio.string() << '"'
        << " --model " << '"' << model.string() << '"'
        << (cpu_only ? " --ml-cpu-only" : "")
        << " --max-seconds " << max_seconds
        << " --dump-features " << '"' << dump_csv.string() << '"'
        << " > " << '"' << log_path.string() << '"' << " 2>&1";
    return std::system(cmd.str().c_str());
}

bool run_cli_and_dump_features(const fs::path& cli,
                               const fs::path& audio,
                               const fs::path& model,
                               double max_seconds,
                               const fs::path& dump_csv) {
    const int rc_default = run_cli_and_dump_features_once(cli, audio, model, max_seconds, dump_csv, false);
    if (rc_default == 0) {
        return true;
    }
    const int rc_cpu = run_cli_and_dump_features_once(cli, audio, model, max_seconds, dump_csv, true);
    return rc_cpu == 0;
}

bool test_fixtures_and_features(const fs::path& audio_dir) {
    const std::vector<std::string> names = {
        "sine_a4_10s.wav",
        "c_major_triad_12s.wav",
        "b_major_triad_12s.wav",
        "e_major_triad_12s.wav",
        "a_minor_triad_12s.wav",
        "c_major_to_a_minor_20s.wav",
        "low_noise_8s.wav",
    };

    keyit::KeyitConfig cfg;
    bool ok = true;

    for (const auto& name : names) {
        const fs::path wav = audio_dir / name;
        WavData data;
        std::string err;
        ok &= expect_true(fs::exists(wav), "fixture exists: " + wav.string());
        ok &= expect_true(load_wav_pcm16_mono(wav, &data, &err), "load fixture WAV: " + name + (err.empty() ? "" : (" (" + err + ")")));
        if (data.samples.empty() || data.sample_rate <= 0) {
            ok &= expect_true(false, "fixture has valid sample content: " + name);
            continue;
        }

        std::size_t frames = 0;
        std::string ferr;
        const std::vector<float> feat = keyit::compute_log_cqt_features_from_samples(
            data.samples,
            static_cast<double>(data.sample_rate),
            cfg,
            &frames,
            &ferr);

        ok &= expect_true(!feat.empty(), "features non-empty: " + name);
        ok &= expect_true(ferr.empty(), "features no error: " + name);
        ok &= expect_true(frames > 0, "feature frames > 0: " + name);
        ok &= expect_true(feat.size() == cfg.cqt_bins * frames, "feature size matches bins*frames: " + name);

        float max_abs = 0.0f;
        for (float v : feat) {
            max_abs = std::max(max_abs, std::abs(v));
        }
        ok &= expect_true(std::isfinite(max_abs), "features finite: " + name);
    }

    return ok;
}

bool test_model_inference_if_available(const fs::path& audio_dir, const fs::path& model_path) {
    if (!fs::exists(model_path)) {
        std::cerr << "WARN: model not found, skipping inference checks: " << model_path << "\n";
        return true;
    }

    const fs::path wav = audio_dir / "c_major_triad_12s.wav";
    WavData data;
    std::string err;
    bool ok = true;
    ok &= expect_true(load_wav_pcm16_mono(wav, &data, &err), "load inference fixture");
    if (!ok) {
        return false;
    }

    keyit::KeyitConfig cfg;
    cfg.model_path = model_path.string();
    cfg.coreml_cpu_only = true;
    keyit::KeyEstimate est = keyit::estimate_key_from_samples(
        data.samples,
        static_cast<double>(data.sample_rate),
        cfg);

    ok &= expect_true(est.ok, std::string("inference ok") + (est.error.empty() ? "" : (": " + est.error)));
    ok &= expect_true(est.probabilities.size() == 24, "inference produces 24 probabilities");
    ok &= expect_true(!est.topk.empty(), "topk non-empty");
    ok &= expect_true(est.class_id >= 0 && est.class_id < 24, "class id in range");
    ok &= expect_true(est.confidence >= 0.0f && est.confidence <= 1.0f, "confidence in [0,1]");

    return ok;
}

bool test_mode_balance_regression_if_available(const fs::path& audio_dir, const fs::path& model_path) {
    if (!fs::exists(model_path)) {
        std::cerr << "WARN: model not found, skipping mode-balance regression checks: " << model_path << "\n";
        return true;
    }

    const std::vector<std::string> names = {
        "c_major_triad_12s.wav",
        "a_minor_triad_12s.wav",
    };

    bool ok = true;
    for (const auto& name : names) {
        const fs::path wav = audio_dir / name;
        WavData data;
        std::string err;
        ok &= expect_true(load_wav_pcm16_mono(wav, &data, &err),
                          "load mode-balance fixture: " + name + (err.empty() ? "" : (" (" + err + ")")));
        if (data.samples.empty() || data.sample_rate <= 0) {
            ok &= expect_true(false, "mode-balance fixture has valid samples: " + name);
            continue;
        }

        keyit::KeyitConfig cfg;
        cfg.model_path = model_path.string();
        cfg.coreml_cpu_only = true;
        keyit::KeyEstimate est = keyit::estimate_key_from_samples(
            data.samples,
            static_cast<double>(data.sample_rate),
            cfg);

        ok &= expect_true(est.ok, "mode-balance inference ok: " + name);
        ok &= expect_true(est.probabilities.size() == 24, "mode-balance has 24 probabilities: " + name);
        if (!est.ok || est.probabilities.size() != 24) {
            continue;
        }

        float a_mass = 0.0f;
        float b_mass = 0.0f;
        for (int i = 0; i < 12; ++i) {
            a_mass += est.probabilities[static_cast<std::size_t>(i)];
            b_mass += est.probabilities[static_cast<std::size_t>(i + 12)];
        }

        // Regression guard: if one mode's probability mass collapses close to zero,
        // we effectively lose major/minor discrimination regardless of top-1.
        ok &= expect_true(a_mass > 0.20f, "A-side probability mass remains present: " + name);
        ok &= expect_true(b_mass > 0.20f, "B-side probability mass remains present: " + name);

    }

    return ok;
}

bool test_b_side_major_predictions_if_available(const fs::path& audio_dir, const fs::path& model_path) {
    if (!fs::exists(model_path)) {
        std::cerr << "WARN: model not found, skipping B-side major prediction checks: " << model_path << "\n";
        return true;
    }

    const std::vector<std::string> names = {
        "b_major_triad_12s.wav",
        "e_major_triad_12s.wav",
    };

    bool ok = true;
    for (const auto& name : names) {
        const fs::path wav = audio_dir / name;
        WavData data;
        std::string err;
        ok &= expect_true(load_wav_pcm16_mono(wav, &data, &err),
                          "load B-side fixture: " + name + (err.empty() ? "" : (" (" + err + ")")));
        if (data.samples.empty() || data.sample_rate <= 0) {
            ok &= expect_true(false, "B-side fixture has valid samples: " + name);
            continue;
        }

        keyit::KeyitConfig cfg;
        cfg.model_path = model_path.string();
        cfg.coreml_cpu_only = true;
        keyit::KeyEstimate est = keyit::estimate_key_from_samples(
            data.samples,
            static_cast<double>(data.sample_rate),
            cfg);

        ok &= expect_true(est.ok, "B-side inference ok: " + name);
        ok &= expect_true(est.class_id >= 12 && est.class_id < 24, "B-side top-1 class id (12..23): " + name);
        ok &= expect_true(!est.camelot.empty() && est.camelot.back() == 'B', "B-side camelot suffix B: " + name);
    }
    return ok;
}

bool test_cli_cap_behavior_if_available(const fs::path& audio_dir,
                                        const fs::path& model_path,
                                        const fs::path& cli_path) {
    if (!fs::exists(model_path) || !fs::exists(cli_path)) {
        std::cerr << "WARN: model or cli not found, skipping CLI cap behavior checks\n";
        return true;
    }

    const fs::path wav = audio_dir / "c_major_to_a_minor_20s.wav";
    const fs::path tmp_dir = fs::temp_directory_path() / "keyit_cli_tests";
    std::error_code ec;
    fs::create_directories(tmp_dir, ec);

    const fs::path short_csv = tmp_dir / "short_features.csv";
    const fs::path long_csv = tmp_dir / "long_features.csv";

    bool ok = true;
    ok &= expect_true(run_cli_and_dump_features(cli_path, wav, model_path, 1.0, short_csv), "CLI run with --max-seconds 1");
    ok &= expect_true(run_cli_and_dump_features(cli_path, wav, model_path, 20.0, long_csv), "CLI run with --max-seconds 20");
    if (!ok) {
        return false;
    }

    const std::size_t short_cols = csv_column_count(short_csv);
    const std::size_t long_cols = csv_column_count(long_csv);

    ok &= expect_true(short_cols > 0, "short feature dump has columns");
    ok &= expect_true(long_cols > 0, "long feature dump has columns");
    ok &= expect_true(short_cols < long_cols, "short max-seconds yields fewer frames than long max-seconds");

    return ok;
}

} // namespace

int main() {
#ifndef KEYIT_TEST_AUDIO_DIR
    std::cerr << "FAIL: KEYIT_TEST_AUDIO_DIR is not defined\n";
    return 1;
#endif
#ifndef KEYIT_TEST_MODEL_PATH
    std::cerr << "FAIL: KEYIT_TEST_MODEL_PATH is not defined\n";
    return 1;
#endif
#ifndef KEYIT_TEST_CLI_PATH
    std::cerr << "FAIL: KEYIT_TEST_CLI_PATH is not defined\n";
    return 1;
#endif

    const fs::path audio_dir = fs::path(KEYIT_TEST_AUDIO_DIR);
    const fs::path model_path = fs::path(KEYIT_TEST_MODEL_PATH);
    const fs::path cli_path = fs::path(KEYIT_TEST_CLI_PATH);

    bool ok = true;
    ok &= test_fixtures_and_features(audio_dir);
    ok &= test_model_inference_if_available(audio_dir, model_path);
    ok &= test_mode_balance_regression_if_available(audio_dir, model_path);
    ok &= test_b_side_major_predictions_if_available(audio_dir, model_path);
    ok &= test_cli_cap_behavior_if_available(audio_dir, model_path, cli_path);

    if (!ok) {
        return 1;
    }

    std::cout << "PASS: keyit_pipeline_tests\n";
    return 0;
}
