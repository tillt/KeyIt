#include "keyit/keyit.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool expect_true(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        return false;
    }
    return true;
}

bool test_label_mapping() {
    bool ok = true;
    ok &= expect_true(keyit::camelot_label(0) == "1A", "camelot label for class 0");
    ok &= expect_true(keyit::camelot_label(11) == "12A", "camelot label for class 11");
    ok &= expect_true(keyit::camelot_label(12) == "1B", "camelot label for class 12");
    ok &= expect_true(keyit::camelot_label(23) == "12B", "camelot label for class 23");
    ok &= expect_true(keyit::camelot_label(-1) == "Unknown", "camelot label for negative class");
    ok &= expect_true(keyit::camelot_label(24) == "Unknown", "camelot label for out-of-range class");

    ok &= expect_true(keyit::key_name_label(0) == "G# minor/Ab minor", "key name for class 0");
    ok &= expect_true(keyit::key_name_label(23) == "E major", "key name for class 23");
    ok &= expect_true(keyit::key_name_label(-1) == "Unknown", "key name for negative class");
    ok &= expect_true(keyit::key_name_label(24) == "Unknown", "key name for out-of-range class");
    return ok;
}

std::vector<float> make_sine(double sample_rate, double hz, double seconds) {
    const std::size_t n = static_cast<std::size_t>(std::llround(sample_rate * seconds));
    std::vector<float> out(n, 0.0f);
    const double w = 2.0 * M_PI * hz / sample_rate;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(0.25 * std::sin(w * static_cast<double>(i)));
    }
    return out;
}

bool test_feature_shape_contract() {
    keyit::KeyitConfig cfg;
    const double sr = 44100.0;
    const std::vector<float> samples = make_sine(sr, 440.0, 10.0);

    std::size_t frames = 0;
    std::string err;
    const std::vector<float> feat =
        keyit::compute_log_cqt_features_from_samples(samples, sr, cfg, &frames, &err);

    bool ok = true;
    ok &= expect_true(!feat.empty(), "features are non-empty");
    ok &= expect_true(err.empty(), "no error string");
    ok &= expect_true(frames > 0, "frame count > 0");
    ok &= expect_true(feat.size() == cfg.cqt_bins * frames, "feature vector size matches bins*frames");

    // 10s at 44.1k / 8820 hop -> raw frames = 51, then trim 2 -> 49
    ok &= expect_true(frames == 49, "expected trimmed frame count for 10s synthetic signal");
    return ok;
}

bool test_feature_determinism() {
    keyit::KeyitConfig cfg;
    const double sr = 44100.0;
    const std::vector<float> samples = make_sine(sr, 261.625565, 8.0);

    std::size_t f1 = 0;
    std::size_t f2 = 0;
    std::string e1;
    std::string e2;
    const std::vector<float> a =
        keyit::compute_log_cqt_features_from_samples(samples, sr, cfg, &f1, &e1);
    const std::vector<float> b =
        keyit::compute_log_cqt_features_from_samples(samples, sr, cfg, &f2, &e2);

    bool ok = true;
    ok &= expect_true(e1.empty() && e2.empty(), "determinism runs return no errors");
    ok &= expect_true(f1 == f2, "determinism runs have same frame count");
    ok &= expect_true(a.size() == b.size(), "determinism runs have same feature size");

    float max_abs = 0.0f;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        max_abs = std::max(max_abs, std::abs(a[i] - b[i]));
    }
    ok &= expect_true(max_abs <= 1e-9f, "determinism max abs diff <= 1e-9");
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok &= test_label_mapping();
    ok &= test_feature_shape_contract();
    ok &= test_feature_determinism();

    if (!ok) {
        return 1;
    }

    std::cout << "PASS: keyit_tests\n";
    return 0;
}
