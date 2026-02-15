#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

#include "keyit/keyit.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::string input_path;
    std::string model_path;
    std::string input_name = "spec";
    std::string output_name = "logits";
    std::string dump_features_path;
    std::string dump_probs_path;
    std::string cqt_log_bias_calibration_path;
    double max_seconds = 8.0 * 60.0;
    bool ml_cpu_only = false;
    bool bench = false;
    bool show_help = false;
    bool verbose = false;
};

void print_usage(const char* exe) {
    std::cout
        << "Usage: " << exe << " --input <audio-file> [options]\n\n"
        << "Options:\n"
        << "  -i, --input <path>     Audio file (wav/aiff/mp3/m4a/...)\n"
        << "  -m, --model <path>     CoreML model path (.mlmodelc/.mlpackage)\n"
        << "  --ml-input <name>      CoreML input feature name (default: spec)\n"
        << "  --ml-output <name>     CoreML output feature name (default: logits)\n"
        << "  --max-seconds <sec>    Cap analysis duration from start (default: 480)\n"
        << "  --ml-cpu-only          Force CoreML CPU-only execution\n"
        << "  --cqt-log-bias-calibration <path>\n"
        << "                          CSV/whitespace float list with one bias value per CQT bin\n"
        << "  --bench                Print benchmark timings\n"
        << "  --dump-features <path> Write computed CQT features as CSV (rows=freq bins)\n"
        << "  --dump-probs <path>    Write output probabilities (24 lines)\n"
        << "  --verbose              Verbose diagnostics\n"
        << "  -h, --help             Show this help\n";
}

bool read_float_vector_file(const std::string& path,
                            std::vector<float>* values,
                            std::string* error) {
    if (!values) {
        if (error) {
            *error = "Internal error: values pointer is null";
        }
        return false;
    }
    values->clear();

    std::ifstream in(path);
    if (!in.is_open()) {
        if (error) {
            *error = "Failed to open calibration file: " + path;
        }
        return false;
    }

    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }
        for (char& ch : line) {
            if (ch == ',') {
                ch = ' ';
            }
        }

        std::istringstream iss(line);
        float v = 0.0f;
        while (iss >> v) {
            values->push_back(v);
        }

        // Detect parse errors on non-whitespace garbage.
        if (!iss.eof()) {
            if (error) {
                *error = "Invalid calibration value at line " + std::to_string(line_no);
            }
            return false;
        }
    }

    if (values->empty()) {
        if (error) {
            *error = "Calibration file has no values: " + path;
        }
        return false;
    }
    return true;
}

bool parse_args(int argc, char** argv, CliOptions* options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                return {};
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            options->show_help = true;
            return false;
        }
        if (arg == "-i" || arg == "--input") {
            options->input_path = next(arg.c_str());
            continue;
        }
        if (arg == "-m" || arg == "--model") {
            options->model_path = next(arg.c_str());
            continue;
        }
        if (arg == "--ml-input") {
            options->input_name = next(arg.c_str());
            continue;
        }
        if (arg == "--ml-output") {
            options->output_name = next(arg.c_str());
            continue;
        }
        if (arg == "--max-seconds") {
            const std::string value = next(arg.c_str());
            try {
                std::size_t idx = 0;
                const double parsed = std::stod(value, &idx);
                if (idx != value.size() || parsed <= 0.0) {
                    std::cerr << "Invalid --max-seconds: " << value << "\n";
                    return false;
                }
                options->max_seconds = parsed;
            } catch (...) {
                std::cerr << "Invalid --max-seconds: " << value << "\n";
                return false;
            }
            continue;
        }
        if (arg == "--dump-features") {
            options->dump_features_path = next(arg.c_str());
            continue;
        }
        if (arg == "--dump-probs") {
            options->dump_probs_path = next(arg.c_str());
            continue;
        }
        if (arg == "--ml-cpu-only") {
            options->ml_cpu_only = true;
            continue;
        }
        if (arg == "--cqt-log-bias-calibration") {
            options->cqt_log_bias_calibration_path = next(arg.c_str());
            continue;
        }
        if (arg == "--bench") {
            options->bench = true;
            continue;
        }
        if (arg == "--verbose") {
            options->verbose = true;
            continue;
        }

        std::cerr << "Unknown option: " << arg << "\n";
        return false;
    }

    if (options->input_path.empty()) {
        std::cerr << "Missing required --input\n";
        return false;
    }
    return true;
}

bool load_audio_file_mono_f32(const std::string& path,
                              std::vector<float>* samples,
                              double* sample_rate,
                              std::string* error) {
    if (!samples || !sample_rate) {
        if (error) {
            *error = "Internal error: output pointers are null";
        }
        return false;
    }

    @autoreleasepool {
        NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:ns_path];
        NSError* ns_error = nil;

        AVAudioFile* file = [[AVAudioFile alloc] initForReading:url
                                                   commonFormat:AVAudioPCMFormatFloat32
                                                    interleaved:NO
                                                          error:&ns_error];
        if (!file || ns_error) {
            if (error) {
                *error = [[NSString stringWithFormat:@"Failed to open audio file: %@",
                           ns_error.localizedDescription] UTF8String];
            }
            return false;
        }

        const AVAudioFrameCount frame_count = static_cast<AVAudioFrameCount>(file.length);
        if (frame_count == 0) {
            if (error) {
                *error = "Audio file is empty";
            }
            return false;
        }

        AVAudioFormat* format = file.processingFormat;
        AVAudioPCMBuffer* src_buffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:format
                                                                        frameCapacity:frame_count];
        if (![file readIntoBuffer:src_buffer error:&ns_error] || ns_error) {
            if (error) {
                *error = [[NSString stringWithFormat:@"Failed reading audio: %@",
                           ns_error.localizedDescription] UTF8String];
            }
            return false;
        }
        if (format.commonFormat != AVAudioPCMFormatFloat32 || src_buffer.floatChannelData == nil) {
            if (error) {
                *error = "Audio decode did not produce float32 PCM";
            }
            return false;
        }

        const AVAudioFrameCount out_frames = src_buffer.frameLength;
        const AVAudioChannelCount channels = format.channelCount;
        if (out_frames == 0 || channels == 0) {
            if (error) {
                *error = "Decoded audio has no frames";
            }
            return false;
        }

        samples->assign(out_frames, 0.0f);
        for (AVAudioChannelCount ch = 0; ch < channels; ++ch) {
            const float* channel_data = src_buffer.floatChannelData[ch];
            if (!channel_data) {
                continue;
            }
            for (AVAudioFrameCount i = 0; i < out_frames; ++i) {
                (*samples)[i] += channel_data[i];
            }
        }
        const float inv_channels = 1.0f / static_cast<float>(channels);
        for (float& s : *samples) {
            s *= inv_channels;
        }

        *sample_rate = format.sampleRate;
        return true;
    }
}

bool write_matrix_csv(const std::string& path,
                      const std::vector<float>& matrix,
                      std::size_t rows,
                      std::size_t cols,
                      std::string* error) {
    std::ofstream out(path);
    if (!out.is_open()) {
        if (error) {
            *error = "Failed to open feature dump path: " + path;
        }
        return false;
    }
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            if (c != 0) {
                out << ",";
            }
            out << matrix[r * cols + c];
        }
        out << "\n";
    }
    return true;
}

bool write_vector_lines(const std::string& path,
                        const std::vector<float>& values,
                        std::string* error) {
    std::ofstream out(path);
    if (!out.is_open()) {
        if (error) {
            *error = "Failed to open probability dump path: " + path;
        }
        return false;
    }
    for (float v : values) {
        out << v << "\n";
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const auto t_start = std::chrono::steady_clock::now();
    CliOptions options;
    if (!parse_args(argc, argv, &options)) {
        return options.show_help ? 0 : 1;
    }

    if (!std::filesystem::exists(options.input_path)) {
        std::cerr << "Input file does not exist: " << options.input_path << "\n";
        return 1;
    }

    std::vector<float> samples;
    double sample_rate = 0.0;
    std::string load_error;
    const auto t_decode_start = std::chrono::steady_clock::now();
    if (!load_audio_file_mono_f32(options.input_path, &samples, &sample_rate, &load_error)) {
        std::cerr << load_error << "\n";
        return 1;
    }
    const auto t_decode_end = std::chrono::steady_clock::now();
    const std::size_t max_samples =
        static_cast<std::size_t>(std::llround(options.max_seconds * sample_rate));
    if (max_samples > 0 && samples.size() > max_samples) {
        samples.resize(max_samples);
        if (options.verbose) {
            std::cerr << "Input capped to first " << options.max_seconds << " seconds.\n";
        }
    }

    keyit::KeyitConfig config;
    config.verbose = options.verbose;
    config.coreml_cpu_only = options.ml_cpu_only;
    config.input_name = options.input_name;
    config.output_name = options.output_name;
    if (!options.model_path.empty()) {
        config.model_path = options.model_path;
    }
    if (!options.cqt_log_bias_calibration_path.empty()) {
        std::string cal_error;
        if (!read_float_vector_file(options.cqt_log_bias_calibration_path,
                                    &config.cqt_log_bias_calibration,
                                    &cal_error)) {
            std::cerr << cal_error << "\n";
            return 1;
        }
    }
    if (options.bench) {
        config.verbose = true;
    }
    if (!options.dump_features_path.empty()) {
        std::size_t frames = 0;
        std::string feature_error;
        const std::vector<float> features = keyit::compute_log_cqt_features_from_samples(
            samples, sample_rate, config, &frames, &feature_error);
        if (features.empty()) {
            std::cerr << "Feature dump failed: " << feature_error << "\n";
            return 2;
        }
        std::string write_error;
        if (!write_matrix_csv(options.dump_features_path, features, config.cqt_bins, frames, &write_error)) {
            std::cerr << write_error << "\n";
            return 2;
        }
    }

    const auto t_analyze_start = std::chrono::steady_clock::now();
    keyit::KeyEstimate estimate = keyit::estimate_key_from_samples(samples, sample_rate, config);
    const auto t_analyze_end = std::chrono::steady_clock::now();
    if (!estimate.ok) {
        std::cerr << "Key analysis failed: " << estimate.error << "\n";
        return 2;
    }

    std::cout << "File: " << options.input_path << "\n";
    std::cout << "Class: " << estimate.class_id << "\n";
    std::cout << "Camelot: " << estimate.camelot << "\n";
    std::cout << "Key: " << estimate.key_name << "\n";
    std::cout << "Confidence: " << estimate.confidence << "\n";
    std::cout << "Ambiguous: " << (estimate.ambiguous ? "yes" : "no")
              << " (margin=" << estimate.ambiguity_margin << ")\n";
    if (estimate.ambiguous && estimate.alternate_class_id >= 0) {
        std::cout << "Alternate: " << estimate.alternate_class_id
                  << "  " << estimate.alternate_camelot
                  << "  " << estimate.alternate_key_name << "\n";
    }
    std::cout << "Top-5:\n";
    for (const auto& cls : estimate.topk) {
        std::cout << "  " << cls.class_id
                  << "  " << cls.camelot
                  << "  " << cls.key_name
                  << "  p=" << cls.probability
                  << "\n";
    }

    if (!options.dump_probs_path.empty()) {
        std::string write_error;
        if (!write_vector_lines(options.dump_probs_path, estimate.probabilities, &write_error)) {
            std::cerr << write_error << "\n";
            return 2;
        }
    }

    if (options.bench) {
        const auto t_end = std::chrono::steady_clock::now();
        const double decode_ms =
            std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
        const double analyze_wall_ms =
            std::chrono::duration<double, std::milli>(t_analyze_end - t_analyze_start).count();
        const double total_ms =
            std::chrono::duration<double, std::milli>(t_end - t_start).count();
        const double analyzed_seconds = sample_rate > 0.0
            ? static_cast<double>(samples.size()) / sample_rate
            : 0.0;
        const double x_realtime = (analyze_wall_ms > 0.0)
            ? (analyzed_seconds / (analyze_wall_ms / 1000.0))
            : 0.0;

        std::cout << "\n[bench]\n";
        std::cout << "decode_ms=" << decode_ms << "\n";
        std::cout << "analysis_feature_ms=" << estimate.timing.feature_ms << "\n";
        std::cout << "analysis_inference_ms=" << estimate.timing.inference_ms << "\n";
        std::cout << "analysis_total_ms=" << estimate.timing.total_ms << "\n";
        std::cout << "analysis_wall_ms=" << analyze_wall_ms << "\n";
        std::cout << "total_ms=" << total_ms << "\n";
        std::cout << "feature_frames=" << estimate.timing.feature_frames << "\n";
        std::cout << "inference_windows=" << estimate.timing.inference_windows << "\n";
        std::cout << "audio_seconds=" << analyzed_seconds << "\n";
        std::cout << "analysis_x_realtime=" << x_realtime << "\n";
    }

    return 0;
}
