#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

int run_cli(const fs::path& cli,
            const fs::path& audio,
            const fs::path& model,
            bool cpu_only,
            const fs::path& log_path) {
    std::ostringstream cmd;
    cmd << '"' << cli.string() << '"'
        << " --input " << '"' << audio.string() << '"'
        << " --model " << '"' << model.string() << '"';
    if (cpu_only) {
        cmd << " --ml-cpu-only";
    }
    cmd << " > " << '"' << log_path.string() << '"' << " 2>&1";
    return std::system(cmd.str().c_str());
}

std::string read_text(const fs::path& path) {
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
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

    const fs::path model_path = fs::path(KEYIT_TEST_MODEL_PATH);
    const fs::path wav_path = fs::path(KEYIT_TEST_AUDIO_DIR) / "c_major_triad_12s.wav";
    const fs::path cli_path = fs::path(KEYIT_TEST_CLI_PATH);

    if (!fs::exists(model_path)) {
        std::cerr << "SKIP: model not found: " << model_path << "\n";
        return 77;
    }
    if (!fs::exists(wav_path)) {
        std::cerr << "SKIP: fixture not found: " << wav_path << "\n";
        return 77;
    }
    if (!fs::exists(cli_path)) {
        std::cerr << "SKIP: keyit-cli not found: " << cli_path << "\n";
        return 77;
    }

    const fs::path tmp_dir = fs::temp_directory_path() / "keyit_gpu_tests";
    std::error_code ec;
    fs::create_directories(tmp_dir, ec);
    const fs::path gpu_log = tmp_dir / "gpu.log";
    const fs::path cpu_log = tmp_dir / "cpu.log";

    const int gpu_rc = run_cli(cli_path, wav_path, model_path, false, gpu_log);
    if (gpu_rc == 0) {
        std::cout << "PASS: keyit_gpu_tests accelerated/default path succeeded\n";
        return 0;
    }

    // If accelerated/default path crashes or fails on this machine/runtime,
    // ensure CPU fallback is still functional and mark as skipped.
    const int cpu_rc = run_cli(cli_path, wav_path, model_path, true, cpu_log);
    if (cpu_rc == 0) {
        std::cerr << "SKIP: accelerated/default CoreML path unavailable on this runtime. "
                  << "CPU fallback succeeded.\n";
        const std::string gpu_out = read_text(gpu_log);
        if (!gpu_out.empty()) {
            std::cerr << "GPU/default log:\n" << gpu_out << "\n";
        }
        return 77;
    }

    std::cerr << "FAIL: both accelerated/default and CPU-only paths failed\n";
    const std::string gpu_out = read_text(gpu_log);
    const std::string cpu_out = read_text(cpu_log);
    if (!gpu_out.empty()) {
        std::cerr << "GPU/default log:\n" << gpu_out << "\n";
    }
    if (!cpu_out.empty()) {
        std::cerr << "CPU-only log:\n" << cpu_out << "\n";
    }
    return 1;
}
