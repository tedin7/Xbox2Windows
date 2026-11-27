#include "direct_compute_cache.h"
#include "shader_dxil_compiler.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace x2w;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point since) {
    return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
}
static void check(bool ok, const char *reason) {
    if (!ok) throw std::runtime_error(reason);
}
int main(int argc, char **argv) {
    try {
        if (argc < 3 || argc > 4) return 2;
        const fs::path input = argv[1], file = argv[2];
        std::vector<fs::path> markers;
        for (auto &it : fs::directory_iterator(input))
            if (it.path().extension() == ".c67") markers.push_back(it.path());
        std::sort(markers.begin(), markers.end());
        auto start = Clock::now();
        auto pack = ComputeFrontendPack::load(file, 0);
        const auto load_ms = elapsed(start);
        std::ofstream report;
        if (argc == 4) {
            report.open(argv[3]);
            report.exceptions(std::ios::failbit | std::ios::badbit);
            report << "key\tstatus\tcold_ms\tlookup_ms\tir_sha256_or_diagnostic\tmanifest_sha256\n";
        }
        size_t success = 0, refused = 0, other = 0;
        double live_total = 0, warm_total = 0;
        bool tested_success = false, tested_refused = false;
        for (const auto &marker : markers) {
            auto r = read_compute_request(marker);
            NativeComputeShader warm, cold;
            std::string warm_error, cold_error;
            start = Clock::now();
            const auto hit = pack.lookup(r, 0, warm, warm_error);
            const double warm_ms = elapsed(start);
            start = Clock::now();
            ComputePackResult expected = ComputePackResult::prepared;
            try { cold = translate_compute(r, 0, true); }
            catch (const UnsupportedComputeShader &e) { expected = ComputePackResult::unsupported; cold_error = e.what(); }
            catch (const std::runtime_error &e) { expected = ComputePackResult::miss; cold_error = e.what(); }
            const double live_ms = elapsed(start);
            live_total += live_ms;
            warm_total += warm_ms;
            check(hit == expected, "result class differs from direct frontend");
            const char *status;
            std::string identity, manifest_digest;
            if (hit == ComputePackResult::prepared) {
                auto manifest = format_compute_runtime_manifest(cold);
                check(warm.dxil_ir == cold.dxil_ir && warm.group == cold.group &&
                      warm.instruction_count == cold.instruction_count && format_compute_runtime_manifest(warm) == manifest,
                      "IR, layout or manifest differs");
                status = "prepared";
                identity = shader_digest(cold.dxil_ir);
                if (manifest) manifest_digest = shader_digest(*manifest);
                ++success;
            } else if (hit == ComputePackResult::unsupported) {
                check(warm_error == cold_error, "refusal differs");
                status = "unsupported";
                identity = cold_error;
                ++refused;
            } else {
                status = "other";
                identity = cold_error;
                ++other;
            }
            if ((hit == ComputePackResult::prepared && !tested_success) ||
                (hit == ComputePackResult::unsupported && !tested_refused)) {
                for (auto *part : {&r.code, &r.data, &r.root, &r.description, &r.sa, &r.sb}) {
                    part->push_back(0);
                    check(pack.lookup(r, 0, warm, warm_error) == ComputePackResult::miss, "stale input accepted");
                    part->pop_back();
                }
                check(pack.lookup(r, 4, warm, warm_error) == ComputePackResult::miss, "stale entry accepted");
                if (hit == ComputePackResult::prepared) tested_success = true;
                else tested_refused = true;
            }
            if (report.is_open())
                report << marker.stem().string() << '\t' << status << '\t' << live_ms << '\t' << warm_ms << '\t'
                       << identity << '\t' << manifest_digest << '\n';
        }
        check(tested_success && tested_refused && pack.size() == success + refused, "missing test coverage");
        std::cout << "prepared=" << success << " unsupported=" << refused << " other=" << other
                  << " load_ms=" << load_ms << " cold_ms=" << live_total << " lookup_ms=" << warm_total << '\n';
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
