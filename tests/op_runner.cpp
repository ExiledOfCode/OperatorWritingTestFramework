#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "op_registry.hpp"
#include "op_runtime.hpp"

namespace {

constexpr const char *kAnsiReset = "\033[0m";
constexpr const char *kAnsiGreen = "\033[32m";
constexpr const char *kAnsiRed = "\033[31m";
constexpr const char *kAnsiCyan = "\033[36m";

void append_json_string(std::ostringstream &oss, const std::string &value) {
    oss << '"';
    for (char ch : value) {
        switch (ch) {
        case '\\':
            oss << "\\\\";
            break;
        case '"':
            oss << "\\\"";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            oss << ch;
            break;
        }
    }
    oss << '"';
}

std::string shapes_to_json(const std::vector<std::vector<size_t>> &shapes) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < shapes.size(); ++i) {
        if (i != 0) {
            oss << ',';
        }
        oss << '[';
        for (size_t j = 0; j < shapes[i].size(); ++j) {
            if (j != 0) {
                oss << ',';
            }
            oss << shapes[i][j];
        }
        oss << ']';
    }
    oss << ']';
    return oss.str();
}

std::string params_to_json(const std::vector<ParamSpec> &params) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < params.size(); ++i) {
        if (i != 0) {
            oss << ',';
        }
        oss << "{\"name\":";
        append_json_string(oss, params[i].name);
        oss << ",\"default_value\":" << params[i].default_value << '}';
    }
    oss << ']';
    return oss.str();
}

std::string strings_to_json(const std::vector<std::string> &values) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            oss << ',';
        }
        append_json_string(oss, values[i]);
    }
    oss << ']';
    return oss.str();
}

std::string metrics_to_json(const std::vector<MetricEntry> &metrics) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < metrics.size(); ++i) {
        if (i != 0) {
            oss << ',';
        }
        oss << "{\"name\":";
        append_json_string(oss, metrics[i].name);
        oss << ",\"value\":" << metrics[i].value << ",\"unit\":";
        append_json_string(oss, metrics[i].unit);
        oss << '}';
    }
    oss << ']';
    return oss.str();
}

std::string correctness_to_json(const CorrectnessResult &result) {
    std::ostringstream oss;
    oss << "{\"ok\":" << (result.ok ? "true" : "false") << ",\"metric\":" << result.metric << ",\"metric_name\":";
    append_json_string(oss, result.metric_name);
    oss << ",\"threshold\":" << result.threshold << ",\"note\":";
    append_json_string(oss, result.note);
    oss << '}';
    return oss.str();
}

std::string performance_to_json(const PerfResult &result) {
    std::ostringstream oss;
    oss << "{\"ms\":" << result.ms << ",\"cpu_ms\":" << result.cpu_ms << ",\"unit_name\":";
    append_json_string(oss, result.unit_name);
    oss << ",\"unit_value\":" << result.unit_value << ",\"note\":";
    append_json_string(oss, result.note);
    oss << ",\"input_size\":" << result.input_size << ",\"output_size\":" << result.output_size << ",\"input_format\":"
        << shapes_to_json(result.input_format) << ",\"output_format\":" << shapes_to_json(result.output_format) << ",\"extra_metrics\":"
        << metrics_to_json(result.extra_metrics) << '}';
    return oss.str();
}

std::string describe_to_json(const OpEntry &entry) {
    std::ostringstream oss;
    oss << "{\"op_name\":";
    append_json_string(oss, entry.meta.name);
    oss << ",\"kind\":";
    append_json_string(oss, entry.meta.kind);
    oss << ",\"params\":" << params_to_json(entry.meta.params);
    oss << ",\"edge_axes\":" << strings_to_json(entry.meta.edge_axes);
    oss << ",\"baselines\":" << strings_to_json(entry.meta.baselines);
    oss << ",\"description\":";
    append_json_string(oss, entry.meta.description);
    oss << ",\"binary_mode\":\"single_op\"}";
    return oss.str();
}

std::string run_to_json(const std::string &op_name, const std::string &case_name, const std::string &mode, const CorrectnessResult *correctness,
                        const PerfResult *performance) {
    std::ostringstream oss;
    oss << "{\"op_name\":";
    append_json_string(oss, op_name);
    oss << ",\"case_name\":";
    append_json_string(oss, case_name);
    oss << ",\"mode\":";
    append_json_string(oss, mode);
    oss << ",\"correctness\":";
    if (correctness == nullptr) {
        oss << "null";
    } else {
        oss << correctness_to_json(*correctness);
    }
    oss << ",\"performance\":";
    if (performance == nullptr) {
        oss << "null";
    } else {
        oss << performance_to_json(*performance);
    }
    oss << '}';
    return oss.str();
}

const OpEntry &get_single_entry() {
    auto &registry = op_registry();
    if (registry.empty()) {
        std::cerr << "No ops registered in this binary.\n";
        std::exit(EXIT_FAILURE);
    }
    if (registry.size() != 1) {
        std::cerr << "Expected exactly one registered op in a single-op runner, but found " << registry.size() << ".\n";
        for (const auto &entry : registry) {
            std::cerr << "  - " << entry.name << "\n";
        }
        std::exit(EXIT_FAILURE);
    }
    return registry.front();
}

void print_usage(const char *argv0) {
    std::cerr << "Usage: " << argv0 << " [--describe] [--json] [--mode correctness|performance|both]\n";
}

void print_human_summary(const OpEntry &entry, const std::string &mode, const CorrectnessResult *correctness, const PerfResult *performance) {
    std::cout << "Op: " << entry.meta.name << "\n";
    std::cout << "Kind: " << entry.meta.kind << "\n";
    std::cout << "Case: " << opfw::current_case_name() << "\n";

    if (correctness != nullptr) {
        const char *color = correctness->ok ? kAnsiGreen : kAnsiRed;
        const char *tag = correctness->ok ? "[AC]" : "[WA]";
        std::cout << color << tag << kAnsiReset << "  " << correctness->metric_name << '=' << correctness->metric << "  thr=" << correctness->threshold;
        if (!correctness->note.empty()) {
            std::cout << "  note=" << correctness->note;
        }
        std::cout << "\n";
    }

    if (performance != nullptr) {
        std::cout << kAnsiCyan << "[PERF]" << kAnsiReset << "  " << std::fixed << std::setprecision(4) << performance->ms << " ms"
                  << "  " << performance->unit_name << '=' << performance->unit_value;
        const double total_mb = static_cast<double>(performance->input_size + performance->output_size) / (1024.0 * 1024.0);
        std::cout << "  total_io=" << std::setprecision(2) << total_mb << " MB\n";
        for (const auto &metric : performance->extra_metrics) {
            std::cout << "  " << metric.name << ": " << metric.value << ' ' << metric.unit << "\n";
        }
    }

    const std::string payload = run_to_json(entry.meta.name, opfw::current_case_name(), mode, correctness, performance);
    std::cout << "RESULT_JSON:" << payload << std::endl;
}

} // namespace

int main(int argc, char **argv) {
    bool describe = false;
    bool json_only = false;
    std::string mode = "both";

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--describe") {
            describe = true;
            continue;
        }
        if (arg == "--json") {
            json_only = true;
            continue;
        }
        if (arg == "--mode") {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            mode = argv[++i];
            continue;
        }
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const OpEntry &entry = get_single_entry();

    if (describe) {
        std::cout << describe_to_json(entry) << std::endl;
        return EXIT_SUCCESS;
    }

    if (mode != "correctness" && mode != "performance" && mode != "both") {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    CorrectnessResult correctness{};
    PerfResult performance{};
    const bool run_correctness = mode == "correctness" || mode == "both";
    const bool run_performance = mode == "performance" || mode == "both";

    if (run_correctness) {
        correctness = entry.correctness();
    }
    if (run_performance) {
        performance = entry.performance();
    }

    const std::string payload = run_to_json(entry.meta.name, opfw::current_case_name(), mode, run_correctness ? &correctness : nullptr,
                                            run_performance ? &performance : nullptr);

    if (json_only) {
        std::cout << payload << std::endl;
    } else {
        print_human_summary(entry, mode, run_correctness ? &correctness : nullptr, run_performance ? &performance : nullptr);
    }

    if (run_correctness && (!correctness.ok || correctness.metric > correctness.threshold)) {
        return 2;
    }
    return EXIT_SUCCESS;
}
