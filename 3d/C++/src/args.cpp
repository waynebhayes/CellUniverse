#include <iostream>
#include <filesystem>
#include <optional>
#include <cstdlib>
#include <ctime>
#include <random>
#include <thread>
#include <cxxopts.hpp>

namespace fs = std::filesystem;

struct Args {
    std::string input;
    fs::path output;
    fs::path config;
    fs::path initial;
    std::optional<fs::path> debug;
    int first_frame = 0;
    int last_frame = -1;
    int workers = -1;
    int jobs = -1;
    std::string cluster;
    bool no_parallel = false;
    bool auto_temp = false;
    std::optional<float> start_temp;
    std::optional<float> end_temp;
    std::optional<fs::path> residual;
    int continue_from = -1;
    std::optional<int> seed;
    int batches = 1;

    void postInit() {
        // Validate arguments
        if (auto_temp && (start_temp.has_value() || end_temp.has_value())) {
            throw std::runtime_error("When auto_temp is on, starting temperature or ending temperature should not be set manually");
        } else if (!auto_temp && (!start_temp.has_value() || !end_temp.has_value())) {
            throw std::runtime_error("When auto_temp is off, starting temperature and ending temperature should be set manually");
        }
        if (first_frame > last_frame && last_frame >= 0) {
            throw std::invalid_argument("Invalid interval: frame_first must be less than frame_last");
        } else if (first_frame < 0) {
            throw std::invalid_argument("Invalid interval: frame_first must be greater or equal to 0");
        }

        // Set seed
        unsigned int seedValue = static_cast<unsigned int>(std::time(nullptr)) % (1U << 32);
        std::default_random_engine generator(seed.has_value() ? *seed : seedValue);
        std::cout << "Seed: " << generator << std::endl;

        if (workers == -1) {
            workers = std::thread::hardware_concurrency();
        }

        if (jobs == -1) {
            if (!cluster.empty()) {
                throw std::invalid_argument("-j/--jobs is required for non-local clusters");
            } else {
                jobs = workers;
            }
        }
    }
};

int main(int argc, char* argv[]) {
    try {
        Args args;

        cxxopts::Options options("Program", "Description");
        options.add_options()
            ("i,input", "Input file pattern (e.g. 'input%03d.png')", cxxopts::value<std::string>(args.input))
            ("o,output", "Path to the output directory", cxxopts::value<std::string>())
            ("c,config", "Path to the config file", cxxopts::value<std::string>())
            ("I,initial", "Path to the initial cell configuration file", cxxopts::value<std::string>())
            ("d,debug", "Path to the debug directory", cxxopts::value<std::string>())
            ("ff,first_frame", "First frame to analyze", cxxopts::value<int>())
            ("lf,last_frame", "Last frame to analyze (defaults to the last frame)", cxxopts::value<int>())
            ("w,workers", "Number of workers to use (defaults to the number of cores)", cxxopts::value<int>())
            ("j,jobs", "Number of jobs to run in parallel (defaults to the number of workers)", cxxopts::value<int>())
            ("C,cluster", "Address of the cluster to connect to", cxxopts::value<std::string>())
            ("no_parallel", "Disable parallelization", cxxopts::value<bool>())
            ("auto_temp", "Automatically determine the starting and ending temperatures", cxxopts::value<bool>())
            ("st,start_temp", "Starting temperature", cxxopts::value<float>())
            ("et,end_temp", "Ending temperature", cxxopts::value<float>())
            ("r,residual", "Path to save the residual directory", cxxopts::value<std::string>())
            ("cf,continue_from", "Frame to start from (defaults to first)", cxxopts::value<int>())
            ("s,seed", "Random seed", cxxopts::value<int>())
            ("b,batches", "Number of batches to run", cxxopts::value<int>());

        cxxopts::ParseResult result = options.parse(argc, argv);

        // Set parsed values to Args struct
        args.input = result["input"].as<std::string>();
        args.output = fs::path(result["output"].as<std::string>());
        args.config = fs::path(result["config"].as<std::string>());
        args.initial = fs::path(result["initial"].as<std::string>());
        if (result.count("debug")) args.debug = fs::path(result["debug"].as<std::string>());
        if (result.count("first_frame")) args.first_frame = result["first_frame"].as<int>();
        if (result.count("last_frame")) args.last_frame = result["last_frame"].as<int>();
        if (result.count("workers")) args.workers = result["workers"].as<int>();
        if (result.count("jobs")) args.jobs = result["jobs"].as<int>();
        if (result.count("cluster")) args.cluster = result["cluster"].as<std::string>();
        if (result.count("no_parallel")) args.no_parallel = result["no_parallel"].as<bool>();
        if (result.count("auto_temp")) args.auto_temp = result["auto_temp"].as<bool>();
        if (result.count("start_temp")) args.start_temp = result["start_temp"].as<float>();
        if (result.count("end_temp")) args.end_temp = result["end_temp"].as<float>();
        if (result.count("residual")) args.residual = fs::path(result["residual"].as<std::string>());
        if (result.count("continue_from")) args.continue_from = result["continue_from"].as<int>();
        if (result.count("seed")) args.seed = result["seed"].as<int>();
        if (result.count("batches")) args.batches = result["batches"].as<int>();

        args.postInit(); // Perform post initialization checks

        // Rest of your program using the parsed arguments
    } catch (const cxxopts::OptionException& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
