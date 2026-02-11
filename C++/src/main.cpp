#include <iostream>
#include "types.hpp"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "ConfigTypes.hpp"
#include "CellFactory.hpp"
#include "yaml-cpp/yaml.h"
#include "Spheroid.hpp"
#include "Lineage.hpp"
#include <chrono>

#if USE_MERSENNE
// Global seed for Mersenne Twister
std::uint32_t global_mt_seed;
#endif

class Args
{
public:
    std::string config{};
    std::string input{};
    int firstFrame = 0;
    int lastFrame = 0;
    std::string initial{};
    std::string output{};
    int continueFrom = -1;
};

PathVec getImageFilePaths(const std::string &input, int firstFrame, int lastFrame)
{
    PathVec paths;

    //printf-style pattern like .../t%03d.tif
    if (input.find('%') != std::string::npos)
    {
<<<<<<< Updated upstream
        char buffer[100];
        sprintf(buffer, inputPattern.c_str(), i);
        fs::path file(buffer);

        if (fs::exists(file) && fs::is_regular_file(file))
=======
        for (int frame = firstFrame; frame <= lastFrame; ++frame)
>>>>>>> Stashed changes
        {
            char buffer[2048];
            std::snprintf(buffer, sizeof(buffer), input.c_str(), frame);
            std::string imagePath = buffer;
            if (std::filesystem::exists(imagePath))
            {
                paths.push_back(imagePath);
            }
            else
            {
                std::cerr << "Missing frame file: " << imagePath << std::endl;
            }
        }
        return paths;
    }

    // a directory containing TIFF files (any naming: t000.tif, frame001.tif, etc.)
    if (std::filesystem::is_directory(input))
    {
        struct Item { std::string path; int idx; std::string name; };
        std::vector<Item> items;

        for (const auto &e : std::filesystem::directory_iterator(input))
        {
            if (!e.is_regular_file()) continue;
            auto ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext != ".tif" && ext != ".tiff") continue;

            std::string name = e.path().filename().string();
            std::string stem = e.path().stem().string();

            // Extract the last run of digits from the stem as the sort key.
            int idx = -1;
            for (int i = (int)stem.size() - 1; i >= 0; --i)
            {
                if (std::isdigit((unsigned char)stem[i]))
                {
                    int j = i;
                    while (j >= 0 && std::isdigit((unsigned char)stem[j])) --j;
                    idx = std::stoi(stem.substr(j + 1, i - j));
                    break;
                }
            }

            items.push_back({e.path().string(), idx, name});
        }

        std::sort(items.begin(), items.end(), [](const Item &a, const Item &b) {
            if (a.idx != -1 && b.idx != -1 && a.idx != b.idx) return a.idx < b.idx;
            return a.name < b.name;
        });

        if (items.empty())
        {
            throw std::runtime_error("No .tif/.tiff files found in directory: " + input);
        }

        if (firstFrame < 0) firstFrame = 0;
        if (lastFrame < 0 || lastFrame >= (int)items.size()) lastFrame = (int)items.size() - 1;
        if (firstFrame > lastFrame)
        {
            throw std::runtime_error("Invalid frame range for directory input: firstFrame > lastFrame");
        }

        for (int i = firstFrame; i <= lastFrame; ++i)
        {
            paths.push_back(items[i].path);
        }

        return paths;
    }

    //a single TIFF file path
    if (std::filesystem::exists(input))
    {
        paths.push_back(input);
        return paths;
    }

    throw std::runtime_error("Input is neither a printf-pattern, nor a directory, nor a file: " + input);
}

// helper function to load the config
void loadConfig(const std::string &path, BaseConfig &config)
{
    YAML::Node node = YAML::LoadFile(path);
    config.explodeConfig(node);
}

int main(int argc, char *argv[])
{
    // Expected fixed order:
    // celluniverse <firstFrame> <lastFrame> <initial.csv> <input_pattern_or_dir> <output_dir> <config.yaml>
    if (argc != 7)
    {
        std::cerr
            << "Usage:\n"
            << "  celluniverse <firstFrame> <lastFrame> <initial.csv> <input_pattern_or_dir> <output_dir> <config.yaml>\n\n"
            << "Examples:\n"
            << "  celluniverse 0 10 initial.csv images/t%03d.tif out config.yaml\n"
            << "  celluniverse 0 10 initial.csv /path/to/tiffs out config.yaml\n";
        return 1;
    }

    Args args;
    try
    {
        args.firstFrame = std::stoi(argv[1]);
        args.lastFrame = std::stoi(argv[2]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing firstFrame/lastFrame: " << e.what() << std::endl;
        return 1;
    }

    args.initial = argv[3];
    args.input = argv[4];
    args.output = argv[5];
    args.config = argv[6];
    args.continueFrom = -1;

    std::cout << "Loading args:\n";
    std::cout << "First frame: " << args.firstFrame << std::endl;
    std::cout << "Last frame: " << args.lastFrame << std::endl;
    std::cout << "Initial CSV path: " << args.initial << std::endl;
    std::cout << "Input (pattern/dir/file): " << args.input << std::endl;
    std::cout << "Output folder: " << args.output << std::endl;
    std::cout << "Config file: " << args.config << std::endl;

    // Load config (YAML)
    BaseConfig config;
    try
    {
        loadConfig(args.config, config);
    }
    catch (const YAML::Exception &e)
    {
        std::cerr << "YAML error while loading config: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error while loading config: " << e.what() << std::endl;
        return 1;
    }

    config.printConfig();

#if USE_MERSENNE
    // Seed the RNG once for reproducibility within a run
    global_mt_seed = static_cast<std::uint32_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::cout << "Mersenne Twister seed: " << global_mt_seed << std::endl;
#else
    std::cout << "Using Linear Congruential generator (seeding every call)" << std::endl;
#endif

    // Resolve which TIFFs to process (pattern OR directory OR file)
    PathVec imageFilePaths;
    try
    {
        imageFilePaths = getImageFilePaths(args.input, args.firstFrame, args.lastFrame);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to resolve input frames: " << e.what() << std::endl;
        return 1;
    }

    if (imageFilePaths.empty())
    {
        std::cerr << "No input frames found. Check your input path/pattern and frame range." << std::endl;
        return 1;
    }

    // Load initial cells from CSV (this is the model initialization)
    CellFactory cellFactory(config);
    std::map<Path, std::vector<Spheroid>> cells =
        cellFactory.createCells(args.initial,
                                config.simulation.z_slices / 2,
                                config.simulation.z_scaling);

    // Build the lineage (frames + optimization state)
    Lineage lineage(cells, imageFilePaths, config, args.output, args.continueFrom);

    // Run optimization per frame and save outputs (real/synth/etc.)
    auto start = std::chrono::steady_clock::now();

    for (int frame = 0; frame < (int)lineage.length(); ++frame)
    {
        lineage.optimize(frame);
        lineage.copyCellsForward(frame + 1);
        lineage.saveFrame(frame);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time elapsed: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}