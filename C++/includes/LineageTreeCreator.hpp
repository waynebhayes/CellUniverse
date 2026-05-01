#pragma once

#include <memory>
#include <string>
#include <vector>

class Spheroid;

class LineageTreeCreator
{
public:
    struct CellViz
    {
        std::string rawName;
        float x = 0.0f;
        float y = 0.0f;
    };

    struct RenderOptions
    {
        int width;
        int height;
        double fps;
        int firstFrame;
        int lastFrame;

        RenderOptions()
            : width(1400), height(1400), fps(4.0), firstFrame(-1), lastFrame(-1)
        {
        }
    };

    LineageTreeCreator();
    ~LineageTreeCreator();

    LineageTreeCreator(LineageTreeCreator &&) noexcept;
    LineageTreeCreator &operator=(LineageTreeCreator &&) noexcept;

    LineageTreeCreator(const LineageTreeCreator &) = delete;
    LineageTreeCreator &operator=(const LineageTreeCreator &) = delete;

    void update(int frameIndex, const std::vector<CellViz> &cells);
    void close();

    static std::vector<CellViz> makeCellViz(const std::vector<Spheroid> &cells);

    static bool renderCsvFiles(const std::vector<std::string> &csvPaths,
                               const std::string &outputPath,
                               const RenderOptions &options = RenderOptions());

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
