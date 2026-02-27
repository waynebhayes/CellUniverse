#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

class LineageViewer
{
public:
    struct CellViz
    {
        std::string rawName; // original long name (from SpheroidParams.name)
        float x = 0.0f;
        float y = 0.0f;
    };

    LineageViewer();

    // Call once per frame. Provide current frame's cell list with x,y.
    void update(int frameIndex, const std::vector<CellViz> &cells);

    void close();

private:
    struct Node
    {
        std::string rawName;    // original name
        std::string parentRaw;  // inferred parent raw name (suffix rule)
        std::string shortLabel; // "0", "0.1", ...
        cv::Scalar colorBGR;    // node color (BGR)
        bool isRoot = true;
    };

    // persistent mapping across frames
    std::unordered_map<std::string, Node> nodesByRaw; // rawName -> Node
    std::unordered_map<std::string, int> rootIndexByRaw; // rootRaw -> rootIndex
    int nextRootIndex = 0;

    bool windowReady = false;
    std::string windowName = "Cell Lineage (Realtime)";

private:
    static bool isDaughterName(const std::string &name);
    static std::string parentNameOf(const std::string &name);

    static cv::Scalar colorFromPalette(int idx);
    static cv::Scalar lighten(const cv::Scalar &bgr, double amount01);

    void ensureNodeExists(const std::string &rawName);

    void render2D(int frameIndex, const std::vector<CellViz> &cells);
};