#include "../includes/LineageViewer.hpp"
#include <algorithm>

/**
 * LineageViewer
 * -------------
 * Real-time visualization module for cell tracking results.
 * Displays current frame cells, split events, and lost cell events.
 * Fully decoupled from core tracking logic (read-only consumer of data).
 * Intended for debugging, validation, and interpretability of lineage behavior.
 */

LineageViewer::LineageViewer()
{
}

bool LineageViewer::isDaughterName(const std::string &name)
{
    if (name.empty()) return false;
    char c = name.back();
    return (c == '0' || c == '1');
}

std::string LineageViewer::parentNameOf(const std::string &name)
{
    if (!isDaughterName(name)) return "";
    return name.substr(0, name.size() - 1);
}

cv::Scalar LineageViewer::colorFromPalette(int idx)
{
    // BGR palette (high contrast on black background)
    static const std::vector<cv::Scalar> palette = {
        cv::Scalar(0, 255, 0),     // green
        cv::Scalar(255, 0, 255),   // purple
        cv::Scalar(0, 255, 255),   // yellow
        cv::Scalar(255, 255, 0),   // cyan
        cv::Scalar(0, 128, 255),   // orange
        cv::Scalar(255, 0, 0),     // blue
        cv::Scalar(0, 0, 255),     // red
        cv::Scalar(180, 180, 180), // gray
        cv::Scalar(255, 128, 0),   // light blue-ish
        cv::Scalar(128, 0, 255)    // magenta-ish
    };
    return palette[idx % (int)palette.size()];
}

cv::Scalar LineageViewer::lighten(const cv::Scalar &bgr, double amount01)
{
    // amount01: 0 -> same, 1 -> white
    double a = std::clamp(amount01, 0.0, 1.0);
    double b = bgr[0] + (255.0 - bgr[0]) * a;
    double g = bgr[1] + (255.0 - bgr[1]) * a;
    double r = bgr[2] + (255.0 - bgr[2]) * a;
    return cv::Scalar(b, g, r);
}

void LineageViewer::ensureNodeExists(const std::string &rawName)
{
    if (nodesByRaw.find(rawName) != nodesByRaw.end()) return;

    Node n;
    n.rawName = rawName;
    n.parentRaw = parentNameOf(rawName);
    n.isRoot = n.parentRaw.empty();

    // root: assign new integer label + palette color
    if (n.isRoot)
    {
        int ridx = nextRootIndex++;
        rootIndexByRaw[rawName] = ridx;
        n.shortLabel = std::to_string(ridx);
        n.colorBGR = colorFromPalette(ridx);
        nodesByRaw[rawName] = n;
        return;
    }

    // daughter: ensure parent exists first
    ensureNodeExists(n.parentRaw);

    // inherit root index from parent
    int rootIdx = 0;
    auto itRoot = rootIndexByRaw.find(n.parentRaw);
    if (itRoot != rootIndexByRaw.end())
    {
        rootIdx = itRoot->second;
    }
    else
    {
        // fallback: treat parent as root
        int ridx = nextRootIndex++;
        rootIndexByRaw[n.parentRaw] = ridx;
        rootIdx = ridx;
    }

    // daughter label: parentShort + ".1/.2" based on last char 0/1
    const Node &parentNode = nodesByRaw[n.parentRaw];
    char c = rawName.back();
    std::string suffix = (c == '0') ? ".1" : ".2";
    n.shortLabel = parentNode.shortLabel + suffix;

    // daughter color: lighter version of root color
    cv::Scalar base = colorFromPalette(rootIdx);
    n.colorBGR = lighten(base, 0.55);

    nodesByRaw[rawName] = n;
}

void LineageViewer::render2D(int frameIndex, const std::vector<CellViz> &cells)
{
    if (!windowReady)
    {
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        windowReady = true;
    }

    // Canvas settings
    const int W = 1100;
    const int H = 650;
    const int margin = 60;

    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(0, 0, 0)); // black background

    //Title
    std::string title = "Realtime Lineage (2D)  |  frame=" + std::to_string(frameIndex) +
                        "  |  nodes(current)=" + std::to_string(cells.size()) +
                        "  |  nodes(total)=" + std::to_string(nodesByRaw.size());
    cv::putText(canvas, title, cv::Point(20, 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    if (cells.empty())
    {
        cv::imshow(windowName, canvas);
        cv::waitKey(1);
        return;
    }

    // Determine bounding box in data space (x,y)
    float minX = cells[0].x, maxX = cells[0].x;
    float minY = cells[0].y, maxY = cells[0].y;
    for (const auto &c : cells)
    {
        minX = std::min(minX, c.x);
        maxX = std::max(maxX, c.x);
        minY = std::min(minY, c.y);
        maxY = std::max(maxY, c.y);
    }

    // Avoid zero range
    float dx = std::max(1.0f, maxX - minX);
    float dy = std::max(1.0f, maxY - minY);

    // Scale to fit canvas (keep aspect)
    double sx = (double)(W - 2 * margin) / dx;
    double sy = (double)(H - 2 * margin) / dy;
    double s = std::min(sx, sy);

    // map data (x,y) to pixel
    auto toPixel = [&](float x, float y) -> cv::Point
    {
        // Map x right, y down (standard screen). If you want y-up, invert here.
        int px = margin + (int)std::lround((x - minX) * s);
        int py = margin + (int)std::lround((y - minY) * s);
        // Keep inside
        px = std::clamp(px, 0, W - 1);
        py = std::clamp(py, 0, H - 1);
        return cv::Point(px, py);
    };
    // Draw axes box
    cv::rectangle(canvas, cv::Rect(margin, margin, W - 2 * margin, H - 2 * margin),
                  cv::Scalar(60, 60, 60), 1, cv::LINE_AA);

    // Draw each cell
    for (const auto &c : cells)
    {
        auto it = nodesByRaw.find(c.rawName);
        if (it == nodesByRaw.end()) continue;
        const Node &n = it->second;
        cv::Point p = toPixel(c.x, c.y);
        // node
        cv::circle(canvas, p, 10, n.colorBGR, -1, cv::LINE_AA);
        cv::circle(canvas, p, 10, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        // label (short)
        cv::putText(canvas, n.shortLabel, p + cv::Point(14, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    cv::imshow(windowName, canvas);
    cv::waitKey(1);
}

void LineageViewer::update(int frameIndex, const std::vector<CellViz> &cells)
{
    // Persist mapping for newly seen raw names
    for (const auto &c : cells)
    {
        ensureNodeExists(c.rawName);
    }

    render2D(frameIndex, cells);
}

void LineageViewer::close()
{
    if (windowReady)
    {
        cv::destroyWindow(windowName);
        windowReady = false;
    }
}