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
    static const std::vector<cv::Scalar> palette = {
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(0, 128, 255),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(180, 180, 180),
        cv::Scalar(255, 128, 0),
        cv::Scalar(128, 0, 255)
    };
    return palette[idx % static_cast<int>(palette.size())];
}

cv::Scalar LineageViewer::lighten(const cv::Scalar &bgr, double amount01)
{
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

    if (n.isRoot)
    {
        int ridx = nextRootIndex++;
        rootIndexByRaw[rawName] = ridx;
        n.shortLabel = std::to_string(ridx);
        n.colorBGR = colorFromPalette(ridx);
        nodesByRaw[rawName] = n;
        return;
    }

    ensureNodeExists(n.parentRaw);

    int rootIdx = 0;
    auto itRoot = rootIndexByRaw.find(n.parentRaw);
    if (itRoot != rootIndexByRaw.end())
    {
        rootIdx = itRoot->second;
    }
    else
    {
        int ridx = nextRootIndex++;
        rootIndexByRaw[n.parentRaw] = ridx;
        rootIdx = ridx;
    }

    const Node &parentNode = nodesByRaw[n.parentRaw];
    char c = rawName.back();
    std::string suffix = (c == '0') ? ".1" : ".2";
    n.shortLabel = parentNode.shortLabel + suffix;

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

    const int W = 1100;
    const int H = 650;
    const int margin = 60;

    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(0, 0, 0));

    std::string title = "Realtime Lineage (2D) | frame=" + std::to_string(frameIndex) +
                        " | nodes(current)=" + std::to_string(cells.size()) +
                        " | nodes(total)=" + std::to_string(nodesByRaw.size());
    cv::putText(canvas, title, cv::Point(20, 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    if (cells.empty())
    {
        cv::imshow(windowName, canvas);
        cv::waitKey(1);
        return;
    }

    float minX = cells[0].x;
    float maxX = cells[0].x;
    float minY = cells[0].y;
    float maxY = cells[0].y;
    for (const auto &c : cells)
    {
        minX = std::min(minX, c.x);
        maxX = std::max(maxX, c.x);
        minY = std::min(minY, c.y);
        maxY = std::max(maxY, c.y);
    }

    float dx = std::max(1.0f, maxX - minX);
    float dy = std::max(1.0f, maxY - minY);

    double sx = static_cast<double>(W - 2 * margin) / dx;
    double sy = static_cast<double>(H - 2 * margin) / dy;
    double s = std::min(sx, sy);

    auto toPixel = [&](float x, float y) -> cv::Point
    {
        int px = margin + static_cast<int>(std::lround((x - minX) * s));
        int py = margin + static_cast<int>(std::lround((y - minY) * s));
        px = std::clamp(px, 0, W - 1);
        py = std::clamp(py, 0, H - 1);
        return cv::Point(px, py);
    };

    cv::rectangle(canvas, cv::Rect(margin, margin, W - 2 * margin, H - 2 * margin),
                  cv::Scalar(60, 60, 60), 1, cv::LINE_AA);

    for (const auto &c : cells)
    {
        auto it = nodesByRaw.find(c.rawName);
        if (it == nodesByRaw.end()) continue;
        const Node &n = it->second;
        cv::Point p = toPixel(c.x, c.y);
        cv::circle(canvas, p, 10, n.colorBGR, -1, cv::LINE_AA);
        cv::circle(canvas, p, 10, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::putText(canvas, n.shortLabel, p + cv::Point(14, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    cv::imshow(windowName, canvas);
    cv::waitKey(1);
}

void LineageViewer::update(int frameIndex, const std::vector<CellViz> &cells)
{
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
