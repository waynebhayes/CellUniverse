#include "../includes/LineageTreeCreator.hpp"

#include "../includes/Ellipsoid.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace fs = std::filesystem;

namespace
{
constexpr double kPi = 3.14159265358979323846;

const std::vector<cv::Scalar> kRootColorsBGR = {
    cv::Scalar(255, 92, 214),  // bright purple
    cv::Scalar(20, 255, 57),   // bright green
    cv::Scalar(59, 212, 255),  // gold
    cv::Scalar(255, 245, 0),   // cyan
};

struct NameParts
{
    bool hasCode = false;
    std::string prefix;
    std::string code;
};

struct Node
{
    std::string rawName;
    std::string parentRaw;
    std::string rootRaw;
    std::string code;
    int depth = 0;
    double angle = 0.0;
    double radius = 0.0;
    std::set<std::string> children;
};

NameParts parseNameParts(const std::string &name)
{
    const size_t pos = name.find_last_of('_');
    if (pos != std::string::npos && pos + 1 < name.size())
    {
        const std::string code = name.substr(pos + 1);
        const bool hasWhitespace = std::any_of(code.begin(), code.end(), [](unsigned char c) {
            return std::isspace(c) != 0;
        });
        if (!code.empty() && !hasWhitespace)
        {
            return {true, name.substr(0, pos + 1), code};
        }
    }
    return {};
}

std::string lineageCodeOf(const std::string &name)
{
    const NameParts parts = parseNameParts(name);
    return parts.hasCode ? parts.code : name;
}

std::string parentNameOf(const std::string &name)
{
    const NameParts parts = parseNameParts(name);
    if (parts.hasCode)
    {
        if (parts.code.size() <= 1)
        {
            return "";
        }
        const char last = parts.code.back();
        if (last != '0' && last != '1')
        {
            return "";
        }
        return parts.prefix + parts.code.substr(0, parts.code.size() - 1);
    }

    if (name.size() > 1 && (name.back() == '0' || name.back() == '1'))
    {
        return name.substr(0, name.size() - 1);
    }
    return "";
}

std::string rootNameOf(const std::string &name)
{
    const NameParts parts = parseNameParts(name);
    if (parts.hasCode && !parts.code.empty())
    {
        return parts.prefix + parts.code.substr(0, 1);
    }

    std::string current = name;
    while (true)
    {
        const std::string parent = parentNameOf(current);
        if (parent.empty())
        {
            return current;
        }
        current = parent;
    }
}

bool codeLess(const std::string &a, const std::string &b)
{
    const std::string ac = lineageCodeOf(a);
    const std::string bc = lineageCodeOf(b);
    const bool aDigits = !ac.empty() && std::all_of(ac.begin(), ac.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
    });
    const bool bDigits = !bc.empty() && std::all_of(bc.begin(), bc.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
    });
    if (aDigits && bDigits && ac.size() != bc.size())
    {
        return ac.size() < bc.size();
    }
    if (ac != bc)
    {
        return ac < bc;
    }
    return a < b;
}

std::vector<std::string> sortedChildren(const std::map<std::string, Node> &nodes,
                                        const std::string &name)
{
    std::vector<std::string> children(nodes.at(name).children.begin(), nodes.at(name).children.end());
    std::sort(children.begin(), children.end(), codeLess);
    return children;
}

std::vector<std::string> sortedRoots(const std::map<std::string, Node> &nodes)
{
    std::vector<std::string> roots;
    for (const auto &[name, node] : nodes)
    {
        if (node.parentRaw.empty())
        {
            roots.push_back(name);
        }
    }
    std::sort(roots.begin(), roots.end(), codeLess);
    return roots;
}

Node &ensureNode(std::map<std::string, Node> &nodes, const std::string &rawName)
{
    auto existing = nodes.find(rawName);
    if (existing != nodes.end())
    {
        return existing->second;
    }

    Node node;
    node.rawName = rawName;
    node.parentRaw = parentNameOf(rawName);
    node.rootRaw = rootNameOf(rawName);
    node.code = lineageCodeOf(rawName);
    node.depth = std::max(0, static_cast<int>(node.code.size()) - 1);

    auto inserted = nodes.emplace(rawName, std::move(node));
    Node &stored = inserted.first->second;

    if (!stored.parentRaw.empty())
    {
        Node &parent = ensureNode(nodes, stored.parentRaw);
        parent.children.insert(rawName);
        stored.rootRaw = parent.rootRaw;
        stored.depth = parent.depth + 1;
    }

    return stored;
}

std::vector<std::string> collectLeaves(const std::map<std::string, Node> &nodes,
                                       const std::string &name)
{
    const std::vector<std::string> children = sortedChildren(nodes, name);
    if (children.empty())
    {
        return {name};
    }

    std::vector<std::string> leaves;
    for (const std::string &child : children)
    {
        std::vector<std::string> childLeaves = collectLeaves(nodes, child);
        leaves.insert(leaves.end(), childLeaves.begin(), childLeaves.end());
    }
    return leaves;
}

double normalizeAngleDiff(double diff)
{
    while (diff > kPi) diff -= 2.0 * kPi;
    while (diff < -kPi) diff += 2.0 * kPi;
    return diff;
}

double assignInternalAngle(std::map<std::string, Node> &nodes, const std::string &name)
{
    const std::vector<std::string> children = sortedChildren(nodes, name);
    if (children.empty())
    {
        return nodes[name].angle;
    }

    double sx = 0.0;
    double sy = 0.0;
    for (const std::string &child : children)
    {
        const double a = assignInternalAngle(nodes, child);
        sx += std::cos(a);
        sy += std::sin(a);
    }
    nodes[name].angle = std::atan2(sy, sx);
    return nodes[name].angle;
}

void assignLayout(std::map<std::string, Node> &nodes, int width, int height)
{
    if (nodes.empty())
    {
        return;
    }

    const std::vector<std::string> roots = sortedRoots(nodes);
    if (roots.empty())
    {
        return;
    }

    int maxDepth = 0;
    for (const auto &[_, node] : nodes)
    {
        maxDepth = std::max(maxDepth, node.depth);
    }

    const double side = static_cast<double>(std::min(width, height));
    const double innerRadius = side * 0.085;
    const double outerRadius = side * 0.455;
    const double depthStep = (outerRadius - innerRadius) / static_cast<double>(std::max(1, maxDepth));
    const double gap = 7.0 * kPi / 180.0;
    const double start = -0.5 * kPi;
    const double sector = (2.0 * kPi) / static_cast<double>(roots.size());

    for (size_t rootIndex = 0; rootIndex < roots.size(); ++rootIndex)
    {
        const std::string &root = roots[rootIndex];
        const std::vector<std::string> leaves = collectLeaves(nodes, root);
        const double a0 = start + static_cast<double>(rootIndex) * sector + gap;
        const double a1 = start + static_cast<double>(rootIndex + 1) * sector - gap;

        if (leaves.size() == 1)
        {
            nodes[leaves.front()].angle = 0.5 * (a0 + a1);
        }
        else
        {
            for (size_t i = 0; i < leaves.size(); ++i)
            {
                const double t = static_cast<double>(i) / static_cast<double>(leaves.size() - 1);
                nodes[leaves[i]].angle = a0 * (1.0 - t) + a1 * t;
            }
        }
        assignInternalAngle(nodes, root);
    }

    for (auto &[_, node] : nodes)
    {
        node.radius = innerRadius + static_cast<double>(node.depth) * depthStep;
    }
}

cv::Point pointAt(const cv::Point &center, double radius, double angle)
{
    return cv::Point(
        static_cast<int>(std::lround(static_cast<double>(center.x) + radius * std::cos(angle))),
        static_cast<int>(std::lround(static_cast<double>(center.y) + radius * std::sin(angle))));
}

std::vector<cv::Point> arcPoints(const cv::Point &center, double radius, double startAngle, double endAngle)
{
    const double diff = normalizeAngleDiff(endAngle - startAngle);
    const int steps = std::max(3, static_cast<int>(std::ceil(std::abs(diff) * radius / 9.0)));

    std::vector<cv::Point> points;
    points.reserve(static_cast<size_t>(steps + 1));
    for (int i = 0; i <= steps; ++i)
    {
        const double t = static_cast<double>(i) / static_cast<double>(steps);
        points.push_back(pointAt(center, radius, startAngle + diff * t));
    }
    return points;
}

cv::Scalar rootColor(const std::map<std::string, Node> &nodes,
                     const std::unordered_map<std::string, int> &rootOrder,
                     const std::string &name)
{
    const std::string &root = nodes.at(name).rootRaw;
    auto it = rootOrder.find(root);
    const int idx = (it == rootOrder.end()) ? 0 : it->second;
    return kRootColorsBGR[static_cast<size_t>(idx) % kRootColorsBGR.size()];
}

void drawEdge(cv::Mat &canvas,
              const std::map<std::string, Node> &nodes,
              const std::unordered_map<std::string, int> &rootOrder,
              const std::string &parentName,
              const std::string &childName,
              const cv::Point &center,
              int thickness)
{
    const Node &parent = nodes.at(parentName);
    const Node &child = nodes.at(childName);
    const cv::Scalar color = rootColor(nodes, rootOrder, childName);

    const cv::Point p0 = pointAt(center, parent.radius, parent.angle);
    const cv::Point elbow = pointAt(center, child.radius, parent.angle);
    const cv::Point p1 = pointAt(center, child.radius, child.angle);
    cv::line(canvas, p0, elbow, color, thickness, cv::LINE_AA);

    const std::vector<cv::Point> arc = arcPoints(center, child.radius, parent.angle, child.angle);
    const std::vector<std::vector<cv::Point>> arcs = {arc};
    cv::polylines(canvas, arcs, false, color, thickness, cv::LINE_AA);
    cv::circle(canvas, p1, std::max(1, thickness / 2), color, -1, cv::LINE_AA);
}

cv::Mat renderTreeImage(std::map<std::string, Node> nodes, int frameIndex, int width, int height)
{
    assignLayout(nodes, width, height);

    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Point center(width / 2, height / 2);
    const std::vector<std::string> roots = sortedRoots(nodes);
    std::unordered_map<std::string, int> rootOrder;
    for (size_t i = 0; i < roots.size(); ++i)
    {
        rootOrder[roots[i]] = static_cast<int>(i);
    }

    const int thickness = std::max(2, std::min(width, height) / 700);
    std::vector<std::string> orderedNames;
    orderedNames.reserve(nodes.size());
    for (const auto &[name, _] : nodes)
    {
        orderedNames.push_back(name);
    }
    std::sort(orderedNames.begin(), orderedNames.end(), [&](const std::string &a, const std::string &b) {
        if (nodes.at(a).depth != nodes.at(b).depth)
        {
            return nodes.at(a).depth < nodes.at(b).depth;
        }
        return codeLess(a, b);
    });

    for (const std::string &parent : orderedNames)
    {
        for (const std::string &child : sortedChildren(nodes, parent))
        {
            drawEdge(canvas, nodes, rootOrder, parent, child, center, thickness);
        }
    }

    for (const std::string &root : roots)
    {
        cv::circle(canvas, pointAt(center, nodes.at(root).radius, nodes.at(root).angle),
                   5, rootColor(nodes, rootOrder, root), -1, cv::LINE_AA);
    }

    std::ostringstream title;
    title << "Lineage Tree | frame=" << frameIndex << " | nodes=" << nodes.size();
    cv::putText(canvas, title.str(), cv::Point(36, 54),
                cv::FONT_HERSHEY_SIMPLEX, 0.95, cv::Scalar(232, 232, 232), 2, cv::LINE_AA);

    const int legendX = 38;
    const int legendY = std::max(80, height - 118);
    for (size_t i = 0; i < roots.size() && i < 4; ++i)
    {
        const int y = legendY + static_cast<int>(i) * 28;
        const cv::Scalar color = kRootColorsBGR[i % kRootColorsBGR.size()];
        cv::line(canvas, cv::Point(legendX, y), cv::Point(legendX + 42, y), color, 4, cv::LINE_AA);
        cv::putText(canvas, lineageCodeOf(roots[i]), cv::Point(legendX + 56, y + 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
    }

    return canvas;
}

std::vector<std::string> splitCsvLine(const std::string &line)
{
    std::vector<std::string> fields;
    std::string current;
    bool inQuotes = false;

    for (size_t i = 0; i < line.size(); ++i)
    {
        const char c = line[i];
        if (c == '"')
        {
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '"')
            {
                current.push_back('"');
                ++i;
            }
            else
            {
                inQuotes = !inQuotes;
            }
        }
        else if (c == ',' && !inQuotes)
        {
            fields.push_back(current);
            current.clear();
        }
        else
        {
            current.push_back(c);
        }
    }
    fields.push_back(current);
    return fields;
}

int frameNumberFromFile(const std::string &fileName)
{
    const std::string stem = fs::path(fileName).stem().string();
    std::string digits;
    for (char c : stem)
    {
        if (std::isdigit(static_cast<unsigned char>(c)) != 0)
        {
            digits.push_back(c);
        }
    }
    return digits.empty() ? -1 : std::stoi(digits);
}

std::map<int, std::set<std::string>> readFrameNames(const std::vector<std::string> &csvPaths)
{
    std::map<int, std::set<std::string>> frameNames;
    for (const std::string &csvPath : csvPaths)
    {
        std::ifstream in(csvPath);
        if (!in.is_open())
        {
            throw std::runtime_error("Could not open lineage CSV: " + csvPath);
        }

        std::string line;
        if (!std::getline(in, line))
        {
            continue;
        }
        const std::vector<std::string> header = splitCsvLine(line);
        int fileCol = -1;
        int nameCol = -1;
        for (size_t i = 0; i < header.size(); ++i)
        {
            if (header[i] == "file") fileCol = static_cast<int>(i);
            if (header[i] == "name") nameCol = static_cast<int>(i);
        }
        if (fileCol < 0 || nameCol < 0)
        {
            throw std::runtime_error("Lineage CSV must contain file,name columns: " + csvPath);
        }

        while (std::getline(in, line))
        {
            if (line.empty())
            {
                continue;
            }
            const std::vector<std::string> fields = splitCsvLine(line);
            if (static_cast<int>(fields.size()) <= std::max(fileCol, nameCol))
            {
                continue;
            }
            const int frame = frameNumberFromFile(fields[static_cast<size_t>(fileCol)]);
            const std::string &name = fields[static_cast<size_t>(nameCol)];
            if (frame >= 0 && !name.empty())
            {
                frameNames[frame].insert(name);
            }
        }
    }
    return frameNames;
}

std::string lowerExtension(const std::string &path)
{
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}
} // namespace

struct LineageTreeCreator::Impl
{
    std::map<std::string, Node> nodesByRaw;
    bool windowReady = false;
    std::string windowName = "Cell Lineage Tree";
    int width = 1100;
    int height = 1100;
};

LineageTreeCreator::LineageTreeCreator()
    : impl(std::make_unique<Impl>())
{
}

LineageTreeCreator::~LineageTreeCreator()
{
    close();
}

LineageTreeCreator::LineageTreeCreator(LineageTreeCreator &&) noexcept = default;

LineageTreeCreator &LineageTreeCreator::operator=(LineageTreeCreator &&) noexcept = default;

void LineageTreeCreator::update(int frameIndex, const std::vector<CellViz> &cells)
{
    for (const auto &cell : cells)
    {
        if (!cell.rawName.empty())
        {
            ensureNode(impl->nodesByRaw, cell.rawName);
        }
    }

    if (!impl->windowReady)
    {
        cv::namedWindow(impl->windowName, cv::WINDOW_AUTOSIZE);
        impl->windowReady = true;
    }

    cv::Mat canvas = renderTreeImage(impl->nodesByRaw, frameIndex, impl->width, impl->height);
    cv::imshow(impl->windowName, canvas);
    cv::waitKey(1);
}

void LineageTreeCreator::close()
{
    if (impl && impl->windowReady)
    {
        cv::destroyWindow(impl->windowName);
        impl->windowReady = false;
    }
}

std::vector<LineageTreeCreator::CellViz> LineageTreeCreator::makeCellViz(const std::vector<Ellipsoid> &cells)
{
    std::vector<CellViz> viz;
    viz.reserve(cells.size());
    for (const auto &cell : cells)
    {
        CellViz item;
        item.rawName = cell.getName();
        item.x = cell.getX();
        item.y = cell.getY();
        viz.push_back(item);
    }
    return viz;
}

bool LineageTreeCreator::renderCsvFiles(const std::vector<std::string> &csvPaths,
                                        const std::string &outputPath,
                                        const RenderOptions &options)
{
    if (csvPaths.empty())
    {
        std::cerr << "No lineage CSV files provided." << std::endl;
        return false;
    }

    const std::map<int, std::set<std::string>> frameNames = readFrameNames(csvPaths);
    if (frameNames.empty())
    {
        std::cerr << "No lineage rows found in CSV file(s)." << std::endl;
        return false;
    }

    const int firstFrame = (options.firstFrame >= 0) ? options.firstFrame : frameNames.begin()->first;
    const int lastFrame = (options.lastFrame >= 0) ? options.lastFrame : frameNames.rbegin()->first;
    if (lastFrame < firstFrame)
    {
        std::cerr << "Lineage render lastFrame is before firstFrame." << std::endl;
        return false;
    }

    const std::string ext = lowerExtension(outputPath);
    const bool isVideo = (ext == ".mp4" || ext == ".avi" || ext == ".mov");
    if (ext == ".gif")
    {
        std::cerr << "GIF encoding is not handled in C++; use C++/scripts/make_lineage_tree_demo.py." << std::endl;
        return false;
    }

    std::map<std::string, Node> nodes;
    cv::VideoWriter writer;
    if (isVideo)
    {
        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(outputPath, fourcc, options.fps, cv::Size(options.width, options.height));
        if (!writer.isOpened())
        {
            std::cerr << "Could not open lineage video writer: " << outputPath << std::endl;
            return false;
        }
    }

    cv::Mat lastImage;
    for (int frame = firstFrame; frame <= lastFrame; ++frame)
    {
        auto it = frameNames.find(frame);
        if (it != frameNames.end())
        {
            for (const std::string &name : it->second)
            {
                ensureNode(nodes, name);
            }
        }

        lastImage = renderTreeImage(nodes, frame, options.width, options.height);
        if (isVideo)
        {
            writer.write(lastImage);
        }
    }

    if (isVideo)
    {
        writer.release();
        return true;
    }

    if (lastImage.empty())
    {
        return false;
    }
    return cv::imwrite(outputPath, lastImage);
}
