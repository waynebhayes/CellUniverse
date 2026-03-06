#ifndef SPHEROID_HPP
#define SPHEROID_HPP

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "Cell.hpp"
#include "ConfigTypes.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class SpheroidParams : public CellParams
{
public:
    float x;
    float y;
    float z;
    float radius;
    float a_axis;
    float b_axis;
    float c_axis;

    SpheroidParams() : CellParams(""), x(0), y(0), z(0), radius(0), a_axis(0), b_axis(0), c_axis(0) {}
    SpheroidParams(const std::string &name, float x, float y, float z, float radius)
        : CellParams(name), x(x), y(y), z(z), radius(radius), a_axis(radius), b_axis(radius), c_axis(radius) {}
    SpheroidParams(const std::string &name, float x, float y, float z, float radius, float a, float b, float c)
        : CellParams(name), x(x), y(y), z(z), radius(radius), a_axis(a), b_axis(b), c_axis(c) {}
};

class Spheroid
{
private:
    std::string _name;
    cv::Point3f _position;
    double _radius;
    bool dormant;
    double _a_axis;
    double _b_axis;
    double _c_axis;

public:
    static SphereConfig cellConfig;
    Spheroid(const SpheroidParams &init_props)
        : _name(init_props.name), _position{init_props.x, init_props.y, init_props.z},
          _radius(init_props.radius), dormant(false),
          _a_axis(init_props.a_axis), _b_axis(init_props.b_axis), _c_axis(init_props.c_axis)
    {
        // printCellInfo();
    }

    Spheroid() : _radius(0), dormant(false), _a_axis(0), _b_axis(0), _c_axis(0) {}

    void printCellInfo() const
    {
        std::cout << "Spheroid name: " << _name << " x: " << _position.x << " y: " << _position.y << " z: " << _position.z 
                  << " radius: " << _radius << " a_axis: " << _a_axis << " b_axis: " << _b_axis << " c_axis: " << _c_axis 
                  << " isDormant: " << dormant << std::endl;
    }

    double getRadiusAt(double z) const;

    void draw(cv::Mat &image, SimulationConfig simulationConfig, cv::Mat *cellMap = nullptr, float z = 0) const;

    void drawOutline(cv::Mat &image, float color, float z = 0) const;

    [[nodiscard]] Spheroid getPerturbedCell() const;

    Spheroid getParameterizedCell(std::unordered_map<std::string, float> params) const;

    std::vector<std::pair<double, cv::Point3d>> performPCA(const std::vector<cv::Point3d> &pts, std::vector<cv::Mat> &frame) const;

    std::tuple<Spheroid, Spheroid, bool> getSplitCells(const std::vector<cv::Mat> &realTiffSlices) const;

    bool checkConstraints() const;

    float getRadiusAt(float z);

    CellParams getCellParams() const;

    [[nodiscard]] std::pair<std::vector<float>, std::vector<float>> calculateCorners() const;

    bool checkIfCellsValid(const std::vector<Spheroid> &spheroids)
    {
        return !checkIfCellsOverlap(spheroids);
    }

    std::pair<std::vector<float>, std::vector<float>> calculateMinimumBox(Spheroid &perturbed_cell) const;

    static bool checkIfCellsOverlap(const std::vector<Spheroid> &spheroids);

    // Additional method specific to spheroids
    double getAxisRadiusAt(double z, double axis_length) const;
};

#endif