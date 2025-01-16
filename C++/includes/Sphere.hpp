#ifndef SPHERE_HPP
#define SPHERE_HPP

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
#define M_PI 3.14159265358979323846

class SphereParams : public CellParams
{
public:
    float x;
    float y;
    float z;
    float radius;

    SphereParams() : CellParams(""), x(0), y(0), z(0), radius(0) {}
    SphereParams(const std::string &name, float x, float y, float z, float radius)
        : CellParams(name), x(x), y(y), z(z), radius(radius) {}

    void parseParams(float x_, float y_, float z_, float radius_)
    {
        x = x_;
        y = y_;
        z = z_;
        radius = radius_;
    }
};

class Sphere
{
private:
    std::string _name;
    cv::Point3f _position;
    double _radius;
    double _rotation;
    bool dormant;

public:
    static SphereParams paramClass;
    static SphereConfig cellConfig;
    Sphere(const SphereParams &init_props)
        : _name(init_props.name), _position{init_props.x, init_props.y, init_props.z},
          _radius(init_props.radius), _rotation(0), dormant(false)
    {
        //        printCellInfo();
    }

    Sphere() : _radius(0), _rotation(0), dormant(false) {}

    void printCellInfo()
    {
        std::cout << "Sphere name: " << _name << " x: " << _position.x << " y: " << _position.y << " z: " << _position.z << " radius: " << _radius << " isDormant: " << dormant << std::endl;
    }

    double getRadiusAt(double z) const;

    void draw(cv::Mat &image, SimulationConfig simulationConfig, cv::Mat *cellMap = nullptr, float z = 0) const;

    void drawOutline(cv::Mat &image, float color, float z = 0) const;

    [[nodiscard]] Sphere getPerturbedCell() const;

    Sphere getParameterizedCell(std::unordered_map<std::string, float> params = {}) const;

    std::tuple<Sphere, Sphere, bool> getSplitCells(const std::vector<cv::Mat> &image) const;
    // std::tuple<Sphere, Sphere, bool> Sphere::getSplitCells() const;

    std::vector<std::pair<float, cv::Vec3f>> performPCA(const std::vector<cv::Point3f> &points) const;
    //std::vector<float> performPCA(const std::vector<cv::Point3f> &points) const;

    bool checkConstraints() const;

    float getRadiusAt(float z);

    CellParams getCellParams() const;

    [[nodiscard]] std::pair<std::vector<float>, std::vector<float>> calculateCorners() const;

    bool checkIfCellsValid(const std::vector<Sphere> &spheres)
    {
        return !checkIfCellsOverlap(spheres);
    }

    std::pair<std::vector<float>, std::vector<float>> calculateMinimumBox(Sphere &perturbed_cell) const;

    static bool checkIfCellsOverlap(const std::vector<Sphere> &spheres);
};

#endif