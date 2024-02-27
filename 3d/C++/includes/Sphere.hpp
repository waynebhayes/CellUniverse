#ifndef SPHERE_HPP
#define SPHERE_HPP

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


class SphereParams: public CellParams{
public:
    float x;
    float y;
    float z;
    float radius;

    SphereParams() : CellParams(""), x(0), y(0), z(0), radius(0) {}
    SphereParams(const std::string& name, float x, float y, float z, float radius)
        : CellParams(name), x(x), y(y), z(z), radius(radius) {}

    void parseParams(float x_, float y_, float z_, float radius_) {
        x = x_;
        y = y_;
        z = z_;
        radius = radius_;
    }
};

class Sphere : public Cell {
private:
    std::string _name;
    cv::Point3f _position;
    double _radius;
    double _rotation;
    bool dormant;
public:
    static SphereParams paramClass;
    static SphereConfig cellConfig;
    Sphere(const SphereParams& init_props)
            : _name(init_props.name), _position{init_props.x, init_props.y, init_props.z},
              _radius(init_props.radius), _rotation(0), dormant(false) {
        printCellInfo();
    }

    Sphere() : _radius(0), _rotation(0), dormant(false) {}

    void printCellInfo() {
        std::cout << "Sphere name: " << _name << " x: " << _position.x << " y: " << _position.y << " z: " << _position.z << " radius: " << _radius << " isDormant: " << dormant << std::endl;
    }

    double get_radius_at(double z) const;

    void draw(cv::Mat & image, SimulationConfig simulationConfig, cv::Mat * cellMap = nullptr, float z = 0) const override;

    void draw_outline(cv::Mat& image, float color, float z = 0) const override;

    Cell* get_perturbed_cell() const override;

    Cell* get_parameterized_cell(std::unordered_map<std::string, float> params = {}) const override;

    std::tuple<Cell*, Cell*, bool> get_split_cells() const override;

    bool check_constraints() const;

    float get_radius_at(float z);

    CellParams get_cell_params() const override;

    std::pair<std::vector<float>, std::vector<float>> calculate_corners() const override;


    std::pair<std::vector<float>, std::vector<float>> calculate_minimum_box(Cell& perturbed_cell) const override;

    static bool check_if_cells_overlap(const std::vector<Sphere>& spheres);
};


#endif