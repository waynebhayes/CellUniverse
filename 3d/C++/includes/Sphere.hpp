#ifndef SPHERE_HPP
#define SPHERE_HPP

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "Cell.hpp"
#define M_PI 3.14159265358979323846

class SphereConfig: public CellConfig {
public:
    PerturbParams x;
    PerturbParams y;
    PerturbParams z;
    PerturbParams radius;
    double minRadius;
    double maxRadius;

    void explodeConfig(const YAML::Node& node) override
    {
        x.explodeParams(node["x"]);
        y.explodeParams(node["y"]);
        z.explodeParams(node["z"]);
        radius.explodeParams(node["radius"]);
        minRadius = node["minRadius"].as<double>();
        maxRadius = node["maxRadius"].as<double>();
    }
};

class SphereParams {
public:
    double x;
    double y;
    double z;
    double radius;

    SphereParams()
            : x(0), y(0), z(0), radius(0) {}

//    SphereParams(std::string name_val , double x_val, double y_val, double z_val, double radius_val)
//            : CellParams{name_val}, x(x_val), y(y_val), z(z_val), radius(radius_val) {}
//
//    void parseParams(const YAML::Node& node) {
//        prob = node["prob"].as<double>();
//        mu = node["mu"].as<double>();
//        sigma = node["sigma"].as<double>();
//    }
};


#endif