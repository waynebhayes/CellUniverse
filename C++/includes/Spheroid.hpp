#ifndef SPHEROID_HPP
#define SPHEROID_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <array>
#include <algorithm>
#include <string>
#include <unordered_map>
#include "Cell.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class SpheroidParams : public CellParams
{
public:
    float x;
    float y;
    float z;
    std::vector<float> x_vec;
    std::vector<float> y_vec;
    std::vector<float> z_vec;
    float majorRadius;
    float minorRadius;
    float theta_x; // rotation about x-axis (radians)
    float theta_y; // rotation about y-axis (radians)
    float theta_z; // rotation about z-axis (radians)

    SpheroidParams() : CellParams(""), x(0), y(0), z(0), majorRadius(0), minorRadius(0), theta_x(0), theta_y(0), theta_z(0) {}
    SpheroidParams(const std::string &name, float x, float y, float z, float majorRadius, float minorRadius) 
        : CellParams(name), x(x), y(y), z(z), majorRadius(majorRadius), minorRadius(minorRadius), theta_x(0), theta_y(0), theta_z(0) {}
    SpheroidParams(const std::string &name, float x, float y, float z, float majorRadius, float minorRadius, float theta_x, float theta_y, float theta_z) 
        : CellParams(name), x(x), y(y), z(z), majorRadius(majorRadius), minorRadius(minorRadius), theta_x(theta_x), theta_y(theta_y), theta_z(theta_z) {}
    SpheroidParams(const std::string &name, float x, float y, float z, std::vector<float> _x_vec, std::vector<float> _y_vec, std::vector<float> _z_vec)
        : CellParams(name), x(x), y(y), z(z), x_vec(_x_vec), y_vec(_y_vec), z_vec(_z_vec)
        {
            // assuming this is the x and y vectors are the same and z is the smaller one
            majorRadius = std::sqrt((_x_vec[0]*_x_vec[0])+(_x_vec[1]*_x_vec[1])+(_x_vec[2]*_x_vec[2]));
            minorRadius = std::sqrt((_z_vec[0]*_z_vec[0])+(_z_vec[1]*_z_vec[1])+(_z_vec[2]*_z_vec[2]));
            
            theta_x = std::atan2(_z_vec[1], _z_vec[2]); // rotation about x from z-vector
            theta_y = std::atan2(_z_vec[0], _z_vec[2]); // rotation about y from z-vector
            theta_z = std::atan2(_x_vec[1], _x_vec[0]); // rotation about z from x-vector
        }

    void parseParams(float x_, float y_, float z_, std::vector<float> _x_vec, std::vector<float> _y_vec, std::vector<float> _z_vec)
    {
        x = x_;
        y = y_;
        z = z_;
        x_vec = _x_vec;
        y_vec = _y_vec;
        z_vec = _z_vec;
        majorRadius = std::sqrt((_x_vec[0]*_x_vec[0])+(_x_vec[1]*_x_vec[1])+(_x_vec[2]*_x_vec[2]));
        minorRadius = std::sqrt((_z_vec[0]*_z_vec[0])+(_z_vec[1]*_z_vec[1])+(_z_vec[2]*_z_vec[2]));
    }

    

};

class Spheroid 
{
    private:
        std::string _name;
        std::vector<double> _x_vec;
        std::vector<double> _y_vec;
        std::vector<double> _z_vec;
        cv::Point3f _position;
        double _major_radius;
        double _minor_radius;
        double _rotation;
        double a, b, c;
        double _theta_x;  // rotation angle about x-axis (radians)
        double _theta_y;  // rotation angle about y-axis (radians)
        double _theta_z;  // rotation angle about z-axis (radians)
        bool dormant;

        // Inverse rotation: transforms world-space displacement back to local (upright) frame
        void inverseRotatePoint(double dx, double dy, double dz,
                                double &lx, double &ly, double &lz) const;
    
    public:
        static SpheroidParams paramClass;
        static SpheroidConfig cellConfig;
        
        Spheroid(const SpheroidParams &init_props);
        
        Spheroid() : _major_radius(0), _minor_radius(0), _rotation(0), _theta_x(0), _theta_y(0), _theta_z(0), dormant(false) {}

        void printCellInfo() const {
            std::cout << "Spheroid name: " << _name << " x: " << _position.x << " y: " << _position.y << " z: " << _position.z << " majorRadius: " << _major_radius << " minorRadius: " << _minor_radius << " theta_x: " << _theta_x << " theta_y: " << _theta_y << " theta_z: " << _theta_z << " isDormant: " << dormant << std::endl;
        }

        std::vector<double> getShapeAt(double z) const;

        void draw(cv::Mat &image, SimulationConfig simulationConfig, cv::Mat *cellMap = nullptr, float z = 0) const;

        void drawOutline(cv::Mat &image, float color, float z = 0) const;

        [[nodiscard]] Spheroid getPerturbedCell() const;

        Spheroid getParameterizedCell(std::unordered_map<std::string, float> params) const;

        std::tuple<Spheroid, Spheroid, bool> getSplitCells(const std::vector<cv::Mat> &image, float z_scaling) const;

        std::vector<std::pair<float, cv::Vec3f>> performPCA(const std::vector<cv::Point3f> &points) const;

        bool checkConstraints() const;

        SpheroidParams getCellParams() const;

        [[nodiscard]] std::pair<std::vector<float>, std::vector<float>> calculateCorners() const;

        bool checkIfCellsValid(const std::vector<Spheroid> &spheroids)
        {
            return !checkIfCellsOverlap(spheroids);
        }

        std::pair<std::vector<float>, std::vector<float>> calculateMinimumBox(Spheroid &perturbed_cell) const;

        static bool checkIfCellsOverlap(const std::vector<Spheroid> &spheroids);

        float major_magnitude();
        
        float minor_magnitude();

        cv::Point3f get_center() const;

        void print() const;

        int get_matrix_size();
};

#endif