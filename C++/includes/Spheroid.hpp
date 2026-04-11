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
    float majorRadius;
    float minorRadius;
    float equatorialAspectRatio;
    float theta_x; // rotation about x-axis (radians)
    float theta_y; // rotation about y-axis (radians)
    float theta_z; // rotation about z-axis (radians)
    float brightness; // per-cell rendering brightness [0,1]

    SpheroidParams() : CellParams(""), x(0), y(0), z(0), majorRadius(0), minorRadius(0), equatorialAspectRatio(1.0f), theta_x(0), theta_y(0), theta_z(0), brightness(0.5f) {}
    SpheroidParams(const std::string &name, float x, float y, float z, float majorRadius, float minorRadius)
        : CellParams(name), x(x), y(y), z(z), majorRadius(majorRadius), minorRadius(minorRadius), equatorialAspectRatio(1.0f), theta_x(0), theta_y(0), theta_z(0), brightness(0.5f) {}
    SpheroidParams(const std::string &name, float x, float y, float z, float majorRadius, float minorRadius, float theta_x, float theta_y, float theta_z, float brightness = 0.5f, float equatorialAspectRatio = 1.0f)
        : CellParams(name), x(x), y(y), z(z), majorRadius(majorRadius), minorRadius(minorRadius), equatorialAspectRatio(equatorialAspectRatio), theta_x(theta_x), theta_y(theta_y), theta_z(theta_z), brightness(brightness) {}
};

struct SplitDiagnostics
{
    float elongationRatio = 1.0f;
    int totalCount = 0;
    int count1 = 0;
    int count2 = 0;
    float balance = 0.0f;
    float separation = 0.0f;
    float separationOverDaughterMajor = 0.0f;
    bool recentered = false;
    float recenterDrift = 0.0f;
    float driftOverParentMajor = 0.0f;
    float daughterMajorRadius = 0.0f;
    float daughterMinorRadius = 0.0f;
    float axisAbsZ = 0.0f;
    // Total bright voxels inside the cell's ellipsoid volume used for split PCA.
    // Used by Frame::trySplitCell's min-size gate to reject boundary cells /
    // clipped cells whose PCA would be unreliable.
    int insideCount = 0;
};

struct PerturbDirections
{
    int brightness = 0;
    int majorRadius = 0;
    int minorRadius = 0;
    int abRatio = 0;
};

class Spheroid 
{
    private:
        std::string _name;
        cv::Point3f _position;
        double _major_radius;
        double _minor_radius;
        double _equatorial_aspect_ratio;
        double a, b, c;
        double _theta_x;  // rotation angle about x-axis (radians)
        double _theta_y;  // rotation angle about y-axis (radians)
        double _theta_z;  // rotation angle about z-axis (radians)
        float _brightness; // per-cell rendering brightness [0,1]
        PerturbParams _majorRadiusPerturbParams;
        PerturbParams _minorRadiusPerturbParams;
        PerturbParams _abRatioPerturbParams;
        PerturbParams _brightnessPerturbParams;

        // Inverse rotation: transforms world-space displacement back to local (upright) frame
        void inverseRotatePoint(double dx, double dy, double dz,
                                double &lx, double &ly, double &lz) const;

        void generateInverseRotationMatrix(std::array<double, 9> &R_T) const;

        bool computeSliceBounds(const cv::Mat &image, float z,
                                int &minX, int &maxX, int &minY, int &maxY) const;

    public:
        static SpheroidConfig cellConfig;
        
        Spheroid(const SpheroidParams &init_props);
        
        Spheroid() : _major_radius(0), _minor_radius(0), _theta_x(0), _theta_y(0), _theta_z(0), _brightness(0.5f) {}

        // Lightweight accessors — avoid getCellParams() copy in tight loops
        const std::string& getName() const { return _name; }
        float getX() const { return _position.x; }
        float getY() const { return _position.y; }
        float getZ() const { return _position.z; }
        float getMajorRadius() const { return static_cast<float>(_major_radius); }
        float getBRadius() const { return static_cast<float>(b); }
        float getMinorRadius() const { return static_cast<float>(_minor_radius); }
        float getEquatorialAspectRatio() const { return static_cast<float>(_equatorial_aspect_ratio); }
        float getBrightness() const { return _brightness; }
        float getMajorRadiusIncreaseProbability() const { return _majorRadiusPerturbParams.increase_prob; }
        float getMajorRadiusDecreaseProbability() const { return _majorRadiusPerturbParams.decrease_prob; }
        float getMinorRadiusIncreaseProbability() const { return _minorRadiusPerturbParams.increase_prob; }
        float getMinorRadiusDecreaseProbability() const { return _minorRadiusPerturbParams.decrease_prob; }
        float getABRatioIncreaseProbability() const { return _abRatioPerturbParams.increase_prob; }
        float getABRatioDecreaseProbability() const { return _abRatioPerturbParams.decrease_prob; }
        float getBrightnessIncreaseProbability() const { return _brightnessPerturbParams.increase_prob; }
        float getBrightnessDecreaseProbability() const { return _brightnessPerturbParams.decrease_prob; }
        void setBrightness(float brightness);
        void setMajorRadiusPerturbProbabilities(float increaseProbability, float decreaseProbability);
        void setMinorRadiusPerturbProbabilities(float increaseProbability, float decreaseProbability);
        void setABRatioPerturbProbabilities(float increaseProbability, float decreaseProbability);
        void setBrightnessPerturbProbabilities(float increaseProbability, float decreaseProbability);
        void blendAdaptivePerturbProbabilitiesWithConfig(float brightnessTrust,
                                                         float majorRadiusTrust,
                                                         float minorRadiusTrust,
                                                         float abRatioTrust);
        void blendBrightnessPerturbProbabilitiesWithConfig(float trust);
        void adjustMajorRadiusPerturbProbability(int direction, float delta);
        void adjustMinorRadiusPerturbProbability(int direction, float delta);
        void adjustABRatioPerturbProbability(int direction, float delta);
        void adjustBrightnessPerturbProbability(int direction, float delta);
        float measureMeanBrightness(const std::vector<cv::Mat> &image) const;
        std::pair<float, float> measureBrightnessStats(const std::vector<cv::Mat> &image) const;

        void printCellInfo() const {
            std::cout << "Spheroid name: " << _name << " x: " << _position.x << " y: " << _position.y << " z: " << _position.z << " majorRadius: " << _major_radius << " bRadius: " << b << " minorRadius: " << _minor_radius << " abRatio: " << _equatorial_aspect_ratio << " theta_x: " << _theta_x << " theta_y: " << _theta_y << " theta_z: " << _theta_z << " brightness: " << _brightness << '\n';
        }

        void draw(cv::Mat &image, const SimulationConfig &simulationConfig, float z = 0) const;

        void drawOutline(cv::Mat &image, float color, float z = 0) const;

        [[nodiscard]] Spheroid getPerturbedCell(PerturbDirections *directions = nullptr) const;

        std::tuple<Spheroid, Spheroid, bool, float, SplitDiagnostics> getSplitCells(const std::vector<cv::Mat> &image, float z_scaling,
            float backgroundColor,
            const std::vector<cv::Point3f> &neighborCenters = {},
            float preOptMajorR = 0.0f, float preOptMinorR = 0.0f,
            float preOptX = 0.0f, float preOptY = 0.0f, float preOptZ = 0.0f,
            float splitSearchRadiusMultiplier = 3.0f,
            float splitMinorAxisAlignmentToleranceDegrees = 180.0f,
            float splitMinorAxisAlignmentFlatnessRatioThreshold = 0.5f,
            float splitMinorAxisAlignmentMinRadiusDisableThreshold = 0.0f) const;

        bool checkConstraints() const;

        SpheroidParams getCellParams() const;

        [[nodiscard]] std::pair<std::vector<float>, std::vector<float>> calculateCorners() const;

        std::pair<std::vector<float>, std::vector<float>> calculateMinimumBox(Spheroid &perturbed_cell) const;

        cv::Point3f get_center() const;

        void print() const;
};

#endif
