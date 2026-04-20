#ifndef ELLIPSOID_HPP
#define ELLIPSOID_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <tuple>

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

class EllipsoidParams : public CellParams
{
public:
    float x;
    float y;
    float z;
    // Triaxial ellipsoid semi-axes. Historical names retained for `a` (major)
    // and `c` (minor). `bRadius` is the new independent equatorial axis; when
    // zero it is treated as oblate (`bRadius == aRadius`) by the Ellipsoid
    // constructor for backward compatibility with pre-triaxial CSVs.
    float aRadius; // a axis
    float bRadius;     // b axis (second equatorial axis, 0 = oblate fallback)
    float cRadius; // c axis
    float theta_x; // rotation about x-axis (radians)
    float theta_y; // rotation about y-axis (radians)
    float theta_z; // rotation about z-axis (radians)
    float brightness; // per-cell rendering brightness [0,1]

    EllipsoidParams() : CellParams(""), x(0), y(0), z(0), aRadius(0), bRadius(0), cRadius(0), theta_x(0), theta_y(0), theta_z(0), brightness(0.5f) {}
    EllipsoidParams(const std::string &name, float x, float y, float z, float aRadius, float cRadius)
        : CellParams(name), x(x), y(y), z(z), aRadius(aRadius), bRadius(0), cRadius(cRadius), theta_x(0), theta_y(0), theta_z(0), brightness(0.5f) {}
    EllipsoidParams(const std::string &name, float x, float y, float z, float aRadius, float cRadius, float theta_x, float theta_y, float theta_z, float brightness = 0.5f)
        : CellParams(name), x(x), y(y), z(z), aRadius(aRadius), bRadius(0), cRadius(cRadius), theta_x(theta_x), theta_y(theta_y), theta_z(theta_z), brightness(brightness) {}
};

// Per-cell snapshot captured at end of each frame. Authoritative source for
// next-frame split decisions. Driven by the triaxial fitted shape —
// shapeElongation = max(a,b,c) / min(a,b,c), splitAxisDir = world-space unit
// vector along the shortest axis (split direction), splitAxisLength = that radius.
struct PreviousFrameSnapshot
{
    bool valid = false;
    float shapeElongation = 1.0f;
    cv::Point3f splitAxisDir{1.0f, 0.0f, 0.0f};
    float splitAxisLength = 0.0f;
    cv::Point3f position{0.0f, 0.0f, 0.0f};
    float aRadius = 0.0f;
    float bRadius = 0.0f;
    float cRadius = 0.0f;
    float thetaX = 0.0f;
    float thetaY = 0.0f;
    float thetaZ = 0.0f;
    float brightness = 0.5f;
};

struct PerturbDirections
{
    int brightness = 0;
    int aRadius = 0;
    int bRadius = 0;
    int cRadius = 0;
};

class Ellipsoid
{
    private:
        std::string _name;
        cv::Point3f _position;
        double _major_radius; // a axis
        double _b_radius;     // b axis (triaxial; oblate fallback = _major_radius when 0)
        double _minor_radius; // c axis
        double a, b, c;
        double _theta_x;  // rotation angle about x-axis (radians)
        double _theta_y;  // rotation angle about y-axis (radians)
        double _theta_z;  // rotation angle about z-axis (radians)
        float _brightness; // per-cell rendering brightness [0,1]
        PerturbParams _aRadiusPerturbParams;
        PerturbParams _bRadiusPerturbParams;
        PerturbParams _cRadiusPerturbParams;
        PerturbParams _brightnessPerturbParams;

        // Inverse rotation: transforms world-space displacement back to local (upright) frame
        void inverseRotatePoint(double dx, double dy, double dz,
                                double &lx, double &ly, double &lz) const;

        bool computeSliceBounds(const cv::Mat &image, float z,
                                int &minX, int &maxX, int &minY, int &maxY) const;

    public:
        void generateInverseRotationMatrix(std::array<double, 9> &R_T) const;
        static EllipsoidConfig cellConfig;
        
        Ellipsoid(const EllipsoidParams &init_props);
        
        Ellipsoid() : _major_radius(0), _b_radius(0), _minor_radius(0), _theta_x(0), _theta_y(0), _theta_z(0), _brightness(0.5f) {}

        // Lightweight accessors — avoid getCellParams() copy in tight loops
        const std::string& getName() const { return _name; }
        float getX() const { return _position.x; }
        float getY() const { return _position.y; }
        float getZ() const { return _position.z; }
        float getARadius() const { return static_cast<float>(_major_radius); }
        float getBRadius() const { return static_cast<float>(_b_radius); }
        float getCRadius() const { return static_cast<float>(_minor_radius); }
        float getMinorRadius() const { return std::min({getARadius(), getBRadius(), getCRadius()}); }
        float getMajorRadius() const { return getARadius(); }
        // Max/min of the three fitted semi-axes. Used by the triaxial plan as
        // the elongation signal for snapshot-based split classification.
        float shapeElongation() const {
            const double mn = std::min({_major_radius, _b_radius, _minor_radius});
            const double mx = std::max({_major_radius, _b_radius, _minor_radius});
            return (mn > 1e-6) ? static_cast<float>(mx / mn) : 1.0f;
        }

        // World-space direction of the shortest fitted semi-axis (the split
        // direction). Daughters separate along this axis:
        //   D1_seed = snapshot.center - 0.5 * splitAxisLength * splitAxisDir
        //   D2_seed = snapshot.center + 0.5 * splitAxisLength * splitAxisDir
        void worldSplitAxis(cv::Point3f &dir, float &length) const;
        // Signal-guided perturbation override: jump cell to a new world-space
        // position (used by the signal-guided perturbation path to teleport
        // cells onto bright clusters). See yp ffc1917.
        void setPosition(float x, float y, float z) {
            _position = cv::Point3f(x, y, z);
        }
        float getBrightness() const { return _brightness; }
        double getVolume() const {
            return static_cast<double>(getARadius()) *
                   static_cast<double>(getBRadius()) *
                   static_cast<double>(getCRadius());
        }
        float getARadiusIncreaseProbability() const { return _aRadiusPerturbParams.increase_prob; }
        float getARadiusDecreaseProbability() const { return _aRadiusPerturbParams.decrease_prob; }
        float getBRadiusIncreaseProbability() const { return _bRadiusPerturbParams.increase_prob; }
        float getBRadiusDecreaseProbability() const { return _bRadiusPerturbParams.decrease_prob; }
        float getCRadiusIncreaseProbability() const { return _cRadiusPerturbParams.increase_prob; }
        float getCRadiusDecreaseProbability() const { return _cRadiusPerturbParams.decrease_prob; }
        float getBrightnessIncreaseProbability() const { return _brightnessPerturbParams.increase_prob; }
        float getBrightnessDecreaseProbability() const { return _brightnessPerturbParams.decrease_prob; }
        void setBrightness(float brightness);
        void setRadii(float newA, float newB, float newC);
        void setRotation(float thetaX, float thetaY, float thetaZ);
        float getThetaX() const { return static_cast<float>(_theta_x); }
        float getThetaY() const { return static_cast<float>(_theta_y); }
        float getThetaZ() const { return static_cast<float>(_theta_z); }
        void setARadiusPerturbProbabilities(float increaseProbability, float decreaseProbability);
        void setBRadiusPerturbProbabilities(float increaseProbability, float decreaseProbability);
        void setCRadiusPerturbProbabilities(float increaseProbability, float decreaseProbability);
        void setBrightnessPerturbProbabilities(float increaseProbability, float decreaseProbability);
        void blendAdaptivePerturbProbabilitiesWithConfig(float brightnessTrust,
                                                         float aRadiusTrust,
                                                         float cRadiusTrust,
                                                         float bRadiusTrust);
        void blendBrightnessPerturbProbabilitiesWithConfig(float trust);
        void adjustARadiusPerturbProbability(int direction, float delta);
        void adjustBRadiusPerturbProbability(int direction, float delta);
        void adjustCRadiusPerturbProbability(int direction, float delta);
        void adjustBrightnessPerturbProbability(int direction, float delta);
        float measureMeanBrightness(const std::vector<cv::Mat> &image,
                                    float topPercentile = 1.0f) const;
        std::pair<float, float> measureBrightnessStats(const std::vector<cv::Mat> &image) const;

        void printCellInfo() const {
            std::cout << "Ellipsoid name: " << _name << " x: " << _position.x << " y: " << _position.y << " z: " << _position.z << " aRadius: " << _major_radius << " bRadius: " << _b_radius << " cRadius: " << _minor_radius << " theta_x: " << _theta_x << " theta_y: " << _theta_y << " theta_z: " << _theta_z << " brightness: " << _brightness << '\n';
        }

        void draw(cv::Mat &image, const SimulationConfig &simulationConfig, float z = 0) const;

        void drawOutline(cv::Mat &image, float color, float z = 0) const;

        // positionScale: multiplier on the position perturbation offset.
        // Used for radius-proportional sigma: scale = maxR / referenceR so
        // large cells take bigger steps and small cells take smaller steps.
        // Default 1.0 = use the base sigma from config unchanged.
        [[nodiscard]] Ellipsoid getPerturbedCell(PerturbDirections *directions = nullptr,
                                                  float positionScale = 1.0f) const;

        // Test if point is inside this cell's (optionally scaled) ellipsoid
        bool isPointInsideEllipsoid(const cv::Point3f &worldPoint,
                                    float scaleFactor = 1.0f) const;

        bool checkConstraints() const;

        EllipsoidParams getCellParams() const;

        [[nodiscard]] std::pair<std::vector<float>, std::vector<float>> calculateCorners() const;

        std::pair<std::vector<float>, std::vector<float>> calculateMinimumBox(Ellipsoid &perturbed_cell) const;

        cv::Point3f get_center() const;

        void print() const;
};

#endif
