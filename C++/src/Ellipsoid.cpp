#include "../includes/Ellipsoid.hpp"
#include <limits>
#include <random>
#include <type_traits>
// #include <iostream>

namespace {
template <typename>
inline constexpr bool alwaysFalseV = false;

template <typename Visitor>
void scanEllipsoidSlice(
    int minX, int maxX, int minY, int maxY,
    const cv::Point3f &position, double z,
    const std::array<double, 9> &R_T,
    double invA2, double invB2, double invC2,
    Visitor &&visitor)
{
    const double stepXx = R_T[0];
    const double stepXy = R_T[3];
    const double stepXz = R_T[6];
    const double dz = z - position.z;
    const double baseDx = static_cast<double>(minX) - position.x;

    for (int y = minY; y <= maxY; ++y) {
        const double dy = static_cast<double>(y) - position.y;
        double lx = R_T[0] * baseDx + R_T[1] * dy + R_T[2] * dz;
        double ly = R_T[3] * baseDx + R_T[4] * dy + R_T[5] * dz;
        double lz = R_T[6] * baseDx + R_T[7] * dy + R_T[8] * dz;

        for (int x = minX; x <= maxX; ++x) {
            const double val = (lx * lx) * invA2
                             + (ly * ly) * invB2
                             + (lz * lz) * invC2;
            visitor(x, y, val);

            lx += stepXx;
            ly += stepXy;
            lz += stepXz;
        }
    }
}

template <typename Visitor>
void scanEllipsoidVolume(
    const std::vector<cv::Mat> &image,
    int minX, int maxX, int minY, int maxY, int minZ, int maxZ,
    const cv::Point3f &position,
    const std::array<double, 9> &R_T,
    double invA2, double invB2, double invC2,
    Visitor &&visitor)
{
    const double stepXx = R_T[0];
    const double stepXy = R_T[3];
    const double stepXz = R_T[6];
    const double baseDx = static_cast<double>(minX) - position.x;

    for (int z = minZ; z <= maxZ; ++z) {
        const double dz = static_cast<double>(z) - position.z;
        for (int y = minY; y <= maxY; ++y) {
            const double dy = static_cast<double>(y) - position.y;
            double lx = R_T[0] * baseDx + R_T[1] * dy + R_T[2] * dz;
            double ly = R_T[3] * baseDx + R_T[4] * dy + R_T[5] * dz;
            double lz = R_T[6] * baseDx + R_T[7] * dy + R_T[8] * dz;

            const float *row = image[z].ptr<float>(y);
            for (int x = minX; x <= maxX; ++x) {
                const double dx = static_cast<double>(x) - position.x;
                const double val = (lx * lx) * invA2
                                 + (ly * ly) * invB2
                                 + (lz * lz) * invC2;
                if constexpr (std::is_invocable_v<Visitor, double, double, double, int, int, int, float, double>) {
                    visitor(dx, dy, dz, x, y, z, row[x], val);
                } else if constexpr (std::is_invocable_v<Visitor, int, int, int, float, double>) {
                    visitor(x, y, z, row[x], val);
                } else {
                    static_assert(alwaysFalseV<Visitor>,
                                  "scanEllipsoidVolume visitor must be invocable with either "
                                  "(dx,dy,dz,x,y,z,pixel,val) or (x,y,z,pixel,val).");
                }

                lx += stepXx;
                ly += stepXy;
                lz += stepXz;
            }
        }
    }
}
} // namespace

EllipsoidConfig Ellipsoid::cellConfig = EllipsoidConfig();

// ---- INVERSE ROTATION ----
// Transforms a world-space displacement vector back into the spheroid's 
// local (upright) coordinate frame. Since the forward rotation is 
// R_total = Rz * Ry * Rx, the inverse is R_total^T = Rx^T * Ry^T * Rz^T.
// We undo z first, then y, then x.
void Ellipsoid::inverseRotatePoint(double dx, double dy, double dz,
                                   double &lx, double &ly, double &lz) const 
{
    std::array<double, 9> R_T;
    generateInverseRotationMatrix(R_T);
    lx = R_T[0] * dx + R_T[1] * dy + R_T[2] * dz;
    ly = R_T[3] * dx + R_T[4] * dy + R_T[5] * dz;
    lz = R_T[6] * dx + R_T[7] * dy + R_T[8] * dz;
}

void Ellipsoid::worldSplitAxis(cv::Point3f &dir, float &length) const
{
    // Pick whichever of (a, b, c) is shortest and rotate the corresponding
    // local unit vector out to world space using the forward rotation
    // R = Rz * Ry * Rx. Since R_T in generateInverseRotationMatrix is R^T
    // stored row-major, column-i of R (which maps local axis i to world) is
    // read out of R_T as (R_T[i], R_T[3 + i], R_T[6 + i]).
    std::array<double, 9> R_T{};
    generateInverseRotationMatrix(R_T);

    // Use the SHORTEST axis as the split direction. For a pancake cell
    // (a ≈ b >> c), daughters separate along c (stacked through the thin
    // dimension). The split plane cuts perpendicular to the short axis.
    int shortestAxis = 0; // 0 = a (local x), 1 = b (local y), 2 = c (local z)
    double shortestValue = a;
    if (b < shortestValue) { shortestAxis = 1; shortestValue = b; }
    if (c < shortestValue) { shortestAxis = 2; shortestValue = c; }

    const double dx = R_T[shortestAxis];
    const double dy = R_T[3 + shortestAxis];
    const double dz = R_T[6 + shortestAxis];

    // The column should already be a unit vector because R is orthonormal,
    // but normalize defensively against floating-point drift.
    const double norm = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (norm > 1e-9) {
        dir = cv::Point3f(
            static_cast<float>(dx / norm),
            static_cast<float>(dy / norm),
            static_cast<float>(dz / norm));
    } else {
        dir = cv::Point3f(1.0f, 0.0f, 0.0f);
    }
    length = static_cast<float>(shortestValue);
}

void Ellipsoid::generateInverseRotationMatrix(std::array<double, 9> &R_T) const {
    const double cx = std::cos(_theta_x), sx = std::sin(_theta_x);
    const double cy = std::cos(_theta_y), sy = std::sin(_theta_y);
    const double cz = std::cos(_theta_z), sz = std::sin(_theta_z);

    // R = Rz * Ry * Rx
    const double R00 = cz * cy;
    const double R01 = cz * sy * sx - sz * cx;
    const double R02 = cz * sy * cx + sz * sx;

    const double R10 = sz * cy;
    const double R11 = sz * sy * sx + cz * cx;
    const double R12 = sz * sy * cx - cz * sx;

    const double R20 = -sy;
    const double R21 = cy * sx;
    const double R22 = cy * cx;

    // Store R^T in row-major order.
    R_T[0] = R00; R_T[1] = R10; R_T[2] = R20;
    R_T[3] = R01; R_T[4] = R11; R_T[5] = R21;
    R_T[6] = R02; R_T[7] = R12; R_T[8] = R22;
}

bool Ellipsoid::computeSliceBounds(const cv::Mat &image, float z,
                                  int &minX, int &maxX, int &minY, int &maxY) const {
    float maxR = std::max({static_cast<float>(a), static_cast<float>(b), static_cast<float>(c)});
    if (std::abs(z - _position.z) > maxR) {
        return false;
    }

    minX = std::max(0, static_cast<int>(std::floor(_position.x - maxR)));
    maxX = std::min(image.cols - 1, static_cast<int>(std::ceil(_position.x + maxR)));
    minY = std::max(0, static_cast<int>(std::floor(_position.y - maxR)));
    maxY = std::min(image.rows - 1, static_cast<int>(std::ceil(_position.y + maxR)));
    return true;
}


// ---- SIMPLIFIED CONSTRUCTOR ----
// No voxel matrix construction needed — drawing is done analytically.
Ellipsoid::Ellipsoid(const EllipsoidParams &init_props)
: _name(init_props.name), _position{init_props.x, init_props.y, init_props.z},
          _major_radius(init_props.aRadius),
          _b_radius(init_props.bRadius > 0.0f ? init_props.bRadius : init_props.aRadius),
          _minor_radius(init_props.cRadius),
          _theta_x(init_props.theta_x), _theta_y(init_props.theta_y), _theta_z(init_props.theta_z),
          _brightness(init_props.brightness),
          _aRadiusPerturbParams(cellConfig.aRadius),
          _cRadiusPerturbParams(cellConfig.cRadius),
          _bRadiusPerturbParams(cellConfig.bRadius),
          _brightnessPerturbParams(cellConfig.brightness)
{
    _major_radius = std::fmax(_major_radius, cellConfig.minARadius);
    _major_radius = std::fmin(_major_radius, cellConfig.maxARadius);
    _minor_radius = std::fmax(_minor_radius, cellConfig.minCRadius);
    _minor_radius = std::fmin(_minor_radius, cellConfig.maxCRadius);
    // Triaxial b-axis is clamped to [minBRadius, maxBRadius] when those bounds
    // are configured. If they are not set (still zero), fall back to the same
    // bounds as the a-axis so an unconfigured run still behaves near-oblate.
    const double minBR = (cellConfig.maxBRadius > 0.0)
        ? cellConfig.minBRadius : cellConfig.minARadius;
    const double maxBR = (cellConfig.maxBRadius > 0.0)
        ? cellConfig.maxBRadius : cellConfig.maxARadius;
    _b_radius = std::fmax(_b_radius, minBR);
    _b_radius = std::fmin(_b_radius, maxBR);

    // Clamp z to the valid interpolated-z range. Without this, perturbation can
    // push cells off the z-stack (where they contribute zero pixels and reduce
    // L2 cost, which the optimizer happily rewards). See run output_jihang_20260408_161444
    // for a concrete example: a daughter drifted to z=266 with inside_count=0.
    _position.z = std::fmax(_position.z, 0.0f);
    _position.z = std::fmin(_position.z, cellConfig.maxZ);

    if (_minor_radius > _major_radius) {
        _minor_radius = _major_radius;
    }
    _brightness = std::fmax(_brightness, static_cast<float>(cellConfig.minBrightness));
    _brightness = std::fmin(_brightness, static_cast<float>(cellConfig.maxBrightness));

    this->a = this->_major_radius;
    this->b = this->_b_radius;
    this->c = this->_minor_radius;

    if (a <= 0 || b <= 0 || c <= 0) {
        throw std::invalid_argument("Ellipsoid radii must be positive");
    }
}

cv::Point3f Ellipsoid::get_center() const {
    return _position; // x, y, z position
}

void Ellipsoid::print() const {
    std::cout << "Ellipsoid: " << _name
              << " pos=(" << _position.x << ", " << _position.y << ", " << _position.z << ")"
              << " a=" << a << " b=" << b << " c=" << c
              << " theta=(" << _theta_x << ", " << _theta_y << ", " << _theta_z << ")"
              << '\n';
}

void Ellipsoid::setBrightness(float brightness)
{
    _brightness = std::clamp(brightness,
                             static_cast<float>(cellConfig.minBrightness),
                             static_cast<float>(cellConfig.maxBrightness));
}

void Ellipsoid::setRadii(float newA, float newB, float newC)
{
    const double minBR = (cellConfig.maxBRadius > 0.0)
        ? cellConfig.minBRadius : cellConfig.minARadius;
    const double maxBR = (cellConfig.maxBRadius > 0.0)
        ? cellConfig.maxBRadius : cellConfig.maxARadius;
    _major_radius = std::clamp(static_cast<double>(newA), cellConfig.minARadius, cellConfig.maxARadius);
    _b_radius     = std::clamp(static_cast<double>(newB), minBR, maxBR);
    _minor_radius = std::clamp(static_cast<double>(newC), cellConfig.minCRadius, cellConfig.maxCRadius);
    a = _major_radius;
    b = _b_radius;
    c = _minor_radius;
}

void Ellipsoid::setRotation(float thetaX, float thetaY, float thetaZ)
{
    _theta_x = static_cast<double>(thetaX);
    _theta_y = static_cast<double>(thetaY);
    _theta_z = static_cast<double>(thetaZ);
}

void Ellipsoid::setARadiusPerturbProbabilities(float increaseProbability, float decreaseProbability)
{
    const float clampedIncrease = std::clamp(increaseProbability, 0.0f, 1.0f);
    _aRadiusPerturbParams.increase_prob = clampedIncrease;
    _aRadiusPerturbParams.decrease_prob =
        std::clamp(decreaseProbability, 0.0f, std::max(0.0f, 1.0f - clampedIncrease));
}

void Ellipsoid::setCRadiusPerturbProbabilities(float increaseProbability, float decreaseProbability)
{
    const float clampedIncrease = std::clamp(increaseProbability, 0.0f, 1.0f);
    _cRadiusPerturbParams.increase_prob = clampedIncrease;
    _cRadiusPerturbParams.decrease_prob =
        std::clamp(decreaseProbability, 0.0f, std::max(0.0f, 1.0f - clampedIncrease));
}

void Ellipsoid::setBRadiusPerturbProbabilities(float increaseProbability, float decreaseProbability)
{
    const float clampedIncrease = std::clamp(increaseProbability, 0.0f, 1.0f);
    _bRadiusPerturbParams.increase_prob = clampedIncrease;
    _bRadiusPerturbParams.decrease_prob =
        std::clamp(decreaseProbability, 0.0f, std::max(0.0f, 1.0f - clampedIncrease));
}

void Ellipsoid::setBrightnessPerturbProbabilities(float increaseProbability, float decreaseProbability)
{
    const float clampedIncrease = std::clamp(increaseProbability, 0.0f, 1.0f);
    _brightnessPerturbParams.increase_prob = clampedIncrease;
    _brightnessPerturbParams.decrease_prob =
        std::clamp(decreaseProbability, 0.0f, std::max(0.0f, 1.0f - clampedIncrease));
}

void Ellipsoid::blendBrightnessPerturbProbabilitiesWithConfig(float trust)
{
    const float clampedTrust = std::clamp(trust, 0.0f, 1.0f);
    const float baseIncrease =
        std::clamp(cellConfig.brightness.increase_prob >= 0.0f ? cellConfig.brightness.increase_prob : 0.0f,
                   0.0f, 1.0f);
    const float baseDecrease =
        std::clamp(cellConfig.brightness.decrease_prob >= 0.0f ? cellConfig.brightness.decrease_prob : 0.0f,
                   0.0f, 1.0f);

    const float blendedIncrease =
        baseIncrease * (1.0f - clampedTrust) + getBrightnessIncreaseProbability() * clampedTrust;
    const float blendedDecrease =
        baseDecrease * (1.0f - clampedTrust) + getBrightnessDecreaseProbability() * clampedTrust;

    setBrightnessPerturbProbabilities(blendedIncrease, blendedDecrease);
}

void Ellipsoid::blendAdaptivePerturbProbabilitiesWithConfig(float brightnessTrust,
                                                           float aRadiusTrust,
                                                           float cRadiusTrust,
                                                           float bRadiusTrust)
{
    const float clampedMajorTrust = std::clamp(aRadiusTrust, 0.0f, 1.0f);
    const float clampedBTrust = std::clamp(bRadiusTrust, 0.0f, 1.0f);
    const float clampedMinorTrust = std::clamp(cRadiusTrust, 0.0f, 1.0f);

    const float baseMajorIncrease =
        std::clamp(cellConfig.aRadius.increase_prob >= 0.0f ? cellConfig.aRadius.increase_prob : 0.0f,
                   0.0f, 1.0f);
    const float baseMajorDecrease =
        std::clamp(cellConfig.aRadius.decrease_prob >= 0.0f ? cellConfig.aRadius.decrease_prob : 0.0f,
                   0.0f, 1.0f);
    setARadiusPerturbProbabilities(
        baseMajorIncrease * (1.0f - clampedMajorTrust) + getARadiusIncreaseProbability() * clampedMajorTrust,
        baseMajorDecrease * (1.0f - clampedMajorTrust) + getARadiusDecreaseProbability() * clampedMajorTrust);

    const float baseBIncrease =
        std::clamp(cellConfig.bRadius.increase_prob >= 0.0f ? cellConfig.bRadius.increase_prob : 0.0f,
                   0.0f, 1.0f);
    const float baseBDecrease =
        std::clamp(cellConfig.bRadius.decrease_prob >= 0.0f ? cellConfig.bRadius.decrease_prob : 0.0f,
                   0.0f, 1.0f);
    setBRadiusPerturbProbabilities(
        baseBIncrease * (1.0f - clampedBTrust) + getBRadiusIncreaseProbability() * clampedBTrust,
        baseBDecrease * (1.0f - clampedBTrust) + getBRadiusDecreaseProbability() * clampedBTrust);

    const float baseMinorIncrease =
        std::clamp(cellConfig.cRadius.increase_prob >= 0.0f ? cellConfig.cRadius.increase_prob : 0.0f,
                   0.0f, 1.0f);
    const float baseMinorDecrease =
        std::clamp(cellConfig.cRadius.decrease_prob >= 0.0f ? cellConfig.cRadius.decrease_prob : 0.0f,
                   0.0f, 1.0f);
    setCRadiusPerturbProbabilities(
        baseMinorIncrease * (1.0f - clampedMinorTrust) + getCRadiusIncreaseProbability() * clampedMinorTrust,
        baseMinorDecrease * (1.0f - clampedMinorTrust) + getCRadiusDecreaseProbability() * clampedMinorTrust);

    blendBrightnessPerturbProbabilitiesWithConfig(brightnessTrust);
}

void Ellipsoid::adjustARadiusPerturbProbability(int direction, float delta)
{
    _aRadiusPerturbParams.adjustSignedProbability(direction, delta);
}

void Ellipsoid::adjustCRadiusPerturbProbability(int direction, float delta)
{
    _cRadiusPerturbParams.adjustSignedProbability(direction, delta);
}

void Ellipsoid::adjustBRadiusPerturbProbability(int direction, float delta)
{
    _bRadiusPerturbParams.adjustSignedProbability(direction, delta);
}

void Ellipsoid::adjustBrightnessPerturbProbability(int direction, float delta)
{
    _brightnessPerturbParams.adjustSignedProbability(direction, delta);
}

// ---- ROTATION-AWARE draw() ----
// Instead of checking a voxel matrix, we analytically test each pixel
// against the rotated spheroid by inverse-transforming back to local coords.
void Ellipsoid::draw(cv::Mat &image, const SimulationConfig &simulationConfig, float z) const{
    int minX, maxX, minY, maxY;
    if (!computeSliceBounds(image, z, minX, maxX, minY, maxY)) return;

    std::array<double, 9> R_T;
    generateInverseRotationMatrix(R_T);

    const double invA2 = 1.0 / (a * a);
    const double invB2 = 1.0 / (b * b);
    const double invC2 = 1.0 / (c * c);

    scanEllipsoidSlice(
        minX, maxX, minY, maxY,
        _position, static_cast<double>(z),
        R_T, invA2, invB2, invC2,
        [&](int x, int y, double val) {
            if (val <= 1.0) {
                image.at<float>(y, x) = _brightness;
            }
        });
}

float Ellipsoid::measureMeanBrightness(const std::vector<cv::Mat> &image,
                                      float topPercentile) const
{
    if (image.empty()) {
        return _brightness;
    }

    const float clampedTopPercentile = std::clamp(topPercentile, 0.0f, 1.0f);
    if (clampedTopPercentile >= 1.0f) {
        return measureBrightnessStats(image).first;
    }

    std::array<double, 9> R_T;
    generateInverseRotationMatrix(R_T);

    const double invA2 = 1.0 / (a * a);
    const double invB2 = 1.0 / (b * b);
    const double invC2 = 1.0 / (c * c);
    const int maxZIndex = static_cast<int>(image.size()) - 1;
    const float maxR = std::max({static_cast<float>(a), static_cast<float>(b), static_cast<float>(c)});

    const int minX = std::max(0, static_cast<int>(std::floor(_position.x - maxR)));
    const int maxX = std::min(image[0].cols - 1, static_cast<int>(std::ceil(_position.x + maxR)));
    const int minY = std::max(0, static_cast<int>(std::floor(_position.y - maxR)));
    const int maxY = std::min(image[0].rows - 1, static_cast<int>(std::ceil(_position.y + maxR)));
    const int minZ = std::max(0, static_cast<int>(std::floor(_position.z - maxR)));
    const int maxZ = std::min(maxZIndex, static_cast<int>(std::ceil(_position.z + maxR)));

    std::vector<float> brightnessValues;
    scanEllipsoidVolume(
        image, minX, maxX, minY, maxY, minZ, maxZ, _position,
        R_T, invA2, invB2, invC2,
        [&](int /*x*/, int /*y*/, int /*z*/, float pixel, double val) {
            if (val <= 1.0) {
                brightnessValues.push_back(pixel);
            }
        });

    if (brightnessValues.empty()) {
        return _brightness;
    }

    const std::size_t keepCount = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(brightnessValues.size() * clampedTopPercentile)));
    const auto keepBegin = brightnessValues.end() - static_cast<std::ptrdiff_t>(keepCount);
    std::nth_element(brightnessValues.begin(), keepBegin, brightnessValues.end());

    double sum = 0.0;
    for (auto it = keepBegin; it != brightnessValues.end(); ++it) {
        sum += *it;
    }

    return static_cast<float>(sum / static_cast<double>(keepCount));
}

std::pair<float, float> Ellipsoid::measureBrightnessStats(const std::vector<cv::Mat> &image) const
{
    if (image.empty()) {
        return {_brightness, 0.0f};
    }

    std::array<double, 9> R_T;
    generateInverseRotationMatrix(R_T);

    const double invA2 = 1.0 / (a * a);
    const double invB2 = 1.0 / (b * b);
    const double invC2 = 1.0 / (c * c);
    const int maxZIndex = static_cast<int>(image.size()) - 1;
    const float maxR = std::max({static_cast<float>(a), static_cast<float>(b), static_cast<float>(c)});

    const int minX = std::max(0, static_cast<int>(std::floor(_position.x - maxR)));
    const int maxX = std::min(image[0].cols - 1, static_cast<int>(std::ceil(_position.x + maxR)));
    const int minY = std::max(0, static_cast<int>(std::floor(_position.y - maxR)));
    const int maxY = std::min(image[0].rows - 1, static_cast<int>(std::ceil(_position.y + maxR)));
    const int minZ = std::max(0, static_cast<int>(std::floor(_position.z - maxR)));
    const int maxZ = std::min(maxZIndex, static_cast<int>(std::ceil(_position.z + maxR)));

    double sum = 0.0;
    double sumSquares = 0.0;
    int count = 0;
    scanEllipsoidVolume(
        image, minX, maxX, minY, maxY, minZ, maxZ, _position,
        R_T, invA2, invB2, invC2,
        [&](int /*x*/, int /*y*/, int /*z*/, float pixel, double val) {
            if (val <= 1.0) {
                sum += pixel;
                sumSquares += static_cast<double>(pixel) * pixel;
                count++;
            }
        });

    if (count == 0) {
        return {_brightness, 0.0f};
    }
    const double mean = sum / count;
    const double variance = std::max(0.0, (sumSquares / count) - mean * mean);
    return {static_cast<float>(mean), static_cast<float>(std::sqrt(variance))};
}


// ---- ROTATION-AWARE drawOutline() ----
// Scans pixels and marks those near the spheroid surface.
void Ellipsoid::drawOutline(cv::Mat &image, float color, float z) const {
    int minX, maxX, minY, maxY;
    if (!computeSliceBounds(image, z, minX, maxX, minY, maxY)) return;

    std::array<double, 9> R_T;
    generateInverseRotationMatrix(R_T);

    const double invA2 = 1.0 / (a * a);
    const double invB2 = 1.0 / (b * b);
    const double invC2 = 1.0 / (c * c);

    const int channels = image.channels();
    const float outlineIntensity = std::clamp(color, 0.0f, 1.0f);

    if (channels == 1) {
        scanEllipsoidSlice(
            minX, maxX, minY, maxY,
            _position, static_cast<double>(z),
            R_T, invA2, invB2, invC2,
            [&](int x, int y, double val) {
                if (val >= 0.95 && val <= 1.05) {
                    image.at<float>(y, x) = outlineIntensity;
                }
            });
    } else if (channels == 3) {
        const cv::Vec3f drawColor(outlineIntensity, outlineIntensity, outlineIntensity);
        scanEllipsoidSlice(
            minX, maxX, minY, maxY,
            _position, static_cast<double>(z),
            R_T, invA2, invB2, invC2,
            [&](int x, int y, double val) {
                if (val >= 0.95 && val <= 1.05) {
                    image.at<cv::Vec3f>(y, x) = drawColor;
                }
            });
    }
}

[[nodiscard]] Ellipsoid Ellipsoid::getPerturbedCell(PerturbDirections *directions,
                                                    float positionScale) const {
    const PerturbParams::Sample aRadiusSample = _aRadiusPerturbParams.samplePerturb();
    const PerturbParams::Sample cRadiusSample = _cRadiusPerturbParams.samplePerturb();
    const PerturbParams::Sample bRadiusSample = _bRadiusPerturbParams.samplePerturb();
    const PerturbParams::Sample brightnessSample = _brightnessPerturbParams.samplePerturb();
    if (directions != nullptr) {
        directions->brightness = brightnessSample.direction;
        directions->aRadius = aRadiusSample.direction;
        directions->bRadius = bRadiusSample.direction;
        directions->cRadius = cRadiusSample.direction;
    }
    EllipsoidParams spheroidParams(
        _name,
        _position.x + positionScale * cellConfig.x.getPerturbOffset(),
        _position.y + positionScale * cellConfig.y.getPerturbOffset(),
        _position.z + positionScale * cellConfig.z.getPerturbOffset(),
        _major_radius + aRadiusSample.offset,
        _minor_radius + cRadiusSample.offset,
        static_cast<float>(_theta_x) + cellConfig.thetaX.getPerturbOffset(),
        static_cast<float>(_theta_y) + cellConfig.thetaY.getPerturbOffset(),
        static_cast<float>(_theta_z) + cellConfig.thetaZ.getPerturbOffset(),
        _brightness + brightnessSample.offset);
    spheroidParams.bRadius = static_cast<float>(_b_radius) + bRadiusSample.offset;
    Ellipsoid perturbedCell(spheroidParams);
    perturbedCell._aRadiusPerturbParams = _aRadiusPerturbParams;
    perturbedCell._bRadiusPerturbParams = _bRadiusPerturbParams;
    perturbedCell._cRadiusPerturbParams = _cRadiusPerturbParams;
    perturbedCell._brightnessPerturbParams = _brightnessPerturbParams;
    return perturbedCell;
}

bool Ellipsoid::isPointInsideEllipsoid(const cv::Point3f &worldPoint,
                                      float scaleFactor) const
{
    // Inverse-rotate the world-space displacement back into the cell's
    // local (upright) frame, then test the scaled ellipsoid inequality
    // (lx/sa)^2 + (ly/sb)^2 + (lz/sc)^2 <= 1.
    const double dx = static_cast<double>(worldPoint.x) - _position.x;
    const double dy = static_cast<double>(worldPoint.y) - _position.y;
    const double dz = static_cast<double>(worldPoint.z) - _position.z;
    double lx = 0.0, ly = 0.0, lz = 0.0;
    inverseRotatePoint(dx, dy, dz, lx, ly, lz);

    const double clampedScale = std::max(1e-3, static_cast<double>(scaleFactor));
    const double sa = a * clampedScale;
    const double sb = b * clampedScale;
    const double sc = c * clampedScale;
    if (sa <= 0.0 || sb <= 0.0 || sc <= 0.0) {
        return false;
    }
    const double fx = lx / sa;
    const double fy = ly / sb;
    const double fz = lz / sc;
    return (fx * fx + fy * fy + fz * fz) <= 1.0;
}



bool Ellipsoid::checkConstraints() const {
    const bool majorOk = (cellConfig.minARadius <= _major_radius) &&
                         (_major_radius <= cellConfig.maxARadius);
    const bool minorOk = (cellConfig.minCRadius <= _minor_radius) &&
                         (_minor_radius <= cellConfig.maxCRadius);
    const bool bConfigured = (cellConfig.maxBRadius > 0.0);
    const bool bOk = !bConfigured ||
        ((cellConfig.minBRadius <= _b_radius) && (_b_radius <= cellConfig.maxBRadius));
    return majorOk && minorOk && bOk;
}

EllipsoidParams Ellipsoid::getCellParams() const {
    EllipsoidParams params(_name, _position.x, _position.y, _position.z,
                          static_cast<float>(_major_radius), static_cast<float>(_minor_radius),
                          static_cast<float>(_theta_x), static_cast<float>(_theta_y), static_cast<float>(_theta_z),
                          _brightness);
    params.bRadius = static_cast<float>(_b_radius);
    return params;
}

// Vincent's getSplitCells() + duplicate checkConstraints/getCellParams
// removed during merge. See git history if needed.


[[nodiscard]] std::pair<std::vector<float>, std::vector<float>> Ellipsoid::calculateCorners() const {
    // Use max radius as conservative bound for rotated spheroid
    float maxR = std::max({static_cast<float>(a), static_cast<float>(b), static_cast<float>(c)});

    std::vector<float> min_corner = {static_cast<float>(_position.x) - maxR,
                                     static_cast<float>(_position.y) - maxR,
                                     static_cast<float>(_position.z) - maxR};

    std::vector<float> max_corner = {static_cast<float>(_position.x) + maxR,
                                     static_cast<float>(_position.y) + maxR,
                                     static_cast<float>(_position.z) + maxR};

    return std::make_pair(min_corner, max_corner);
}

std::pair<std::vector<float>, std::vector<float>> Ellipsoid::calculateMinimumBox(Ellipsoid &perturbed_cell) const {
    auto [cell1_min_corner, cell1_max_corner] = calculateCorners();
    auto [cell2_min_corner, cell2_max_corner] = perturbed_cell.calculateCorners();

    std::vector<float> min_corner, max_corner;
    for (int i = 0; i < 3; ++i)
    {
        min_corner.push_back(std::min(cell1_min_corner[i], cell2_min_corner[i]));
        max_corner.push_back(std::max(cell1_max_corner[i], cell2_max_corner[i]));
    }
    return std::make_pair(min_corner, max_corner);
}
