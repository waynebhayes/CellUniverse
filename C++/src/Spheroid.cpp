#include "../includes/Spheroid.hpp"
#include <limits>
#include <random>
#include <type_traits>
// #include <iostream>

namespace {
template <typename>
inline constexpr bool alwaysFalseV = false;

template <typename Visitor>
void scanSpheroidSlice(
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
void scanSpheroidVolume(
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
                                  "scanSpheroidVolume visitor must be invocable with either "
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

SpheroidConfig Spheroid::cellConfig = SpheroidConfig();

// ---- INVERSE ROTATION ----
// Transforms a world-space displacement vector back into the spheroid's 
// local (upright) coordinate frame. Since the forward rotation is 
// R_total = Rz * Ry * Rx, the inverse is R_total^T = Rx^T * Ry^T * Rz^T.
// We undo z first, then y, then x.
void Spheroid::inverseRotatePoint(double dx, double dy, double dz,
                                   double &lx, double &ly, double &lz) const 
{
    std::array<double, 9> R_T;
    generateInverseRotationMatrix(R_T);
    lx = R_T[0] * dx + R_T[1] * dy + R_T[2] * dz;
    ly = R_T[3] * dx + R_T[4] * dy + R_T[5] * dz;
    lz = R_T[6] * dx + R_T[7] * dy + R_T[8] * dz;
}

void Spheroid::worldLongAxis(cv::Point3f &dir, float &length) const
{
    // Pick whichever of (a, b, c) is largest and rotate the corresponding
    // local unit vector out to world space using the forward rotation
    // R = Rz * Ry * Rx. Since R_T in generateInverseRotationMatrix is R^T
    // stored row-major, column-i of R (which maps local axis i to world) is
    // read out of R_T as (R_T[i], R_T[3 + i], R_T[6 + i]).
    std::array<double, 9> R_T{};
    generateInverseRotationMatrix(R_T);

    int longestAxis = 0; // 0 = a (local x), 1 = b (local y), 2 = c (local z)
    double longestValue = a;
    if (b > longestValue) { longestAxis = 1; longestValue = b; }
    if (c > longestValue) { longestAxis = 2; longestValue = c; }

    // Column `longestAxis` of R = (R_T[longestAxis], R_T[3 + longestAxis], R_T[6 + longestAxis]).
    const double dx = R_T[longestAxis];
    const double dy = R_T[3 + longestAxis];
    const double dz = R_T[6 + longestAxis];

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
    length = static_cast<float>(longestValue);
}

void Spheroid::generateInverseRotationMatrix(std::array<double, 9> &R_T) const {
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

bool Spheroid::computeSliceBounds(const cv::Mat &image, float z,
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
Spheroid::Spheroid(const SpheroidParams &init_props)
: _name(init_props.name), _position{init_props.x, init_props.y, init_props.z},
          _major_radius(init_props.majorRadius),
          _b_radius(init_props.bRadius > 0.0f ? init_props.bRadius : init_props.majorRadius),
          _minor_radius(init_props.minorRadius),
          _theta_x(init_props.theta_x), _theta_y(init_props.theta_y), _theta_z(init_props.theta_z),
          _brightness(init_props.brightness),
          _majorRadiusPerturbParams(cellConfig.majorRadius),
          _minorRadiusPerturbParams(cellConfig.minorRadius),
          _abRatioPerturbParams(cellConfig.bRadius),
          _brightnessPerturbParams(cellConfig.brightness)
{
    _major_radius = std::fmax(_major_radius, cellConfig.minMajorRadius);
    _major_radius = std::fmin(_major_radius, cellConfig.maxMajorRadius);
    _minor_radius = std::fmax(_minor_radius, cellConfig.minMinorRadius);
    _minor_radius = std::fmin(_minor_radius, cellConfig.maxMinorRadius);
    // Triaxial b-axis is clamped to [minBRadius, maxBRadius] when those bounds
    // are configured. If they are not set (still zero), fall back to the same
    // bounds as the a-axis so an unconfigured run still behaves near-oblate.
    const double minBR = (cellConfig.maxBRadius > 0.0)
        ? cellConfig.minBRadius : cellConfig.minMajorRadius;
    const double maxBR = (cellConfig.maxBRadius > 0.0)
        ? cellConfig.maxBRadius : cellConfig.maxMajorRadius;
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
        throw std::invalid_argument("Spheroid radii must be positive");
    }

    // No matrix construction needed — draw() uses analytic inverse rotation
}

cv::Point3f Spheroid::get_center() const {
    return _position; // x, y, z position
}

void Spheroid::print() const {
    std::cout << "Spheroid: " << _name
              << " pos=(" << _position.x << ", " << _position.y << ", " << _position.z << ")"
              << " a=" << a << " b=" << b << " c=" << c
              << " theta=(" << _theta_x << ", " << _theta_y << ", " << _theta_z << ")"
              << '\n';
}

void Spheroid::setBrightness(float brightness)
{
    _brightness = std::clamp(brightness,
                             static_cast<float>(cellConfig.minBrightness),
                             static_cast<float>(cellConfig.maxBrightness));
}


void Spheroid::setMajorRadiusPerturbProbabilities(float increaseProbability, float decreaseProbability)
{
    const float clampedIncrease = std::clamp(increaseProbability, 0.0f, 1.0f);
    _majorRadiusPerturbParams.increase_prob = clampedIncrease;
    _majorRadiusPerturbParams.decrease_prob =
        std::clamp(decreaseProbability, 0.0f, std::max(0.0f, 1.0f - clampedIncrease));
}

void Spheroid::setMinorRadiusPerturbProbabilities(float increaseProbability, float decreaseProbability)
{
    const float clampedIncrease = std::clamp(increaseProbability, 0.0f, 1.0f);
    _minorRadiusPerturbParams.increase_prob = clampedIncrease;
    _minorRadiusPerturbParams.decrease_prob =
        std::clamp(decreaseProbability, 0.0f, std::max(0.0f, 1.0f - clampedIncrease));
}

void Spheroid::setABRatioPerturbProbabilities(float increaseProbability, float decreaseProbability)
{
    const float clampedIncrease = std::clamp(increaseProbability, 0.0f, 1.0f);
    _abRatioPerturbParams.increase_prob = clampedIncrease;
    _abRatioPerturbParams.decrease_prob =
        std::clamp(decreaseProbability, 0.0f, std::max(0.0f, 1.0f - clampedIncrease));
}

void Spheroid::setBrightnessPerturbProbabilities(float increaseProbability, float decreaseProbability)
{
    const float clampedIncrease = std::clamp(increaseProbability, 0.0f, 1.0f);
    _brightnessPerturbParams.increase_prob = clampedIncrease;
    _brightnessPerturbParams.decrease_prob =
        std::clamp(decreaseProbability, 0.0f, std::max(0.0f, 1.0f - clampedIncrease));
}

void Spheroid::blendBrightnessPerturbProbabilitiesWithConfig(float trust)
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

void Spheroid::blendAdaptivePerturbProbabilitiesWithConfig(float brightnessTrust,
                                                           float majorRadiusTrust,
                                                           float minorRadiusTrust,
                                                           float abRatioTrust)
{
    const float clampedMajorTrust = std::clamp(majorRadiusTrust, 0.0f, 1.0f);
    const float clampedMinorTrust = std::clamp(minorRadiusTrust, 0.0f, 1.0f);
    const float clampedABTrust = std::clamp(abRatioTrust, 0.0f, 1.0f);

    const float baseMajorIncrease =
        std::clamp(cellConfig.majorRadius.increase_prob >= 0.0f ? cellConfig.majorRadius.increase_prob : 0.0f,
                   0.0f, 1.0f);
    const float baseMajorDecrease =
        std::clamp(cellConfig.majorRadius.decrease_prob >= 0.0f ? cellConfig.majorRadius.decrease_prob : 0.0f,
                   0.0f, 1.0f);
    setMajorRadiusPerturbProbabilities(
        baseMajorIncrease * (1.0f - clampedMajorTrust) + getMajorRadiusIncreaseProbability() * clampedMajorTrust,
        baseMajorDecrease * (1.0f - clampedMajorTrust) + getMajorRadiusDecreaseProbability() * clampedMajorTrust);

    const float baseMinorIncrease =
        std::clamp(cellConfig.minorRadius.increase_prob >= 0.0f ? cellConfig.minorRadius.increase_prob : 0.0f,
                   0.0f, 1.0f);
    const float baseMinorDecrease =
        std::clamp(cellConfig.minorRadius.decrease_prob >= 0.0f ? cellConfig.minorRadius.decrease_prob : 0.0f,
                   0.0f, 1.0f);
    setMinorRadiusPerturbProbabilities(
        baseMinorIncrease * (1.0f - clampedMinorTrust) + getMinorRadiusIncreaseProbability() * clampedMinorTrust,
        baseMinorDecrease * (1.0f - clampedMinorTrust) + getMinorRadiusDecreaseProbability() * clampedMinorTrust);

    const float baseABIncrease =
        std::clamp(cellConfig.abRatio.increase_prob >= 0.0f ? cellConfig.abRatio.increase_prob : 0.0f,
                   0.0f, 1.0f);
    const float baseABDecrease =
        std::clamp(cellConfig.abRatio.decrease_prob >= 0.0f ? cellConfig.abRatio.decrease_prob : 0.0f,
                   0.0f, 1.0f);
    setABRatioPerturbProbabilities(
        baseABIncrease * (1.0f - clampedABTrust) + getABRatioIncreaseProbability() * clampedABTrust,
        baseABDecrease * (1.0f - clampedABTrust) + getABRatioDecreaseProbability() * clampedABTrust);

    blendBrightnessPerturbProbabilitiesWithConfig(brightnessTrust);
}

void Spheroid::adjustMajorRadiusPerturbProbability(int direction, float delta)
{
    _majorRadiusPerturbParams.adjustSignedProbability(direction, delta);
}

void Spheroid::adjustMinorRadiusPerturbProbability(int direction, float delta)
{
    _minorRadiusPerturbParams.adjustSignedProbability(direction, delta);
}

void Spheroid::adjustABRatioPerturbProbability(int direction, float delta)
{
    _abRatioPerturbParams.adjustSignedProbability(direction, delta);
}

void Spheroid::adjustBrightnessPerturbProbability(int direction, float delta)
{
    _brightnessPerturbParams.adjustSignedProbability(direction, delta);
}

// ---- ROTATION-AWARE draw() ----
// Instead of checking a voxel matrix, we analytically test each pixel
// against the rotated spheroid by inverse-transforming back to local coords.
void Spheroid::draw(cv::Mat &image, const SimulationConfig &simulationConfig, float z) const{
    int minX, maxX, minY, maxY;
    if (!computeSliceBounds(image, z, minX, maxX, minY, maxY)) return;

    std::array<double, 9> R_T;
    generateInverseRotationMatrix(R_T);

    const double invA2 = 1.0 / (a * a);
    const double invB2 = 1.0 / (b * b);
    const double invC2 = 1.0 / (c * c);

    scanSpheroidSlice(
        minX, maxX, minY, maxY,
        _position, static_cast<double>(z),
        R_T, invA2, invB2, invC2,
        [&](int x, int y, double val) {
            if (val <= 1.0) {
                image.at<float>(y, x) = _brightness;
            }
        });
}

float Spheroid::measureMeanBrightness(const std::vector<cv::Mat> &image) const
{
    return measureBrightnessStats(image).first;
}

std::pair<float, float> Spheroid::measureBrightnessStats(const std::vector<cv::Mat> &image) const
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
    scanSpheroidVolume(
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
void Spheroid::drawOutline(cv::Mat &image, float color, float z) const {
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
        scanSpheroidSlice(
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
        scanSpheroidSlice(
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

[[nodiscard]] Spheroid Spheroid::getPerturbedCell(PerturbDirections *directions) const {
    const PerturbParams::Sample majorRadiusSample = _majorRadiusPerturbParams.samplePerturb();
    const PerturbParams::Sample minorRadiusSample = _minorRadiusPerturbParams.samplePerturb();
    const PerturbParams::Sample abRatioSample = _abRatioPerturbParams.samplePerturb();
    const PerturbParams::Sample brightnessSample = _brightnessPerturbParams.samplePerturb();
    if (directions != nullptr) {
        directions->brightness = brightnessSample.direction;
        directions->majorRadius = majorRadiusSample.direction;
        directions->minorRadius = minorRadiusSample.direction;
        directions->abRatio = abRatioSample.direction;
    }
    SpheroidParams spheroidParams(
        _name,
        _position.x + cellConfig.x.getPerturbOffset(),
        _position.y + cellConfig.y.getPerturbOffset(),
        _position.z + cellConfig.z.getPerturbOffset(),
        _major_radius + majorRadiusSample.offset,
        _minor_radius + minorRadiusSample.offset,
        static_cast<float>(_theta_x) + cellConfig.thetaX.getPerturbOffset(),
        static_cast<float>(_theta_y) + cellConfig.thetaY.getPerturbOffset(),
        static_cast<float>(_theta_z) + cellConfig.thetaZ.getPerturbOffset(),
        _brightness + brightnessSample.offset);
    // Triaxial b-axis perturbation via the adaptive abRatio PerturbParams
    // (which tracks bRadius in our triaxial model). Applied as a separate
    // field assignment because SpheroidParams constructors don't include bRadius.
    spheroidParams.bRadius = static_cast<float>(_b_radius) + abRatioSample.offset;
    Spheroid perturbedCell(spheroidParams);
    perturbedCell._majorRadiusPerturbParams = _majorRadiusPerturbParams;
    perturbedCell._minorRadiusPerturbParams = _minorRadiusPerturbParams;
    perturbedCell._abRatioPerturbParams = _abRatioPerturbParams;
    perturbedCell._brightnessPerturbParams = _brightnessPerturbParams;
    return perturbedCell;
}

bool Spheroid::isPointInsideEllipsoid(const cv::Point3f &worldPoint,
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



bool Spheroid::checkConstraints() const {
    const bool majorOk = (cellConfig.minMajorRadius <= _major_radius) &&
                         (_major_radius <= cellConfig.maxMajorRadius);
    const bool minorOk = (cellConfig.minMinorRadius <= _minor_radius) &&
                         (_minor_radius <= cellConfig.maxMinorRadius);
    const bool bConfigured = (cellConfig.maxBRadius > 0.0);
    const bool bOk = !bConfigured ||
        ((cellConfig.minBRadius <= _b_radius) && (_b_radius <= cellConfig.maxBRadius));
    return majorOk && minorOk && bOk;
}

SpheroidParams Spheroid::getCellParams() const {
    SpheroidParams params(_name, _position.x, _position.y, _position.z,
                          static_cast<float>(_major_radius), static_cast<float>(_minor_radius),
                          static_cast<float>(_theta_x), static_cast<float>(_theta_y), static_cast<float>(_theta_z),
                          _brightness);
    params.bRadius = static_cast<float>(_b_radius);
    return params;
}

// Vincent's getSplitCells() + duplicate checkConstraints/getCellParams
// removed during merge. See git history if needed.


[[nodiscard]] std::pair<std::vector<float>, std::vector<float>> Spheroid::calculateCorners() const {
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

std::pair<std::vector<float>, std::vector<float>> Spheroid::calculateMinimumBox(Spheroid &perturbed_cell) const {
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
