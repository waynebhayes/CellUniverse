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
          _major_radius(init_props.majorRadius), _minor_radius(init_props.minorRadius),
          _theta_x(init_props.theta_x), _theta_y(init_props.theta_y), _theta_z(init_props.theta_z),
          _brightness(init_props.brightness)
{
    _major_radius = std::fmax(_major_radius, cellConfig.minMajorRadius);
    _major_radius = std::fmin(_major_radius, cellConfig.maxMajorRadius);
    _minor_radius = std::fmax(_minor_radius, cellConfig.minMinorRadius);
    _minor_radius = std::fmin(_minor_radius, cellConfig.maxMinorRadius);

    if (_minor_radius > _major_radius) {
        _minor_radius = _major_radius;
    }
    _brightness = std::fmax(_brightness, static_cast<float>(cellConfig.minBrightness));
    _brightness = std::fmin(_brightness, static_cast<float>(cellConfig.maxBrightness));

    this->a = this->_major_radius;
    this->b = this->a; // oblate: a == b
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
    const float outlineIntensity = 0.25f;

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

[[nodiscard]] Spheroid Spheroid::getPerturbedCell() const {
    SpheroidParams spheroidParams(
        _name,
        _position.x + cellConfig.x.getPerturbOffset(),
        _position.y + cellConfig.y.getPerturbOffset(),
        _position.z + cellConfig.z.getPerturbOffset(),
        _major_radius + cellConfig.majorRadius.getPerturbOffset(),
        _minor_radius + cellConfig.minorRadius.getPerturbOffset(),
        _theta_x + cellConfig.thetaX.getPerturbOffset(),
        _theta_y + cellConfig.thetaY.getPerturbOffset(),
        _theta_z + cellConfig.thetaZ.getPerturbOffset(),
        _brightness + cellConfig.brightness.getPerturbOffset());
    return Spheroid(spheroidParams);
}

std::tuple<Spheroid, Spheroid, bool, float> Spheroid::getSplitCells(const std::vector<cv::Mat> &image, float z_scaling,
    float backgroundColor,
    const std::vector<cv::Point3f> &neighborCenters,
    float preOptMajorR, float preOptMinorR,
    float preOptX, float preOptY, float preOptZ,
    float splitSearchRadiusMultiplier,
    float splitMinorAxisAlignmentToleranceDegrees,
    float splitMinorAxisAlignmentFlatnessRatioThreshold,
    float splitMinorAxisAlignmentMinRadiusDisableThreshold) const {
    // Step 1: Get the bounding box, expanded for split detection.
    // Use pre-optimization radii if available (Phase 1 may collapse the cell).
    // Use pre-optimization position if available (Phase 1 may shift the cell
    // toward one blob, making PCA look spherical from the shifted center).

    float effA = (preOptMajorR > 0.0f) ? std::max(static_cast<float>(a), preOptMajorR) : static_cast<float>(a);
    float effB = (preOptMajorR > 0.0f) ? std::max(static_cast<float>(b), preOptMajorR) : static_cast<float>(b);
    float effC = (preOptMinorR > 0.0f) ? std::max(static_cast<float>(c), preOptMinorR) : static_cast<float>(c);
    float maxR = std::max({effA, effB, effC});

    const float clampedSplitSearchRadiusMultiplier = std::max(0.1f, splitSearchRadiusMultiplier);
    float splitSearchRadius = maxR * clampedSplitSearchRadiusMultiplier;

    // PCA center: use pre-opt position when available so PCA sees both blobs
    // from the original midpoint, not the Phase-1-shifted position.
    cv::Point3f pcaCenter = _position;
    if (preOptMajorR > 0.0f) {
        pcaCenter = cv::Point3f(preOptX, preOptY, preOptZ);
    }

    if (preOptMajorR > 0.0f) {
        std::cout << "[Split PreOpt] " << _name
                  << " current=(" << a << "," << b << "," << c << ")"
                  << " preOpt=(" << preOptMajorR << "," << preOptMajorR << "," << preOptMinorR << ")"
                  << " effective=(" << effA << "," << effB << "," << effC << ")"
                  << " searchRadius=" << splitSearchRadius
                  << " pcaCenter=(" << pcaCenter.x << "," << pcaCenter.y << "," << pcaCenter.z << ")"
                  << " currentPos=(" << _position.x << "," << _position.y << "," << _position.z << ")"
                  << '\n';
    }

    int minX = std::max(0, static_cast<int>(std::floor(pcaCenter.x - splitSearchRadius)));
    int maxX = std::min(static_cast<int>(image[0].cols) - 1, static_cast<int>(std::ceil(pcaCenter.x + splitSearchRadius)));

    int minY = std::max(0, static_cast<int>(std::floor(pcaCenter.y - splitSearchRadius)));
    int maxY = std::min(static_cast<int>(image[0].rows) - 1, static_cast<int>(std::ceil(pcaCenter.y + splitSearchRadius)));

    int minZ = std::max(0, static_cast<int>(std::floor(pcaCenter.z - splitSearchRadius)));
    int maxZ = std::min(static_cast<int>(image.size()) - 1, static_cast<int>(std::ceil(pcaCenter.z + splitSearchRadius)));

    // Step 2: Collect bright pixels from the REAL IMAGE near this cell.
    // Previously we used the spheroid boundary (geometric shape), but for
    // oblate spheroids (a == b) PCA of a symmetric shape gives a random
    // axis — useless for finding the split direction.
    // Instead, PCA on real-image bright pixels finds where the actual cell
    // mass is. If the cell has split, there are two clusters of bright
    // pixels and PCA finds the axis between them.

    // First pass: collect brightness values inside the spheroid boundary.
    // We use these to derive a percentile threshold for the brightest pixels.
    // Precompute constants for fast evaluation
    std::array<double, 9> R_T;
    generateInverseRotationMatrix(R_T);

    const double invA2 = 1.0 / (a * a);
    const double invB2 = 1.0 / (b * b);
    const double invC2 = 1.0 / (c * c);

    std::vector<float> insideBrightnessValues;
    insideBrightnessValues.reserve((maxX - minX + 1) * (maxY - minY + 1));

    scanSpheroidVolume(
        image, minX, maxX, minY, maxY, minZ, maxZ, _position,
        R_T, invA2, invB2, invC2,
        [&](int /*x*/, int /*y*/, int /*z*/, float pixel, double val) {
            if (val <= 1.0 && pixel >= backgroundColor) {
                insideBrightnessValues.push_back(pixel);
            }
        });

    const float brightestFraction =
        std::clamp(cellConfig.splitBrightestFraction, 0.0f, 1.0f);
    float pixelThreshold = 0.4f;
    if (!insideBrightnessValues.empty()) {
        if (brightestFraction <= 0.0f) {
            pixelThreshold = std::numeric_limits<float>::infinity();
        } else if (brightestFraction >= 1.0f) {
            pixelThreshold = *std::min_element(insideBrightnessValues.begin(), insideBrightnessValues.end());
        } else {
            const size_t total = insideBrightnessValues.size();
            const size_t selectedCount = std::max<size_t>(
                1, static_cast<size_t>(std::ceil(brightestFraction * static_cast<float>(total))));
            const size_t thresholdIndex = total - selectedCount;
            std::nth_element(
                insideBrightnessValues.begin(),
                insideBrightnessValues.begin() + static_cast<std::ptrdiff_t>(thresholdIndex),
                insideBrightnessValues.end());
            pixelThreshold = insideBrightnessValues[thresholdIndex];
        }
    }

    // Second pass: collect bright pixels within an expanded boundary (2.0x radius).
    // The expansion captures daughter blobs that may extend beyond the parent's boundary.
    // Only the top configurable fraction of brightest pixels are included.
    // Skip pixels closer to a neighbor cell than to this cell (prevents PCA contamination).
    // Store raw image-space coordinates for centroid-based daughter placement.
    std::vector<cv::Point3f> rawPoints;
    rawPoints.reserve((maxX - minX + 1) * (maxY - minY + 1));

    // No ellipsoidal boundary — the bounding box (3×maxR) and neighbor
    // exclusion naturally limit the search area. Removing the ellipsoidal
    // gate fixes the inconsistency where the bounding box used pre-opt
    // radii but the ellipsoid check used current (possibly collapsed) radii.

    scanSpheroidVolume(
        image, minX, maxX, minY, maxY, minZ, maxZ, _position,
        R_T, invA2, invB2, invC2,
        [&](double /*dx*/, double /*dy*/, double /*dz*/, int x, int y, int z, float pixel, double /*val*/) {
            if (pixel >= pixelThreshold && pixel >= backgroundColor) {
                // Skip pixel if it's closer to any neighbor than to pcaCenter
                // (use pcaCenter = pre-opt position so distance is measured from
                // the original cell midpoint, not the Phase-1-shifted position)
                    float selfDx = static_cast<float>(x) - pcaCenter.x;
                    float selfDy = static_cast<float>(y) - pcaCenter.y;
                    float selfDz = static_cast<float>(z) - pcaCenter.z;
                    float distSqToSelf = selfDx * selfDx + selfDy * selfDy + selfDz * selfDz;
                    bool closerToNeighbor = false;
                    float ndx, ndy, ndz, distSqToNeighbor;
                    for (const auto &nc : neighborCenters) {
                        ndx = static_cast<float>(x) - nc.x;
                        ndy = static_cast<float>(y) - nc.y;
                        ndz = static_cast<float>(z) - nc.z;
                        distSqToNeighbor = ndx * ndx + ndy * ndy + ndz * ndz;
                        if (distSqToNeighbor < distSqToSelf) {
                            closerToNeighbor = true;
                            break;
                        }
                    }
                    if (closerToNeighbor) return;

                    rawPoints.emplace_back(
                        static_cast<float>(x),
                        static_cast<float>(y),
                        static_cast<float>(z));
            }
        });

    // Step 2b: Iterative centroid refinement.
    // If pcaCenter drifted onto one blob (Phase 1 of previous frame dragged
    // it there), the neighbor exclusion clips the far blob. Compute the
    // centroid of collected pixels; if it differs significantly from
    // pcaCenter, re-collect using the centroid as the new center. This
    // shifts the Voronoi boundary to capture both blobs more evenly.
    if (rawPoints.size() >= 10) {
        cv::Point3f centroid(0, 0, 0);
        for (const auto &pt : rawPoints) {
            centroid.x += pt.x;
            centroid.y += pt.y;
            centroid.z += pt.z;
        }
        centroid *= (1.0f / rawPoints.size());

        float driftSq = (centroid.x - pcaCenter.x) * (centroid.x - pcaCenter.x)
                       + (centroid.y - pcaCenter.y) * (centroid.y - pcaCenter.y)
                       + (centroid.z - pcaCenter.z) * (centroid.z - pcaCenter.z);

        if (driftSq > 25.0f) { // > 5 pixels drift
            std::cout << "[PCA Recenter] " << _name
                      << " pcaCenter=(" << pcaCenter.x << "," << pcaCenter.y << "," << pcaCenter.z << ")"
                      << " centroid=(" << centroid.x << "," << centroid.y << "," << centroid.z << ")"
                      << " drift=" << std::sqrt(driftSq) << '\n';
            pcaCenter = centroid;
            rawPoints.clear();

            scanSpheroidVolume(
                image, minX, maxX, minY, maxY, minZ, maxZ, _position,
                R_T, invA2, invB2, invC2,
                [&](double /*dx*/, double /*dy*/, double /*dz*/, int x, int y, int z, float pixel, double /*val*/) {
                    if (pixel >= pixelThreshold && pixel >= backgroundColor) {
                        float selfDx = static_cast<float>(x) - pcaCenter.x;
                        float selfDy = static_cast<float>(y) - pcaCenter.y;
                        float selfDz = static_cast<float>(z) - pcaCenter.z;
                        float distSqToSelf = selfDx * selfDx + selfDy * selfDy + selfDz * selfDz;
                        bool closerToNeighbor = false;
                        for (const auto &nc : neighborCenters) {
                            float ndx = static_cast<float>(x) - nc.x;
                            float ndy = static_cast<float>(y) - nc.y;
                            float ndz = static_cast<float>(z) - nc.z;
                            float distSqToNeighbor = ndx * ndx + ndy * ndy + ndz * ndz;
                            if (distSqToNeighbor < distSqToSelf) {
                                closerToNeighbor = true;
                                break;
                            }
                        }
                        if (closerToNeighbor) return;
                        rawPoints.emplace_back(
                            static_cast<float>(x),
                            static_cast<float>(y),
                            static_cast<float>(z));
                    }
                });
        }
    }

    // Step 3: Run PCA with isotropic normalization.
    cv::Point3f split_axis;
    float elongationRatio = 1.0f;

    if (rawPoints.size() >= 3) {
        int n = static_cast<int>(rawPoints.size());
        cv::Mat data(n, 3, CV_32F);
        for (int i = 0; i < n; ++i) {
            data.at<float>(i, 0) = rawPoints[i].x - pcaCenter.x;
            data.at<float>(i, 1) = rawPoints[i].y - pcaCenter.y;
            data.at<float>(i, 2) = rawPoints[i].z - pcaCenter.z;
        }

        // Isotropic normalization: divide all axes by majorR (effA).
        // Per-axis normalization (effA, effB, effC) suppresses x-y splits
        // in pancaked cells because the small z-radius amplifies z-noise,
        // inflating all eigenvalues while keeping the ratio low.
        // Isotropic normalization preserves the true shape of the bright
        // pixel distribution regardless of cell aspect ratio.
        float normR = (effA > 1e-6f) ? effA : 1.0f;
        float invS = 1.0f / normR;

        for (int i = 0; i < n; ++i) data.at<float>(i, 0) *= invS;
        for (int i = 0; i < n; ++i) data.at<float>(i, 1) *= invS;
        for (int i = 0; i < n; ++i) data.at<float>(i, 2) *= invS;

        cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

        float lambda1 = pca.eigenvalues.at<float>(0);
        float lambda2 = pca.eigenvalues.at<float>(1);
        float lambda3 = pca.eigenvalues.at<float>(2);

        // First eigenvector (max variance direction) in normalized space
        cv::Point3f ev_norm(
            pca.eigenvectors.at<float>(0, 0),
            pca.eigenvectors.at<float>(0, 1),
            pca.eigenvectors.at<float>(0, 2));

        // With isotropic normalization, eigenvector direction is unchanged
        cv::Point3f ev_image(
            ev_norm.x * normR,
            ev_norm.y * normR,
            ev_norm.z * normR);

        // Normalize to unit vector
        float norm = std::sqrt(ev_image.x * ev_image.x +
                               ev_image.y * ev_image.y +
                               ev_image.z * ev_image.z);

        if (norm > 1e-6f) {
            split_axis = ev_image * (1.0f / norm);
        } else {
            thread_local std::mt19937 rng{std::random_device{}()};
            std::uniform_real_distribution<double> thetaDist(0.0, 2.0 * M_PI);
            std::uniform_real_distribution<double> phiDist(0.0, M_PI);
            double theta = thetaDist(rng);
            double phi = phiDist(rng);
            split_axis = cv::Point3f(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
        }

        elongationRatio = (effC > 1e-6f) ? (effA / effC) : 1.0f;
        std::cout << "[PCA Split] " << _name
                  << " elongation_ratio=" << elongationRatio
                  << " split_axis=(" << split_axis.x << ", " << split_axis.y << ", " << split_axis.z << ")"
                  << " eigenvalues=(" << lambda1 << ", " << lambda2 << ", " << lambda3 << ")"
                  << " num_bright_pixels=" << rawPoints.size()
                  << " normR=" << normR
                  << " threshold=" << pixelThreshold
                  << " selected_fraction=" << brightestFraction
                  << " inside_count=" << insideBrightnessValues.size()
                  << '\n';
    } else {
        std::cout << "[PCA Split] " << _name
                  << " only " << rawPoints.size() << " bright pixels found. Using random split axis."
                  << '\n';
        thread_local std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<double> thetaDist(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<double> phiDist(0.0, M_PI);
        double theta = thetaDist(rng);
        double phi = phiDist(rng);
        split_axis = cv::Point3f(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
    }

    double effMajorR = (preOptMajorR > 0.0f) ? std::max(_major_radius, static_cast<double>(preOptMajorR)) : _major_radius;
    double effMinorR = (preOptMinorR > 0.0f) ? std::max(_minor_radius, static_cast<double>(preOptMinorR)) : _minor_radius;
    const double cx = std::cos(_theta_x), sx = std::sin(_theta_x);
    const double cy = std::cos(_theta_y), sy = std::sin(_theta_y);
    const double cz = std::cos(_theta_z), sz = std::sin(_theta_z);
    const cv::Point3f localZAxis(
        static_cast<float>(cz * sy * cx + sz * sx),
        static_cast<float>(sz * sy * cx - cz * sx),
        static_cast<float>(cy * cx));
    const float flatnessRatio = (effMajorR > 1e-6) ? static_cast<float>(effMinorR / effMajorR) : 1.0f;
    const bool disableMinorAxisAlignmentForSmallCell =
        effMajorR < splitMinorAxisAlignmentMinRadiusDisableThreshold &&
        effMinorR < splitMinorAxisAlignmentMinRadiusDisableThreshold;
    const bool enforceMinorAxisAlignment =
        !disableMinorAxisAlignmentForSmallCell &&
        flatnessRatio <= splitMinorAxisAlignmentFlatnessRatioThreshold;
    if (enforceMinorAxisAlignment) {
        const float alignmentDot = std::clamp(std::abs(split_axis.dot(localZAxis)), 0.0f, 1.0f);
        const float alignmentAngleDegrees =
            static_cast<float>(std::acos(alignmentDot) * 180.0 / M_PI);
        if (alignmentAngleDegrees > splitMinorAxisAlignmentToleranceDegrees) {
            const float signedAlignment = split_axis.dot(localZAxis);
            split_axis = (signedAlignment >= 0.0f) ? localZAxis : (localZAxis * -1.0f);
            std::cout << "[Split Align] " << _name
                      << " flatness_ratio=" << flatnessRatio
                      << " threshold=" << splitMinorAxisAlignmentFlatnessRatioThreshold
                      << " minor-axis alignment angle=" << alignmentAngleDegrees
                      << " > tolerance=" << splitMinorAxisAlignmentToleranceDegrees
                      << " forcing split_axis to local_z=(" << split_axis.x << ", "
                      << split_axis.y << ", " << split_axis.z << ")"
                      << '\n';
        }
    } else if (disableMinorAxisAlignmentForSmallCell) {
        std::cout << "[Split Align Skip] " << _name
                  << " major_radius=" << effMajorR
                  << " minor_radius=" << effMinorR
                  << " disable_threshold=" << splitMinorAxisAlignmentMinRadiusDisableThreshold
                  << '\n';
    }

    // Step 4: Centroid-based daughter placement.
    // Project each bright pixel onto the split axis through the cell center.
    // Split into positive/negative groups and compute 3D centroids.
    // This places daughters where the real bright pixel clusters are,
    // instead of at a fixed offset that may not match the data.
    // Use pre-optimization radii for daughter sizes when available.
    // Phase 1 may collapse the parent (e.g. majorR 31→22 when fitting one cell
    // to two blobs), making daughters too small to produce meaningful cost improvement.
    // The pre-optimization size reflects the true biological cell size before collapse.
    // Use pre-opt radii for daughter sizing when available. Phase 1 may
    // collapse the parent (e.g. 28→21) when fitting one sphere to a two-cell
    // image, making current-radius daughters too small for meaningful cost
    // improvement. Pre-opt radii reflect the cell size before collapse.
    // (Previously this was blocked by daughter-daughter overlap gates, now removed.)
    double volumeScale = std::cbrt(0.5);
    double daughterMajorRadius = effMajorR * volumeScale;
    double daughterMinorRadius = effMinorR * volumeScale;

    cv::Point3f centroid1(0, 0, 0), centroid2(0, 0, 0);
    int count1 = 0, count2 = 0;

    for (const auto &pt : rawPoints) {
        const double dx = pt.x - pcaCenter.x;
        const double dy = pt.y - pcaCenter.y;
        const double dz = pt.z - pcaCenter.z;
        float projection = dx * split_axis.x + dy * split_axis.y + dz * split_axis.z;

        if (projection >= 0) {
            centroid1 += cv::Point3f(pt.x, pt.y, pt.z);
            count1++;
        } else {
            centroid2 += cv::Point3f(pt.x, pt.y, pt.z);
            count2++;
        }
    }

    cv::Point3f new_position1, new_position2;

    if (count1 > 0 && count2 > 0) {
        // Use actual centroid positions — this is where the bright pixel
        // clusters really are. Daughters may initially overlap each other,
        // which is expected since they split from a single parent. The
        // overlap check in trySplitCell skips the daughter-daughter pair
        // and burn-in handles separation.
        centroid1 *= (1.0f / count1);
        centroid2 *= (1.0f / count2);
        const cv::Point3f centroidOffset1 = centroid1 - pcaCenter;
        const cv::Point3f centroidOffset2 = centroid2 - pcaCenter;
        float projection1 = centroidOffset1.dot(split_axis);
        float projection2 = centroidOffset2.dot(split_axis);

        if (projection1 < projection2) {
            std::swap(projection1, projection2);
        }

        new_position1 = pcaCenter + split_axis * projection1;
        new_position2 = pcaCenter + split_axis * projection2;

        float sep = std::sqrt(
            (new_position1.x - new_position2.x) * (new_position1.x - new_position2.x) +
            (new_position1.y - new_position2.y) * (new_position1.y - new_position2.y) +
            (new_position1.z - new_position2.z) * (new_position1.z - new_position2.z));

        std::cout << "[Split Placement] centroid-based:"
                  << " c1=(" << new_position1.x << "," << new_position1.y << "," << new_position1.z << ")"
                  << " c2=(" << new_position2.x << "," << new_position2.y << "," << new_position2.z << ")"
                  << " sep=" << sep
                  << " line_axis=(" << split_axis.x << "," << split_axis.y << "," << split_axis.z << ")"
                  << " daughterMajorR=" << daughterMajorRadius
                  << " daughterMinorR=" << daughterMinorRadius << '\n';
    } else {
        // All pixels on one side — fall back to small offset along split axis
        float offset = static_cast<float>(daughterMajorRadius * 0.5);
        new_position1 = pcaCenter + split_axis * offset;
        new_position2 = pcaCenter - split_axis * offset;
        std::cout << "[Split Placement] one-sided (" << count1 << "/" << count2
                  << "), using fixed offset=" << offset << '\n';
    }


    //TODO use a batter way to represent the cell relationships

    // Inherit parent rotation angles and brightness
    Spheroid cell1(SpheroidParams(
        _name + "0", new_position1.x, new_position1.y, new_position1.z,
        daughterMajorRadius, daughterMinorRadius, _theta_x, _theta_y, _theta_z, _brightness));
    Spheroid cell2(SpheroidParams(
        _name + "1", new_position2.x, new_position2.y, new_position2.z,
        daughterMajorRadius, daughterMinorRadius, _theta_x, _theta_y, _theta_z, _brightness));

    bool constraints = cell1.checkConstraints() && cell2.checkConstraints();
    return std::make_tuple(Spheroid(cell1), Spheroid(cell2), constraints, elongationRatio);
}

bool Spheroid::checkConstraints() const {
    return (cellConfig.minMajorRadius <= _major_radius) && (_major_radius <= cellConfig.maxMajorRadius) && (cellConfig.minMinorRadius <= _minor_radius) && (_minor_radius <= cellConfig.maxMinorRadius);
}

SpheroidParams Spheroid::getCellParams() const {
    return SpheroidParams(_name, _position.x, _position.y, _position.z, _major_radius, _minor_radius, _theta_x, _theta_y, _theta_z, _brightness);
}

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
