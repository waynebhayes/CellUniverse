#include "../../includes/Spheroid.hpp"
#include <random>
// #include <iostream>

SpheroidParams Spheroid::paramClass = SpheroidParams();
SpheroidConfig Spheroid::cellConfig = SpheroidConfig();



static double _get_magnitude(std::vector<double> vec){
    return std::sqrt((vec[0]*vec[0])+(vec[1]*vec[1])+(vec[2]*vec[2]));
}

// ---- INVERSE ROTATION ----
// Transforms a world-space displacement vector back into the spheroid's 
// local (upright) coordinate frame. Since the forward rotation is 
// R_total = Rz * Ry * Rx, the inverse is R_total^T = Rx^T * Ry^T * Rz^T.
// We undo z first, then y, then x.
void Spheroid::inverseRotatePoint(double dx, double dy, double dz,
                                   double &lx, double &ly, double &lz) const 
{
    // Undo rotation around z (apply Rz^T)
    double cos_z = std::cos(_theta_z);
    double sin_z = std::sin(_theta_z);
    double tx =  cos_z * dx + sin_z * dy;
    double ty = -sin_z * dx + cos_z * dy;
    double tz = dz;

    // Undo rotation around y (apply Ry^T)
    double cos_y = std::cos(_theta_y);
    double sin_y = std::sin(_theta_y);
    double ux =  cos_y * tx + (-sin_y) * tz;
    double uy = ty;
    double uz =  sin_y * tx + cos_y * tz;

    // Undo rotation around x (apply Rx^T)
    double cos_x = std::cos(_theta_x);
    double sin_x = std::sin(_theta_x);
    lx = ux;
    ly =  cos_x * uy + sin_x * uz;
    lz = -sin_x * uy + cos_x * uz;
}

// ---- SIMPLIFIED CONSTRUCTOR ----
// No voxel matrix construction needed — drawing is done analytically.
Spheroid::Spheroid(const SpheroidParams &init_props) 
: _name(init_props.name), _position{init_props.x, init_props.y, init_props.z},
          _major_radius(init_props.majorRadius), _minor_radius(init_props.minorRadius),  // BUG FIX: was majorRadius
          _rotation(0), 
          _theta_x(init_props.theta_x), _theta_y(init_props.theta_y), _theta_z(init_props.theta_z),
          dormant(false)
{
    _major_radius = std::fmax(_major_radius, cellConfig.minMajorRadius);
    _major_radius = std::fmin(_major_radius, cellConfig.maxMajorRadius);
    _minor_radius = std::fmax(_minor_radius, cellConfig.minMinorRadius);
    _minor_radius = std::fmin(_minor_radius, cellConfig.maxMinorRadius);

    if (_minor_radius > _major_radius) {
        _minor_radius = _major_radius;
    }

    this->a = this->_major_radius;
    this->b = this->a; // oblate: a == b
    this->c = this->_minor_radius;

    if (a <= 0 || b <= 0 || c <= 0) {
        throw std::invalid_argument("Spheroid radii must be positive");
    }

    // No matrix construction needed — draw() uses analytic inverse rotation
    // // DEBUG: Print to verify values — remove after confirming correctness
    // std::cout << "[Spheroid INIT] " << _name 
    //         << " a=" << a << " b=" << b << " c=" << c
    //         << " theta=(" << _theta_x << ", " << _theta_y << ", " << _theta_z << ")"
    //         << std::endl;
}

float Spheroid::major_magnitude(){
    return _get_magnitude(_x_vec); // semi-major radius
}

float Spheroid::minor_magnitude(){
    return _get_magnitude(_z_vec); // semi-minor radius
}

cv::Point3f Spheroid::get_center() const {
    return _position; // x, y, z position
}

void Spheroid::print() const {
    std::cout << "Spheroid: " << _name 
              << " pos=(" << _position.x << ", " << _position.y << ", " << _position.z << ")"
              << " a=" << a << " b=" << b << " c=" << c
              << " theta=(" << _theta_x << ", " << _theta_y << ", " << _theta_z << ")"
              << std::endl;
}

int Spheroid::get_matrix_size(){
    // Return approximate diameter for backward compatibility
    return (int)std::ceil(2 * std::max({a, b, c}));
}

std::vector<double> Spheroid::getShapeAt(double z) const
{
    // need to find axes and radii using z value
    std::vector<double> vec = {_major_radius, _minor_radius, _position.x, _position.y};
    return vec;
}

// ---- ROTATION-AWARE draw() ----
// Instead of checking a voxel matrix, we analytically test each pixel
// against the rotated spheroid by inverse-transforming back to local coords.
void Spheroid::draw(cv::Mat &image, SimulationConfig simulationConfig, cv::Mat *cellMap, float z) const{
    if (dormant)
    {
        return;
    }

    // TEMPORARY DEBUG — remove after testing
    // std::cout << "[DRAW] " << _name 
    //         << " a=" << a << " b=" << b << " c=" << c
    //         << " theta_y=" << _theta_y 
    //         << " z=" << z << " pos_z=" << _position.z << std::endl;

    // Conservative bounding box: the max reach in any direction after rotation
    float maxR = std::max({(float)a, (float)b, (float)c});

    // Quick check: does this z-slice potentially intersect?
    if (std::abs(z - _position.z) > maxR) return;

    int minX = std::max(0, (int)std::floor(_position.x - maxR));
    int maxX = std::min(image.cols - 1, (int)std::ceil(_position.x + maxR));
    int minY = std::max(0, (int)std::floor(_position.y - maxR));
    int maxY = std::min(image.rows - 1, (int)std::ceil(_position.y + maxR));

    for (int i = minX; i <= maxX; ++i) {
        for (int j = minY; j <= maxY; ++j) {
            // Displacement from spheroid center in world space
            double dx = (double)i - _position.x;
            double dy = (double)j - _position.y;
            double dz = (double)z - _position.z;

            // Transform to local (upright) coordinate frame
            double lx, ly, lz;
            inverseRotatePoint(dx, dy, dz, lx, ly, lz);

            // Check against upright spheroid equation: x²/a² + y²/b² + z²/c² <= 1
            double val = (lx * lx) / (a * a) 
                       + (ly * ly) / (b * b) 
                       + (lz * lz) / (c * c);

            if (val <= 1.0) {
                image.at<float>(j, i) = simulationConfig.cell_color;
            }
        }
    }
}

// ---- ROTATION-AWARE drawOutline() ----
// Scans pixels and marks those near the spheroid surface.
void Spheroid::drawOutline(cv::Mat &image, float color, float z) const {
    if (dormant) return;

    float maxR = std::max({(float)a, (float)b, (float)c});
    if (std::abs(z - _position.z) > maxR) return;

    int minX = std::max(0, (int)std::floor(_position.x - maxR));
    int maxX = std::min(image.cols - 1, (int)std::ceil(_position.x + maxR));
    int minY = std::max(0, (int)std::floor(_position.y - maxR));
    int maxY = std::min(image.rows - 1, (int)std::ceil(_position.y + maxR));

    for (int i = minX; i <= maxX; ++i) {
        for (int j = minY; j <= maxY; ++j) {
            double dx = (double)i - _position.x;
            double dy = (double)j - _position.y;
            double dz = (double)z - _position.z;

            double lx, ly, lz;
            inverseRotatePoint(dx, dy, dz, lx, ly, lz);

            double val = (lx * lx) / (a * a) 
                       + (ly * ly) / (b * b) 
                       + (lz * lz) / (c * c);

            // Outline: only draw pixels near the surface
            if (val >= 0.95 && val <= 1.05) {
                if (image.channels() == 1) {
                    image.at<float>(j, i) = color;
                } else if (image.channels() == 3) {
                    // BGR float image (from cvtColor of CV_32F grayscale)
                    image.at<cv::Vec3f>(j, i) = cv::Vec3f(0, color, 0);
                }
            }
        }
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
        _theta_z + cellConfig.thetaZ.getPerturbOffset());
    return Spheroid(spheroidParams);
}

Spheroid Spheroid::getParameterizedCell(std::unordered_map<std::string, float> params) const {
    float xOffset = params["x"];
    float yOffset = params["y"];
    float zOffset = params["z"];
    float majorRadiusOffset = params["majorRadius"];
    float minorRadiusOffset = params["minorRadius"];
    float thetaXOffset = params["thetaX"];
    float thetaYOffset = params["thetaY"];
    float thetaZOffset = params["thetaZ"];

    if (params.empty())
    {
        xOffset = Spheroid::cellConfig.x.getPerturbOffset();
        yOffset = Spheroid::cellConfig.y.getPerturbOffset();
        zOffset = Spheroid::cellConfig.z.getPerturbOffset();
        majorRadiusOffset = Spheroid::cellConfig.majorRadius.getPerturbOffset();
        minorRadiusOffset = Spheroid::cellConfig.minorRadius.getPerturbOffset();
        thetaXOffset = Spheroid::cellConfig.thetaX.getPerturbOffset();
        thetaYOffset = Spheroid::cellConfig.thetaY.getPerturbOffset();
        thetaZOffset = Spheroid::cellConfig.thetaZ.getPerturbOffset();
    }

    float newMajorRadius = fmin(fmax(Spheroid::cellConfig.minMajorRadius, _major_radius + majorRadiusOffset), Spheroid::cellConfig.maxMajorRadius);
    float newMinorRadius = fmin(fmax(Spheroid::cellConfig.minMinorRadius, _minor_radius + minorRadiusOffset), Spheroid::cellConfig.maxMinorRadius);
    SpheroidParams spheroidParams(
        _name,
        _position.x + xOffset,
        _position.y + yOffset,
        _position.z + zOffset,
        newMajorRadius,
        newMinorRadius,
        _theta_x + thetaXOffset,
        _theta_y + thetaYOffset,
        _theta_z + thetaZOffset);
    return Spheroid(spheroidParams);
}

std::tuple<Spheroid, Spheroid, bool> Spheroid::getSplitCells(const std::vector<cv::Mat> &image, float z_scaling) const {
    // Step 1: Get the bounding box
    auto [min_corner, max_corner] = calculateCorners();

    int minX = std::max(0, static_cast<int>(std::floor(min_corner[0])));
    int maxX = std::min(static_cast<int>(image[0].cols) - 1, static_cast<int>(std::ceil(max_corner[0])));
    int minY = std::max(0, static_cast<int>(std::floor(min_corner[1])));
    int maxY = std::min(static_cast<int>(image[0].rows) - 1, static_cast<int>(std::ceil(max_corner[1])));
    int minZ = std::max(0, static_cast<int>(std::floor(min_corner[2])));
    int maxZ = std::min(static_cast<int>(image.size()) - 1, static_cast<int>(std::ceil(max_corner[2])));

    // Step 2: Collect bright pixels from the REAL IMAGE near this cell.
    // Previously we used the spheroid boundary (geometric shape), but for
    // oblate spheroids (a == b) PCA of a symmetric shape gives a random
    // axis — useless for finding the split direction.
    // Instead, PCA on real-image bright pixels finds where the actual cell
    // mass is. If the cell has split, there are two clusters of bright
    // pixels and PCA finds the axis between them.

    // First pass: compute mean brightness inside the spheroid boundary
    double brightnessSum = 0;
    int brightnessCount = 0;
    for (int z = minZ; z <= maxZ; ++z) {
        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                double dx = (double)x - _position.x;
                double dy = (double)y - _position.y;
                double dz = (double)z - _position.z;
                double lx, ly, lz;
                inverseRotatePoint(dx, dy, dz, lx, ly, lz);
                double val = (lx * lx) / (a * a) + (ly * ly) / (b * b) + (lz * lz) / (c * c);
                if (val <= 1.0) {
                    brightnessSum += image[z].at<float>(y, x);
                    brightnessCount++;
                }
            }
        }
    }
    float meanBrightness = (brightnessCount > 0) ? (float)(brightnessSum / brightnessCount) : 0.4f;

    // Second pass: collect bright pixels within an expanded boundary (1.5x radius).
    // The expansion captures daughter blobs that may extend beyond the parent's boundary.
    // Only pixels brighter than the mean are included (cell tissue, not background).
    // Store raw image-space coordinates for centroid-based daughter placement.
    std::vector<cv::Point3f> rawPoints;
    for (int z = minZ; z <= maxZ; ++z) {
        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                float pixel = image[z].at<float>(y, x);
                if (pixel <= meanBrightness) continue;

                double dx = (double)x - _position.x;
                double dy = (double)y - _position.y;
                double dz = (double)z - _position.z;
                double lx, ly, lz;
                inverseRotatePoint(dx, dy, dz, lx, ly, lz);

                // Expanded boundary: val <= 2.25 means ~1.5x the radius
                double val = (lx * lx) / (a * a) + (ly * ly) / (b * b) + (lz * lz) / (c * c);
                if (val <= 2.25) {
                    rawPoints.emplace_back(
                        static_cast<float>(x),
                        static_cast<float>(y),
                        static_cast<float>(z));
                }
            }
        }
    }

    // Step 3: Run PCA with per-axis stddev normalization.
    // Normalize each axis to unit standard deviation before PCA so that
    // the axis with the largest absolute spread doesn't dominate. For
    // oblate spheroids, bright pixels cluster near (center_x, center_y)
    // in each z-slice but span many z-slices, making z-variance dominate.
    // Per-axis normalization equalizes the axes so PCA detects bimodal
    // structure (two blobs = split) regardless of direction.
    cv::Point3f split_axis;

    if (rawPoints.size() >= 3) {
        // Build centered data matrix
        int n = static_cast<int>(rawPoints.size());
        cv::Mat data(n, 3, CV_32F);
        for (int i = 0; i < n; ++i) {
            data.at<float>(i, 0) = rawPoints[i].x - _position.x;
            data.at<float>(i, 1) = rawPoints[i].y - _position.y;
            data.at<float>(i, 2) = rawPoints[i].z - _position.z;
        }

        // Compute per-axis standard deviation (data is already zero-centered)
        float sx = 0, sy = 0, sz = 0;
        for (int i = 0; i < n; ++i) {
            float vx = data.at<float>(i, 0);
            float vy = data.at<float>(i, 1);
            float vz = data.at<float>(i, 2);
            sx += vx * vx;
            sy += vy * vy;
            sz += vz * vz;
        }
        sx = std::sqrt(sx / n);
        sy = std::sqrt(sy / n);
        sz = std::sqrt(sz / n);

        // Normalize each axis to unit variance
        if (sx > 1e-6f) for (int i = 0; i < n; ++i) data.at<float>(i, 0) /= sx;
        if (sy > 1e-6f) for (int i = 0; i < n; ++i) data.at<float>(i, 1) /= sy;
        if (sz > 1e-6f) for (int i = 0; i < n; ++i) data.at<float>(i, 2) /= sz;

        cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

        float lambda1 = pca.eigenvalues.at<float>(0);
        float lambda2 = pca.eigenvalues.at<float>(1);
        float lambda3 = pca.eigenvalues.at<float>(2);

        // First eigenvector (max variance direction) in normalized space
        cv::Point3f ev_norm(
            pca.eigenvectors.at<float>(0, 0),
            pca.eigenvectors.at<float>(0, 1),
            pca.eigenvectors.at<float>(0, 2));

        // Transform back to image space: multiply by per-axis stddev
        cv::Point3f ev_image(
            ev_norm.x * sx,
            ev_norm.y * sy,
            ev_norm.z * sz);

        // Normalize to unit vector
        float norm = std::sqrt(ev_image.x * ev_image.x +
                               ev_image.y * ev_image.y +
                               ev_image.z * ev_image.z);
        if (norm > 1e-6f) {
            split_axis = ev_image * (1.0f / norm);
        } else {
            double theta = ((double)rand() / RAND_MAX) * 2 * M_PI;
            double phi = ((double)rand() / RAND_MAX) * M_PI;
            split_axis = cv::Point3f(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
        }

        float elongationRatio = (lambda2 > 1e-6f) ? (lambda1 / lambda2) : 1.0f;
        std::cout << "[PCA Split] " << _name
                  << " elongation_ratio=" << elongationRatio
                  << " split_axis=(" << split_axis.x << ", " << split_axis.y << ", " << split_axis.z << ")"
                  << " eigenvalues=(" << lambda1 << ", " << lambda2 << ", " << lambda3 << ")"
                  << " num_bright_pixels=" << rawPoints.size()
                  << " stddev=(" << sx << ", " << sy << ", " << sz << ")"
                  << std::endl;
    } else {
        std::cout << "[PCA Split] " << _name
                  << " only " << rawPoints.size() << " bright pixels found. Using random split axis."
                  << std::endl;
        double theta = ((double)rand() / RAND_MAX) * 2 * M_PI;
        double phi = ((double)rand() / RAND_MAX) * M_PI;
        split_axis = cv::Point3f(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
    }

    // Step 4: Centroid-based daughter placement.
    // Project each bright pixel onto the split axis through the cell center.
    // Split into positive/negative groups and compute 3D centroids.
    // This places daughters where the real bright pixel clusters are,
    // instead of at a fixed offset that may not match the data.
    double volumeScale = std::cbrt(0.5);
    double daughterMajorRadius = _major_radius * volumeScale;
    double daughterMinorRadius = _minor_radius * volumeScale;

    cv::Point3f centroid1(0, 0, 0), centroid2(0, 0, 0);
    int count1 = 0, count2 = 0;

    for (const auto &pt : rawPoints) {
        float dx = pt.x - _position.x;
        float dy = pt.y - _position.y;
        float dz = pt.z - _position.z;
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
        new_position1 = centroid1;
        new_position2 = centroid2;

        float sep = std::sqrt(
            (new_position1.x - new_position2.x) * (new_position1.x - new_position2.x) +
            (new_position1.y - new_position2.y) * (new_position1.y - new_position2.y) +
            (new_position1.z - new_position2.z) * (new_position1.z - new_position2.z));

        std::cout << "[Split Placement] centroid-based:"
                  << " c1=(" << new_position1.x << "," << new_position1.y << "," << new_position1.z << ")"
                  << " c2=(" << new_position2.x << "," << new_position2.y << "," << new_position2.z << ")"
                  << " sep=" << sep << std::endl;
    } else {
        // All pixels on one side — fall back to small offset along split axis
        float offset = static_cast<float>(daughterMajorRadius * 0.5);
        new_position1 = _position + split_axis * offset;
        new_position2 = _position - split_axis * offset;
        std::cout << "[Split Placement] one-sided (" << count1 << "/" << count2
                  << "), using fixed offset=" << offset << std::endl;
    }

    // Inherit parent rotation angles
    Spheroid cell1(SpheroidParams(
        _name + "0", new_position1.x, new_position1.y, new_position1.z,
        daughterMajorRadius, daughterMinorRadius, _theta_x, _theta_y, _theta_z));
    Spheroid cell2(SpheroidParams(
        _name + "1", new_position2.x, new_position2.y, new_position2.z,
        daughterMajorRadius, daughterMinorRadius, _theta_x, _theta_y, _theta_z));

    bool constraints = cell1.checkConstraints() && cell2.checkConstraints();
    return std::make_tuple(Spheroid(cell1), Spheroid(cell2), constraints);
}

std::vector<std::pair<float, cv::Vec3f>> Spheroid::performPCA(const std::vector<cv::Point3f> &points) const {
    if (points.empty())
    {
        throw std::invalid_argument("No points provided for PCA.");
    }

    // Create a matrix from the points
    cv::Mat data(points.size(), 3, CV_32F);
    for (size_t i = 0; i < points.size(); ++i)
    {
        data.at<float>(i, 0) = points[i].x;
        data.at<float>(i, 1) = points[i].y;
        data.at<float>(i, 2) = points[i].z;
    }

    // Perform PCA
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

    // Extract the eigenvalues and eigenvectors
    cv::Mat eigenvalues = pca.eigenvalues;
    cv::Mat eigenvectors = pca.eigenvectors;

    // Prepare the result
    std::vector<std::pair<float, cv::Vec3f>> eigenPairs;
    for (int i = 0; i < std::min(3, eigenvalues.rows); ++i)
    {
        float eigenvalue = eigenvalues.at<float>(i);
        cv::Vec3f eigenvector(
            eigenvectors.at<float>(i, 0),
            eigenvectors.at<float>(i, 1),
            eigenvectors.at<float>(i, 2)
        );
        eigenPairs.emplace_back(eigenvalue, eigenvector);
    }

    return eigenPairs;
}

bool Spheroid::checkConstraints() const {
    return (cellConfig.minMajorRadius <= _major_radius) && (_major_radius <= cellConfig.maxMajorRadius) && (cellConfig.minMinorRadius <= _minor_radius) && (_minor_radius <= cellConfig.maxMinorRadius);
}

SpheroidParams Spheroid::getCellParams() const {
    return SpheroidParams(_name, _position.x, _position.y, _position.z, _major_radius, _minor_radius, _theta_x, _theta_y, _theta_z);
}

[[nodiscard]] std::pair<std::vector<float>, std::vector<float>> Spheroid::calculateCorners() const {
    // Use max radius as conservative bound for rotated spheroid
    float maxR = std::max({(float)a, (float)b, (float)c});

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

bool Spheroid::checkIfCellsOverlap(const std::vector<Spheroid> &spheroids) {
    std::vector<std::vector<float>> positions;
    std::vector<std::pair<float, float>> radii;

    for (const auto &cell : spheroids)
    {
        positions.push_back({cell._position.x, cell._position.y, cell._position.z});
        radii.push_back({cell._major_radius * 0.95, cell._minor_radius * 0.95});
    }

    std::vector<std::vector<float>> distances;
    for (const auto &position1 : positions)
    {
        std::vector<float> distance_row;
        for (const auto &position2 : positions)
        {
            float distance = 0.0f;
            for (int i = 0; i < 3; ++i)
            {
                distance += pow(position1[i] - position2[i], 2);
            }
            distance = sqrt(distance);
            distance_row.push_back(distance);
        }
        distances.push_back(distance_row);
    }

    std::vector<std::vector<std::pair<float, float>>> radii_sums;
    for (const auto &radius1 : radii)
    {
        std::vector<std::pair<float, float>> radii_row;
        for (const auto &radius2 : radii)
        {
            radii_row.push_back({radius1.first + radius2.first, radius1.second + radius2.second});
        }
        radii_sums.push_back(radii_row);
    }

    bool overlap = false;
    for (std::size_t i = 0; i < spheroids.size(); ++i)
    {
        for (std::size_t j = 0; j < spheroids.size(); ++j)
        {
            if (i != j && (distances[i][j] < radii_sums[i][j].first) && (distances[i][j] < radii_sums[i][j].second))
            {
                overlap = true;
                break;
            }
        }
        if (overlap)
        {
            break;
        }
    }

    return overlap;
}