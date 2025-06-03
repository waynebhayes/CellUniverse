#include "Spheroid.hpp"
#include <random>

SphereConfig Spheroid::cellConfig = SphereConfig();

double Spheroid::getRadiusAt(double z) const
{
    // For spheroid, use the c_axis (z-axis) for z calculations
    if (std::abs(_position.z - z) > _c_axis)
    {
        return 0;
    }
    
    // Calculate the elliptical cross-section at z level
    double z_factor = (_position.z - z) / _c_axis;
    double scale_factor = std::sqrt(1 - z_factor * z_factor);

    // Return the average of a and b axes scaled by z factor
    return (_a_axis + _b_axis) * 0.5 * scale_factor;
}

double Spheroid::getAxisRadiusAt(double z, double axis_length) const
{
    if (std::abs(_position.z - z) > _c_axis)
    {
        return 0;
    }
    double z_factor = (_position.z - z) / _c_axis;
    double scale_factor = std::sqrt(1 - z_factor * z_factor);
    return axis_length * scale_factor;
}

void Spheroid::draw(cv::Mat &image, SimulationConfig simulationConfig, cv::Mat *cellMap, float z) const
{
    if (dormant)
    {
        return;
    }

    // Calculate the semi-axes lengths at this z-slice
    double a_at_z = getAxisRadiusAt(z, _a_axis);
    double b_at_z = getAxisRadiusAt(z, _b_axis);
    
    if (a_at_z <= 0 || b_at_z <= 0)
    {
        return;
    }

    float background_color = simulationConfig.background_color;
    float cell_color = simulationConfig.cell_color;

    cv::Point center(_position.x, _position.y);
    
    // Draw ellipses with rotation angle
    cv::Size axes(static_cast<int>(a_at_z), static_cast<int>(b_at_z));
    cv::ellipse(image, center, axes, _rotation, 0, 360, cv::Scalar(cell_color), -1);
}

void Spheroid::drawOutline(cv::Mat &image, float color, float z) const
{
    if (dormant)
    {
        return;
    }
    float outlineColor = color;
    double a_at_z = getAxisRadiusAt(z, _a_axis);
    double b_at_z = getAxisRadiusAt(z, _b_axis);

    if (a_at_z <= 0 || b_at_z <= 0)
    {
        return;
    }
    cv::Point center(_position.x, _position.y);
    cv::Size axes(static_cast<int>(a_at_z), static_cast<int>(b_at_z));
    int thickness = 1;
    cv::ellipse(image, center, axes, _rotation, 0, 360, cv::Scalar(outlineColor), thickness, cv::LINE_AA);
}

void Spheroid::rotateCell(const std::vector<cv::Point3d> &points)
{
    if (points.empty())
    {
        return; // Keep current rotation if no points
    }

    // Prepare data for PCA
    int sz = static_cast<int>(points.size());
    cv::Mat data_pts = cv::Mat(sz, 3, CV_64F);
    for (int i = 0; i < data_pts.rows; i++)
    {
        data_pts.at<double>(i, 0) = points[i].x;
        data_pts.at<double>(i, 1) = points[i].y;
        data_pts.at<double>(i, 2) = points[i].z;
    }

    // Perform PCA analysis
    cv::PCA pca_analysis(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);

    // Get the principal components
    cv::Point2d principal_direction;
    if (pca_analysis.eigenvectors.rows >= 2)
    {
        // For 2D rotation, we're interested in the first eigenvector's direction in the XY plane
        principal_direction.x = pca_analysis.eigenvectors.at<double>(0, 0);
        principal_direction.y = pca_analysis.eigenvectors.at<double>(0, 1);
        
        // Calculate the angle in degrees
        double angle = atan2(principal_direction.y, principal_direction.x) * 180.0 / M_PI;
        
        // OpenCV's ellipse angle is measured clockwise from horizontal
        _rotation = angle;
    }
}

Spheroid Spheroid::getPerturbedCell() const
{
    SpheroidParams spheroidParams(
        _name,

        // FIXME: we should choose only ONE of these, uniformly at random, to perturb in each iteration.
        _position.x + cellConfig.x.getPerturbOffset(),
        _position.y + cellConfig.y.getPerturbOffset(),
        _position.z + cellConfig.z.getPerturbOffset(),
        _radius + cellConfig.radius.getPerturbOffset(),

        //We use smaller perturbation for axes
        _a_axis + cellConfig.radius.getPerturbOffset() * 0.5,
        _b_axis + cellConfig.radius.getPerturbOffset() * 0.5,
        _c_axis + cellConfig.radius.getPerturbOffset() * 0.5,
        _rotation);
    return Spheroid(spheroidParams);
}

Spheroid Spheroid::getParameterizedCell(std::unordered_map<std::string, float> params) const  {
    float xOffset = params["x"];
    float yOffset = params["y"];
    float zOffset = params["z"];
    float radiusOffset = params["radius"];

    if (params.empty())
    {
        xOffset = Spheroid::cellConfig.x.getPerturbOffset();
        yOffset = Spheroid::cellConfig.y.getPerturbOffset();
        zOffset = Spheroid::cellConfig.z.getPerturbOffset();
        radiusOffset = Spheroid::cellConfig.radius.getPerturbOffset();
    }

    float newRadius = fmin(fmax(Spheroid::cellConfig.minRadius, _radius + radiusOffset), Spheroid::cellConfig.maxRadius);
    float newAAxis = fmin(fmax(Spheroid::cellConfig.minRadius, _a_axis + radiusOffset * 0.5), Spheroid::cellConfig.maxRadius);
    float newBAxis = fmin(fmax(Spheroid::cellConfig.minRadius, _b_axis + radiusOffset * 0.5), Spheroid::cellConfig.maxRadius);
    float newCAxis = fmin(fmax(Spheroid::cellConfig.minRadius, _c_axis + radiusOffset * 0.5), Spheroid::cellConfig.maxRadius);
    
    SpheroidParams spheroidParams(
        _name,
        _position.x + xOffset,
        _position.y + yOffset,
        _position.z + zOffset,
        newRadius,
        newAAxis,
        newBAxis,
        newCAxis,
        _rotation);
    return Spheroid(spheroidParams);
}

std::vector<std::pair<double, cv::Point3d>> Spheroid::performPCA(const std::vector<cv::Point3d> &pts, std::vector<cv::Mat> &frame) const
{
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    cv::Mat data_pts = cv::Mat(sz, 3, CV_64F);
    for (int i = 0; i < data_pts.rows; i++)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
        data_pts.at<double>(i, 2) = pts[i].z;
    }
    //Perform PCA analysis
    cv::PCA pca_analysis(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);

    //Store the center of the object
    cv::Point3d cntr = cv::Point3d(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
    static_cast<int>(pca_analysis.mean.at<double>(0, 1)),
    static_cast<int>(pca_analysis.mean.at<double>(0, 2)));

    //Store the eigenvalues and eigenvectors
    std::vector<cv::Point3d> eigen_vecs(3);
    std::vector<double> eigen_val(3);
    std::vector<std::pair<double, cv::Point3d>> eigen_pairs;
    for (int i = 0; i < 3; i++)
    {
    eigen_vecs[i] = cv::Point3d(pca_analysis.eigenvectors.at<double>(i, 0),
                    pca_analysis.eigenvectors.at<double>(i, 1),
                    pca_analysis.eigenvectors.at<double>(i, 2));

    eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
    std::cout << "eigenval " << i << ": " << eigen_val[i] << std::endl;
    std::cout << "eigenvec " << i << ": " << eigen_vecs[i] << std::endl;
    std::pair<double, cv::Point3d> eigen_pair {std::make_pair(eigen_val[i], eigen_vecs[i])};
    eigen_pairs.push_back(eigen_pair);

    }
    return eigen_pairs; // return std::vector<eigenvalue, eigenvector>
}

std::tuple<Spheroid, Spheroid, bool> Spheroid::getSplitCells(const std::vector<cv::Mat> &realTiffSlices) const
{
    // remove calback function
    // make z scale same as x and y scale
    // 16 z slices per x-y
    // interpolate (make up) z slices convert 400x400x30 to 400x400x400
    // using brightness values interpolate
    auto [min_corner, max_corner] = calculateCorners();

    double scaleFactor = Spheroid::cellConfig.boundingBoxScale;

    for (int i = 0; i < 3; ++i){
        float midPoint = 0.5 * (min_corner[i] + max_corner[i]);
        float halfBox = 0.5 * (max_corner[i] - min_corner[i]);
        float scaledBox = halfBox * scaleFactor;

        min_corner[i] = midPoint - scaledBox;
        max_corner[i] = midPoint + scaledBox;
    }

    int minX = std::max(0, static_cast<int>(std::floor(min_corner[0])));
    int maxX = std::min(static_cast<int>(realTiffSlices[0].cols), static_cast<int>(std::ceil(max_corner[0])));
    int minY = std::max(0, static_cast<int>(std::floor(min_corner[1])));
    int maxY = std::min(static_cast<int>(realTiffSlices[0].rows), static_cast<int>(std::ceil(max_corner[1])));
    int minZ = std::max(0, static_cast<int>(std::floor(min_corner[2])));
    int maxZ = std::min(static_cast<int>(realTiffSlices.size()-1), static_cast<int>(std::ceil(max_corner[2])));

    std::cout << "minX: " << minX << " maxX: " << maxX << std::endl;
    std::cout << "minY: " << minY << " maxY: " << maxY << std::endl;
    std::cout << "minZ: " << minZ << " maxZ: " << maxZ << std::endl;

    cv::Range yRange(minY, maxY); // y
    cv::Range xRange(minX, maxX); // x
    cv::Range zRange(minZ, maxZ); // z

    cv::Point3f splitAxis;

    std::vector<cv::Mat> subTiffSlices;
    if (maxZ > minZ && maxX > minX && maxY > minY) {
    // iterate through z levels
    for(unsigned n = zRange.start; n < zRange.end; ++n) {
        cv::Mat nSlice = realTiffSlices[n]; 
        // generate subslice
        cv::Mat subNSlice = nSlice(
            cv::Range(yRange.start, yRange.end),
            cv::Range(xRange.start, xRange.end)
        );
        subTiffSlices.push_back(subNSlice);
        }

    std::vector<std::vector<cv::Point3d>> contours3D;
    for(int n = 0; n < subTiffSlices.size(); ++n)
    {
        cv::Mat &sliceN = subTiffSlices[n];
        sliceN.convertTo(sliceN, CV_8UC1);
        std::vector<std::vector<cv::Point>> contours;
        findContours(sliceN, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

        cv::cvtColor(sliceN, sliceN, cv::COLOR_GRAY2BGR);

        for (size_t i = 0; i < contours.size(); i++)
        {
            // Calculate the area of each contour
            double area = cv::contourArea(contours[i]);
            // // Ignore contours that are too small or too large
            if (area < 1e2 || 1e5 < area) continue; 

            // Create 3D contour and then add it to contours3D
            std::vector<cv::Point3d> contour3D;
            for(const auto& point : contours[i])
            {
                contour3D.push_back(cv::Point3d(point.x, point.y, n)); // n being nth slice
            }
            contours3D.push_back(contour3D);
        }
    }
        
        // flatten contours to one 3D object
        std::vector<cv::Point3d> allPoints;
        for(const auto& contour : contours3D) {
            allPoints.insert(allPoints.end(), contour.begin(), contour.end());
        }

        // If PCA cannot be performed, print message and retun a fallback
        if (allPoints.empty()) {
            std::cout << "No points for PCA!" << std::endl;
            return std::make_tuple(*this, *this, false);
        }

        // return pair of (eigenval, eigenvector)
        auto eigenPair(performPCA(allPoints, subTiffSlices));
        if (eigenPair.size() >= 2) {
            cv::Point3d v1 = eigenPair[0].second;
            cv::Point3d v2 = eigenPair[1].second;

            // computes cross product of largest two vectors and normalizes
            cv::Point3d crossProd(
                v1.y * v2.z - v1.z * v2.y,
                v1.z * v2.x - v1.x * v2.z,
                v1.x * v2.y - v1.y * v2.x);
                
            double norm = std::sqrt(crossProd.x * crossProd.x + crossProd.y * crossProd.y + crossProd.z * crossProd.z);
            if (norm != 0) {crossProd.x /= norm; crossProd.y /= norm; crossProd.z /= norm;}

            splitAxis = cv::Point3f(static_cast<float>(crossProd.x), static_cast<float>(crossProd.y),static_cast<float>(crossProd.z));
    } else {
        std::cout << "Invalid bounding box. No split will be performed." << std::endl;
        return std::make_tuple(*this, *this, false);
    }

    // Split axis is used to determine new cell positions
    cv::Point3f offset = splitAxis * (_radius / 2.0);
    cv::Point3f new_position1 = _position + offset;
    cv::Point3f new_position2 = _position - offset;

    double halfRadius = _radius / 2.0;
    double halfAAxis = _a_axis / 2.0;
    double halfBAxis = _b_axis / 2.0;
    double halfCAxis = _c_axis / 2.0;

    Spheroid cell1(SpheroidParams(_name + "0", new_position1.x, new_position1.y, new_position1.z, halfRadius, halfAAxis, halfBAxis, halfCAxis, _rotation));
    Spheroid cell2(SpheroidParams(_name + "1", new_position2.x, new_position2.y, new_position2.z, halfRadius, halfAAxis, halfBAxis, halfCAxis, _rotation));

    bool constraints = cell1.checkConstraints() && cell2.checkConstraints();

    return std::make_tuple(Spheroid(cell1), Spheroid(cell2), constraints);
    }

    return std::make_tuple(*this, *this, false);
}


bool Spheroid::checkConstraints() const
{
    SphereConfig config;
    return (config.minRadius <= _radius) && (_radius <= config.maxRadius) &&
           (config.minRadius <= _a_axis) && (_a_axis <= config.maxRadius) &&
           (config.minRadius <= _b_axis) && (_b_axis <= config.maxRadius) &&
           (config.minRadius <= _c_axis) && (_c_axis <= config.maxRadius);
}

float Spheroid::getRadiusAt(float z)
{
    // For spheroid, use the c_axis (z-axis) for z calculations
    if (std::abs(_position.z - z) > _c_axis)
    {
        return 0;
    }
    // Calculate elliptical cross-section at z level
    double z_factor = (_position.z - z) / _c_axis;
    double scale_factor = std::sqrt(1 - z_factor * z_factor);
    // Return average of a and b axes scaled by z factor
    return (_a_axis + _b_axis) * 0.5 * scale_factor;
}

CellParams Spheroid::getCellParams() const
{
    return SpheroidParams(_name, _position.x, _position.y, _position.z, _radius, _a_axis, _b_axis, _c_axis, _rotation);
}

std::pair<std::vector<float>, std::vector<float>> Spheroid::calculateCorners() const
{
    std::vector<float> min_corner = {static_cast<float>(_position.x) - static_cast<float>(_a_axis),
                                     static_cast<float>(_position.y) - static_cast<float>(_b_axis),
                                     static_cast<float>(_position.z) - static_cast<float>(_c_axis)};

    std::vector<float> max_corner = {static_cast<float>(_position.x) + static_cast<float>(_a_axis),
                                     static_cast<float>(_position.y) + static_cast<float>(_b_axis),
                                     static_cast<float>(_position.z) + static_cast<float>(_c_axis)};

    return std::make_pair(min_corner, max_corner);
}

std::pair<std::vector<float>, std::vector<float>> Spheroid::calculateMinimumBox(Spheroid &perturbed_cell) const
{
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

bool Spheroid::checkIfCellsOverlap(const std::vector<Spheroid> &spheroids)
{
    std::vector<std::vector<float>> positions;
    std::vector<float> radii;

    for (const auto &cell : spheroids)
    {
        positions.push_back({cell._position.x, cell._position.y, cell._position.z});
        // Use average of all axes for overlap detection
        radii.push_back((cell._a_axis + cell._b_axis + cell._c_axis) / 3.0 * 0.95);
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

    std::vector<std::vector<float>> radii_sums;
    for (const auto &radius1 : radii)
    {
        std::vector<float> radii_row;
        for (const auto &radius2 : radii)
        {
            radii_row.push_back(radius1 + radius2);
        }
        radii_sums.push_back(radii_row);
    }

    bool overlap = false;
    for (std::size_t i = 0; i < spheroids.size(); ++i)
    {
        for (std::size_t j = 0; j < spheroids.size(); ++j)
        {
            if (i != j && distances[i][j] < radii_sums[i][j])
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