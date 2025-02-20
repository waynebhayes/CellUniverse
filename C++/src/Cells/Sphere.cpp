#include "Sphere.hpp"
#include <random>

SphereParams Sphere::paramClass = SphereParams();
SphereConfig Sphere::cellConfig = SphereConfig();

double Sphere::getRadiusAt(double z) const
{
    if (std::abs(_position.z - z) > _radius)
    {
        return 0;
    }
    return std::sqrt((_radius * _radius) - ((_position.z - z) * (_position.z - z)));
}

void Sphere::draw(cv::Mat &image, SimulationConfig simulationConfig, cv::Mat *cellMap, float z) const
{
    if (dormant)
    {
        return;
    }

    float currentRadius = getRadiusAt(z);
    if (currentRadius <= 0)
    {
        return;
    }

    float background_color = simulationConfig.background_color;
    float cell_color = simulationConfig.cell_color;

    cv::Point center(_position.x, _position.y);
    cv::circle(image, center, static_cast<int>(currentRadius), cv::Scalar(cell_color), -1);
}

void Sphere::drawOutline(cv::Mat &image, float color, float z) const
{
    if (dormant)
    {
        return;
    }
    float outlineColor = color;
    float currentRadius = getRadiusAt(z);

    if (currentRadius <= 0)
    {
        return;
    }
    cv::Point center(_position.x, _position.y);
    int thickness = 1;
    cv::circle(image, center, static_cast<int>(round(currentRadius)), cv::Scalar(outlineColor), thickness, cv::LINE_AA);
}

Sphere Sphere::getPerturbedCell() const
{
    SphereParams sphereParams(
        _name,

        // FIXME: we should choose only ONE of these, uniformly at random, to perturb in each iteration.
        _position.x + cellConfig.x.getPerturbOffset(),
        _position.y + cellConfig.y.getPerturbOffset(),
        _position.z + cellConfig.z.getPerturbOffset(),
        _radius + cellConfig.radius.getPerturbOffset());
    return Sphere(sphereParams);
}

Sphere Sphere::getParameterizedCell(std::unordered_map<std::string, float> params) const  {
    float xOffset = params["x"];
    float yOffset = params["y"];
    float zOffset = params["z"];
    float radiusOffset = params["radius"];

    if (params.empty())
    {
        xOffset = Sphere::cellConfig.x.getPerturbOffset();
        yOffset = Sphere::cellConfig.y.getPerturbOffset();
        zOffset = Sphere::cellConfig.z.getPerturbOffset();
        radiusOffset = Sphere::cellConfig.radius.getPerturbOffset();
    }

    float newRadius = fmin(fmax(Sphere::cellConfig.minRadius, _radius + radiusOffset), Sphere::cellConfig.maxRadius);
    SphereParams sphereParams(
        _name,
        _position.x + xOffset,
        _position.y + yOffset,
        _position.z + zOffset,
        newRadius);
    return Sphere(sphereParams);
}

double Sphere::getOrientation3D(const std::vector<cv::Point3d> &pts, std::vector<cv::Mat> &frame) const
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
    for (int i = 0; i < 3; i++)
    {
    eigen_vecs[i] = cv::Point3d(pca_analysis.eigenvectors.at<double>(i, 0),
                    pca_analysis.eigenvectors.at<double>(i, 1),
                    pca_analysis.eigenvectors.at<double>(i, 2));

    eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
    std::cout << "eigenval " << i << ": " << eigen_val[i] << std::endl;
    std::cout << "eigenvec " << i << ": " << eigen_vecs[i] << std::endl;

    }
    return 0.0; // TODO: change to angle
}

std::tuple<Sphere, Sphere, bool> Sphere::getSplitCells(const std::vector<cv::Mat> &image) const
{
    // remove calback function
    // make z scale same as x and y scale
    // 16 z slices per x-y
    // interpolate (make up) z slices convert 400x400x30 to 400x400x400
    // using brightness values interpolate
    auto [min_corner, max_corner] = calculateCorners();

    int minX = std::max(0, static_cast<int>(min_corner[0]));
    int maxX = std::min(static_cast<int>(image[0].cols), static_cast<int>(max_corner[0]));
    int minY = std::max(0, static_cast<int>(min_corner[1]));
    int maxY = std::min(static_cast<int>(image[0].rows), static_cast<int>(max_corner[1]));
    int minZ = std::max(0, static_cast<int>(min_corner[2]));
    int maxZ = std::min(static_cast<int>(image.size()), static_cast<int>(max_corner[2]));

    std::cout << "minX: " << minX << " maxX: " << maxX << std::endl;
    std::cout << "minY: " << minY << " maxY: " << maxY << std::endl;
    std::cout << "minZ: " << minZ << " maxZ: " << maxZ << std::endl;

    cv::Range yRange(minY, maxY); // y
    cv::Range xRange(minX, maxX); // x
    cv::Range zRange(minZ, maxZ); // z

    std::vector<cv::Mat> subTiffSlices;
    // iterate through z levels
    for(unsigned n = zRange.start; n < zRange.end; ++n) {
        cv::Mat nSlice = image[n];
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
    std::vector<cv::Point3d> allPoints;
    // flatten contours to one 3D object
    for(const auto& contour : contours3D) {
        allPoints.insert(allPoints.end(), contour.begin(), contour.end());
    }
    // std::cout << allPoints << std::endl;
    if(allPoints.size() > 0)
    {
        getOrientation3D(allPoints, subTiffSlices);
    }
    else {std::cout << "No points for PCA!" << std::endl;}

    double theta = ((double)rand() / RAND_MAX) * 2 * M_PI;
    double phi = ((double)rand() / RAND_MAX) * M_PI;

    cv::Point3f split_axis(
        sin(phi) * cos(theta),
        sin(phi) * sin(theta),
        cos(phi));

    cv::Point3f offset = split_axis * (_radius / 2.0);
    cv::Point3f new_position1 = _position + offset;
    cv::Point3f new_position2 = _position - offset;

    double halfRadius = _radius / 2.0;

    Sphere cell1(SphereParams(_name + "0", new_position1.x, new_position1.y, new_position1.z, halfRadius));
    Sphere cell2(SphereParams(_name + "1", new_position2.x, new_position2.y, new_position2.z, halfRadius));

    bool constraints = cell1.checkConstraints() && cell2.checkConstraints();

    // Cell cell1Base = static_cast<Cell>(cell1);
    // Cell cell2Base = static_cast<Cell>(cell2);//convert Sphere to Cell

    return std::make_tuple(Sphere(cell1), Sphere(cell2), constraints);
}

std::vector<std::pair<float, cv::Vec3f>> Sphere::performPCA(const std::vector<cv::Point3f> &points) const
{
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

// std::vector<float> Sphere::performPCA(const std::vector<cv::Point3f> &points) const
// {
//     if (points.empty())
//     {
//         throw std::invalid_argument("No points provided for PCA.");
//     }

//     // Create a matrix from the points
//     cv::Mat data(points.size(), 3, CV_32F);
//     for (size_t i = 0; i < points.size(); ++i)
//     {
//         data.at<float>(i, 0) = points[i].x;
//         data.at<float>(i, 1) = points[i].y;
//         data.at<float>(i, 2) = points[i].z;
//     }

//     // Perform PCA
//     cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

//     // Extract the eigenvalues
//     cv::Mat eigenvalues = pca.eigenvalues;

//     // Convert eigenvalues to a vector
//     std::vector<float> eigenvaluesVec;
//     eigenvalues.copyTo(eigenvaluesVec);

//     // Sort the eigenvalues in descending order
//     std::sort(eigenvaluesVec.begin(), eigenvaluesVec.end(), std::greater<float>());

//     // Return the top 3 eigenvalues
//     return std::vector<float>(eigenvaluesVec.begin(), eigenvaluesVec.begin() + 3);
// }

bool Sphere::checkConstraints() const
{
    SphereConfig config;
    return (config.minRadius <= _radius) && (_radius <= config.maxRadius);
}

float Sphere::getRadiusAt(float z)
{
    if (std::abs(_position.z - z) > _radius)
    {
        return 0;
    }
    return std::sqrt((_radius * _radius) - ((_position.z - z) * (_position.z - z)));
}

CellParams Sphere::getCellParams() const
{
    return SphereParams(_name, _position.x, _position.y, _position.z, _radius);
}

std::pair<std::vector<float>, std::vector<float>> Sphere::calculateCorners() const
{
    std::vector<float> min_corner = {static_cast<float>(_position.x) - static_cast<float>(_radius),
                                     static_cast<float>(_position.y) - static_cast<float>(_radius),
                                     static_cast<float>(_position.z) - static_cast<float>(_radius)};

    std::vector<float> max_corner = {static_cast<float>(_position.x) + static_cast<float>(_radius),
                                     static_cast<float>(_position.y) + static_cast<float>(_radius),
                                     static_cast<float>(_position.z) + static_cast<float>(_radius)};

    return std::make_pair(min_corner, max_corner);
}

std::pair<std::vector<float>, std::vector<float>> Sphere::calculateMinimumBox(Sphere &perturbed_cell) const
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

bool Sphere::checkIfCellsOverlap(const std::vector<Sphere> &spheres)
{
    std::vector<std::vector<float>> positions;
    std::vector<float> radii;

    for (const auto &cell : spheres)
    {
        positions.push_back({cell._position.x, cell._position.y, cell._position.z});
        radii.push_back(cell._radius * 0.95);
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
    for (std::size_t i = 0; i < spheres.size(); ++i)
    {
        for (std::size_t j = 0; j < spheres.size(); ++j)
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