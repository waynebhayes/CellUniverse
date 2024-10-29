#include "Sphere.hpp"

SphereParams Sphere::paramClass = SphereParams();
SphereConfig Sphere::cellConfig = SphereConfig();



double Sphere::getRadiusAt(double z) const {
    if (std::abs(_position.z - z) > _radius) {
        return 0;
    }
    return std::sqrt((_radius * _radius) - ((_position.z - z) * (_position.z - z)));
}


void Sphere::draw(cv::Mat & image, SimulationConfig simulationConfig, cv::Mat * cellMap, float z) const  {
    if (dormant) {
        return;
    }

    float currentRadius = getRadiusAt(z);
    if (currentRadius <= 0) {
        return;
    }

    float background_color = simulationConfig.background_color;
    float cell_color = simulationConfig.cell_color;

    cv::Point center(_position.x, _position.y);
    cv::circle(image, center, static_cast<int>(currentRadius), cv::Scalar(cell_color), -1);
}

void Sphere::drawOutline(cv::Mat& image, float color, float z) const {
    if (dormant) {
        return;
    }
    float outlineColor = color;
    float currentRadius = getRadiusAt(z);
    if (currentRadius <= 0) {
        return;
    }
    cv::Point center(_position.x, _position.y);
    int thickness = 1;
    cv::circle(image, center, static_cast<int>(round(currentRadius)), cv::Scalar(outlineColor), thickness, cv::LINE_AA);
}

Sphere Sphere::getPerturbedCell() const {
    SphereParams sphereParams(
            _name,
	    // FIXME: we should choose only ONE of these, uniformly at random, to perturb in each iteration.
            _position.x + cellConfig.x.getPerturbOffset(),
            _position.y + cellConfig.y.getPerturbOffset(),
            _position.z + cellConfig.z.getPerturbOffset(),
            _radius + cellConfig.radius.getPerturbOffset()
    );
    return Sphere(sphereParams);
}

Sphere Sphere::getParameterizedCell(std::unordered_map<std::string, float> params) const  {
    float xOffset = params["x"];
    float yOffset = params["y"];
    float zOffset = params["z"];
    float radiusOffset = params["radius"];

    if (params.empty()) {
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
            newRadius
    );
    return Sphere(sphereParams);
}

std::tuple<Sphere, Sphere, bool> Sphere::getSplitCells() const  {
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

    //Cell cell1Base = static_cast<Cell>(cell1);
    //Cell cell2Base = static_cast<Cell>(cell2);//convert Sphere to Cell

    return std::make_tuple(Sphere(cell1), Sphere(cell2), constraints);
}

bool Sphere::checkConstraints() const {
    SphereConfig config;
    return (config.minRadius <= _radius) && (_radius <= config.maxRadius);
}

float Sphere::getRadiusAt(float z) {
    if (std::abs(_position.z - z) > _radius) {
        return 0;
    }
    return std::sqrt((_radius * _radius) - ((_position.z - z) * (_position.z - z)));
}

CellParams Sphere::getCellParams() const  {
    return SphereParams(_name, _position.x, _position.y, _position.z, _radius);
}

std::pair<std::vector<float>, std::vector<float>> Sphere::calculateCorners() const  {
    std::vector<float> min_corner = { static_cast<float>(_position.x) - static_cast<float>(_radius),
                                      static_cast<float>(_position.y) - static_cast<float>(_radius),
                                      static_cast<float>(_position.z) - static_cast<float>(_radius) };

    std::vector<float> max_corner = { static_cast<float>(_position.x) + static_cast<float>(_radius),
                                      static_cast<float>(_position.y) + static_cast<float>(_radius),
                                      static_cast<float>(_position.z) + static_cast<float>(_radius) };

    return std::make_pair(min_corner, max_corner);
}


std::pair<std::vector<float>, std::vector<float>> Sphere::calculateMinimumBox(Sphere &perturbed_cell) const  {
    auto [cell1_min_corner, cell1_max_corner] = calculateCorners();
    auto [cell2_min_corner, cell2_max_corner] = perturbed_cell.calculateCorners();

    std::vector<float> min_corner, max_corner;
    for (int i = 0; i < 3; ++i) {
        min_corner.push_back(std::min(cell1_min_corner[i], cell2_min_corner[i]));
        max_corner.push_back(std::max(cell1_max_corner[i], cell2_max_corner[i]));
    }
    return std::make_pair(min_corner, max_corner);
}


bool Sphere::checkIfCellsOverlap(const std::vector<Sphere>& spheres) {
    std::vector<std::vector<float>> positions;
    std::vector<float> radii;

    for (const auto& cell : spheres) {
        positions.push_back({ cell._position.x, cell._position.y, cell._position.z });
        radii.push_back(cell._radius * 0.95);
    }

    std::vector<std::vector<float>> distances;
    for (const auto& position1 : positions) {
        std::vector<float> distance_row;
        for (const auto& position2 : positions) {
            float distance = 0.0f;
            for (int i = 0; i < 3; ++i) {
                distance += pow(position1[i] - position2[i], 2);
            }
            distance = sqrt(distance);
            distance_row.push_back(distance);
        }
        distances.push_back(distance_row);
    }

    std::vector<std::vector<float>> radii_sums;
    for (const auto& radius1 : radii) {
        std::vector<float> radii_row;
        for (const auto& radius2 : radii) {
            radii_row.push_back(radius1 + radius2);
        }
        radii_sums.push_back(radii_row);
    }

    bool overlap = false;
    for (std::size_t i = 0; i < spheres.size(); ++i) {
        for (std::size_t j = 0; j < spheres.size(); ++j) {
            if (i != j && distances[i][j] < radii_sums[i][j]) {
                overlap = true;
                break;
            }
        }
        if (overlap) {
            break;
        }
    }

    return overlap;
}
