#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "cell.cpp"
#define M_PI 3.14159265358979323846

class SphereConfig : public CellConfig {
public:
    PerturbParams x;
    PerturbParams y;
    PerturbParams z;
    PerturbParams radius;
    double minRadius;
    double maxRadius;

    SphereConfig(const YAML::node& node)
    {
        x.parseParams(node["x"]);
        y.parseParams(node["y"]);
        z.parseParams(node["z"]);
        radius.parseParams(node["radius"]);
        minRadius = node["minRadius"].as<double>();
        maxRadius = node["maxRadius"].as<double>();
    }
};

class SphereParams: public CellParams {
public:
    double x;
    double y;
    double z;
    double radius;

    SphereParams()
        : x(0), y(0), z(0), radius(0) {}

    SphereParams(std::string name_val , double x_val, double y_val, double z_val, double radius_val)
        : CellParams{name_val}, x(x_val), y(y_val), z(z_val), radius(radius_val) {}

    void parseParams(const YAML::Node& node) {
        prob = node["prob"].as<double>();
        mu = node["mu"].as<double>();
        sigma = node["sigma"].as<double>();
    }
};

class Sphere : public Cell {
private:
    std::string _name;
    cv::Point3f _position;
    double _radius;
    double _rotation;
    bool dormant;
    SphereParams paramClass;
    SphereConfig cellConfig;

public:
    Sphere(const SphereParams& init_props)
        : Cell(init_props), _name(init_props.name), _position({ static_cast<float>(init_props.x), static_cast<float>(init_props.y), static_cast<float>(init_props.z) }),
        _radius(static_cast<float>(init_props.radius)), _rotation(0), dormant(false) {}
    //default constructor ?
    Sphere() : _radius(0), _rotation(0), dormant(false) {}

    double get_radius_at(double z) const {
        if (std::abs(_position.z - z) > _radius) {
            return 0;
        }
        return std::sqrt((_radius * _radius) - ((_position.z - z) * (_position.z - z)));
    }


    void Cell::draw(cv::Mat & image, SimulationConfig simulationConfig, cv::Mat * cellMap = nullptr, float z = 0) const override {
        if (dormant) {
            return;
        }

        float currentRadius = get_radius_at(z);
        if (currentRadius <= 0) {
            return;
        }

        cv::Scalar backgroundColor(simulationConfig.backgroundColor, simulationConfig.backgroundColor, simulationConfig.backgroundColor); //unsure
        cv::Scalar cellColor(simulationConfig.cellColor, simulationConfig.cellColor, simulationConfig.cellColor);

        cv::Point center(static_cast<int>(_position.x), static_cast<int>(_position.y));
        cv::Size axes(static_cast<int>(currentRadius), static_cast<int>(currentRadius));
        cv::ellipse(image, center, axes, 0, 0, 360, cellColor, -1);
    }

    void Cell::draw_outline(cv::Mat& image, cv::Scalar color, float z = 0) const override {
        if (dormant) {
            return;
        }

        float currentRadius = get_radius_at(z);
        if (currentRadius <= 0) {
            return;
        }

        cv::Point center(static_cast<int>(_position.x), static_cast<int>(_position.y));
        cv::Size axes(static_cast<int>(currentRadius), static_cast<int>(currentRadius));
        cv::ellipse(image, center, axes, 0, 0, 360, color, 1);
    }

    Cell* Cell::get_perturbed_cell() const override {
        SphereParams sphereParams(
            _name,
            _position.x + cellConfig.x.get_perturb_offset(),
            _position.y + cellConfig.y.get_perturb_offset(),
            _position.z + cellConfig.z.get_perturb_offset(),
            _radius + cellConfig.radius.get_perturb_offset()
        );
        return new Sphere(sphereParams);
    }

    Cell* Cell::get_parameterized_cell(std::unordered_map<std::string, float> params = {}) const override {
        float xOffset = params["x"];
        float yOffset = params["y"];
        float zOffset = params["z"];
        float radiusOffset = params["radius"];

        if (params.empty()) {
            xOffset = Sphere::cellConfig.x.get_perturb_offset();
            yOffset = Sphere::cellConfig.y.get_perturb_offset();
            zOffset = Sphere::cellConfig.z.get_perturb_offset();
            radiusOffset = Sphere::cellConfig.radius.get_perturb_offset();
        }

        float newRadius = fmin(fmax(Sphere::cellConfig.minRadius, _radius + radiusOffset), Sphere::cellConfig.maxRadius);
        SphereParams sphereParams(
            _name,
            _position.x + xOffset,
            _position.y + yOffset,
            _position.z + zOffset,
            newRadius
        );
        return new Sphere(sphereParams);
    }

    std::tuple<Cell*, Cell*, bool> Cell::get_split_cells() const override {
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

        bool constraints = cell1.check_constraints() && cell2.check_constraints();

        //Cell cell1Base = static_cast<Cell>(cell1);
        //Cell cell2Base = static_cast<Cell>(cell2);//convert Sphere to Cell

        return std::make_tuple(new Sphere(cell1), new Sphere(cell2), constraints);
    }

    bool check_constraints() const {
        SphereConfig config;
        return (config.minRadius <= _radius) && (_radius <= config.maxRadius);
    }

    float get_radius_at(float z) {
        if (std::abs(_position.z - z) > _radius) {
            return 0;
        }
        return std::sqrt((_radius * _radius) - ((_position.z - z) * (_position.z - z)));
    }

    CellParams get_cell_params() const override {
        return SphereParams(_name, _position.x, _position.y, _position.z, _radius);
    }

    std::pair<std::vector<float>, std::vector<float>> Cell::calculate_corners() const override {
        std::vector<float> min_corner = { static_cast<float>(_position.x) - static_cast<float>(_radius),
                                          static_cast<float>(_position.y) - static_cast<float>(_radius),
                                          static_cast<float>(_position.z) - static_cast<float>(_radius) };

        std::vector<float> max_corner = { static_cast<float>(_position.x) + static_cast<float>(_radius),
                                          static_cast<float>(_position.y) + static_cast<float>(_radius),
                                          static_cast<float>(_position.z) + static_cast<float>(_radius) };

        return std::make_pair(min_corner, max_corner);
    }


    std::pair<std::vector<float>, std::vector<float>> Cell::calculate_minimum_box(Cell& perturbed_cell) const override {
        auto [cell1_min_corner, cell1_max_corner] = calculate_corners();
        auto [cell2_min_corner, cell2_max_corner] = perturbed_cell.calculate_corners();

        std::vector<float> min_corner, max_corner;
        for (int i = 0; i < 3; ++i) {
            min_corner.push_back(std::min(cell1_min_corner[i], cell2_min_corner[i]));
            max_corner.push_back(std::max(cell1_max_corner[i], cell2_max_corner[i]));
        }
        return std::make_pair(min_corner, max_corner);
    }

    static bool check_if_cells_overlap(const std::vector<Sphere>& spheres) {
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
};
