#include "Bacilli.hpp"
#include "Cell.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

Bacilli::Bacilli(std::string name, double x, double y, double width, double length, double rotation, double split_alpha, double opacity)
    : name(name), _position(Vector(x, y, 0)), _width(width), _length(length), _rotation(rotation), _split_alpha(split_alpha), _opacity(opacity), _needs_refresh(true), dormant(false) {}
}

void Bacilli::_refresh() {
    Vector direction(cos(_rotation), sin(_rotation), 0);
    double distance = (_length - _width) / 2;
    Vector displacement = distance * direction;
    _head_center = _position + displacement;
    _tail_center = _position - displacement;
    Vector side(-sin(_rotation), cos(_rotation), 0);
    double radius = _width / 2;
    _head_right = _head_center + radius * side;
    _head_left = _head_center - radius * side;
    _tail_right = _tail_center + radius * side;
    _tail_left = _tail_center - radius * side;
    _region = Rectangle(std::floor(std::min(_head_center.x, _tail_center.x) - radius), std::floor(std::min(_head_center.y, _tail_center.y) - radius), std::ceil(std::max(_head_center.x, _tail_center.x) + radius) + 1, std::ceil(std::max(_head_center.y, _tail_center.y) + radius) + 1);
    //Might need MathHelper for Vector/Rectangle classes
    _needs_refresh = false;
}

vvoid Bacilli::draw(cv::Mat& image, cv::Mat& cell_map, bool is_cell, cv::Scalar background_color, cv::Scalar cell_color, std::map<std::string, std::string> simulation_config) {
    if (dormant)
        return;

    if (_needs_refresh)
        _refresh();

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);

    int top = _region.top;
    int bottom = _region.bottom;
    int left = _region.left;
    int right = _region.right;
    int width = right - left;
    int height = bottom - top;

    cv::Mat body_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::fillPoly(body_mask, std::vector<std::vector<cv::Point>>{{cv::Point(_head_left.x - left, _head_left.y - top),
        cv::Point(_head_right.x - left, _head_right.y - top),
        cv::Point(_tail_right.x - left, _tail_right.y - top),
        cv::Point(_tail_left.x - left, _tail_left.y - top)}}, cv::Scalar(255));
    cv::Mat body_mask_up = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::fillPoly(body_mask_up, std::vector<std::vector<cv::Point>>{{cv::Point(_head_left.x - _region.left, _head_left.y - _region.top),
        cv::Point((std::ceil((_head_right.x + _head_left.x) / 2) - _region.left), (std::ceil((_head_right.y + _head_left.y) / 2) - _region.top)),
        cv::Point((std::ceil((_tail_right.x + _tail_left.x) / 2) - _region.left), (std::ceil((_tail_right.y + _tail_left.y) / 2) - _region.top)),
        cv::Point(_tail_left.x - _region.left, _tail_left.y - _region.top)}}, cv::Scalar(255));
    cv::Mat body_mask_middle = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::fillPoly(body_mask_middle, std::vector<std::vector<cv::Point>>{{cv::Point((std::ceil((_head_right.x + _head_left.x * 2) / 3) - _region.left),
        (std::ceil((_head_right.y + _head_left.y * 2) / 3) - _region.top)),
        cv::Point((std::ceil((_head_right.x * 2 + _head_left.x) / 3) - _region.left),
            (std::ceil((_head_right.y * 2 + _head_left.y) / 3) - _region.top)),
        cv::Point((std::ceil((_tail_right.x * 2 + _tail_left.x) / 3) - _region.left),
            (std::ceil((_tail_right.y * 2 + _tail_left.y) / 3) - _region.top)),
        cv::Point((std::ceil((_tail_right.x + _tail_left.x * 2) / 3) - _region.left),
            (std::ceil((_tail_right.y + _tail_left.y * 2) / 3) - _region.top))}}, cv::Scalar(255));
    cv::Mat head_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::circle(head_mask, cv::Point(_head_center.x - left, _head_center.y - top), _width / 2, cv::Scalar(255), -1);
    cv::Mat tail_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::circle(tail_mask, cv::Point(_tail_center.x - left, _tail_center.y - top), _width / 2, cv::Scalar(255), -1);

    try {
        std::string image_type = simulation_config["image.type"];
        double diffraction_strength = std::stod(simulation_config["light.diffraction.strength"]);
        double cell_opacity = (_opacity != "auto" && _opacity != "None" && _opacity != "") ? std::stod(_opacity) : std::stod(simulation_config["cell.opacity"]);

        if (image_type == "phaseContrast") {
            if (!is_cell) {
                mask.setTo(0);
                body_mask.copyTo(mask);
                head_mask.copyTo(mask);
                tail_mask.copyTo(mask);
                image(_region)[mask] = 0.39 * 255;  // 0.39 * 255 = 100
            }
            else {
                mask.setTo(0);
                body_mask.copyTo(mask);
                image(_region)[mask] = 0.25 * 255;  // 0.25 * 255 = 65

                mask.setTo(0);
                head_mask.copyTo(mask);
                image(_region)[mask] = 0.25 * 255;

                mask.setTo(0);
                tail_mask.copyTo(mask);
                image(_region)[mask] = 0.25 * 255;

                mask.setTo(0);
                body_mask_up.copyTo(mask);
                image(_region)[mask] = 0.63 * 255;  // 0.63 * 255 = 160

                mask.setTo(0);
                body_mask_middle.copyTo(mask);
                image(_region)[mask] = 0.39 * 255;  // 0.39 * 255 = 100
            }
        }
        else if (image_type == "graySynthetic") {
            mask.setTo(0);
            body_mask.copyTo(mask);
            head_mask.copyTo(mask);
            tail_mask.copyTo(mask);
            image(_region)[mask] += diffraction_strength;

            cv::Mat diffraction_mask = cv::Mat::zeros(image.size(), CV_8UC1);
            diffraction_mask.setTo(background_color);
            cell_map(_region)[mask] += 1;
            cv::GaussianBlur(cell_map, diffraction_mask, cv::Size(0, 0), std::stod(simulation_config["light.diffraction.sigma"]));
            diffraction_mask.setTo(cell_color + cell_opacity * diffraction_mask);
            image(_region) = diffraction_mask;
        }
        else if (image_type == "binary") {
            mask.setTo(0);
            body_mask.copyTo(mask);
            head_mask.copyTo(mask);
            tail_mask.copyTo(mask);
            if (is_cell) {
                image(_region)[mask] += 1.0;
                cell_map(_region)[mask] += 1;
            }
            else {
                image(_region)[mask] -= 1.0;
                cell_map(_region)[mask] -= 1;
            }
        }
    }
    catch (std::exception& e) {
        dormant = true;
    }
}

void Bacilli::draw_outline(cv::Mat& image, cv::Scalar color) {
    if (_needs_refresh)
        _refresh();

    cv::line(image, cv::Point(_tail_left.x, _tail_left.y), cv::Point(_head_left.x, _head_left.y), color);
    cv::line(image, cv::Point(_tail_right.x, _tail_right.y), cv::Point(_head_right.x, _head_right.y), color);

    double t0 = std::atan2(_head_right.y - _head_center.y, _head_right.x - _head_center.x);
    double t1 = std::atan2(_head_left.y - _head_center.y, _head_left.x - _head_center.x);
    cv::ellipse(image, cv::Point(_head_center.x, _head_center.y), cv::Size(_width / 2, _width / 2), 0, t0, t1, color);

    t0 = std::atan2(_tail_right.y - _tail_center.y, _tail_right.x - _tail_center.x);
    t1 = std::atan2(_tail_left.y - _tail_center.y, _tail_left.x - _tail_center.x);
    cv::ellipse(image, cv::Point(_tail_center.x, _tail_center.y), cv::Size(_width / 2, _width / 2), 0, t0, t1, color);
}



std::pair<Bacilli*, Bacilli*> Bacilli::split(double alpha) {
    if (_needs_refresh)
        _refresh();

    Vector direction(cos(_rotation), sin(_rotation), 0);
    Vector unit = _length * direction;

    Vector front = _position + unit / 2;
    Vector back = _position - unit / 2;
    Vector center = _position + (0.5 - alpha) * unit;

    Vector position1 = (front + center) / 2;
    Vector position2 = (center + back) / 2;

    Bacilli* cell1 = new Bacilli(name + '0', position1.x, position1.y, _width, _length * alpha, _rotation, alpha, _opacity);
    Bacilli* cell2 = new Bacilli(name + '1', position2.x, position2.y, _width, _length * (1 - alpha), _rotation, alpha, _opacity);

    return std::make_pair(cell1, cell2);
}

Bacilli* Bacilli::combine(Bacilli* cell) {
    if (_needs_refresh)
        _refresh();

    if (cell->_needs_refresh)
        cell->_refresh();

    Vector separation = _position - cell->_position;
    Vector direction = separation / sqrt(separation.dot(separation));

    Vector direction1(cos(_rotation), sin(_rotation), 0);
    double distance1 = _length - _width;
    Vector head1 = direction1.dot(direction) >= 0 ? _position + distance1 * direction1 / 2 : _position - distance1 * direction1 / 2;
    Vector extent1 = head1 + _width * direction / 2;
    Vector front = _position + (extent1 - _position).dot(direction) * direction;

    Vector direction2(cos(cell->_rotation), sin(cell->_rotation), 0);
    double distance2 = cell->_length - cell->_width;
    Vector tail2 = direction2.dot(direction) >= 0 ? cell->_position - distance2 * direction2 / 2 : cell->_position + distance2 * direction2 / 2;
    Vector extent2 = tail2 - cell->_width * direction / 2;
    Vector back = cell->_position + (extent2 - cell->_position).dot(direction) * direction;

    Vector position = (front + back) / 2;
    double rotation = atan2(direction.y, direction.x);
    double width = (_width + cell->_width) / 2;
    double length = sqrt(pow((front - back).dot(front - back), 2));

    return new Bacilli(name.substr(0, name.size() - 1), position.x, position.y, width, length, rotation, 0, (_opacity + cell->_opacity) / 2);
}

std::string Bacilli::toString() {
    return "Bacilli(name=\"" + name + "\", x=" + std::to_string(_position.x) + ", y=" + std::to_string(_position.y) + ", width=" + std::to_string(_width) + ", length=" + std::to_string(_length) + ", rotation=" + std::to_string(_rotation) + ")";
}

void Bacilli::simulated_region() {
    if (_needs_refresh)
        _refresh();

    // Implement simulated_region function

    throw std::logic_error("Unsupported image type");
}

Rectangle Bacilli::getRegion() {
    if (_needs_refresh)
        _refresh();
    return _region;
}

Vector Bacilli::getPosition() {
    return _position;
}

double Bacilli::getX() {
    return _position.x;
}

void Bacilli::setX(double value) {
    if (value != _position.x) {
        _position.x = value;
        _needs_refresh = true;
    }
}

double Bacilli::getY() {
    return _position.y;
}

void Bacilli::setY(double value) {
    if (value != _position.y) {
        _position.y = value;
        _needs_refresh = true;
    }
}

double Bacilli::getWidth() {
    return _width;
}

void Bacilli::setWidth(double value) {
    if (value != _width) {
        _width = value;
        _needs_refresh = true;
    }
}

double Bacilli::getLength() {
    return _length;
}

void Bacilli::setLength(double value) {
    if (value != _length) {
        _length = value;
        _needs_refresh = true;
    }
}

double Bacilli::getRotation() {
    return _rotation;
}

void Bacilli::setRotation(double value) {
    if (value != _rotation) {
        _rotation = value;
        _needs_refresh = true;
    }
}

double Bacilli::getSplitAlpha() {
    return _split_alpha;
}

void Bacilli::setSplitAlpha(double value) {
    _split_alpha = value;
}

double Bacilli::getOpacity() {
    return _opacity;
}

void Bacilli::setOpacity(double value) {
    _opacity = value;
}
