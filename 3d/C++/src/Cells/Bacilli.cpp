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

void Bacilli::draw() {
    if (dormant)
        return;

    if (_needs_refresh)
        _refresh();

    // Implement the draw function
}

void Bacilli::draw_outline() {
    if (_needs_refresh)
        _refresh();

    // Implement the draw_outline function
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
