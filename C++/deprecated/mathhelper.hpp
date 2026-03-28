//
// Created by yuant on 2/19/2024.
//

#ifndef CELLUNIVERSE_MATHHELPER_HPP
#define CELLUNIVERSE_MATHHELPER_HPP

class Vector {
public:
    Vector(): x(0), y(0), z(0) {}
    Vector(double x, double y, double z)
    : x(x), y(y), z(z) {
    }
    // redefine + operator
    Vector operator+(const Vector& other) const {
        Vector result;
        result.x = other.x + x;
        result.y = other.y + y;
        result.z = other.z + z;
        return result;
    }
    // overload + operator to support numpy like operation
    Vector operator+(double val) const {
        Vector result;
        result.x = val + x;
        result.y = val + y;
        result.z = val + z;
        return result;
    }

    double dotProduct(const Vector& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    double x;
    double y;
    double z;
};

#endif //CELLUNIVERSE_MATHHELPER_HPP
