#ifndef BACILLI_H
#define BACILLI_H

#include <string>

class Bacilli : public Cell {
private:
    std::string name;
    Vector position;
    double width, length, rotation, opacity, split_alpha;
    bool _needs_refresh, dormant;
    Vector _position, _head_center, _tail_center, _head_right, _head_left, _tail_right, _tail_left;
    Rectangle _region;

public:
    Bacilli(std::string name, double x, double y, double width, double length, double rotation, double split_alpha = 0, double opacity = 0);
    void _refresh();
    void draw() override;
    void drawOutline() override;
    void split(double alpha) override;
    void combine(Bacilli* cell) override;
    void simulated_region() override;
};

#endif // BACILLI_H
