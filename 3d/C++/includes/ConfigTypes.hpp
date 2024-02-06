// ConfigTypes.hpp
#ifndef CONFIGTYPES_HPP
#define CONFIGTYPES_HPP

#include "Cell.hpp"

class BaseModel {

};

class GenericModel {

};

class SimulationConfig: public BaseModel {
public:
    int iterationsPerCell;
};

class ProbabilityConfig: public BaseModel {

};

class BaseConfig: public GenericModel {
public:
    SimulationConfig simulation;
};

#endif