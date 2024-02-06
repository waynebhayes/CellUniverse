// Frame.hpp
#ifndef FRAME_HPP
#define FRAME_HPP

#include "Cell.hpp"
#include "Config.hpp"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "types.hpp"
#include "ConfigTypes.hpp"
#include <random>
#include <functional>


class Frame {
public:
    Frame(const ImageStack& realImageStack, const SimulationConfig& simulationConfig, const std::vector<Cell>& cells, const Path& outputPath, const std::string& imageName);

    // Method declarations
    void padRealImage();
    ImageStack generateSynthImages();
    ImageStack generateSynthImagesFast(Cell &oldCell, Cell &newCell);
    Cost calculateCost(const ImageStack& synthImageStack);
    ImageStack generateOutputImage();
    ImageStack generateOutputSynthImage();
    // DataFrame getCellsAsParams();
    size_t length() const;
    CostCallbackPair perturb();
    CostCallbackPair split();

private:
    std::vector<double> zSlices;
    std::vector<Cell> cells;
    SimulationConfig simulationConfig;
    std::string outputPath;
    std::string imageName;
    ImageStack _realImageStack;
    ImageStack _realImageStackCopy; // copy of realImageStack
    ImageStack _synthImageStack;
    cv::Size getImageShape();
    Cost costOfPerturb(const std::string& perturbParam, float perturbVal, size_t index, const Cell& oldCell);
    ParamImageMap getSynthPerturbedCells(size_t index, const ParamValMap& params, float perturbLength, const Cell& oldCell);

};



#endif // FRAME_H