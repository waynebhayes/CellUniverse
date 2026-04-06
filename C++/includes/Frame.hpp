// Frame.hpp
#ifndef FRAME_HPP
#define FRAME_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "types.hpp"
#include "ConfigTypes.hpp"
#include <random>
#include <functional>
#include "Spheroid.hpp"
#include <opencv2/core/mat.hpp>

void interpolateSlices(const cv::Mat& slice1, const cv::Mat& slice2, std::vector<cv::Mat>& processedSlices, int numInterpolations);
class Frame
{
public:
    // FIXME: Get rid of ImageStack entirely, use mat3D instead.
    // Also, the word "image" should never be used to represent the stack of 2D images in a TIF file; instead these should
    // be called "frames". Please try to find and change all such variable names to use "frame" rather than "image". An
    // "image" should only refer to 2D image. (Even if OpenCV is lazy... let's try to keep our names less confusing.)
    Frame(const std::vector<cv::Mat> &realFrame, const SimulationConfig &simulationConfig, const std::vector<Spheroid> &cells, const Path &outputPath, const std::string &imageName);

    // Method declarations
    std::vector<cv::Mat> generateSynthFrame();
    std::vector<cv::Mat> generateSynthFrameFast(Spheroid &oldCell, Spheroid &newCell);
    Cost calculateCost(const std::vector<cv::Mat> &synthFrame);
    std::vector<cv::Mat> generateOutputFrame();
    std::vector<cv::Mat> generateOutputSynthFrame();
    // DataFrame getCellsAsParams();
    size_t length() const;
    CostCallbackPair perturbCell(size_t index, float overlapWeight = 1000.0f,
                                 float sizeReductionWeight = 0.0f);
    double computeOverlapPenalty(float weight) const;
    double computeOverlapForCell(size_t cellIdx, float weight) const;

    std::map<std::string, float> computeElongationRatios() const;
    float computeElongationForCell(size_t cellIdx) const;
    CostCallbackPair trySplitCell(size_t cellIndex, float preOptMajorR = 0.0f, float preOptMinorR = 0.0f,
                                  float preOptX = 0.0f, float preOptY = 0.0f, float preOptZ = 0.0f,
                                  float splitElongationThreshold = 1.3f,
                                  float overlapWeight = 1000.0f,
                                  float fakeSplitOverlapVolumeFractionThreshold = 0.30f,
                                  float fakeSplitRadiusRatioThreshold = 2.0f,
                                  float splitSearchRadiusMultiplier = 3.0f,
                                  float splitMinorAxisAlignmentToleranceDegrees = 20.0f,
                                  float splitMinorAxisAlignmentFlatnessRatioThreshold = 0.5f,
                                  float splitMinorAxisAlignmentMinRadiusDisableThreshold = 0.0f,
                                  float splitFakeBridgeBrightnessSimilarityThreshold = 0.9f);
    std::vector<cv::Mat> getSynthFrame();
    const std::vector<cv::Mat>& getRealFrame() const { return _realFrame; }
    void setBackgroundColor(float backgroundColor) { simulationConfig.background_color = backgroundColor; }
    void regenerateSynthFrame() { _synthFrame = generateSynthFrame(); _currentCost = calculateCost(_synthFrame); }
    std::string getImageName() const { return imageName; }
    std::vector<Spheroid> cells;

private:
    std::vector<double> z_slices;
    SimulationConfig simulationConfig;
    std::string outputPath;
    std::string imageName;
    std::vector<cv::Mat> _realFrame;
    std::vector<cv::Mat> _synthFrame;
    double _currentCost = -1.0; // cached L2 image cost of _synthFrame
    cv::Size getImageShape();
};
#endif // FRAME_H
