// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <memory>

#include "core/ParamsBase.h"

#include "DepthMapTransform.h"
#include "DepthVideo.h"
#include "FrameRange.h"

namespace ceres {
  class Problem;
}

namespace facebook {
namespace cp {

class FlowConstraintsCollection;

enum class StaticLossType {
  Euclidean,
  ReproDisparity,
  ReproDepthRatio,
  ReproLogDepth,
};

extern const EnumStrings<StaticLossType> staticLossTypeStrs;

// The smoothness loss compares points a, b, c from three consecutive frames.
// There are two types:
// Euclidean:
//   L = a + c - 2 * b
// Consistency:
//   L = d(a - b, c - b)
enum class SmoothLossType {
  EuclideanLaplacian,
  ReproDisparityLaplacian,
  ReproDepthRatioConsistency,
  ReproLogDepthConsistency,
};

extern const EnumStrings<SmoothLossType> smoothLossTypeStrs;

enum class IntrinsicsOptimization {
  Fixed,
  Shared,
  PerFrame,
};

class DepthVideoPoseOptimizer {
 public:
  struct Params : public ParamsBase {
    FrameRange frameRange;
    int maxIterations = 1000;
    int numThreads = 12;
    int numSteps = 4;
    double robustness = 0.5;

    StaticLossType staticLossType = StaticLossType::ReproDisparity;
    double staticSpatialWeight = 1.0;
    double staticDepthWeight = 1.0;

    SmoothLossType smoothLossType = SmoothLossType::ReproDisparityLaplacian;
    double smoothStaticWeight = 0.0;
    double smoothDynamicWeight = 0.0;

    double positionReg = 0.0;
    double scaleReg = 1.0;
    int scaleRegGridSize = 10;
    double depthDeformRegInitial = 1.0;
    double depthDeformRegFinal = 0.1;
    double adaptiveDeformationCost = 0.0;
    double spatialDeformReg = 1.0;
    bool graduateDepthDeformReg = false;
    double focalReg = 1.0;

    // When enabling coarse-to-fine the depth grid will be gradually subdivided
    // over the specified number of optimization steps.
    bool coarseToFine = true;
    int ctfLong = 17;
    int ctfShort = 10;

    // When using deferred spatial optimization we will first optimize without
    // spatial transforms (i.e., using identity transform), and then perform
    // a final extra optimization step with a bicubic spatial transform grid.
    bool deferredSpatialOpt = false;
    int dsoLong = 4;
    int dsoShort = 3;

    // The default focal length corresponds to an iPhone 7 portrait mode photo
    // (38.187 degrees field of view on the long image side.)
    double focalLong = 0.3461538376301239;
    IntrinsicsOptimization intrOpt = IntrinsicsOptimization::PerFrame;

    bool fixPoses = false;
    bool fixDepthXforms = false;
    bool fixSpatialXforms = false;

    // When enabling this we normalize depth **only using the first frame**, and
    // copy the optimized depth xform parameters to all other frames.
    bool normalizeDepthFromFirstFrame = true;

    void addCommandLineOptions() override;
    void printParams() const override;
    void resolve(const int numFrames);
  };

  DepthVideoPoseOptimizer(DepthVideo* const video, const int depthStream);
  ~DepthVideoPoseOptimizer();

  // Pose optimization, similar to the Instant3D paper.
  void poseOptimization(
      const Params& params, const FlowConstraintsCollection& constraints);
  void poseOptimizationStep(
      const Params& params,
      const FlowConstraintsCollection& constraints,
      const double depthDeformReg);

  // Find a good initialization of depth transforms, using the constraint that
  // flow-connected samples have similar depth.
  void normalizeDepth(
      const Params& params, const FlowConstraintsCollection& constraints);

 private:
  constexpr static int kStride = ValueXform::kStride;

  void addStaticSceneLoss(
    const Params& params, const FlowConstraintsCollection& constraints);
  void addSceneFlowSmoothnessLoss(
    const Params& params, const FlowConstraintsCollection& constraints);
  void addScaleRegularization(const Params& params);
  void addPositionRegularization(const Params& params);
  void addDepthDeformRegularization(const Params& params, const double weight);
  void addSpatialDeformRegularization(const Params& params);
  void addFocalRegularization(const Params& params);

  DepthVideo* const video_;
  const int depthStream_;

  std::unique_ptr<ceres::Problem> problem_;
  int numFrames_ = 0;

  // For each camera a 7-tuple storing the following:
  // indices 0 - 2: (x, y, z) position,
  // indices 3 - 5: (r1, r2, r3) rotation in angle-axis representation,
  // index 6: vertical focal length (defined as focal = tan(fov / 2)).
  std::vector<std::array<double, 7>> poseParams_;
};

}} // namespace facebook::cp
