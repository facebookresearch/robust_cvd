// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "core/ParamsBase.h"
#include "core/TrackTable.h"

#include "DepthVideo.h"
#include "FrameRange.h"
#include "PoseOptimizer.h"

namespace facebook {
namespace cp {

class FlowConstraintsCollection;

// 2D track observation
struct DepthVideoObs {
  Vector2fna loc = Vector2fna::Zero();
  DepthVideoObs() = default;
  explicit DepthVideoObs(const Vector2fna& loc) : loc(loc) {}
  DepthVideoObs(const float x, const float y) : loc(x, y) {}
};

using DepthVideoTrack = TrackBaseSequential<DepthVideoObs>;
using DepthVideoTrackedFrame = FrameBase;
using DepthVideoTrackTable =
    TrackTable<DepthVideoObs, DepthVideoTrack, DepthVideoTrackedFrame>;

class DepthVideoProcessor {
 public:
  enum class Op {
      // This operation does nothing..
      None,
      // The next few operations work on the "processed" depth images.
      Reset,
      Copy,
      BilateralFilter,
      FlowGuidedFilter,
      ClipMaxDepth,
      // Flow constraint ops.
      ComputeConstraints,
      ResetConstraintStaticFlag,
      SetConstraintStaticFlagFromDynamicMask,
      PruneConstraintStaticFlag,
      // Compute long tracks for visualizing stability.
      ComputeTracks,
      // Pose optimize ops.
      GridXformSplit,
      ResetPoses,
      ResetDepthXforms,
      ResetSpatialXforms,
      NormalizeDepth,
      OptimizePoses,
      // This convienience op performs the following ops in order:
      // reset (poses + depth + spatial), normalize depth, optimize poses.
      ResetNormalizeOptimize,
  };

  struct Params : public ParamsBase {
    Op op = Op::None;
    FrameRange frameRange;
    int colorStream = 0;
    int depthStream = 0;
    int sourceDepthStream = 0;
    int spatialRadius = 0;
    int frameRadius = 2;
    float depthSigma = 0.3f;
    float colorSigma = 0.0f;
    bool median = false;
    bool farConnections = false;
    float maxDepth = 1000.f;

    // Minimum distance between two matches (in pixels).
    int matchSeparation = 10;
    float flowConsistancyThresh = 0.05f;
    int trackSpawnDistance = 20;
    int trackPruneDistance = 5;
    int minDynamicDistance = 3;
    int minTrackLength = 4;

    XformDescriptor depthXformDesc;
    XformDescriptor spatialXformDesc;

    DepthVideoPoseOptimizer::Params poseOptimizer;

    void addCommandLineOptions() override;
    void printParams() const override;
    void resolve(const int numFrames);
  };

  DepthVideoProcessor(DepthVideo* const video);

  void process(const Params& params);

  void reset(const Params& params);
  void copy(const Params& params);
  void bilateralFilter(const Params& params);
  void flowGuidedFilter(const Params& params);
  void clipMaxDepth(const Params& params);
  std::unique_ptr<FlowConstraintsCollection> computeConstraints(
      const Params& params);
  void resetConstraintStaticFlag(FlowConstraintsCollection& constraints);
  void setConstraintStaticFlagFromDynamicMask(
      const Params& params, FlowConstraintsCollection& constraints);
  void pruneConstraintStaticFlag(
      const Params& params, FlowConstraintsCollection& constraints);
  std::unique_ptr<DepthVideoTrackTable> computeTracks(const Params& params);
  void gridXformSplit(const Params& params);
  void resetPoses(const Params& params);
  void resetDepthXforms(const Params& params);
  void resetSpatialXforms(const Params& params);
  void normalizeDepth(
      const Params& params, const FlowConstraintsCollection& constraints);
  void optimizePoses(
      const Params& params, const FlowConstraintsCollection& constraints);
  void resetNormalizeOptimize(
      const Params& params, const FlowConstraintsCollection& constraints);

 private:
  using TrackTable = DepthVideoTrackTable;
  using Obs = DepthVideoObs;
  using Track = DepthVideoTrack;

  DepthVideo* const video_;
};

extern const EnumStrings<DepthVideoProcessor::Op> depthVideoProcessorOpStrs;

}} // namespace facebook::cp
