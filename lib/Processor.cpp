// Copyright 2004-present Facebook. All Rights Reserved.

#include "Processor.h"

#include <stdlib.h>

#include <boost/filesystem.hpp>
#include <fmt/format.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/CvUtil.h"
#include "core/Enum.h"
#include "core/Enum_impl.h"
#include "core/MathUtil.h"
#include "core/TrackTable.h"

#include "ColorStream.h"
#include "DepthStream.h"
#include "FlowConstraints.h"
#include "PoseOptimizer.h"

using namespace Eigen;
using namespace cv;

namespace fs = boost::filesystem;

namespace facebook {
namespace cp {

using Op = DepthVideoProcessor::Op;

const EnumStrings<Op> depthVideoProcessorOpStrs {
  {Op::None, "None"},
  {Op::Reset, "Reset"},
  {Op::Copy, "Copy"},
  {Op::BilateralFilter, "BilateralFilter"},
  {Op::FlowGuidedFilter, "FlowGuidedFilter"},
  {Op::ClipMaxDepth, "ClipMaxDepth"},
  {Op::ComputeConstraints, "ComputeConstraints"},
  {Op::ResetConstraintStaticFlag, "ResetConstraintStaticFlag"},
  {Op::SetConstraintStaticFlagFromDynamicMask,
      "SetConstraintStaticFlagFromDynamicMask"},
  {Op::PruneConstraintStaticFlag, "PruneConstraintStaticFlag"},
  {Op::ComputeTracks, "ComputeTracks"},
  {Op::GridXformSplit, "GridXformSplit"},
  {Op::ResetPoses, "ResetPoses"},
  {Op::ResetDepthXforms, "ResetDepthXforms"},
  {Op::ResetSpatialXforms, "ResetSpatialXforms"},
  {Op::NormalizeDepth, "NormalizeDepth"},
  {Op::OptimizePoses, "OptimizePoses"},
  {Op::ResetNormalizeOptimize, "ResetNormalizeOptimize"},
};
MAKE_VALIDATOR(Op, depthVideoProcessorOpStrs);

void DepthVideoProcessor::Params::addCommandLineOptions() {
  addOption("op", &op);
  addOption("frameRange", &frameRange);
  addOption("colorStream", &colorStream);
  addOption("depthStream", &depthStream);
  addOption("sourceDepthStream", &sourceDepthStream);
  addOption("spatialRadius", &spatialRadius);
  addOption("frameRadius", &frameRadius);
  addOption("depthSigma", &depthSigma);
  addOption("colorSigma", &colorSigma);
  addOption("median", &median);
  addOption("farConnections", &farConnections);
  addOption("maxDepth", &maxDepth);
  addOption("matchSeparation", &matchSeparation);
  addOption("flowConsistancyThresh", &flowConsistancyThresh);
  addOption("trackSpawnDistance", &trackSpawnDistance);
  addOption("trackPruneDistance", &trackPruneDistance);
  addOption("minDynamicDistance", &minDynamicDistance);
  addOption("minTrackLength", &minTrackLength);
  addSubParamsOptions(depthXformDesc, "depthXform");
  addSubParamsOptions(spatialXformDesc, "spatialXform");
  addSubParamsOptions(poseOptimizer, "poseOptimizer");
}

void DepthVideoProcessor::Params::printParams() const {
  DepthVideoProcessor::Params defaultParams;

  printParam("op", op, depthVideoProcessorOpStrs);
  printParam("frameRange",
      (frameRange.frames.empty() ? "" : frameRange.toString()));
  PRINT_PARAM_IF_NEQ(colorStream)
  PRINT_PARAM(depthStream)
  PRINT_PARAM(sourceDepthStream)
  PRINT_PARAM_IF_NEQ(spatialRadius)
  PRINT_PARAM_IF_NEQ(frameRadius)
  PRINT_PARAM_IF_NEQ(depthSigma)
  PRINT_PARAM_IF_NEQ(colorSigma)
  PRINT_PARAM_IF_NEQ(median);
  PRINT_PARAM_IF_NEQ(farConnections);
  PRINT_PARAM_IF_NEQ(maxDepth);
  PRINT_PARAM_IF_NEQ(matchSeparation)
  PRINT_PARAM_IF_NEQ(flowConsistancyThresh)
  PRINT_PARAM_IF_NEQ(trackSpawnDistance)
  PRINT_PARAM_IF_NEQ(trackPruneDistance)
  PRINT_PARAM_IF_NEQ(minDynamicDistance)
  PRINT_PARAM_IF_NEQ(minTrackLength)
  depthXformDesc.printParams();
  spatialXformDesc.printParams();
  poseOptimizer.printParams();
}

void DepthVideoProcessor::Params::resolve(const int numFrames) {
  frameRange.resolve(numFrames);
}

DepthVideoProcessor::DepthVideoProcessor(DepthVideo* const video)
    : video_(video) {
}

void DepthVideoProcessor::process(const Params& params) {
  LOG(INFO) << "------------------------";
  params.printParams();

  if (params.op == Op::None) {
    // This does nothing :)
  } else if (params.op == Op::Reset) {
    reset(params);
  } else if (params.op == Op::Copy) {
    copy(params);
  } else if (params.op == Op::BilateralFilter) {
    bilateralFilter(params);
  } else if (params.op == Op::FlowGuidedFilter) {
    flowGuidedFilter(params);
  } else if (params.op == Op::ClipMaxDepth) {
    clipMaxDepth(params);
  } else if (params.op == Op::GridXformSplit) {
    gridXformSplit(params);
  } else if (params.op == Op::ResetPoses) {
    resetPoses(params);
  } else if (params.op == Op::ResetDepthXforms) {
    resetDepthXforms(params);
  } else if (params.op == Op::ResetSpatialXforms) {
    resetSpatialXforms(params);
  } else {
    throw std::runtime_error("Unsupported operation selected.");
  }

  LOG(INFO) << "";
}

void DepthVideoProcessor::reset(const Params& params) {
  for (int frame : params.frameRange) {
    video_->depthFrame(params.depthStream, frame).clear();
  }
}

void DepthVideoProcessor::copy(const Params& params) {
  if (params.sourceDepthStream < 0 ||
      params.sourceDepthStream >= video_->numDepthStreams()) {
    throw std::runtime_error("Source depth stream out of range.");
  }

  if (params.sourceDepthStream == params.depthStream) {
    throw std::runtime_error(
        "Source and destination depth stream cannot be identical.");
  }

  const DepthStream& srcDs = video_->depthStream(params.sourceDepthStream);
  DepthStream& dstDs = video_->depthStream(params.depthStream);

  LOG(INFO) << fmt::format("Copying depth stream {:d} ({:s}) -> {:d} ({:s})",
      params.sourceDepthStream, srcDs.name(),
      params.depthStream, dstDs.name());

  for (int frame : params.frameRange) {
    const DepthFrame& src = srcDs.frame(frame);
    DepthFrame& dst = dstDs.frame(frame);

    const Mat1f* depth = src.depth();
    CHECK_NOTNULL(depth);
    dst.setDepth(*depth);

    dst.intrinsics = src.intrinsics;
    dst.extrinsics = src.extrinsics;
  }
}

void DepthVideoProcessor::bilateralFilter(const Params& params) {
  const float depthSigma2 = sqr(params.depthSigma);
  const float colorSigma2 = sqr(params.colorSigma);

  const bool useDepthRange = (params.depthSigma > 0.f);
  const bool useColorRange = (params.colorSigma > 0.f);

  const DepthStream& ds = video_->depthStream(0);
  const ColorStream& cs = video_->colorStream("down");

  for (int frame : params.frameRange) {
    LOG(INFO) << "Processing frame " << frame <<
        " of range " << params.frameRange.toString() << "...";

    // This is the image we're filtering.
    const Mat1f& referenceDepth = *ds.frame(frame).depth();
    const int w = referenceDepth.cols;
    const int h = referenceDepth.rows;

    // We have a reference color image, too.
    const Mat* referenceColor = cs.frame(frame).image();
    CHECK_EQ(referenceColor->cols, w);
    CHECK_EQ(referenceColor->rows, h);

    // The output image.
    Mat1f filteredDepth(h, w);

    // Fetching a stack of (2 * frameRadius + 1) *guidance images*, temporally
    // centered around the current frame.
    const int f0 = std::max(0, frame - params.frameRadius);
    const int f1 = std::min(video_->numFrames()-1, frame + params.frameRadius);
    const int temporalWindowSize = f1 - f0 + 1;
    std::vector<const Mat1f*> guideDepthStack(temporalWindowSize);
    std::vector<const Mat3f*> guideColorStack(temporalWindowSize);
    for (int i = 0; i < temporalWindowSize; ++i) {
      guideDepthStack[i] = ds.frame(f0 + i).depth();
      CHECK_EQ(guideDepthStack[i]->cols, w);
      CHECK_EQ(guideDepthStack[i]->rows, h);
      guideColorStack[i] = cs.frame(f0 + i).image3f();
      CHECK_EQ(guideColorStack[i]->cols, w);
      CHECK_EQ(guideColorStack[i]->rows, h);
    }

    // Vector of samples for the median filter
    std::vector<std::pair<float, float>> depthWeightSamples;

    // Looping over the output image pixels.
    for (int y = 0; y < h; ++y) {
      const int y0 = std::max(0, y - params.spatialRadius);
      const int y1 = std::min(h - 1, y + params.spatialRadius);

      const float* const refDepthPtr =
          (const float* const)referenceDepth.ptr(y);
      const Vec3f* const refColorPtr =
          (const Vec3f* const)referenceColor->ptr(y);
      float* const dstPtr = (float* const)filteredDepth.ptr(y);

      for (int x = 0; x < w; ++x) {
        const int x0 = std::max(0, x - params.spatialRadius);
        const int x1 = std::min(w - 1, x + params.spatialRadius);

        const float referenceDepth = refDepthPtr[x];
        const Vec3f& referenceColor = refColorPtr[x];

        float sumDepth = 0.f;
        float sumWeight = 0.f;

        if (params.median) {
          depthWeightSamples.clear();
        }

        // "Inner loop" over the spatio-temporal kernel.
        for (int wf = 0; wf < temporalWindowSize; ++wf) {
          for (int wy = y0; wy <= y1; ++wy) {
            const float* const guideDepthPtr =
                (const float* const)guideDepthStack[wf]->ptr(wy);
            const Vec3f* const guideColorPtr =
                (const Vec3f* const)guideColorStack[wf]->ptr(wy);

            for (int wx = x0; wx <= x1; ++wx) {
              const float depth = guideDepthPtr[wx];
              float exponent = 0.f;

              if (useDepthRange) {
                const float diff2 = sqr(depth - referenceDepth);
                exponent += -diff2 / depthSigma2;
              }

              if (useColorRange) {
                const Vec3f& color = guideColorPtr[wx];
                float diff2 =
                    sqr(color(0) - referenceColor(0)) +
                    sqr(color(1) - referenceColor(1)) +
                    sqr(color(2) - referenceColor(2));
                exponent += -diff2 / colorSigma2;
              }

              const float weight = (exponent != 0.f ? expf(exponent) : 1.f);

              if (params.median) {
                depthWeightSamples.emplace_back(depth, weight);
              } else {
                sumDepth += depth * weight;
              }
              sumWeight += weight;
            }
          }
        }

        if (params.median) {
          float halfWeight = sumWeight / 2.f;
          std::sort(depthWeightSamples.begin(), depthWeightSamples.end());
          const int N = depthWeightSamples.size();
          float cumWeight = 0.f;
          for (int i = 0; i < N; ++i) {
            const auto& dw = depthWeightSamples[i];
            cumWeight += dw.second;
            if (cumWeight >= halfWeight) {
              dstPtr[x] = dw.first;
              break;
            }
          }
        } else {
          dstPtr[x] = (sumWeight > 0.f ? sumDepth / sumWeight : 0.f);
        }
      }
    }

    video_->depthFrame(params.depthStream, frame).setDepth(filteredDepth);
  }
}

void DepthVideoProcessor::flowGuidedFilter(const Params& params) {
  LOG(INFO) << "Applying flow guided filter...";

  if (!params.frameRange.isConsecutive()) {
    throw std::runtime_error("Frame range must be consecutive.");
  }

  const ColorStream& cs = video_->colorStream("down");
  const int w = cs.width();
  const int h = cs.height();

  std::vector<std::pair<int, int>> flowPairs;
  std::string flowPath = fmt::format("{:s}/flow", video_->path());
  for (auto& entry :
      boost::make_iterator_range(fs::directory_iterator(flowPath), {})) {
    std::string fileName = entry.path().stem().string();
    if (fileName.size() != 18 ||
        fileName.substr(0, 5) != "flow_") {
      continue;
    }
    int f0 = std::stoi(fileName.substr(5, 6));
    int f1 = std::stoi(fileName.substr(12, 6));
    flowPairs.emplace_back(f0, f1);
  }

  auto loadFlow = [&](const int frame0, const int frame1)
      -> std::unique_ptr<Mat2f> {
    std::string flowFile = fmt::format(
        "{:s}/flow/flow_{:06d}_{:06d}.raw",
        video_->path(), frame0, frame1);
    if (!fs::exists(flowFile)) {
      return nullptr;
    }
    std::unique_ptr<Mat2f> flow = std::make_unique<Mat2f>();
    freadim(flowFile, *flow);
    if (flow->cols != w || flow->rows != h) {
      return nullptr;
    }

    return flow;
  };

  auto loadFlowMask = [&](const int frame0, const int frame1)
      -> std::unique_ptr<Mat1b> {
    std::string maskFile = fmt::format(
         "{:s}/flow_mask/mask_{:06d}_{:06d}.png",
         video_->path(), frame0, frame1);
    if (!fs::exists(maskFile)) {
      return nullptr;
    }
    std::unique_ptr<Mat1b> mask = std::make_unique<Mat1b>();
    *mask = imread(maskFile, IMREAD_GRAYSCALE);
    if (mask->cols != w || mask->rows != h) {
      return nullptr;
    }

    return mask;
  };

  const DepthStream& srcDs = video_->depthStream(params.sourceDepthStream);
  DepthStream& dstDs = video_->depthStream(params.depthStream);

  for (int frame : params.frameRange) {
    LOG(INFO) << "Processing frame " << frame <<
        " of range " << params.frameRange.toString() << "...";

    const DepthFrame& refDf = srcDs.frame(frame);
    const Vector3f refPosition = refDf.extrinsics.position;
    const Vector3f refForward = refDf.extrinsics.forward();

    Mat1f filteredDepth(h, w);

    // Fetching a stack of (2 * frameRadius + 1) *guidance images*, temporally
    // centered around the current frame.
    const int f0 = std::max(0, frame - params.frameRadius);
    const int f1 =
        std::min(params.frameRange.lastFrame(), frame + params.frameRadius);
    const int temporalWindowSize = f1 - f0 + 1;
    const int middleFrame = frame - f0;
    std::vector<std::unique_ptr<Mat2f>> forwardFlowStack(temporalWindowSize);
    std::vector<std::unique_ptr<Mat1b>> forwardMaskStack(temporalWindowSize);
    std::vector<std::unique_ptr<Mat2f>> backwardFlowStack(temporalWindowSize);
    std::vector<std::unique_ptr<Mat1b>> backwardMaskStack(temporalWindowSize);
    for (int i = 0; i < temporalWindowSize; ++i) {
      if (i >= middleFrame && i < temporalWindowSize - 1) {
        forwardFlowStack[i] = loadFlow(f0 + i, f0 + i + 1);
        forwardMaskStack[i] = loadFlowMask(f0 + i, f0 + i + 1);
        CHECK(forwardFlowStack[i]);
        CHECK(forwardMaskStack[i]);
      }

      if (i <= middleFrame && i > 0) {
        backwardFlowStack[i] = loadFlow(f0 + i, f0 + i - 1);
        backwardMaskStack[i] = loadFlowMask(f0 + i, f0 + i - 1);
        CHECK(backwardFlowStack[i]);
        CHECK(backwardMaskStack[i]);
      }
    }

    std::vector<std::unique_ptr<Mat2f>> farFlowStack;
    std::vector<std::unique_ptr<Mat1b>> farMaskStack;
    std::vector<int> farIndex;
    if (params.farConnections) {
      for (const auto& pair : flowPairs) {
        const int fi = pair.second;
        if (pair.first == frame && (fi < f0 || fi > f1)) {
          farIndex.push_back(fi);
          farFlowStack.push_back(loadFlow(frame, fi));
          farMaskStack.push_back(loadFlowMask(frame, fi));
        }
      }
    }

    struct SampleInfo {
      int frame;
      Vector2f loc;
      float depth;
      float weight;
    };
    std::vector<SampleInfo> samples;

    auto addSample = [&](const Vector2f& loc, const int fi) {
      samples.emplace_back();
      SampleInfo& s = samples.back();
      s.frame = fi;
      s.loc = loc;
      Vector2f ndc(loc.x() / w, loc.y() / h * video_->invAspect());
      Vector3f position =
          video_->project(params.sourceDepthStream, fi, ndc, false);
      s.depth = (position - refPosition).dot(refForward);
    };

    // Looping over the output image pixels.
    for (int y = 0; y < h; ++y) {
      const int y0 = std::max(0, y - params.spatialRadius);
      const int y1 = std::min(h - 1, y + params.spatialRadius);

      float* const dstPtr = (float* const)filteredDepth.ptr(y);
      for (int x = 0; x < w; ++x) {
        const int x0 = std::max(0, x - params.spatialRadius);
        const int x1 = std::min(w - 1, x + params.spatialRadius);

        samples.clear();

        float referenceDepth = FLT_MAX;
        for (int wy = y0; wy <= y1; ++wy) {
          for (int wx = x0; wx <= x1; ++wx) {
            addSample(Vector2f(wx, wy), frame);

            if (wx == x && wy == y) {
              assert(!samples.empty()); // For lint.
              referenceDepth = samples.back().depth;
            }

            // Forward pass
            Vector2f loc = Vector2f(wx, wy);
            for (int fi = frame + 1; fi <= f1; ++fi) {
              const int i = fi - f0;
              Mat2f& flow = *forwardFlowStack[i - 1];
              Mat1b& mask = *forwardMaskStack[i - 1];

              int ix = std::min(int(loc.x() + 0.5f), w - 1);
              int iy = std::min(int(loc.y() + 0.5f), h - 1);

              if (!mask(iy, ix)) {
                break;
              }

              const Vec2f& f = flow(iy, ix);
              loc.x() += f(0);
              loc.y() += f(1);
              ix = loc.x() + 0.5f;
              iy = loc.y() + 0.5f;
              if (ix < 0 || ix >= w || iy < 0 || iy >= h) {
                break;
              }

              addSample(loc, fi);
            }

            // Backward pass
            loc = Vector2f(wx, wy);
            for (int fi = frame - 1; fi >= f0; --fi) {
              const int i = fi - f0;
              Mat2f& flow = *backwardFlowStack[i + 1];
              Mat1b& mask = *backwardMaskStack[i + 1];

              int ix = std::min(int(loc.x() + 0.5f), w - 1);
              int iy = std::min(int(loc.y() + 0.5f), h - 1);

              if (!mask(iy, ix)) {
                break;
              }

              const Vec2f& f = flow(iy, ix);
              loc.x() += f(0);
              loc.y() += f(1);
              ix = loc.x() + 0.5f;
              iy = loc.y() + 0.5f;
              if (ix < 0 || ix >= w || iy < 0 || iy >= h) {
                break;
              }

              addSample(loc, fi);
            }

            for (int i = 0; i < farIndex.size(); ++i) {
              assert(!farFlowStack.empty()); // For lint.
              assert(!farMaskStack.empty()); // For lint.
              Mat2f& flow = *farFlowStack[i];
              Mat1b& mask = *farMaskStack[i];

              loc = Vector2f(wx, wy);
              int ix = std::min(int(loc.x() + 0.5f), w - 1);
              int iy = std::min(int(loc.y() + 0.5f), h - 1);

              if (!mask(iy, ix)) {
                break;
              }

              const Vec2f& f = flow(iy, ix);
              loc.x() += f(0);
              loc.y() += f(1);
              ix = loc.x() + 0.5f;
              iy = loc.y() + 0.5f;
              if (ix < 0 || ix >= w || iy < 0 || iy >= h) {
                break;
              }

              addSample(loc, farIndex[i]);
            }
          }
        }

        // Get depth samples
        float depthSum = 0.f;
        float weightSum = 0.f;
        for (SampleInfo& s : samples) {
          const float value =
              std::max(s.depth, referenceDepth) /
              std::min(s.depth, referenceDepth);

          s.weight = expf(-value * 3.f);

          depthSum += s.depth * s.weight;

          weightSum += s.weight;
        }

        if (params.median) {
          float halfWeight = weightSum / 2.f;
          std::sort(samples.begin(), samples.end(),
              [&](SampleInfo& lhs, SampleInfo& rhs) {
            return (lhs.depth < rhs.depth);
          });
          float cumWeight = 0.f;
          for (const SampleInfo& s : samples) {
            cumWeight += s.weight;
            if (cumWeight >= halfWeight) {
              dstPtr[x] = s.depth;
              break;
            }
          }
        } else {
          if (weightSum > 0.f) {
            dstPtr[x] = depthSum / weightSum;
          } else {
            dstPtr[x] = 0.f;
          }
        }
      }
    }

    dstDs.frame(frame).setDepth(filteredDepth);
  }
}

void DepthVideoProcessor::clipMaxDepth(const Params& params) {
  DepthStream& ds = video_->depthStream(params.depthStream);

  for (int frame : params.frameRange) {
    LOG(INFO) << "Processing frame " << frame <<
        " of range " << params.frameRange.toString() << "...";
    DepthFrame& df = ds.frame(frame);
    if (!df.enabled) {
      continue;
    }
    const Mat1f* depth = df.depth();
    if (!depth) {
      continue;
    }

    Mat1f clipped;
    depth->copyTo(clipped);

    for (int y = 0; y < clipped.rows; ++y) {
      float* ptr = clipped.ptr<float>(y);
      for (int x = 0; x < clipped.cols; ++x) {
        ptr[x] = std::min(ptr[x], params.maxDepth);
      }
    }

    df.setDepth(clipped);
  }
}

std::unique_ptr<FlowConstraintsCollection>
    DepthVideoProcessor::computeConstraints(const Params& params) {
  FlowConstraintsParams fcParams;
  fcParams.frameRange = params.frameRange;
  fcParams.matchSeparation = params.matchSeparation;
  fcParams.minDynamicDistance = params.minDynamicDistance;
  fcParams.doNotUseCache = true;
  return std::make_unique<FlowConstraintsCollection>(*video_, fcParams);
}

void DepthVideoProcessor::resetConstraintStaticFlag(
    FlowConstraintsCollection& constraints) {
  constraints.resetStaticFlag();
}

void DepthVideoProcessor::setConstraintStaticFlagFromDynamicMask(
    const Params& params, FlowConstraintsCollection& constraints) {
  constraints.setStaticFlagFromDynamicMask(params.minDynamicDistance);
}

void DepthVideoProcessor::pruneConstraintStaticFlag(
    const Params& params, FlowConstraintsCollection& constraints) {
  constraints.pruneStaticFlag(params.trackPruneDistance);
}

std::unique_ptr<DepthVideoTrackTable> DepthVideoProcessor::computeTracks(
    const Params& params) {
  const ColorStream& cs = video_->colorStream("down");
  const int w = cs.width();
  const int h = cs.height();

  const ColorStream* dynamicMaskStream = nullptr;
  Vector2f dynamicMaskScale(1.f, 1.f);
  if (video_->hasColorStream("dynamic_mask")) {
    dynamicMaskStream = &video_->colorStream("dynamic_mask");
    dynamicMaskScale = Vector2f(
        dynamicMaskStream->width() / float(w),
        dynamicMaskStream->height() / float(h));
  }

  struct Pixel {
    float cornerStrength;
    Vector2fna pos;
    Pixel(const float c, const Vector2fna& pos) : cornerStrength(c), pos(pos) {
    }
    bool operator<(const Pixel& other) const {
      return cornerStrength > other.cornerStrength;
    }
  };

  auto loadFlow = [&](const int frame) -> std::unique_ptr<Mat2f> {
    std::string flowFile = fmt::format(
        "{:s}/flow/flow_{:06d}_{:06d}.raw",
        video_->path(), frame, frame + 1);
    if (!fs::exists(flowFile)) {
      return nullptr;
    }
    std::unique_ptr<Mat2f> flow = std::make_unique<Mat2f>();
    freadim(flowFile, *flow);
    if (flow->cols != w || flow->rows != h) {
      return nullptr;
    }

    return flow;
  };

  auto loadFlowMask = [&](const int frame) -> std::unique_ptr<Mat1b> {
    std::string maskFile = fmt::format(
         "{:s}/flow_mask/mask_{:06d}_{:06d}.png",
         video_->path(), frame, frame + 1);
    if (!fs::exists(maskFile)) {
      return nullptr;
    }
    std::unique_ptr<Mat1b> mask = std::make_unique<Mat1b>();
    *mask = imread(maskFile, IMREAD_GRAYSCALE);
    if (mask->cols != w || mask->rows != h) {
      return nullptr;
    }

    return mask;
  };

  auto createDiskKernel = [&](const int radius) {
    const int size = 2 * radius + 1;
    Mat1b kernel(size, size);
    for (int y = 0; y < size; ++y) {
      uint8_t* ptr = kernel.ptr<uint8_t>(y);
      for (int x = 0; x < size; ++x) {
        const int rx = x - radius;
        const int ry = y - radius;
        ptr[x] = (sqr(rx) + sqr(ry) <= sqr(radius) ? 255 : 0);
      }
    }
    return kernel;
  };

  auto splatKernel = [&](
      Mat1b& mask, const Mat1b& kernel, const int x, const int y) {
    const int radius = kernel.cols / 2;
    const int mx0 = std::max(0, x - radius);
    const int mx1 = std::min(w - 1, x + radius);
    const int my0 = std::max(0, y - radius);
    const int my1 = std::min(h - 1, y + radius);
    for (int my = my0; my <= my1; ++my) {
      const int dy = my - (y - radius);
      const uint8_t* kernelPtr = kernel.ptr<const uint8_t>(dy);
      uint8_t* maskPtr = mask.ptr<uint8_t>(my);
      for (int mx = mx0; mx <= mx1; ++mx) {
        const int dx = mx - (x - radius);
        if (kernelPtr[dx]) {
          maskPtr[mx] = 255;
        }
      }
    }
  };

  std::vector<Pixel> pixels;
  pixels.reserve(w * h);

  Mat1b spawnKernel = createDiskKernel(params.trackSpawnDistance);
  Mat1b pruneKernel = createDiskKernel(params.trackPruneDistance);

  std::unique_ptr<DepthVideoTrackTable> tracks =
      std::make_unique<DepthVideoTrackTable>();

  std::unique_ptr<Mat2f> flow;
  std::unique_ptr<Mat1b> flowMask;

  for (int frame = 0; frame < video_->numFrames(); ++frame) {
    tracks->addFrame();

    if (!params.frameRange.inRange(frame)) {
      LOG(INFO) <<
          "Skipping frame << " << frame << " (not in specified range).";
      continue;
    }

    LOG(INFO) << "Tracking frame " << frame << "...";

    const Mat* color = cs.frame(frame).image();
    if (!color) {
      continue;
    }

    Mat1f dynamicDistance(h, w);

    if (dynamicMaskStream) {
      const Mat1b* dynamicMask = dynamicMaskStream->frame(frame).image1b();

      const int w = dynamicMask->cols;
      const int h = dynamicMask->rows;

      Mat1b binarized(h, w);
      for (int y = 0; y < h; ++y) {
        const uint8_t* srcPtr = dynamicMask->ptr<const uint8_t>(y);
        uint8_t* dstPtr = binarized.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
          dstPtr[x] = (srcPtr[x] < 127 ? 0 : 255);
        }
      }

      distanceTransform(binarized, dynamicDistance, DIST_L2, DIST_MASK_5);
    } else {
      dynamicDistance = FLT_MAX;
    }

    Mat1b spawnMask(h, w, uint8_t(0));
    Mat1b pruneMask(h, w, uint8_t(0));

    // First, continue tracks from previous frame.
    if (frame > params.frameRange.firstFrame()) {
      std::unique_ptr<Mat2f> flow = loadFlow(frame - 1);
      std::unique_ptr<Mat1b> flowMask = loadFlowMask(frame - 1);

      if (flow && flowMask) {
        for (int trackId : tracks->frame(frame - 1).tracks) {
          auto& t = tracks->track(trackId);
          auto& o0 = t.obs(frame - 1);

          const float fx0 = o0.loc(0) * w;
          const float fy0 = o0.loc(1) / video_->invAspect() * h;
          const int ix0 = std::min(int(fx0 + 0.5f), w - 1);
          const int iy0 = std::min(int(fy0 + 0.5f), h - 1);

          if (!flowMask->at<uint8_t>(iy0, ix0)) {
            continue;
          }

          const Vec2f& f = flow->at<Vec2f>(iy0, ix0);
          const float fx1 = fx0 + f(0);
          const float fy1 = fy0 + f(1);
          const int ix1 = fx1 + 0.5f;
          const int iy1 = fy1 + 0.5f;
          if (ix1 >= 0 && ix1 < w && iy1 >= 0 && iy1 < h) {
            const int ix1s = fx1 * dynamicMaskScale.x();
            const int iy1s = fy1 * dynamicMaskScale.y();
            if (!pruneMask(iy1, ix1) &&
                dynamicDistance(iy1s, ix1s) >= params.minDynamicDistance) {
              Obs o1(fx1 / w, fy1 / h * video_->invAspect());
              tracks->addObs(trackId, frame, o1);
              splatKernel(pruneMask, pruneKernel, ix1, iy1);
              splatKernel(spawnMask, spawnKernel, ix1, iy1);
            }
          }
        }
      }
    }

    // Then, spawn some new tracks if there are large gaps.
    if (frame < params.frameRange.lastFrame()) {
      std::unique_ptr<Mat1b> flowMask = loadFlowMask(frame - 1);

      Mat1f gray;
      cvtColor(*color, gray, COLOR_BGR2GRAY);

      const int blockSize = 3;
      Mat1f cornerResponse;
      cornerMinEigenVal(gray, cornerResponse, blockSize);

      pixels.clear();
      for (int y = 0; y < h; ++y) {
        const float* const cornerPtr = cornerResponse.ptr<const float>(y);
        const uint8_t* const flowMaskPtr =
            (flowMask ? flowMask->ptr<const uint8_t>(y) : nullptr);
        const float* const dynamicDistancePtr =
            dynamicDistance.ptr<const float>(y * dynamicMaskScale.y());
        for (int x = 0; x < w; ++x) {
          const float dd = dynamicDistancePtr[int(x * dynamicMaskScale.x())];
          if ((!flowMaskPtr || flowMaskPtr[x]) &&
              dd > params.minDynamicDistance) {
            pixels.emplace_back(cornerPtr[x],
                Vector2fna(x / float(w), y / float(h) * video_->invAspect()));
          }
        }
      }

      // Sort pixels so that the most corner-like ones come first.
      std::sort(pixels.begin(), pixels.end());

      for (const Pixel& p : pixels) {
        const int x = p.pos.x() * w;
        const int y = p.pos.y() / float(video_->invAspect()) * h;

        if (spawnMask(y, x)) {
          continue;
        }

        tracks->createTrack(frame, Obs(p.pos.x(), p.pos.y()));

        splatKernel(spawnMask, spawnKernel, x, y);
      }
    }
  }

  // Prune short tracks
  for (int trackId = 0; trackId < tracks->numTracks(); ++trackId) {
    if (tracks->hasTrack(trackId)) {
      const Track& t = tracks->track(trackId);
      if (t.length() < params.minTrackLength) {
        tracks->deleteTrack(trackId);
      }
    }
  }

  return tracks;
}

void DepthVideoProcessor::gridXformSplit(const Params& params) {
  // Validate compatibility of old and new xform types.
  if (params.depthXformDesc.depthType != DepthXformType::Grid) {
    throw std::runtime_error("Transform type must be a grid type.");
  }

  DepthStream& ds = video_->depthStream(params.depthStream);
  const XformDescriptor prevDesc = ds.depthXformDesc();
  if (prevDesc.depthType != DepthXformType::Global &&
      prevDesc.depthType != DepthXformType::Grid) {
    throw std::runtime_error("Can only split global or grid type transforms.");
  }

  if (params.depthXformDesc.valueXform != prevDesc.valueXform) {
    throw std::runtime_error(
        "Old and new transforms must use same value transform.");
  }

  if (prevDesc.depthType != DepthXformType::Global && (
      prevDesc.gridSize.x() > params.depthXformDesc.gridSize.x() ||
      prevDesc.gridSize.y() > params.depthXformDesc.gridSize.y())) {
    throw std::runtime_error(
        "New transform must have at least the same number"
        " of rows and columns as the old transform.");
  }

  // Create a copy of the current transforms
  std::vector<std::unique_ptr<Xform>> prevXforms;
  for (int frame = 0; frame < video_->numFrames(); ++frame) {
    prevXforms.push_back(ds.frame(frame).depthXform().clone());
  }

  ds.resetDepthXforms(params.depthXformDesc);

  for (int frame = 0; frame < video_->numFrames(); ++frame) {
    const std::vector<double*>& prevParamBlocks =
        prevXforms[frame]->paramBlocks();
    const std::vector<int>& prevParamBlockSizes =
        prevXforms[frame]->paramBlockSizes();

    const DepthXform& newXform = ds.frame(frame).depthXform();
    const std::vector<double*>& newParamBlocks = newXform.paramBlocks();
    const std::vector<int>& newParamBlockSizes = newXform.paramBlockSizes();

    for (int row = 0; row < params.depthXformDesc.gridSize.y(); ++row) {
      for (int col = 0; col < params.depthXformDesc.gridSize.x(); ++col) {
        const int idx = col + row * params.depthXformDesc.gridSize.x();
        const int N = newParamBlockSizes[idx];
        if (prevDesc.depthType == DepthXformType::Global) {
          assert(prevParamBlocks.size() == 1);
          assert(prevParamBlockSizes[0] == N);
          memcpy(newParamBlocks[idx], prevParamBlocks[0], N * sizeof(double));
        } else if (prevDesc.depthType == DepthXformType::Grid) {
          const int prevRows = prevDesc.gridSize.y();
          const int prevCols = prevDesc.gridSize.x();
          assert(prevRows >= 2);
          assert(prevCols >= 2);

          const int newRows = params.depthXformDesc.gridSize.y();
          const int newCols = params.depthXformDesc.gridSize.x();

          const double maxx = std::nextafter(prevCols - 1, 0.0);
          const double maxy = std::nextafter(prevRows - 1, 0.0);

          const double sx = std::min(
              col / double(newCols - 1) * (prevCols - 1), maxx);
          const double sy = std::min(
              row / double(newRows - 1) * (prevRows - 1), maxy);

          const int ix = static_cast<int>(sx);
          const int iy = static_cast<int>(sy);
          assert(ix >= 0 && ix < newCols - 1);
          assert(iy >= 0 && iy < newRows - 1);

          const double rx = sx - ix;
          const double ry = sy - iy;

          double* b0 = prevParamBlocks[ix + iy * prevCols];
          double* b1 = prevParamBlocks[(ix + 1) + iy * prevCols];
          double* b2 = prevParamBlocks[ix + (iy + 1) * prevCols];
          double* b3 = prevParamBlocks[(ix + 1) + (iy + 1) * prevCols];

          const double w0 = (1.f - rx) * (1.f - ry);
          const double w1 = rx * (1.f - ry);
          const double w2 = (1.f - rx) * ry;
          const double w3 = rx * ry;

          for (int i = 0; i < N; ++i) {
            newParamBlocks[idx][i] =
                b0[i] * w0 + b1[i] * w1 + b2[i] * w2 + b3[i] * w3;
          }
        } else {
          throw std::runtime_error("Unsupported transform type.");
        }
      }
    }
  }
}

void DepthVideoProcessor::resetPoses(const Params& params) {
  DepthStream& ds = video_->depthStream(params.depthStream);
  for (int frame = 0; frame < video_->numFrames(); ++frame) {
    DepthFrame& df = ds.frame(frame);
    df.extrinsics.position = Vector3f::Zero();
    df.extrinsics.orientation = DepthPhoto::Extrinsics::kDefaultOrientation;

    const float focal = params.poseOptimizer.focalLong;
    if (video_->aspect() >= 1.f) {
      df.intrinsics.hFov = std::atan(focal) * 2.f;
      df.intrinsics.vFov = std::atan(focal / video_->aspect()) * 2.f;
    } else {
      df.intrinsics.hFov = std::atan(focal * video_->aspect()) * 2.f;
      df.intrinsics.vFov = std::atan(focal) * 2.f;
    }
  }
}

void DepthVideoProcessor::resetDepthXforms(const Params& params) {
  DepthStream& ds = video_->depthStream(params.depthStream);
  ds.resetDepthXforms(params.depthXformDesc);
}

void DepthVideoProcessor::resetSpatialXforms(const Params& params) {
  DepthStream& ds = video_->depthStream(params.depthStream);
  ds.resetSpatialXforms(params.spatialXformDesc);
}

void DepthVideoProcessor::normalizeDepth(
    const Params& params, const FlowConstraintsCollection& constraints) {
  DepthVideoPoseOptimizer optimizer(video_, params.depthStream);
  optimizer.normalizeDepth(params.poseOptimizer, constraints);
}

void DepthVideoProcessor::optimizePoses(
    const Params& params, const FlowConstraintsCollection& constraints) {
  DepthVideoPoseOptimizer optimizer(video_, params.depthStream);
  optimizer.poseOptimization(params.poseOptimizer, constraints);
}

void DepthVideoProcessor::resetNormalizeOptimize(
    const Params& params, const FlowConstraintsCollection& constraints) {
  resetPoses(params);
  resetDepthXforms(params);
  resetSpatialXforms(params);
  normalizeDepth(params, constraints);
  optimizePoses(params, constraints);
}

}} // namespace facebook::cp
