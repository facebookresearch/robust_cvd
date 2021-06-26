// Copyright 2004-present Facebook. All Rights Reserved.

#include "PoseOptimizer.h"

#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/rotation.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "core/CvUtil.h"
#include "core/Enum_impl.h"
#include "core/Platform.h"
// #include "x3d/photo3d/core/DepthMapConverter.h"

#include "ColorStream.h"
#include "DepthStream.h"
#include "DepthVideo.h"
#include "FlowConstraints.h"
#include "Processor.h"

// Undefine this to enable checking for NaN values in the ceres loss functions.
//#define DEBUG_LOSSES

using namespace Eigen;
using namespace cv;

template <typename T>
using Vector2 = Matrix<T, 2, 1>;

template <typename T>
using Vector3 = Matrix<T, 3, 1>;

namespace fs = boost::filesystem;

namespace facebook {
namespace cp {

using Intrinsics = DepthPhoto::Intrinsics;
using Extrinsics = DepthPhoto::Extrinsics;

const EnumStrings<StaticLossType> staticLossTypeStrs = {
    ENUM_STR(StaticLossType, Euclidean),
    ENUM_STR(StaticLossType, ReproDisparity),
    ENUM_STR(StaticLossType, ReproLogDepth),
    ENUM_STR(StaticLossType, ReproDepthRatio),
};
MAKE_VALIDATOR(StaticLossType, staticLossTypeStrs);

const EnumStrings<SmoothLossType> smoothLossTypeStrs = {
    ENUM_STR(SmoothLossType, EuclideanLaplacian),
    ENUM_STR(SmoothLossType, ReproDisparityLaplacian),
    ENUM_STR(SmoothLossType, ReproDepthRatioConsistency),
    ENUM_STR(SmoothLossType, ReproLogDepthConsistency),
};
MAKE_VALIDATOR(SmoothLossType, smoothLossTypeStrs);

const EnumStrings<IntrinsicsOptimization> intrinsicsOptimizationStrs = {
    ENUM_STR(IntrinsicsOptimization, Fixed),
    ENUM_STR(IntrinsicsOptimization, Shared),
    ENUM_STR(IntrinsicsOptimization, PerFrame),
};
MAKE_VALIDATOR(IntrinsicsOptimization, intrinsicsOptimizationStrs);

template <typename T>
void logValue(const std::string& name, const T& value) {
  LOG(INFO) << name << ": " << value;
}

template <typename T>
void logValue2(const std::string& name, const T& value) {
  LOG(INFO) << name << ": " << value.x() << ", " << value.y();
}

template <typename T>
void logValue3(const std::string& name, const T& value) {
  LOG(INFO) << name << ": " <<
      value.x() << ", " << value.y() << ", " << value.z();
}

// Insert all elements of src at the end of dst.
template<typename T>
void insert(std::vector<T>& dst, const std::vector<T>& src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

// This structure contains information about an observation (i.e., an element of
// a track or pairwise match).
struct Observation {
  Vector2f locNdc; // Scaled to [-1,1] x [-1,1]

  float sourceDepth;

  std::unique_ptr<DepthFunctor> depthFn;
  std::unique_ptr<SpatialFunctor> spatialFn;

  // Sizes and pointers of pose-, depth-xform-, and spatial-xform-parameters.
  std::vector<int> paramBlockSizes;
  std::vector<double*> paramBlockPtrs;

  // Input loc is in [0,1] x [0,invAsepct] format.
  Observation(DepthFrame& df, double* poseParamBlock, const Vector2f& loc) {
    locNdc = Vector2f(
        -1.f + 2.f * loc.x(), 1.f - 2.f * loc.y() / df.invAspect());

    const Mat1f* depthImg = df.sourceDepth();
    if (!depthImg) {
      throw std::runtime_error("Missing depth image.");
    }

    const int locImgX = loc.x() * depthImg->cols;
    const int locImgY = loc.y() / df.invAspect() * depthImg->rows;
    sourceDepth = depthImg->at<float>(locImgY, locImgX);

    depthFn = df.depthXform().createFunctor(sourceDepth, locNdc);
    spatialFn = df.spatialXform().createFunctor(locNdc);

    paramBlockSizes.push_back(6);
    insert(paramBlockSizes, depthFn->paramBlockSizes());
    insert(paramBlockSizes, spatialFn->paramBlockSizes());

    paramBlockPtrs.push_back(poseParamBlock);
    insert(paramBlockPtrs, depthFn->paramBlocks());
    insert(paramBlockPtrs, spatialFn->paramBlocks());
  }
};

namespace {

// This struct has the unpacked parameters for an observation.
template <typename T>
struct ObservationParams {
  T const* pose; // 6-tuple with camera position and rotation.
  T const* const* depthXform; // Multiple N-tuples with depth xform params.
  T const* const* spatialXform; // Multiple N-tuples with spatial xform params.
};

// This helper function fetches the unpacked parameters for an observation from
// a stacked vector, starting at and updating the provided offset.
template <typename T>
ObservationParams<T> unpack(
    int& offset, T const* const* params, const Observation& obs) {
  ObservationParams<T> p;

  p.pose = params[offset];
  offset += 1;

  p.depthXform = &params[offset];
  offset += obs.depthFn->numParamBlocks();

  p.spatialXform = &params[offset];
  offset += obs.spatialFn->numParamBlocks();

  return p;
}

// Obtain the camera-coordinate point of an observation. The xy-components
// contain the ndc in [-1, 1] x [-1, 1] range, and the z-component the linear
// depth.
template <typename T>
Vector3<T> obsToCamera(
    const Observation& obs, ObservationParams<T>& obsParams) {
  T depth = (*obs.depthFn)(obsParams.depthXform);
  Vector2<T> warp = (*obs.spatialFn)(obsParams.spatialXform);
  return Vector3<T> (
      T(obs.locNdc.x()) + warp.x(),
      T(obs.locNdc.y()) + warp.y(),
      depth);
}

// Project a camera-coordinate point out to world-space.
template <typename T>
Vector3<T> cameraToWorld(
    const Vector3<T>& pointCam,
    const Vector2<T>& focal,
    T const* poseParams) {
  T dirCam[3] {
      pointCam.x() * focal.x(),
      pointCam.y() * focal.y(),
      T(-1.0) };

  T dirWorld[3];
  ceres::AngleAxisRotatePoint(poseParams + 3, dirCam, dirWorld);

  const T& depth = pointCam.z();
  return Vector3<T>(
      poseParams[0] + dirWorld[0] * depth,
      poseParams[1] + dirWorld[1] * depth,
      poseParams[2] + dirWorld[2] * depth);
}

// Project a world-space point to a camera's coordinate system.
template <typename T>
Vector3<T> worldToCamera(
    const Vector3<T>& pointWorld,
    const Vector2<T>& focal,
    T const* poseParams) {
  T pointRel[3];
  for (int i = 0; i < 3; ++i) {
    pointRel[i] = pointWorld(i) - poseParams[i];
  }

  T cameraRotationInv[3];
  for (int i = 0; i < 3; ++i) {
    cameraRotationInv[i] = -poseParams[i + 3];
  }

  T pointCam[3];
  ceres::AngleAxisRotatePoint(cameraRotationInv, pointRel, pointCam);

  // Note: dividing by the negative Z component, since we flipped the front
  // vector to make a rotation that is representable in angle-axis form.
  const T& depth = -pointCam[2];

  return Vector3<T>(
      pointCam[0] / depth / focal.x(),
      pointCam[1] / depth / focal.y(),
      depth);
}

struct StaticSceneCost {
  template <typename T> using Vector2 = Matrix<T, 2, 1>;
  template <typename T> using Vector3 = Matrix<T, 3, 1>;

  StaticSceneCost(
      std::shared_ptr<Observation> obs0, std::shared_ptr<Observation> obs1,
      const double fixedVFocal, const double aspect,
      const IntrinsicsOptimization intrOpt, const StaticLossType lossType,
      const double spatialWeight, const double depthWeight)
      : obs0(obs0), obs1(obs1), fixedVFocal(fixedVFocal),
        aspect(aspect), intrOpt(intrOpt), lossType(lossType),
        spatialWeight(spatialWeight), depthWeight(depthWeight) {
  }

  template <typename T>
  bool operator()(T const* const* params, T* residuals) const {
    int paramOffset = 0;

    // Unpack the observation parameters.
    ObservationParams<T> params0 = unpack(paramOffset, params, *obs0);
    ObservationParams<T> params1 = unpack(paramOffset, params, *obs1);

    // Get the focal lengths.
    Vector2<T> focal0, focal1;
    if (intrOpt == IntrinsicsOptimization::Shared) {
      focal0.y() = focal1.y() = params[paramOffset++][0];
    } else if (intrOpt == IntrinsicsOptimization::PerFrame) {
      focal0.y() = params[paramOffset++][0];
      focal1.y() = params[paramOffset++][0];
    } else {
      focal0.y() = focal1.y() = T(fixedVFocal);
    }

    focal0.x() = focal0.y() * aspect;
    focal1.x() = focal1.y() * aspect;

    Vector3<T> pointCam0 = obsToCamera(*obs0, params0);
    Vector3<T> pointWorld0 =
        cameraToWorld(pointCam0, focal0, params0.pose);

    Vector3<T> pointCam1 = obsToCamera(*obs1, params1);

    Map<Vector3<T>> staticLoss(residuals);

    if (lossType == StaticLossType::Euclidean) {
      // Computing the static loss in world space coordinates --- this does not
      // produce good results. I am not sure why...?
      Vector3<T> pointWorld1 = cameraToWorld(pointCam1, focal1, params1.pose);

      staticLoss = pointWorld1 - pointWorld0;

      // I tried to weigh the loss by the point disparity. This didn't help...
      //constexpr static double epsilon = 1e-6;
      //T disp0 = 1.0 / max(pointCam0.z(), T(epsilon));
      //staticLoss *= disp0;
    } else {
      // Computing the static loss in camera 1's coordinate system.
      Vector3<T> pointCam0To1 =
          worldToCamera(pointWorld0, focal1, params1.pose);

      staticLoss.x() = (pointCam0To1.x() - pointCam1.x()) * T(spatialWeight);
      staticLoss.y() = (pointCam0To1.y() - pointCam1.y()) * T(spatialWeight);

      // Measuring the depth error in some way that avoids bias is important.
      if (lossType == StaticLossType::ReproDisparity) {
        constexpr static double epsilon = 1e-6;
        T reproDisp = 1.0 / max(pointCam0To1.z(), T(epsilon));
        T disp1 = 1.0 / max(pointCam1.z(), T(epsilon));

        staticLoss.z() = (reproDisp - disp1) * T(depthWeight);
      } else {
        T maxDepth = max(pointCam0To1.z(), pointCam1.z());
        T minDepth = min(pointCam0To1.z(), pointCam1.z());

        if (lossType == StaticLossType::ReproDepthRatio) {
          staticLoss.z() = (maxDepth / minDepth - 1.0) * depthWeight;
        } else if (lossType == StaticLossType::ReproLogDepth) {
          staticLoss.z() = log(minDepth / maxDepth) * depthWeight;
        } else {
          throw std::runtime_error("Invalid loss type.");
        }
      }
    }

    return true;
  }

  std::shared_ptr<Observation> obs0;
  std::shared_ptr<Observation> obs1;

  const double fixedVFocal;
  const double aspect;
  const IntrinsicsOptimization intrOpt;
  const StaticLossType lossType;
  const double spatialWeight;
  const double depthWeight;
};

struct SceneFlowSmoothnessLoss {
  SceneFlowSmoothnessLoss(
      std::shared_ptr<Observation> obs0,
      std::shared_ptr<Observation> obs1,
      std::shared_ptr<Observation> obs2,
      const double fixedVFocal, const double aspect,
      const IntrinsicsOptimization intrOpt, SmoothLossType lossType)
      : obs0(obs0), obs1(obs1), obs2(obs2), fixedVFocal(fixedVFocal),
        aspect(aspect), intrOpt(intrOpt), lossType(lossType) {
  }

  template <typename T>
  bool operator()(T const* const* params, T* residuals) const {
    int paramOffset = 0;

    // Unpack the observation parameters.
    ObservationParams<T> params0 = unpack(paramOffset, params, *obs0);
    Vector3<T> pointCam0 = obsToCamera(*obs0, params0);

    ObservationParams<T> params1 = unpack(paramOffset, params, *obs1);
    Vector3<T> pointCam1 = obsToCamera(*obs1, params1);

    ObservationParams<T> params2 = unpack(paramOffset, params, *obs2);
    Vector3<T> pointCam2 = obsToCamera(*obs2, params2);

    // Get the focal lengths.
    Vector2<T> focal0, focal1, focal2;
    if (intrOpt == IntrinsicsOptimization::Shared) {
      focal0.y() = focal1.y() = focal2.y() = params[paramOffset++][0];
    } else if (intrOpt == IntrinsicsOptimization::PerFrame) {
      focal0.y() = params[paramOffset++][0];
      focal1.y() = params[paramOffset++][0];
      focal2.y() = params[paramOffset++][0];
    } else {
      focal0.y() = focal1.y() = focal2.y() = T(fixedVFocal);
    }

    focal0.x() = focal0.y() * aspect;
    focal1.x() = focal1.y() * aspect;
    focal2.x() = focal2.y() * aspect;

    Map<Vector3<T>> smoothLoss(residuals);

    if (lossType == SmoothLossType::EuclideanLaplacian) {
      // Similaryly to the static scene loss, this does not work well if we
      // compare points in world-space coordinates. I don't understand why that is
      // the case... Even weighting the loss by disparity does not fix the
      // problem.
      Vector3<T> pointWorld0 = cameraToWorld(pointCam0, focal0, params0.pose);
      Vector3<T> pointWorld1 = cameraToWorld(pointCam1, focal1, params1.pose);
      Vector3<T> pointWorld2 = cameraToWorld(pointCam2, focal2, params2.pose);

      smoothLoss = pointWorld0 + pointWorld2 - 2.0 * pointWorld1;
    } else {
      Vector3<T> pointWorld0 = cameraToWorld(pointCam0, focal0, params0.pose);
      Vector3<T> pointWorld2 = cameraToWorld(pointCam2, focal2, params2.pose);

      Vector3<T> pointCam0To1 = worldToCamera(pointWorld0, focal1, params1.pose);
      Vector3<T> pointCam2To1 = worldToCamera(pointWorld2, focal1, params1.pose);

      smoothLoss.x() =
          (pointCam0To1.x() + pointCam2To1.x() - pointCam1.x() * 2.0) /
          focal1.y();
      smoothLoss.y() =
          (pointCam0To1.y() + pointCam2To1.y() - pointCam1.y() * 2.0) /
          focal1.y();

      if (lossType == SmoothLossType::ReproDisparityLaplacian) {
        constexpr static double epsilon = 1e-6;
        T repro0To1Disp = 1.0 / max(pointCam0To1.z(), T(epsilon));
        T disp1 = 1.0 / max(pointCam1.z(), T(epsilon));
        T repro2To1Disp = 1.0 / max(pointCam2To1.z(), T(epsilon));

        smoothLoss.z() = repro0To1Disp + repro2To1Disp - disp1 * 2.0;
      } else {
        T baseDepth = pointCam1.z();
        T otherDepth = pointCam0To1.z() + pointCam2To1.z() - pointCam1.z();

        T maxDepth = max(baseDepth, otherDepth);
        T minDepth = min(baseDepth, otherDepth);

        if (lossType == SmoothLossType::ReproDepthRatioConsistency) {
          smoothLoss.z() = (maxDepth / minDepth - 1.0);
        } else if (lossType == SmoothLossType::ReproLogDepthConsistency) {
          smoothLoss.z() = log(minDepth / maxDepth);
        } else {
          throw std::runtime_error("Invalid loss type.");
        }
      }
    }

    return true;
  }

  std::shared_ptr<Observation> obs0;
  std::shared_ptr<Observation> obs1;
  std::shared_ptr<Observation> obs2;

  const double fixedVFocal;
  const double aspect;
  const IntrinsicsOptimization intrOpt;
  const SmoothLossType lossType;
};

struct DisparityDissimilarityCost {
  DisparityDissimilarityCost(
      std::unique_ptr<DepthFunctor>&& depth0Functor,
      std::unique_ptr<DepthFunctor>&& depth1Functor)
      : depth0Functor(std::move(depth0Functor)),
        depth1Functor(std::move(depth1Functor)) {
  }

  template<typename T>
  bool operator()(T const* const* params, T* residuals) const {
    int offset = 0;
    T const* const* xform0Params = &params[offset];
    offset += depth0Functor->numParamBlocks();
    T const* const* xform1Params = &params[offset];

    T depth0 = (*depth0Functor)(xform0Params);
    T depth1 = (*depth1Functor)(xform1Params);

    T epsilon = T(1e-6);
    T disp0 = 1.0 / max(depth0, epsilon);
    T disp1 = 1.0 / max(depth1, epsilon);

    residuals[0] = disp0 - disp1;

#ifdef DEBUG_LOSSES
    if (ceres::IsNaN(depth0) || ceres::IsNaN(depth1) ||
        ceres::IsNaN(disp0) || ceres::IsNaN(disp1) ||
        ceres::IsNaN(residuals[0])) {
      LOG(ERROR) << "NaN in DisparityDissimilarityCost.";
    }
#endif

    return true;
  }

  std::unique_ptr<DepthFunctor> depth0Functor;
  std::unique_ptr<DepthFunctor> depth1Functor;
};

struct ParameterRegularizationCost {
  explicit ParameterRegularizationCost(const int size) : size(size) {
  }

  template<typename T>
  bool operator()(T const* const* params, T* residuals) const {
    for (int i = 0; i < size; ++i) {
      residuals[i] = params[0][i] - T(2.0) * params[1][i] + params[2][i];
#ifdef DEBUG_LOSSES
      if (ceres::IsNaN(residuals[i])) {
        LOG(ERROR) << "NaN in ParameterRegularizationCost.";
      }
#endif
    }

    return true;
  }

  int size = 0;
};

// This cost function constrains the transformed depth at specific points to
// match a target value. The error is measured in disparity space, i.e., the
// reciprocal of depth.
struct TargetDisparityCost {
  TargetDisparityCost(
      std::unique_ptr<DepthFunctor>&& depthFunctor,
      const double targetDisparity)
      : depthFunctor(std::move(depthFunctor)),
        targetDisparity(targetDisparity) {
  }

  template<typename T>
  bool operator()(T const* const* params, T* residuals) const {
    T depth = (*depthFunctor)(params);

    T epsilon = T(1e-6);
    T disparity = 1.0 / max(depth, epsilon);

    residuals[0] = disparity - targetDisparity;

#ifdef DEBUG_LOSSES
    if (ceres::IsNaN(depth) || ceres::IsNaN(disparity) ||
        ceres::IsNaN(residuals[0])) {
      LOG(ERROR) << "NaN in TargetDisparityCost.";
    }
#endif

    return true;
  }

  std::unique_ptr<DepthFunctor> depthFunctor;
  double targetDisparity = 0.0;
};

// This cost function constraints the focal length to match a target value.
struct TargetFocalCost {
  TargetFocalCost(const double targetFocal) : targetFocal(targetFocal) {
  }

  template <typename T>
  bool operator()(T const* const* params, T* residuals) const {
    T focal = params[0][0];
    residuals[0] = focal - T(targetFocal);

    return true;
  }

  double targetFocal = 0.0;
};

// This cost function constrains the deformation cost of a frame's transform.
struct DeformationCost {
  DeformationCost(Xform* xform, const double baseWeight)
      : xform(xform), baseWeight(baseWeight) {}

  template<typename T>
  bool operator()(T const* const* params, T* residuals) const {
    xform->computeDeformationCost(params, residuals);

    const int numResiduals = xform->numDeformationCostResiduals();
    for (int i = 0; i < numResiduals; ++i) {
      residuals[i] *= baseWeight;
    }

    return true;
  }

  Xform* xform;
  double baseWeight = 0.0;
};

// This cost is similar to the (non-adaptive) DeformationCost above, but we're
// applying spatially varying weights, to apply more smoothness to the dynamic
// parts of the scene.
struct AdaptiveDeformationCost {
  AdaptiveDeformationCost(
      Xform* xform, const Mat1b& dynamicMask,
      const double baseWeight, const double adaptiveWeight)
      : xform(xform), baseWeight(baseWeight), adaptiveWeight(adaptiveWeight) {
    const XformDescriptor& desc = xform->desc();
    gridSize = desc.gridSize;

    if (desc.type != XformType::Depth ||
        desc.depthType != DepthXformType::Grid) {
      throw std::runtime_error(
          "Adaptive deformation cost is only implemented for grid transforms.");
    }

    const int gw = desc.gridSize.x();
    const int gh = desc.gridSize.y();

    Mat1d dynamicWeights(gh, gw);
    dynamicWeights = 0.0;

    Mat1d staticWeights(gh, gw);
    staticWeights = 0.0;

    const int dw = dynamicMask.cols;
    const int dh = dynamicMask.rows;

    for (int y = 0; y < dh; ++y) {
      double fy = double(y) * (gh - 1) / dh;
      int iy = int(fy);
      double ry = fy - iy;

      const uint8_t* dynamicMaskPtr = dynamicMask.ptr<const uint8_t>(y);

      for (int x = 0; x < dw; ++x) {
        double fx = double(x) * (gw - 1) / dw;
        int ix = int(fx);
        double rx = fx - ix;

        const double w0 = (1.0 - rx) * (1.0 - ry);
        const double w1 = rx * (1.0 - ry);
        const double w2 = (1.0 - rx) * ry;
        const double w3 = rx * ry;

        // Either add to the static or dynamic weights, depending on the mask.
        Mat1d& w = (dynamicMaskPtr[x] > 127 ? staticWeights : dynamicWeights);

        w(iy, ix) += w0;
        w(iy, ix + 1) += w1;
        w(iy + 1, ix) += w2;
        w(iy + 1, ix + 1) += w3;
      }
    }

    weights.create(gh, gw);
    for (int y = 0; y < gh; ++y) {
      for (int x = 0; x < gw; ++x) {
        weights(y, x) =
            dynamicWeights(y, x) / (dynamicWeights(y, x) + staticWeights(y, x));
      }
    }
  }

  template<typename T>
  bool operator()(T const* const* params, T* residuals) const {
    xform->computeDeformationCost(params, residuals);

    // Modulate the residuals by the adaptive weights. Note, that the order of
    // the constraints is the same as in computeGridDeformationCost().
    int idx = 0;
    for (int z = 0; z < gridSize.z(); ++z) {
      for (int y = 0; y < gridSize.y(); ++y) {
        for (int x = 0; x < gridSize.x(); ++x) {
          const double w0 = weights(y, x);

          if (x > 0) {
            const double w1 = weights(y, x - 1);
            residuals[idx++] *= baseWeight + std::max(w0, w1) * adaptiveWeight;
          }
          if (y > 0) {
            const double w1 = weights(y - 1, x);
            residuals[idx++] *= baseWeight + std::max(w0, w1) * adaptiveWeight;
          }
          if (z > 0) {
            residuals[idx++] *= baseWeight + w0 * adaptiveWeight;
          }
        }
      }
    }

    return true;
  }

  Xform* xform;
  double baseWeight = 0.0;
  double adaptiveWeight = 0.0;
  Vector3i gridSize;
  Mat1d weights;
};

} // Anonymous namespace.

void DepthVideoPoseOptimizer::Params::addCommandLineOptions() {
  ADD_OPTION(frameRange)
  ADD_OPTION(maxIterations)
  ADD_OPTION(numThreads)
  ADD_OPTION(numSteps)
  ADD_OPTION(robustness)

  ADD_OPTION(staticLossType)
  ADD_OPTION(staticSpatialWeight)
  ADD_OPTION(staticDepthWeight)

  ADD_OPTION(smoothLossType)
  ADD_OPTION(smoothStaticWeight)
  ADD_OPTION(smoothDynamicWeight)

  ADD_OPTION(positionReg)
  ADD_OPTION(scaleReg)
  ADD_OPTION(scaleRegGridSize)
  ADD_OPTION(depthDeformRegInitial)
  ADD_OPTION(depthDeformRegFinal)
  ADD_OPTION(adaptiveDeformationCost)
  ADD_OPTION(spatialDeformReg)
  ADD_OPTION(graduateDepthDeformReg)
  ADD_OPTION(focalReg)
  ADD_OPTION(coarseToFine)
  ADD_OPTION(ctfLong)
  ADD_OPTION(ctfShort)
  ADD_OPTION(deferredSpatialOpt)
  ADD_OPTION(dsoLong)
  ADD_OPTION(dsoShort)
  ADD_OPTION(focalLong)
  ADD_OPTION(intrOpt)
  ADD_OPTION(fixPoses)
  ADD_OPTION(fixDepthXforms)
  ADD_OPTION(fixSpatialXforms)
  ADD_OPTION(normalizeDepthFromFirstFrame)
}

void DepthVideoPoseOptimizer::Params::printParams() const {
  DepthVideoPoseOptimizer::Params defaultParams;

  printParam("frameRange",
      (frameRange.frames.empty() ? "" : frameRange.toString()));
  PRINT_PARAM_IF_NEQ(maxIterations)
  PRINT_PARAM_IF_NEQ(numThreads)
  PRINT_PARAM_IF_NEQ(numSteps)
  PRINT_PARAM_IF_NEQ(robustness)

  printParamIfNeq(
      "staticLossType", staticLossType,
      defaultParams.staticLossType, staticLossTypeStrs);
  PRINT_PARAM_IF_NEQ(staticSpatialWeight)
  PRINT_PARAM_IF_NEQ(staticDepthWeight)

  printParamIfNeq(
      "smoothLossType", smoothLossType,
      defaultParams.smoothLossType, smoothLossTypeStrs);
  PRINT_PARAM_IF_NEQ(smoothStaticWeight)
  PRINT_PARAM_IF_NEQ(smoothDynamicWeight)

  PRINT_PARAM_IF_NEQ(positionReg)
  PRINT_PARAM_IF_NEQ(scaleReg)
  PRINT_PARAM_IF_NEQ(scaleRegGridSize)
  PRINT_PARAM_IF_NEQ(depthDeformRegInitial)
  PRINT_PARAM_IF_NEQ(depthDeformRegFinal)
  PRINT_PARAM_IF_NEQ(adaptiveDeformationCost)
  PRINT_PARAM_IF_NEQ(spatialDeformReg)
  PRINT_PARAM_IF_NEQ(graduateDepthDeformReg)
  PRINT_PARAM_IF_NEQ(focalReg)
  PRINT_PARAM_IF_NEQ(coarseToFine)
  PRINT_PARAM_IF_NEQ(ctfLong)
  PRINT_PARAM_IF_NEQ(ctfShort)
  PRINT_PARAM_IF_NEQ(deferredSpatialOpt)
  PRINT_PARAM_IF_NEQ(dsoLong)
  PRINT_PARAM_IF_NEQ(dsoShort)
  PRINT_PARAM_IF_NEQ(focalLong)
  printParamIfNeq(
      "intrOpt", intrOpt, defaultParams.intrOpt, intrinsicsOptimizationStrs);
  PRINT_PARAM_IF_NEQ(fixPoses)
  PRINT_PARAM_IF_NEQ(fixDepthXforms)
  PRINT_PARAM_IF_NEQ(fixSpatialXforms)
  PRINT_PARAM_IF_NEQ(normalizeDepthFromFirstFrame)
}

void DepthVideoPoseOptimizer::Params::resolve(const int numFrames) {
  frameRange.resolve(numFrames);
}

DepthVideoPoseOptimizer::DepthVideoPoseOptimizer(
    DepthVideo* const video, const int depthStream)
    : video_(video), depthStream_(depthStream) {
  numFrames_ = video_->numFrames();

  // Convert poses to optimizer internal representation
  poseParams_.resize(numFrames_);
  for (int frame = 0; frame < numFrames_; ++frame) {
    const DepthFrame& df = video_->depthFrame(depthStream_, frame);
    std::array<double, 7>& pose = poseParams_[frame];

    // Storing camera position in components [0, 1, 2].
    pose[0] = df.extrinsics.position.x();
    pose[1] = df.extrinsics.position.y();
    pose[2] = df.extrinsics.position.z();

    // Storing camera orientation in angle-axis representation in components
    // [3, 4, 5]. Note that we are flipping the front vector to make "rotation"
    // a true rotation matrix that can be represented (otherwise "rotation"
    // would represent a flipped rotation that is not representable in
    // quaternion or angle-axis format.)
    const Quaterniond orientation = df.extrinsics.orientation.cast<double>();
    const Vector3d right = orientation * Vector3d::UnitX();
    const Vector3d up = orientation * Vector3d::UnitY();
    const Vector3d front = orientation * -Vector3d::UnitZ();

    Matrix3d rotation;
    rotation.col(0) = right;
    rotation.col(1) = up;
    rotation.col(2) = -front;

    ceres::RotationMatrixToAngleAxis(rotation.data(), &pose[3]);

    pose[6] = std::tan(df.intrinsics.vFov / 2.0);
  }
}

DepthVideoPoseOptimizer::~DepthVideoPoseOptimizer() {
}

void DepthVideoPoseOptimizer::poseOptimization(
    const Params& params, const FlowConstraintsCollection& constraints) {
  LOG(INFO) << "------------------------";
  LOG(INFO) << "Pose optimization (depth stream " << depthStream_ << ")...";

  params.printParams();

  int ctfRows = params.ctfLong;
  int ctfCols = params.ctfShort;
  int dsoRows = params.dsoLong;
  int dsoCols = params.dsoShort;
  if (video_->aspect() >= 1.f) {
    std::swap(ctfCols, ctfRows);
    std::swap(dsoCols, dsoRows);
  }

  auto gridSize = [&](const XformDescriptor& desc) {
    if (desc.depthType == DepthXformType::Grid) {
      return Vector3i(desc.gridSize.x(), desc.gridSize.y(), desc.gridSize.z());
    } else {
      return Vector3i(1, 1, 1);
    }
  };

  // Get initial grid size.
  const DepthStream& ds = video_->depthStream(depthStream_);
  const Vector3i initGrid = gridSize(ds.depthXformDesc());

  if (params.deferredSpatialOpt) {
    LOG(INFO) << "Setting spatial transforms to identity transform.";
    DepthVideoProcessor::Params pp;
    pp.depthStream = depthStream_;
    pp.spatialXformDesc.type = XformType::Spatial;
    pp.spatialXformDesc.spatialType = SpatialXformType::Identity;

    DepthVideoProcessor processor(video_);
    processor.resetSpatialXforms(pp);
  }

  for (int step = 0; step < params.numSteps; ++step) {
    LOG(INFO) << "----------------";
    LOG(INFO) << "Step " << (step + 1) << " / " << params.numSteps << "...";

    const double stepIter =
        (params.numSteps > 1 ? step / double(params.numSteps - 1) : 0.0);

    double depthDeformReg = params.depthDeformRegFinal;

    if (params.graduateDepthDeformReg) {
      const double defRegInit = log(params.depthDeformRegInitial);
      const double defRegFinal = log(params.depthDeformRegFinal);

      depthDeformReg = exp(defRegInit + (defRegFinal - defRegInit) * stepIter);
    }

    LOG(INFO) << "Depth Deformation regularization: " << depthDeformReg;

    poseOptimizationStep(params, constraints, depthDeformReg);

    if (params.coarseToFine && step < params.numSteps - 1) {
      double ctfIter = (step + 1) / double(params.numSteps - 1);
      const Vector3i curGrid = gridSize(ds.depthXformDesc());

      DepthVideoProcessor::Params splitParams;
      splitParams.depthStream = depthStream_;
      splitParams.depthXformDesc = ds.depthXformDesc();
      if (splitParams.depthXformDesc.depthType == DepthXformType::Global) {
        splitParams.depthXformDesc.depthType = DepthXformType::Grid;
      }

      Vector3i& newGrid = splitParams.depthXformDesc.gridSize;
      newGrid.x() =
          int(initGrid.x() + (ctfCols - initGrid.x()) * ctfIter + 0.5);
      newGrid.y() =
          int(initGrid.y() + (ctfRows - initGrid.y()) * ctfIter + 0.5);
      newGrid.z() = initGrid.z();
      LOG(INFO) << "Splitting grid " <<
          curGrid.x() << " x " << curGrid.y() << " x " << curGrid.z() <<
          " --> " <<
          newGrid.x() << " x " << newGrid.y() << " x " << newGrid.z() << "...";

      DepthVideoProcessor processor(video_);
      processor.gridXformSplit(splitParams);
    }
  }

  if (params.deferredSpatialOpt) {
    LOG(INFO) << "Setting spatial transforms to bicubic grid.";
    DepthVideoProcessor::Params pp;
    pp.depthStream = depthStream_;
    pp.spatialXformDesc.type = XformType::Spatial;
    pp.spatialXformDesc.spatialType = SpatialXformType::BicubicGrid;
    pp.spatialXformDesc.gridSize.y() = dsoRows;
    pp.spatialXformDesc.gridSize.x() = dsoCols;

    DepthVideoProcessor processor(video_);
    processor.resetSpatialXforms(pp);

    poseOptimizationStep(params, constraints, params.depthDeformRegFinal);
  }
}

void DepthVideoPoseOptimizer::poseOptimizationStep(
    const Params& params,
    const FlowConstraintsCollection& constraints,
    const double depthDeformReg) {
  LOG(INFO) << "Building problem...";
  problem_ = std::make_unique<ceres::Problem>();

  addStaticSceneLoss(params, constraints);

  if (params.smoothStaticWeight > 0.0 || params.smoothDynamicWeight > 0.0) {
    addSceneFlowSmoothnessLoss(params, constraints);
  }

  if (params.positionReg > 0.0) {
    addPositionRegularization(params);
  }

  if (depthDeformReg > 0.0) {
    addDepthDeformRegularization(params, depthDeformReg);
  }

  if (params.spatialDeformReg > 0.0) {
    addSpatialDeformRegularization(params);
  }

  if (params.fixPoses) {
    for (int frame : params.frameRange) {
      double* values = poseParams_[frame].data();
      if (problem_->HasParameterBlock(values)) {
        problem_->SetParameterBlockConstant(values);
      }
    }
  }

  if (params.fixDepthXforms) {
    DepthStream& ds = video_->depthStream(depthStream_);
    for (int frame : params.frameRange) {
      DepthXform& x = ds.frame(frame).depthXform();
      for (double* block : x.paramBlocks()) {
        problem_->SetParameterBlockConstant(block);
      }
    }
  } else {
    if (params.scaleReg > 0.0) {
      // Compute a good initialization for the first frame's transform and fix
      // it.
      addScaleRegularization(params);
    }
  }

  if (params.fixSpatialXforms) {
    DepthStream& ds = video_->depthStream(depthStream_);
    for (int frame : params.frameRange) {
      SpatialXform& x = ds.frame(frame).spatialXform();
      for (double* block : x.paramBlocks()) {
        problem_->SetParameterBlockConstant(block);
      }
    }
  }

  if (params.focalReg > 0.0) {
    addFocalRegularization(params);
  }

  LOG(INFO) << "Solving...";
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = params.maxIterations;
  options.num_threads = params.numThreads;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &*problem_, &summary);
  LOG(INFO) << summary.BriefReport();

  // Writing back camera poses
  for (int frame : params.frameRange) {
    const std::array<double, 7>& pose = poseParams_[frame];

    Extrinsics extr;
    extr.position = Vector3f(pose[0], pose[1], pose[2]);

    Matrix3d rotation;
    ceres::AngleAxisToRotationMatrix(&pose[3], rotation.data());

    extr.orientation = Quaterniond(rotation).cast<float>();
    DepthFrame& df = video_->depthStream(depthStream_).frame(frame);
    df.extrinsics = extr;
    df.clearXformedCache();

    if (params.intrOpt == IntrinsicsOptimization::Shared) {
      df.intrinsics.vFov = std::atan(poseParams_[0][6]) * 2.f;
      df.intrinsics.hFov =
          std::atan(poseParams_[0][6] * video_->aspect()) * 2.f;
    } else {
      df.intrinsics.vFov = std::atan(pose[6]) * 2.f;
      df.intrinsics.hFov = std::atan(pose[6] * video_->aspect()) * 2.f;
    }
  }

  LOG(INFO) << "";
}

void DepthVideoPoseOptimizer::normalizeDepth(
    const Params& params, const FlowConstraintsCollection& constraints) {
  LOG(INFO) << "------------------------";
  LOG(INFO) << "Depth Normalization (depth stream " << depthStream_ << ")...";

  params.printParams();

  LOG(INFO) << "Building problem...";
  problem_ = std::make_unique<ceres::Problem>();

  DepthStream& ds = video_->depthStream(depthStream_);
  const float invAspect = video_->invAspect();

  for (auto it = constraints.pairBegin(); it != constraints.pairEnd(); it++) {
    const PairKey& pair = *it;
    const int i0 = pair.first;
    const int i1 = pair.second;

    if (!params.frameRange.inRange(i0) || !params.frameRange.inRange(i1)) {
      continue;
    }

    if (params.normalizeDepthFromFirstFrame) {
      // When we normalize from the first frame, we don't need any pairs, but
      // instead rely on addScaleRegularization() below.
      break;
    }

    DepthFrame& df0 = ds.frame(i0);
    DepthFrame& df1 = ds.frame(i1);
    const Mat1f* depthImg0Ptr = df0.sourceDepth();
    const Mat1f* depthImg1Ptr = df1.sourceDepth();
    if (!depthImg0Ptr || !depthImg1Ptr) {
      throw std::runtime_error("Missing depth image.");
    }
    const Mat1f& depthImg0 = *depthImg0Ptr;
    const Mat1f& depthImg1 = *depthImg1Ptr;

    const PairFlowConstraints& pairConstraints = constraints(pair);
    for (const PairFlowConstraints::Constraint& c : pairConstraints) {
      const Vector2fna& obs0 = c[0];
      const Vector2fna& obs1 = c[1];

      const int x0 = obs0.x() * depthImg0.cols;
      const int y0 = obs0.y() / invAspect * depthImg0.rows;
      const float depth0 = depthImg0(y0, x0);

      const int x1 = obs1.x() * depthImg1.cols;
      const int y1 = obs1.y() / invAspect * depthImg1.rows;
      const float depth1 = depthImg1(y1, x1);

      if (!std::isfinite(depth0) || depth0 <= 0 ||
          !std::isfinite(depth1) || depth1 <= 0) {
        continue;
      }

      const Vector2f obs0Scaled(
          -1.f + 2.f * obs0.x(), 1.f - 2.f * obs0.y() / invAspect);
      const Vector2f obs1Scaled(
          -1.f + 2.f * obs1.x(), 1.f - 2.f * obs1.y() / invAspect);

      std::unique_ptr<DepthFunctor> depth0Functor =
          df0.depthXform().createFunctor(depth0, obs0Scaled);
      std::unique_ptr<DepthFunctor> depth1Functor =
          df1.depthXform().createFunctor(depth1, obs1Scaled);

      std::vector<int> depth0ParamBlockSizes = depth0Functor->paramBlockSizes();
      std::vector<double*> depth0ParamBlocks = depth0Functor->paramBlocks();
      std::vector<int> depth1ParamBlockSizes = depth1Functor->paramBlockSizes();
      std::vector<double*> depth1ParamBlocks = depth1Functor->paramBlocks();

      // Create the cost function
      using CostFunction = ceres::DynamicAutoDiffCostFunction<
          DisparityDissimilarityCost, kStride>;
      CostFunction* costFunction =
          new CostFunction(new DisparityDissimilarityCost(
              std::move(depth0Functor), std::move(depth1Functor)));

      // Set the parameter block sizes for the two cameras and depth functors.
      for (int size : depth0ParamBlockSizes) {
        costFunction->AddParameterBlock(size);
      }
      for (int size : depth1ParamBlockSizes) {
        costFunction->AddParameterBlock(size);
      }

      costFunction->SetNumResiduals(1);

      ceres::LossFunction* lossFunction =
          new ceres::CauchyLoss(params.robustness);

      // Create a vector of all the parameter blocks for the two cameras and
      // depth functors, and add the cost to the problem.
      std::vector<double*> paramBlocks;
      for (double* block : depth0ParamBlocks) {
        paramBlocks.push_back(block);
      }
      for (double* block : depth1ParamBlocks) {
        paramBlocks.push_back(block);
      }

      problem_->AddResidualBlock(costFunction, lossFunction, paramBlocks);
    }
  }

  if (params.scaleReg > 0.0) {
    addScaleRegularization(params);
  }

  if (params.depthDeformRegInitial > 0.0) {
    addDepthDeformRegularization(params, params.depthDeformRegInitial);
  }

  // I was having some problem with normalization producing some negative scale
  // parameters, because the initialization was too far off. Setting the lower
  // bound for these parameters resolves this issue.
  for (const int frame : params.frameRange) {
    DepthFrame& df = ds.frame(frame);
    for (double* paramBlock : df.depthXform().paramBlocks()) {
      if (problem_->HasParameterBlock(paramBlock)) {
        problem_->SetParameterLowerBound(paramBlock, 0, 0.0);
      }
    }
  }

  LOG(INFO) << "Solving...";
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = params.maxIterations;
  options.num_threads = params.numThreads;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &*problem_, &summary);
  LOG(INFO) << summary.BriefReport();

  if (params.normalizeDepthFromFirstFrame) {
    // Copy first frame's depth xform to all other frames.
    const int firstFrame = params.frameRange.firstFrame();
    const DepthXform& firstXform =
        video_->depthFrame(depthStream_, firstFrame).depthXform();
    for (const int frame : params.frameRange) {
      if (frame == firstFrame) {
        continue;
      }
      video_->depthFrame(depthStream_, frame).depthXform().copyFrom(firstXform);
    }
  }

  // Clear depth frame transform caches.
  for (int frame : params.frameRange) {
    DepthFrame& df = video_->depthStream(depthStream_).frame(frame);
    df.clearXformedCache();
  }

  LOG(INFO) << "";
}

void DepthVideoPoseOptimizer::addStaticSceneLoss(
    const Params& params, const FlowConstraintsCollection& constraints) {
  LOG(INFO) << "  Adding static scene loss...";

  DepthStream& ds = video_->depthStream(depthStream_);

  const double aspect = video_->aspect();
  const double vFocal =
      (aspect >= 1.f ? params.focalLong / aspect : params.focalLong);

  int pairCount = 0;
  int constraintCount = 0;

  for (auto it = constraints.pairBegin(); it != constraints.pairEnd(); it++) {
    const PairKey& pair = *it;
    const int frame0 = pair.first;
    const int frame1 = pair.second;

    if (!params.frameRange.inRange(frame0) ||
        !params.frameRange.inRange(frame1)) {
      continue;
    }
    ++pairCount;

    DepthFrame& df0 = ds.frame(frame0);
    DepthFrame& df1 = ds.frame(frame1);

    const PairFlowConstraints& pairConstraints = constraints(pair);
    for (const PairConstraint& c : pairConstraints) {
      if (!c.isStatic) {
        continue;
      }

      const Vector2fna& loc0 = c[0];
      const Vector2fna& loc1 = c[1];

      auto obs0 = std::make_shared<Observation>(
          df0, poseParams_[frame0].data(), loc0);
      auto obs1 = std::make_shared<Observation>(
          df1, poseParams_[frame1].data(), loc1);

      if (!std::isfinite(obs0->sourceDepth) || obs0->sourceDepth <= 0 ||
          !std::isfinite(obs1->sourceDepth) || obs1->sourceDepth <= 0) {
        continue;
      }

      // Create the cost function.
      ceres::DynamicCostFunction* costFunction = nullptr;
      using LossType = StaticSceneCost;
      using CostFnType = ceres::DynamicAutoDiffCostFunction<LossType, kStride>;
      costFunction = new CostFnType(new LossType(
          obs0, obs1, vFocal, aspect, params.intrOpt,
          params.staticLossType,
          params.staticSpatialWeight, params.staticDepthWeight));

      costFunction->SetNumResiduals(3);

      for (int size : obs0->paramBlockSizes) {
        costFunction->AddParameterBlock(size);
      }
      for (int size : obs1->paramBlockSizes) {
        costFunction->AddParameterBlock(size);
      }
      if (params.intrOpt == IntrinsicsOptimization::Shared) {
        costFunction->AddParameterBlock(1);
      } else if (params.intrOpt == IntrinsicsOptimization::PerFrame) {
        costFunction->AddParameterBlock(1);
        costFunction->AddParameterBlock(1);
      }

      ceres::LossFunction* lossFunction =
          new ceres::CauchyLoss(params.robustness);

      std::vector<double*> paramBlocks;
      insert(paramBlocks, obs0->paramBlockPtrs);
      insert(paramBlocks, obs1->paramBlockPtrs);
      if (params.intrOpt == IntrinsicsOptimization::Shared) {
        paramBlocks.push_back(&poseParams_[0][6]);
      } else if (params.intrOpt == IntrinsicsOptimization::PerFrame) {
        paramBlocks.push_back(&poseParams_[frame0][6]);
        paramBlocks.push_back(&poseParams_[frame1][6]);
      }

      problem_->AddResidualBlock(costFunction, lossFunction, paramBlocks);

      ++constraintCount;
    }
  }

  LOG(INFO) << "    Using " << pairCount << " frame pairs.";
  LOG(INFO) << "    Added " << constraintCount << " constraints.";
}

void DepthVideoPoseOptimizer::addSceneFlowSmoothnessLoss(
    const Params& params, const FlowConstraintsCollection& constraints) {
  LOG(INFO) << "  Adding scene flow smoothness loss...";

  DepthStream& ds = video_->depthStream(depthStream_);

  const double aspect = video_->aspect();
  const double vFocal =
      (aspect >= 1.f ? params.focalLong / aspect : params.focalLong);

  int tripletCount = 0;
  int constraintCount = 0;

  for (int frame = params.frameRange.firstFrame();
       frame < params.frameRange.lastFrame() - 1; ++frame) {
    if (!params.frameRange.inRange(frame) ||
        !params.frameRange.inRange(frame + 1) ||
        !params.frameRange.inRange(frame + 2)) {
      continue;
    }
    ++tripletCount;

    DepthFrame& df0 = ds.frame(frame + 0);
    DepthFrame& df1 = ds.frame(frame + 1);
    DepthFrame& df2 = ds.frame(frame + 2);

    const int triplet = frame + 1;
    const TripletFlowConstraints& tripletConstraints = constraints(triplet);
    for (const TripletConstraint& c : tripletConstraints) {
      const Vector2fna& loc0 = c[0];
      auto obs0 = std::make_shared<Observation>(
          df0, poseParams_[frame + 0].data(), loc0);

      const Vector2fna& loc1 = c[1];
      auto obs1 = std::make_shared<Observation>(
          df1, poseParams_[frame + 1].data(), loc1);

      const Vector2fna& loc2 = c[2];
      auto obs2 = std::make_shared<Observation>(
          df2, poseParams_[frame + 2].data(), loc2);

      if (!std::isfinite(obs0->sourceDepth) || obs0->sourceDepth <= 0 ||
          !std::isfinite(obs1->sourceDepth) || obs1->sourceDepth <= 0 ||
          !std::isfinite(obs2->sourceDepth) || obs2->sourceDepth <= 0) {
        continue;
      }

      using LossType = SceneFlowSmoothnessLoss;
      using CostFnType = ceres::DynamicAutoDiffCostFunction<LossType, kStride>;
      auto* costFunction = new CostFnType(new LossType(
          obs0, obs1, obs2, vFocal, aspect, params.intrOpt,
          params.smoothLossType));

      costFunction->SetNumResiduals(3);

      for (int size : obs0->paramBlockSizes) {
        costFunction->AddParameterBlock(size);
      }
      for (int size : obs1->paramBlockSizes) {
        costFunction->AddParameterBlock(size);
      }
      for (int size : obs2->paramBlockSizes) {
        costFunction->AddParameterBlock(size);
      }
      if (params.intrOpt == IntrinsicsOptimization::Shared) {
        costFunction->AddParameterBlock(1);
      } else if (params.intrOpt == IntrinsicsOptimization::PerFrame) {
        costFunction->AddParameterBlock(1);
        costFunction->AddParameterBlock(1);
        costFunction->AddParameterBlock(1);
      }

      double lossWeight = (c.isStatic ?
          params.smoothStaticWeight : params.smoothDynamicWeight);
      ceres::LossFunction* lossFunction = new ceres::ScaledLoss(
          nullptr, lossWeight, ceres::TAKE_OWNERSHIP);

      std::vector<double*> paramBlocks;
      insert(paramBlocks, obs0->paramBlockPtrs);
      insert(paramBlocks, obs1->paramBlockPtrs);
      insert(paramBlocks, obs2->paramBlockPtrs);
      if (params.intrOpt == IntrinsicsOptimization::Shared) {
        paramBlocks.push_back(&poseParams_[0][6]);
      } else if (params.intrOpt == IntrinsicsOptimization::PerFrame) {
        paramBlocks.push_back(&poseParams_[frame + 0][6]);
        paramBlocks.push_back(&poseParams_[frame + 1][6]);
        paramBlocks.push_back(&poseParams_[frame + 2][6]);
      }

      problem_->AddResidualBlock(costFunction, lossFunction, paramBlocks);

      ++constraintCount;
    }
  }

  LOG(INFO) << "    Using " << tripletCount << " frame triplets.";
  LOG(INFO) << "    Added " << constraintCount << " constraints.";
}

void DepthVideoPoseOptimizer::addScaleRegularization(const Params& params) {
  LOG(INFO) << "  Adding scale regularization loss ...";

  DepthStream& ds = video_->depthStream(depthStream_);

  // Setting the grid size according to the video aspect ratio
  int gridSizeX = params.scaleRegGridSize;
  int gridSizeY = round(float(gridSizeX) * video_->invAspect());
  if (video_->aspect() <= 1.f) {
    std::swap(gridSizeX, gridSizeY);
  }

  for (const int frame : params.frameRange) {

    DepthFrame& df = ds.frame(frame);

    const Mat1f* depthImgPtr = df.sourceDepth();
    if (!depthImgPtr) {
      throw std::runtime_error("Missing depth image.");
    }
    const Mat1f& depthImg = *depthImgPtr;

    std::vector<float> depthSamples(depthImg.cols * depthImg.rows);

    for (int y = 0; y < depthImg.rows; ++y) {
      const float* const srcPtr = depthImg.ptr<float>(y);
      float* const dstPtr = &depthSamples[y * depthImg.cols];
      memcpy(dstPtr, srcPtr, depthImg.cols * sizeof(float));
    }

    std::nth_element(
        depthSamples.begin(),
        depthSamples.begin() + depthSamples.size() / 2,
        depthSamples.end());
    double medianDepth = depthSamples[depthSamples.size() / 2];

    // We add absolute depth constraints for a regular grid on each frame
    // to constrain the scale of the scene. This effectively constrains
    // the transform for this frame to a global scale as a side-effect (that's
    // not desirable...)

    for (int y = 0; y < gridSizeY; ++y) {
      for (int x = 0; x < gridSizeX; ++x) {
        Vector2f loc(
            -1.f + 2.f * x / (gridSizeX - 1), -1.f + 2.f * y / (gridSizeY - 1));

        std::unique_ptr<DepthFunctor> depthFunctor =
            df.depthXform().createFunctor(medianDepth, loc);

        std::vector<int> paramBlockSizes = depthFunctor->paramBlockSizes();
        std::vector<double*> paramBlocks = depthFunctor->paramBlocks();

        // Create the cost function.
        const double targetDisparity = 1.0;
        using CostFunction =
            ceres::DynamicAutoDiffCostFunction<TargetDisparityCost, kStride>;
        CostFunction* costFunction = new CostFunction(new TargetDisparityCost(
            std::move(depthFunctor), targetDisparity));

        // Set the parameter block sizes for the two cameras and depth functors.
        for (int size : paramBlockSizes) {
          costFunction->AddParameterBlock(size);
        }
        costFunction->SetNumResiduals(1);

        const double weight = params.scaleReg;

        ceres::LossFunction* lossFunction =
            new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP);

        problem_->AddResidualBlock(costFunction, lossFunction, paramBlocks);
      }
    }
  }
}

void DepthVideoPoseOptimizer::addPositionRegularization(const Params& params) {
  LOG(INFO) << "  Adding position regularization loss...";

  for (int frame = params.frameRange.firstFrame();
       frame < params.frameRange.lastFrame() - 1; ++frame) {
    if (!params.frameRange.inRange(frame) ||
        !params.frameRange.inRange(frame + 1) ||
        !params.frameRange.inRange(frame + 2)) {
      continue;
    }

    using CostFunction =
        ceres::DynamicAutoDiffCostFunction<ParameterRegularizationCost, 3>;
    CostFunction* costFunction =
        new CostFunction(new ParameterRegularizationCost(3));

    ceres::LossFunction* lossFunction = new ceres::ScaledLoss(
        nullptr, params.positionReg, ceres::TAKE_OWNERSHIP);

    costFunction->AddParameterBlock(6);
    costFunction->AddParameterBlock(6);
    costFunction->AddParameterBlock(6);
    costFunction->SetNumResiduals(3);

    std::vector<double*> paramBlocks {
        poseParams_[frame].data(),
        poseParams_[frame + 1].data(),
        poseParams_[frame + 2].data()};
    problem_->AddResidualBlock(costFunction, lossFunction, paramBlocks);
  }
}

void DepthVideoPoseOptimizer::addDepthDeformRegularization(
    const Params& params, const double baseWeight) {
  LOG(INFO) << "  Adding depth deform regularization loss... (base weight = "
      << baseWeight << ", adaptive weight = " <<
      params.adaptiveDeformationCost << ")";

  int dynamicMaskStreamIndex = -1;
  if (video_->hasColorStream("dynamic_mask")) {
    dynamicMaskStreamIndex = video_->colorStreamIndex("dynamic_mask");
  }

  DepthStream& ds = video_->depthStream(depthStream_);
  for (const int frame : params.frameRange) {
    DepthFrame& df = ds.frame(frame);
    DepthXform& xform = df.depthXform();

    if (xform.numDeformationCostResiduals() <= 0) {
      continue;
    }

    ceres::DynamicCostFunction* costFunction;
    if (params.adaptiveDeformationCost > 0.0) {
      if (dynamicMaskStreamIndex < 0) {
        throw std::runtime_error(
            "Adaptive smoothness requires a dynamic mask stream.");
      }

      using CostFunction =
          ceres::DynamicAutoDiffCostFunction<AdaptiveDeformationCost, kStride>;
      const Mat1b* dynamicMask =
          video_->colorFrame(dynamicMaskStreamIndex, frame).image1b();
      costFunction = new CostFunction(new AdaptiveDeformationCost(
          &xform, *dynamicMask, baseWeight, params.adaptiveDeformationCost));
    } else {
      using CostFunction =
          ceres::DynamicAutoDiffCostFunction<DeformationCost, kStride>;
      costFunction = new CostFunction(new DeformationCost(&xform, baseWeight));
    }

    for (const int size : xform.paramBlockSizes()) {
      costFunction->AddParameterBlock(size);
    }
    costFunction->SetNumResiduals(xform.numDeformationCostResiduals());

    problem_->AddResidualBlock(costFunction, nullptr, xform.paramBlocks());
  }
}

void DepthVideoPoseOptimizer::addSpatialDeformRegularization(
    const Params& params) {
  LOG(INFO) << "  Adding spatial deform regularization loss...";

  DepthStream& ds = video_->depthStream(depthStream_);
  for (const int frame : params.frameRange) {
    DepthFrame& df = ds.frame(frame);
    SpatialXform& xform = df.spatialXform();

    if (xform.numDeformationCostResiduals() <= 0) {
      continue;
    }

    using CostFunction =
        ceres::DynamicAutoDiffCostFunction<DeformationCost, kStride>;
    CostFunction* costFunction =
        new CostFunction(new DeformationCost(&xform, params.spatialDeformReg));

    for (const int size : xform.paramBlockSizes()) {
      costFunction->AddParameterBlock(size);
    }
    costFunction->SetNumResiduals(xform.numDeformationCostResiduals());

    problem_->AddResidualBlock(costFunction, nullptr, xform.paramBlocks());
  }
}

void DepthVideoPoseOptimizer::addFocalRegularization(const Params& params) {
  if (params.intrOpt == IntrinsicsOptimization::Fixed) {
    return;
  }

  LOG(INFO) << "  Adding focal regularization loss...";

  const double aspect = video_->aspect();
  const double vFocal =
      (aspect >= 1.f ? params.focalLong / aspect : params.focalLong);

  for (const int frame : params.frameRange) {
    using CostFunction =
        ceres::DynamicAutoDiffCostFunction<TargetFocalCost, kStride>;
    CostFunction* costFunction = new CostFunction(new TargetFocalCost(vFocal));

    costFunction->AddParameterBlock(1);
    costFunction->SetNumResiduals(1);

    ceres::LossFunction* lossFunction = new ceres::ScaledLoss(
        nullptr, params.focalReg, ceres::TAKE_OWNERSHIP);

    problem_->AddResidualBlock(
        costFunction, lossFunction, &poseParams_[frame][6]);
  }
}

}} // namespace facebook::cp
