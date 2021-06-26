// Copyright 2004-present Facebook. All Rights Reserved.

// This file provides a class hierarchy for representing depth map
// transformations.

#pragma once

#include <opencv2/core.hpp>

#include "core/Enum.h"
#include "core/ParamsBase.h"

#include "ValueTransform.h"

namespace facebook {
namespace cp {

// Forward declarations.
class DepthFrame;

// Returns the min and max depth values, ignoring invalid (<= 0) values.
std::tuple<float, float> computeDepthRange(const cv::Mat1f& depth);

enum class XformType {
  Depth,
  Spatial,
};
extern const EnumStrings<XformType> xformTypeStrs;

enum class DepthXformType {
  None,
  Identity,
  Global,
  Grid,
};
extern const EnumStrings<DepthXformType> depthXformTypeStrs;

enum class SpatialXformType {
  None,
  Identity,
  VerticalLinear,
  CornersBilinear,
  BilinearGrid,
  BicubicGrid,
};
extern const EnumStrings<SpatialXformType> spatialXformTypeStrs;

// Type descriptor for a depth map transform. Not just a simple enum, because
// we support complex transforms with parameters, e.g., grid size, etc.
struct XformDescriptor : public ParamsBase {
  XformType type = XformType::Depth;

  // Only one of these should be not-none.
  DepthXformType depthType = DepthXformType::Identity;
  SpatialXformType spatialType = SpatialXformType::None;

  ValueXformType valueXform = ValueXformType::None;
  bool cubicInterpolation = false;

  // Grid dimensions for grid-based transforms. The first two components are
  // the spatial grid size (columns, rows), and last component is the depth-wise
  // grid size.
  Eigen::Vector3i gridSize = Eigen::Vector3i::Zero();

  // The depth range for bilateral transforms.
  Eigen::Vector2d depthMinMax = Eigen::Vector2d::Zero();

  XformDescriptor() = default;
  explicit XformDescriptor(std::istream& is);

  void reset(const XformType type = XformType::Depth);

  std::string str() const;
  void parse(const std::string& str);

  void fread(std::istream& is);
  void fwrite(std::ostream& os) const;

  void addCommandLineOptions() override;
  void printParams() const override;

  bool operator==(const XformDescriptor& other) const;
  bool operator!=(const XformDescriptor& other) const;
};

// Abstract base for all transform functors. We'll specialize depth and spatial
// transforms below. The purpose of this is to hide the implementation of a
// particular transform from calling code. For example, we can pass a functor
// into a Ceres cost function and it can optimize the transform parameters
// without having to care about the implementation details.
class XformFunctor {
 public:
  // The Jet type is used for optimization by Ceres.
  using Jet = ValueXform::Jet;
  using Vector2Jet = Eigen::Matrix<Jet, 2, 1>;

  XformFunctor() = default;
  virtual ~XformFunctor() = default;

  // Disable copy and assignment
  XformFunctor(const XformFunctor&) = delete;
  XformFunctor& operator=(const XformFunctor&) = delete;

  // (Optionally) overload this function to return a string with debugging
  // information.
  virtual std::string info() const { return ""; }

  // Return information about the parameter blocks that this functor depends on.
  // For example, in a bilinear grid transform these would be the parameters of
  // the value transforms at the four surrounding vertices.
  const std::vector<double*>& paramBlocks() const;
  const std::vector<int>& paramBlockSizes() const;
  int numParamBlocks() const;

 protected:
  std::vector<double*> paramBlocks_ {};
  std::vector<int> paramBlockSizes_ {};
};

// A depth functor transforms a *specific depth value* at a *specific image
// location*, i.e., out_depth = fn(in_depth, x, y).
class DepthFunctor : public XformFunctor {
 public:
  // Compute the transformed depth value using the provided parameters.
  // We need to provide variants for double and for the Jet type. Ideally we'd
  // use a template function to reduce redundancy; however, templates cannot be
  // virtual.
  virtual double operator()(double const* const* params) const = 0;
  virtual Jet operator()(Jet const* const* params) const = 0;
};

// A spatial functor computes a 2D image-space warp at a specific image
// location.
class SpatialFunctor : public XformFunctor {
 public:
  // Compute the spatial deformation using the provided parameters.
  // We need to provide variants for double and for the Jet type. Ideally we'd
  // use a template function to reduce redundancy; however, templates cannot be
  // virtual.
  virtual Eigen::Vector2d operator()(double const* const* params) const = 0;
  virtual Vector2Jet operator()(Jet const* const* params) const = 0;
};

// Abstract base class for all transforms.
class Xform {
 public:
  using Jet = ValueXform::Jet;

  Xform() = default;
  virtual ~Xform() = default;

  // Disable copy and assignment
  Xform(const Xform&) = delete;
  Xform& operator=(const Xform&) = delete;

  // We can use these explicit functions to clone and copy, though.
  std::unique_ptr<Xform> clone() const;
  void copyFrom(const Xform& other);

  // Return some information about this depth transform.
  const XformDescriptor& desc() const { return desc_; }
  std::string str() const;

  // Override these functions if the transform can compute a deformation cost.
  virtual int numDeformationCostResiduals() const { return 0; }
  virtual void computeDeformationCost(
      double const* const* params, double* residuals) const {};
  virtual void computeDeformationCost(
      Jet const* const* params, Jet* residuals) const {};

  std::vector<double>& params() { return params_; }
  const std::vector<double>& params() const { return params_; }
  double* data() { return params_.data(); }
  const double* data() const { return params_.data(); }
  int numParams() const { return params_.size(); }

  // This function returns a list of pointers into the params_ array, organized
  // into parameter blocks, i.e., units of optimization for Ceres.
  const std::vector<double*>& paramBlocks() const { return paramBlocks_; }
  const std::vector<int>& paramBlockSizes() const { return paramBlockSizes_; }
  const int numParamBlocks() const { return paramBlocks_.size(); }

  bool operator==(const Xform& other) const;
  bool operator!=(const Xform& other) const;

 protected:
  XformDescriptor desc_;
  std::vector<double> params_;
  std::vector<double*> paramBlocks_;
  std::vector<int> paramBlockSizes_;
};

// A depth transform computes a value-wise (i.e., 1D) transformation of all the
// depth values in a depth map.
class DepthXform : public Xform {
 public:
  // Return a map of per-pixel transform parameters.
  virtual cv::Mat paramMap(const DepthFrame& df) const;

  // Compute a transformed depth map.
  std::unique_ptr<cv::Mat1f> apply(const cv::Mat1f& src) const;

  // Create a functor for transforming a specific source depth value at a
  // specific image location. This functor can be passed to Ceres for optimizing
  // its parameters.
  virtual std::unique_ptr<DepthFunctor> createFunctor(
      const float srcDepth, const Eigen::Vector2f& loc) const = 0;
};

// A spatial transform computes a 2D image-space displacement across the image
// domain.
class SpatialXform : public Xform {
 public:
  // Compute a warp map for an image domain.
  cv::Mat2f warp(const int h, const int w) const;

  // Create a functor that computes the warp at a specific image location. This
  // functor can be passed to Ceres for optimizing its parameters.
  virtual std::unique_ptr<SpatialFunctor> createFunctor(
      const Eigen::Vector2f& loc) const = 0;
};

// A very simple rigid depth transform without any parameters.
class IdentityDepthXform : public DepthXform {
 public:
  IdentityDepthXform();

  std::unique_ptr<DepthFunctor> createFunctor(
      const float srcDepth, const Eigen::Vector2f& loc) const override;
};

// Global transforms apply the *same* value-transform to every pixel.
class GlobalDepthXform : public DepthXform {
 public:
  explicit GlobalDepthXform(const ValueXformType valueXform);

  std::unique_ptr<DepthFunctor> createFunctor(
      const float srcDepth, const Eigen::Vector2f& loc) const override;

 private:
  const ValueXform& valueXform_;
};

// This transform interpolates transforms specified on a regular grid of handles
// across the whole domain. The grid can be spatial (2D), depth-wise (1D), or
// bilateral (3D). There are *linear* and *cubic* specializations defined below.
// The outer grid points are on the edges of the image / the provided depth
// interval, so that every source depth value is guaranteed to be within a grid
// cell.
class GridDepthXform : public DepthXform {
 public:
  GridDepthXform(
      const ValueXformType valueXform,
      const bool cubicInterpolation,
      const Eigen::Vector3i gridSize,
      const Eigen::Vector2d depthMinMax = Eigen::Vector2d::Zero());

  cv::Mat paramMap(const DepthFrame& df) const override;

  int numDeformationCostResiduals() const override;
  void computeDeformationCost(
      double const* const* params, double* residuals) const override;
  void computeDeformationCost(
      Jet const* const* params, Jet* residuals) const override;

  std::unique_ptr<DepthFunctor> createFunctor(
      const float srcDepth, const Eigen::Vector2f& loc) const override;

 private:
  void linearGather(
    const float srcDepth, const Eigen::Vector2f& loc,
    std::vector<double*>& paramBlocks, std::vector<double>& weights) const;

  void cubicGather(
    const float srcDepth, const Eigen::Vector2f& loc,
    std::vector<double*>& paramBlocks, std::vector<double>& weights) const;

  const ValueXform& valueXform_;
  const bool cubicInterpolation_;
  const Eigen::Vector3i gridSize_;

  Eigen::Vector2d depthMinMax_ = Eigen::Vector2d::Zero();
  double depthRange_ = 0.0;
  Eigen::Vector2d disparityMinMax_ = Eigen::Vector2d::Zero();
  double disparityRange_ = 0.0;
  double disparityRangeInv_ = 0.0;
  std::vector<double> handleSrcDisparity_;
  double handleSrcDisparityInterval_ = 0.0;
};

// This transform returns a zero-displacement everywhere.
class IdentitySpatialXform : public SpatialXform {
 public:
  IdentitySpatialXform();
  std::unique_ptr<SpatialFunctor> createFunctor(
      const Eigen::Vector2f& loc) const override;
};

// This transform has two handles, at the top and bottom of the image, that are
// linearly interpolated.
class VerticalLinearSpatialXform : public SpatialXform {
 public:
  VerticalLinearSpatialXform();
  std::unique_ptr<SpatialFunctor> createFunctor(
      const Eigen::Vector2f& loc) const override;

  int numDeformationCostResiduals() const override;
  void computeDeformationCost(
      double const* const* params, double* residuals) const override;
  void computeDeformationCost(
      Jet const* const* params, Jet* residuals) const override;
};

// This transform has a handle in each image corner that are bilinearly
// interpolated.
class CornersBilinearSpatialXform : public SpatialXform {
 public:
  CornersBilinearSpatialXform();
  std::unique_ptr<SpatialFunctor> createFunctor(
      const Eigen::Vector2f& loc) const override;

  int numDeformationCostResiduals() const override;
  void computeDeformationCost(
      double const* const* params, double* residuals) const override;
  void computeDeformationCost(
      Jet const* const* params, Jet* residuals) const override;
};

// Spatial grid transforms. Refer to the descriptions for the depth grid
// transforms.
class GridSpatialXform : public SpatialXform {
 public:
  GridSpatialXform(const int rows, const int cols);

  int numDeformationCostResiduals() const override;
  void computeDeformationCost(
      double const* const* params, double* residuals) const override;
  void computeDeformationCost(
      Jet const* const* params, Jet* residuals) const override;
};

class BilinearGridSpatialXform : public GridSpatialXform {
 public:
  BilinearGridSpatialXform(const int rows, const int cols);
  std::unique_ptr<SpatialFunctor> createFunctor(
      const Eigen::Vector2f& loc) const override;
};

class BicubicGridSpatialXform : public GridSpatialXform {
 public:
  BicubicGridSpatialXform(const int rows, const int cols);
  std::unique_ptr<SpatialFunctor> createFunctor(
      const Eigen::Vector2f& loc) const override;
};

// Initializes a new transform using the specified method.
std::unique_ptr<Xform> createXform(const XformDescriptor& desc);
std::unique_ptr<DepthXform> createDepthXform(const XformDescriptor& desc);
std::unique_ptr<SpatialXform> createSpatialXform(const XformDescriptor& desc);

std::unique_ptr<Xform> readXform(std::istream& is);
void writeXform(std::ostream& os, const Xform& xform);

}} // namespace facebook::cp
