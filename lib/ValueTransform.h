// Copyright 2004-present Facebook. All Rights Reserved.

// This file provides classes for transforming *single* (depth) values. These
// form building blocks for the depth map transforms defined in another file.

#pragma once

#include <ceres/ceres.h>

#include "core/Enum.h"

namespace facebook {
namespace cp {

// Enumerating different supported transformations.
enum class ValueXformType {
  None,
  Scale,
  ScaleShift,
};

extern const EnumStrings<ValueXformType> valueXformStrs;
DECLARE_VALIDATOR(ValueXformType, valueXformStrs);

// Abstract base class for all value transforms. These transforms are stateless
// (i.e., everything is const). However, because of the polymorphism we need
// to actually create instances; use getInstance().
class ValueXform {
 public:
  // This is the stride parameter for ceres::DynamicAutoDiffCostFunction.
  constexpr static int kStride = 4;

  // The Jet type is used for optimization by Ceres.
  using Jet = ceres::Jet<double, kStride>;

  // Get a singleton instance of a value transform.
  static const ValueXform& getInstance(const ValueXformType& type);

  virtual ~ValueXform() = default;
  virtual ValueXformType type() const = 0;
  virtual int numParams() const = 0;

  // These functions apply the transform to a depth value. The number of
  // parameters must match numParams(). We need to provide variants for
  // double and for the Jet type. Ideally we'd use a template function to reduce
  // redundancy; however, templates cannot be virtual.
  virtual double operator()(
      const double& src, const double* const params) const = 0;
  virtual Jet operator()(const Jet& src, const Jet* const params) const = 0;

 protected:
  // Can't be constructed directly, only by ValueXform::getInstance.
  ValueXform() = default;
};

// Scale transform: dst = src * params[0].
struct ScaleXform : public ValueXform {
  template <typename T>
  static T function(const T& src, const T* const params) {
    return src * params[0];
  }

  double operator()(
      const double& src, const double* const params) const override {
    return function(src, params);
  }

  Jet operator()(const Jet& src, const Jet* const params) const override {
    return function(src, params);
  }

  ValueXformType type() const override { return ValueXformType::Scale; }
  virtual int numParams() const override { return 1; }
};

// Scale-and-shift (affine) transform: dst = src * params[0] + params[1].
struct ScaleShiftXform : public ValueXform {
  template <typename T>
  static T function(const T& src, const T* const params) {
    return (src * params[0]) + params[1];
  }

  double operator()(
      const double& src, const double* const params) const override {
    return function(src, params);
  }

  Jet operator()(const Jet& src, const Jet* const params) const override {
    return function(src, params);
  }

  ValueXformType type() const override { return ValueXformType::ScaleShift; }
  virtual int numParams() const override { return 2; }
};

}} // namespace facebook::cp
