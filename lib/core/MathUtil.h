// Copyright 2004-present Facebook. All Rights Reserved.
//
//
// This file is formatted with Nuclide command-shift-C (clang-format -i)
//

#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include <algorithm>
#include <iostream>

#include <Eigen/Dense>
// #include <folly/lang/Exception.h>
#include <opencv2/core/core.hpp>

namespace facebook {
namespace cp {

static constexpr float M_PI_F = float(M_PI);
static constexpr float M_PI_2_F = float(M_PI_2);
static constexpr float M_SQRT2_F = float(M_SQRT2);
static constexpr float M_SQRT1_2_F = float(M_SQRT1_2);

// Clamp a value to an interval
template<typename T, typename T1, typename T2> static inline
T clamp(const T& x, const T1& min, const T2& max) {
  return (x < min ? min : (x > max ? max : x));
}

// Positive modulus
template<typename T> static inline
T posmod(const T& i, const T& n) {
  static_assert(std::is_integral<T>::value, "posmod requires integer type");
  return (i % n + n) % n;
}

template<typename T> static inline
T divideRoundDown(const T numerator, const T denominator) {
  static_assert(
      std::is_integral<T>::value, "divideRoundDown requires integer type");
  T quotient = numerator / denominator;
  if ((numerator < T(0)) == (denominator < T(0))) {
    // Numerator and denominator are either both negative or both positive.
    return quotient;
  } else {
    // One is positive, the other negative.
    return (numerator % denominator == T(0) ? quotient : quotient - T(1));
  }
}

// Round n down to the nearest integer multiple of k.
template<typename T> static inline
T roundDownToMultiple(const T n, const T k) {
  static_assert(
      std::is_integral<T>::value, "roundDownToMultiple requires integer type");
  return n - posmod(n, std::abs(k));
};

// Round n up to the nearest integer multiple of k.
template<typename T> static inline
T roundUpToMultiple(const T n, const T k) {
  static_assert(
      std::is_integral<T>::value, "roundUpToMultiple requires integer type");
  T absk = std::abs(k);
  T r = posmod(n, absk);
  return (r == T(0) ? n : n + absk - r);
};

// Return 'true' if 'r' is an empty rect, 'false' otherwise
template <class T>
bool empty(const cv::Rect_<T>& r) {
  return r.width <= 0 || r.height <= 0;
}

// Returns squared value
template<typename T> static inline
T sqr(const T& x) {
  return x * x;
}

// Returns the fractional part of a value
template<typename T> static inline
T fract(const T& x) {
  static_assert(std::is_floating_point<T>::value,
      "fract requires floating point type");
  return x - floor(x);
}

// Linear interpolation between two values
template<typename TVAL, typename TINTER> static inline
TVAL lerp(const TVAL& v0, const TVAL& v1, const TINTER& ival) {
  static_assert(std::is_floating_point<TINTER>::value,
      "lerp requires floating point type");
  return v0 + (v1-v0)*ival;
}

// Compute smallest power-of-two larger than n.
uint32_t nextPowerOfTwo(uint32_t n);

// Check if a number is power-of-two
bool isPowerOfTwo(uint32_t n);

// Shrink (or erode) 'r' in all directions by 'amount'
inline cv::Rect erode(const cv::Rect& r, int amount) {
  cv::Rect er(
      r.x + amount, r.y + amount, r.width - 2 * amount, r.height - 2 * amount);
  return (er.width <= 0 || er.height <= 0) ? cv::Rect(0, 0, 0, 0) : er;
}

// Grow (or dilate) 'r' in all directions by 'amount'
inline cv::Rect dilate(const cv::Rect& r, int amount) {
  return erode(r, -amount);
}

// Convert degrees to radians
template<typename T> static inline
T deg2rad(const T& x) {
  static_assert(std::is_floating_point<T>::value,
      "deg2rad requires floating point type");
  return x * T(M_PI / 180.0);
}

// Convert radians to degrees
template<typename T> static inline
T rad2deg(const T& x) {
  static_assert(std::is_floating_point<T>::value,
      "rad2deg requires floating point type");
  return x * T(180.0 / M_PI);
}

// Return atan2 in the range 0:2*pi instead of the normal -pi:pi
inline float atan2f_adj(float y, float x) {
  float result = atan2f(y, x);
  return (result < 0) ? 2.f * M_PI_F + result : result;
}

// Distance between two angles. Result is in range [0, PI).
template<typename T> static inline
T angleDist(const T& a, const T& b) {
  static_assert(std::is_floating_point<T>::value,
      "angleDist requires floating point type");
  T d = std::fmod(a - b, T(2.0 * M_PI));
  if (d < T(0)) {
    d += T(2.0 * M_PI);
  }

  return (d < T(M_PI) ? d : T(2.0 * M_PI) - d);
}

// The angle of rotation required to get from one quaternion orientation to
// another. A useful measure of distance for quaternions.
template<typename T>
T angle(const Eigen::Quaternion<T>& qa, const Eigen::Quaternion<T>& qb) {
  double cosAngle = 2.0 * sqr(qa.coeffs().dot(qb.coeffs())) - 1.0;
  return acos(clamp(cosAngle, -1.0, 1.0));
}

// Updates a value, moving it closer to some target value. This function can be
// called once per from in interactive applications to implement a hysteresis
// ("delayed update") behavior, e.g. mouse glide, lagged tracking, or adaptive
// exposure compensation. The speed of update is *independent* of the frame
// rate.
template <typename T>
void springUpdate(T& value, const T& target,
                  const double timeDelta, const double dampFactor) {
  value =  target + (value - target) * (T)exp(-timeDelta * dampFactor);
}

template <typename T, int m, int n>
void springUpdate(Eigen::Matrix<T, m, n>& value,
                  const Eigen::Matrix<T, m, n>& target,
                  const double timeDelta, const double dampFactor) {
  value =  target + (value - target) * (T)exp(-timeDelta * dampFactor);
}

// smoothStep performs smooth Hermite interpolation between 0 and 1 when
// edge0 < x < edge1. This is useful in cases where a threshold function with
// a smooth transition is desired. This implementation matches the one in GLSL.
template <typename T>
T smoothStep(const T& edge0, const T& edge1, const T& x) {
  T t = clamp((x - edge0) / (edge1 - edge0), T(0.0), T(1.0));
  return t * t * (T(3.0) - T(2.0) * t);
}

// Compute the median value of a vector
template <typename T>
T median(std::vector<T>& seq) {
  if (seq.empty()) {
    // folly::throw_exception<std::runtime_error>(
    //     "median called on an empty sequence."
    // );
    std::cerr << "median called on an empty sequence.";
  }

  size_t n = seq.size() / 2;

  // Partially sort the sequence, just enough for computing the median.
  std::nth_element(seq.begin(), seq.begin() + n, seq.end());

  if (seq.size() % 2 != 0) {
    // Odd number of elements, just return the median.
    return seq[n];
  } else {
    // Even number of elements, need to average two elements. Note that
    // std::nth_element does not guarantee the elements before the nth one are
    // sorted, just that they are all less than one in n. So, we use
    // std::max_element to get the second-largest element.
    auto it = std::max_element(seq.begin(), seq.begin() + n);
    return (*it + seq[n]) / T(2);
  }
}

}} // facebook::cp
