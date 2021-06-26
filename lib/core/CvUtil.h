// Copyright 2004-present Facebook. All Rights Reserved.
//
// Author: Johannes Kopf (jkopf@fb.com)

#pragma once

#include <opencv2/core/core.hpp>

namespace facebook {
namespace cp {

const char* szDepth(const int depth);

// Return unit-length vector
template<typename T, int m, int n> static inline
cv::Matx<T, m, n> unit(const cv::Matx<T, m, n>& M) {
  T length = norm(M);
  if (length == T(0.0)) { return M; }
  return M * (T(1.0) / length);
}

template<typename T, int n> static inline
cv::Vec<T, n> unit(const cv::Vec<T, n>& V) {
  T length = (T)norm(V);
  if (length == T(0.0)) { return V; }
  return V * (T(1.0) / length);
}

// Extract row
template<typename T, int m, int n> static inline
cv::Vec<T, n> row(const cv::Matx<T, m, n>& M, const int i) {
  cv::Vec<T, n> res;
  memcpy(res.val, M.val+i*n, n*sizeof(T));
  return res;
}

// Extract column
template<typename T, int m, int n> static inline
cv::Vec<T, m> col(const cv::Matx<T, m, n>& M, const int i) {
  cv::Vec<T, m> res;
  T * p = ((T*)M.val) + i;
  for (int k = 0; k < m; ++k) {
    res(k) = *p;
    p += n;
  }
  return res;
}

// Bilinear pixel fetch
template <typename T>
bool bilinearFetch(
    T& res, const cv::Mat_<T>& img, const float x, const float y) {
  const int w = img.cols;
  const int h = img.rows;
  const int ix = x;
  const int iy = y;
  if (ix < 0 || ix >= w - 1 || iy < 0 || iy >= h - 1) {
    return false;
  }

  const float fx = x - ix;
  const float fy = y - iy;

  const T* const rowPtr = (const T*)img.ptr(iy);
  const T* const nextRowPtr = (const T*)((const uint8_t*)rowPtr + img.step[0]);

  res = (rowPtr[ix] * (1.f - fx) + rowPtr[ix + 1] * fx) * (1.f - fy) +
        (nextRowPtr[ix] * (1.f - fx) + nextRowPtr[ix + 1] * fx) * fy;
  return true;
}

// Read/write contents of an image to/from a file
void freadim(FILE * fin, cv::Mat& dst);
void freadim(const std::string& fileName, cv::Mat& dst);
void readim(std::istream& is, cv::Mat& dst);
void fwriteim(FILE * fout, const cv::Mat& src);
void fwriteim(const std::string& fileName, const cv::Mat& src);
void writeim(std::ostream& os, const cv::Mat& src);

}} // namespace facebook::cp
