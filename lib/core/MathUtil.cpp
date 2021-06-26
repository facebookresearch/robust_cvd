// Copyright 2004-present Facebook. All Rights Reserved.

#include "MathUtil.h"

namespace facebook {
namespace cp {

// See http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
uint32_t nextPowerOfTwo(uint32_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;

  return n;
}

// http://www.graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
bool isPowerOfTwo(uint32_t n) {
  return (n & (n - 1)) == 0;
}

}} // namespace facebook::cp
