// Copyright 2004-present Facebook. All Rights Reserved.
//
// Author: Johannes Kopf (jkopf@fb.com)

#include <stdexcept>
#include <stdio.h>

namespace facebook {
namespace cp {

void error(const char* msg) {
  printf("%s\n", msg);

#ifdef _WIN32
  __debugbreak();
#endif

  // TODO: print stack trace, or at least last function, file, and line no
  throw std::runtime_error(msg);
}

void error() {
  error("");
}

void ensure(const bool cond, const char* message) {
  if (!cond) {
    error(message);
  }
}

void ensure(const bool cond) {
  if (!cond) {
    error("Ensure condition failed");
  }
}

}} // namespace facebook::cp
