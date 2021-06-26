// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <set>
#include <string>
#include <vector>

// Forward declaration.
namespace boost {
class any;
}

namespace facebook {
namespace cp {

// A frame range is an ordered set of frame indices. It can be used to restrict
// various operations to a subset of frames. It can be constructed from a string
// representation, e.g., "1,3,5-7" results in the range [1, 3, 5, 6, 7]. Before
// it is used the resolve function should be called. It ensures that the frame
// set is not empty (by filling it with ALL frames if it is...)
struct FrameRange {
  using Container = std::set<int>;

  FrameRange();
  explicit FrameRange(const std::string& str);
  FrameRange(const std::string& str, const int numFrames);

  void fromString(const std::string& str);
  std::string toString() const;

  // Fill set with all frames if it's empty, otherwise do nothing. Optionally
  // remove out of bounds frames.
  void resolve(const int numFrames, const bool clip = false);

  // Get information about this range.
  bool isEmpty() const;
  int firstFrame() const;
  int lastFrame() const;
  int count() const;
  bool isConsecutive() const;
  bool inRange(const int frame) const;

  // Iterators.
  Container::const_iterator begin() const {
    return frames.begin();
  }

  Container::const_iterator end() const {
    return frames.end();
  }

  // Throw if the range is empty. It needs to be resolved.
  void checkEmpty() const;

  std::set<int> frames;
};

// Validator for boost::program_options
void validate(
    boost::any& v, const std::vector<std::string>& values, FrameRange*, int);

}} // namespace facebook::cp
