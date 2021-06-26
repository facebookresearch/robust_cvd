// Copyright 2004-present Facebook. All Rights Reserved.

#include "FrameRange.h"

#include <boost/program_options.hpp>
#include <fmt/format.h>

#include "core/Misc.h"

namespace facebook {
namespace cp {

void validate(
    boost::any& v, const std::vector<std::string>& values, FrameRange*, int) {
  namespace po = boost::program_options;
  if (values.size() != 1) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }
  FrameRange value(values[0]);
  v = boost::any(value);
}

FrameRange::FrameRange() {
}

FrameRange::FrameRange(const std::string& str) {
  fromString(str);
}

FrameRange::FrameRange(const std::string& str, const int numFrames) {
  fromString(str);
  resolve(numFrames);
}

void FrameRange::fromString(const std::string& str) {
  frames.clear();

  std::vector<std::string> pieces = explode(str, ',');
  for (const std::string& piece : pieces) {
    std::vector<std::string> subPieces = explode(piece, '-');
    int count = subPieces.size();
    if (count < 1 || count > 2) {
      throw std::runtime_error("Malformed range piece.");
    }
    std::vector<int> subPiecesInt;
    for (std::string& s : subPieces) {
      subPiecesInt.push_back(std::stoi(s));
    }
    assert(!subPiecesInt.empty()); // For Lint.
    int start = subPiecesInt[0];
    int end = start;
    if (subPieces.size() > 1) {
      end = subPiecesInt[1];
    }
    for (int frame = start; frame <= end; ++frame) {
      frames.insert(frame);
    }
  }
}

std::string FrameRange::toString() const {
  if (isEmpty()) {
    return "";
  }

  std::string res;

  auto it = frames.begin();
  int start = *it;
  ++it;
  int lastIndex = start;

  auto addRange = [&]() {
    if (!res.empty()) {
      res += ",";
    }
    if (lastIndex == start) {
      res += fmt::format("{:d}", start);
    } else {
      res += fmt::format("{:d}-{:d}", start, lastIndex);
    }
  };

  for (; it != frames.end(); ++it) {
    int frame = *it;
    if (frame - lastIndex > 1) {
      addRange();
      start = frame;
    }
    lastIndex = frame;
  }
  addRange();

  return res;
}

void FrameRange::resolve(const int numFrames, const bool clip) {
  if (clip) {
    std::set<int> clippedFrames;
    for (const int frame : frames) {
      if (frame >= 0 && frame < numFrames) {
        clippedFrames.insert(frame);
      }
    }
    frames = clippedFrames;
  }

  if (frames.empty()) {
    for (int frame = 0; frame < numFrames; ++frame) {
      frames.insert(frame);
    }
  }

  if (firstFrame() < 0 || lastFrame() >= numFrames) {
    throw std::runtime_error(
        "Frame range contains out-of-range frame indices.");
  }
}

bool FrameRange::isEmpty() const {
  return frames.empty();
}

int FrameRange::firstFrame() const {
  checkEmpty();
  return *(frames.begin());
}

int FrameRange::lastFrame() const {
  checkEmpty();
  return *(frames.rbegin());
}

int FrameRange::count() const {
  checkEmpty();
  return frames.size();
}

bool FrameRange::isConsecutive() const {
  checkEmpty();
  return (lastFrame() - firstFrame() + 1) == frames.size();
}

bool FrameRange::inRange(const int frame) const {
  checkEmpty();
  return (frames.find(frame) != frames.end());
}

void FrameRange::checkEmpty() const {
  if (frames.empty()) {
    throw std::runtime_error("Frame set is empty. Forgot to call resolve()?");
  }
}

}} // namespace facebook::cp
