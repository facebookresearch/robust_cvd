// Copyright 2004-present Facebook. All Rights Reserved.

#include "Photo3dIo.h"

#include "core/FileIo.h"

#include "DepthVideo.h"

namespace facebook {
namespace cp {

Photo3dReader::Photo3dReader(const std::string& fileName, ReadFn readFn)
    : readFn_(readFn) {
  stream_ = std::ifstream(fileName, std::ios::binary);
  if (stream_.fail()) {
    return;
  }

  int numFrames = facebook::cp::read<int>(stream_);
  frameOffset_.resize(numFrames + 1);
  for (int frame = 0; frame < numFrames + 1; ++frame) {
    frameOffset_[frame] = facebook::cp::read<size_t>(stream_);
  }
}

bool Photo3dReader::read(const int frame) {
  if (frameOffset_.size() <= frame) {
    return false;
  }

  size_t offset = frameOffset_[frame];
  size_t length = frameOffset_[frame + 1] - offset;

  if (length == 0) {
    return false;
  }

  stream_.seekg(offset);

  readFn_(frame, stream_);

  return true;
}

Photo3dWriter::Photo3dWriter(
    const std::string& fileName, WriteFn writeFn, const int numFrames) {
  std::ofstream os(fileName, std::ios::binary);
  if (os.fail()) {
    throw std::runtime_error("Cannot write to file '" + fileName +"'.");
  }

  // Write offset table
  write(os, numFrames);
  for (int frame = 0; frame < numFrames + 1; ++frame) {
    write(os, size_t(0));
  }

  std::vector<size_t> offsets(numFrames + 1);

  // Write frames
  for (int frame = 0; frame < numFrames; ++frame) {
    offsets[frame] = os.tellp();
    writeFn(frame, os);
  }

  offsets[numFrames] = os.tellp();

  // Write actual offsets to offset table
  os.seekp(sizeof(int));
  for (int frame = 0; frame < numFrames + 1; ++frame) {
    write(os, offsets[frame]);
  }
}

}} // namespace facebook::cp
