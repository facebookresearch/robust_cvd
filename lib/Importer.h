// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "DepthVideo.h"

namespace facebook {
namespace cp {

class DepthVideoImporter {
 public:
  static void importVideo(
      DepthVideo& video,
      const std::string& path,
      const bool discoverStreams = true);
  static void importMetaFrames(
      const std::string& path, int& width, int& height,
      std::vector<std::unique_ptr<MetaFrame>>& frames);
  static void importColmapRecon(
      DepthVideo& video, const std::string& colmapFile,
      const int stream, const bool silent = false);
  static void importColmapDepth(DepthVideo& video);
  static void importPoses(
      DepthVideo& video, const std::string& posesFile, const int stream);
  static void importTracks(DepthVideo& video, const std::string& trackFile);
  static float loadScale(const std::string& path);
};

}} // namespace facebook::cp
