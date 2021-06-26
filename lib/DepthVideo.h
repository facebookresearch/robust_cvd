// Copyright 2004-present Facebook. All Rights Reserved.

// This file provides a container for *depth videos*, i.e., a video that
// contains color and depth. It is essentially the video analog to a depth
// photo.
//
// A depth video can contain one or multiple color streams, for example, to
// store different resolutions.
//
// A depth video can contain one or multiple depth streams, for example, to
// store multiple "reconstructions" computed with different methods. The depth
// frames of a depth stream also store the camera pose.
//
// You can access general metadata about a frame like this:
//   const MetaFrame& frame = video.frame(index);
//   LOG(INFO) << "Timestamp: " << frame.pts();
//
// You can access color frames like this:
//   const ColorFrame& cf = video.colorFrame(streamIndex, frameIndex);
// or like this:
//   const ColorStream& cs = video.colorStream(streamIndex);
//   const ColorFrame& cf = cs.frame(frameIndex);
//
// And the image data therein, like this:
//   const Mat3f* image = cf.image();
//
// As well as a GL texture:
//   const GlTexture* texture = cf.glImage();
//
// Color streams can also be accessed by name:
//   const ColorStream& cs = video.colorStream("full");
//
// You can access depth frames like this:
//   const DepthFrame& df = video.depthFrame(streamIndex, frameIndex);
// or like this:
//   const DepthStream& ds = video.depthStream(streamIndex);
//   const DepthFrame& df = ds.frame(frameIndex);
//
// And the depth data therein, like this:
//   const Mat1f* linearDepth = df.depth();
//
// As well as a GL texture:
//   const GlTexture* depthTexture = df.glDepth();
//
// Depth streams can also be accessed by name:
//   const DepthStream& ds = video.depthStream("consistent_depth");
//
// For more details refer to the inline documentation below.

#pragma once

#include <opencv2/core.hpp>

#include "core/EigenNoAlignTypes.h"
#include "DepthPhoto.h"

namespace facebook {
namespace cp {

class ColorFrame;
class ColorStream;
class DepthFrame;
class DepthStream;
class DepthVideo;
// struct GopTable;
struct QuantTable;

//*************************
//****                 ****
//****    MetaFrame    ****
//****                 ****
//*************************

// This class represents some general information about depth video frames.
class MetaFrame {
  friend class DepthVideo;

 public:
  MetaFrame(const float pts) : pts_(pts) {}

  // "Presentation timestamp", i.e., time into the video at which the frame
  // is to be displayed.
  float pts() const { return pts_; }

  // Length the frame will be presented.
  float duration() const { return duration_; }

 private:
  float pts_ = 0.f;
  float duration_ = 0.f;
};

//**************************
//****                  ****
//****    DepthVideo    ****
//****                  ****
//**************************

// This class is a container for a video that has color and depth streams.
// Similar to how DepthPhoto is a container for a color and depth image pair.
class DepthVideo {
 public:
  static constexpr uint32_t kFileFormatVersion = 13;
  static constexpr uint32_t kMinSupportedFileFormat = 9;

  DepthVideo();
  ~DepthVideo();

  // Disable copy and assignment
  DepthVideo(const DepthVideo&) = delete;
  DepthVideo& operator=(const DepthVideo&) = delete;

  void printInfo() const;

  void reset();
  void init(
      const std::string& path, const int width, const int height,
      std::vector<std::unique_ptr<MetaFrame>>&& frames);
  void load(const std::string& path);

  // This saves only the metadata (general information about the video, streams,
  // frames, camera poses, etc.).
  void save();

  int width() const { return width_; }
  int height() const { return height_; }
  float aspect() const { return aspect_; }
  float invAspect() const { return invAspect_; }
  const std::string& path() const { return path_; }

  const MetaFrame& frame(const int index) const { return *metaFrames_[index]; }
  int numFrames() const { return metaFrames_.size(); }
  float duration() const { return duration_; }
  int timeToFrame(const float time) const;
  float time(const int frame) const { return metaFrames_[frame]->pts(); }

  // Parse a frame range from a string, e.g.: 1-10,15,21-40,51-62.
  static std::set<int> parseFrameRange(
      const std::string& str, const int numFrames);
  std::set<int> parseFrameRange(const std::string& str);

  int numColorStreams() const { return colorStreams_.size(); }
  bool hasColorStream(const std::string& name) const;
  int colorStreamIndex(const std::string& name) const;
  ColorStream& colorStream(const int stream);
  const ColorStream& colorStream(const int stream) const;
  ColorStream& colorStream(const std::string& name);
  const ColorStream& colorStream(const std::string& name) const;
  void createColorStream(
      const std::string& name,
      const std::string& dir,
      const std::string& extension,
      const int type,
      const std::pair<int, int>& size = {-1, -1});
//   void createColorStream(
//       const std::string& name,
//       const std::string& dir,
//       const GopTable& gopTable,
//       const std::pair<int, int>& size = {-1, -1});
  ColorFrame& colorFrame(const int stream, const int frame);

  int numDepthStreams() const { return depthStreams_.size(); }
  bool hasDepthStream(const std::string& name) const;
  int depthStreamIndex(const std::string& name) const;
  DepthStream& depthStream(const int stream);
  const DepthStream& depthStream(const int stream) const;
  DepthStream& depthStream(const std::string& name);
  const DepthStream& depthStream(const std::string& name) const;
  void createDepthStream(
      const std::string& name,
      const std::string& dir,
      const std::pair<int, int>& size = {-1, -1});
//   void createDepthStream(
//       const std::string& name,
//       const std::string& dir,
//       const GopTable& gopTable,
//       const QuantTable& quantTable,
//       const std::pair<int, int>& size = {-1, -1});
  DepthFrame& depthFrame(const int stream, const int frame);
  const DepthFrame& depthFrame(const int stream, const int frame) const;
  void clearDepthCaches();

  // Write the depth maps for a stream, for example, after updating them with
  // DepthFrame::setDepth().
  void saveDepth(const int stream);

  // Project an image space point to world space using a camera pose and depth
  // value. The image space location is parameterized in the range
  // [0, 1] x [0, invAspect].
  Eigen::Vector3f project(
      const DepthPhoto::Extrinsics& extr, const DepthPhoto::Intrinsics& intr,
      const Eigen::Vector2f& loc, const float depth) const;

  Eigen::Vector3f project(
      const int stream, const int frame,
      const Eigen::Vector2f& loc, const bool useWarp = true) const;

 private:
  std::string path_;

  std::vector<std::unique_ptr<MetaFrame>> metaFrames_;
  std::vector<std::unique_ptr<ColorStream>> colorStreams_;
  std::vector<std::unique_ptr<DepthStream>> depthStreams_;

  float duration_ = 0.f;
  int width_ = 0;
  int height_ = 0;
  float aspect_ = 0.f;
  float invAspect_ = 0.f;
};

}} // namespace facebook::cp
