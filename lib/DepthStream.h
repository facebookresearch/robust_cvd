// Copyright 2004-present Facebook. All Rights Reserved.

// This file provides a depth stream for depth videos. Refer to `DepthVideo.h`
// for a usage example.

#pragma once

#include <memory>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "DepthPhoto.h"
#include "DepthMapTransform.h"

namespace facebook {
namespace cp {

class DepthStream;
class DepthVideo;
// class GlTexture;
// class GopDecoder;
// struct GopTable;
struct QuantTable;

struct QuantDescriptor {
  float minDisparity = 0.f;
  float maxDisparity = 0.f;
};

struct QuantTable {
  std::vector<QuantDescriptor> gops;

  void fread(std::istream& is, const int format);
  void fwrite(std::ostream& os) const;
};

cv::Mat1f dequantize(const cv::Mat3b& quantized, const QuantDescriptor& desc);

//**************************
//****                  ****
//****    DepthFrame    ****
//****                  ****
//**************************

class DepthFrame {
  friend class DepthVideo;

 public:
  ~DepthFrame();

  // Disable copy and assignment
  DepthFrame(const DepthFrame&) = delete;
  DepthFrame& operator=(const DepthFrame&) = delete;

  // The functions below return the transformed depth.
  const cv::Mat1f* depth() const;
  // const GlTexture* glDepth();

  // The functions below returns the source depth (i.e., _not_ transformed), and
  // a map with the transform parameters.
  const cv::Mat1f* sourceDepth() const;
  const cv::Mat* depthXformParamMap() const;

  // Get the extreme depth values of the source depth map.
  Eigen::Vector2f sourceDepthMinMax() const;

  // Set the source depth.
  void setDepth(const cv::Mat1f& depth);

  // This function returns the spatial warp field.
  const cv::Mat2f* warp() const;

  // Reset everything.
  void clear();

  // Clear only the caches, to force a reload of the depth map on the next
  // access. The poses remains unchanged.
  void clearCache();

  // Clear only the transform cache, to force a re-computation of the
  // post-transform depth map.
  void clearXformedCache();

  DepthXform& depthXform();
  const DepthXform& depthXform() const;
  void resetDepthXform();

  SpatialXform& spatialXform();
  const SpatialXform& spatialXform() const;
  void resetSpatialXform();

  // Public fields, read and write.
  DepthPhoto::Intrinsics intrinsics;
  DepthPhoto::Extrinsics extrinsics;

  // Set to false if this frame is missing.
  bool enabled = true;

  // Convenience functions that return information from the parent stream or
  // parent video, so you can get all information that might be needed with just
  // passing the depth frame around.
  int width() const;
  int height() const;
  float aspect() const;
  float invAspect() const;

 private:
  // Can't be constructed directly, only by friends.
  DepthFrame(
      DepthVideo& parentVideo, DepthStream& parentStream, const int index);

  // These are called by the corresponding public const functions. They are
  // non-const, because they are modifying the cache.
  const cv::Mat1f* getSourceDepth();
  Eigen::Vector2f getSourceDepthMinMax();
  const cv::Mat1f* getDepth();
  const cv::Mat* getDepthXformParamMap();
  const cv::Mat2f* getWarp();

  void loadSourceDepthFromFile(cv::Mat1f& sourceDepth);
  // void loadSourceDepthFromGop(cv::Mat1f& sourceDepth);

  void checkAndUpdateStreamDimensions(const int width, const int height);

  DepthVideo& parentVideo_;
  DepthStream& parentStream_;
  int index_ = -1;

  std::unique_ptr<cv::Mat1f> sourceDepth_;
  Eigen::Vector2f sourceDepthMinMax_ = Eigen::Vector2f::Zero();

  std::unique_ptr<cv::Mat1f> xformedDepth_;
  std::unique_ptr<cv::Mat> depthXformParamMap_;
  std::unique_ptr<DepthXform> depthXform_;
  XformDescriptor appliedDepthXformDesc_;
  std::vector<double> appliedDepthXformParams_;

  std::unique_ptr<cv::Mat2f> warp_;
  std::unique_ptr<SpatialXform> spatialXform_;
  XformDescriptor appliedSpatialXformDesc_;
  std::vector<double> appliedSpatialXformParams_;

  // std::unique_ptr<GlTexture> glDepth_;
};

//***************************
//****                   ****
//****    DepthStream    ****
//****                   ****
//***************************

class DepthStream {
   // We're friends with our family---children and parents.
   friend class DepthFrame;
   friend class DepthVideo;

 public:
  ~DepthStream();

  // Disable copy and assignment
  DepthStream(const DepthStream&) = delete;
  DepthStream& operator=(const DepthStream&) = delete;

  DepthFrame& frame(const int index) { return *frames_[index]; }
  const DepthFrame& frame(const int index) const { return *frames_[index]; }

  const std::string& name() const { return name_; }
  const std::string& path() const { return path_; }

  const XformDescriptor depthXformDesc() const { return depthXformDesc_; }
  const XformDescriptor spatialXformDesc() const { return spatialXformDesc_; }

  int width() const;
  int height() const;

  void setDir(const std::string& dir);
  void resetDepthXforms(const XformDescriptor& desc);
  void resetSpatialXforms(const XformDescriptor& desc);

  void clearCache();

 private:
  // Can't be constructed directly, only by friends.
  explicit DepthStream(DepthVideo& parentVideo);

  void initDimensions();

  DepthVideo& parentVideo_;

  std::string name_;
  std::string dir_;
  std::string path_;
  XformDescriptor depthXformDesc_;
  XformDescriptor spatialXformDesc_;

  int width_ = -1;
  int height_ = -1;

  // std::unique_ptr<GopDecoder> gopDecoder_;
  // std::unique_ptr<GopTable> gopTable_;
  std::unique_ptr<QuantTable> quantTable_;

  std::vector<std::unique_ptr<DepthFrame>> frames_;
};

}} // namespace facebook::cp
