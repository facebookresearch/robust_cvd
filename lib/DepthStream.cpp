// Copyright 2004-present Facebook. All Rights Reserved.

#include "DepthStream.h"

#include <boost/filesystem.hpp>

#include <fmt/format.h>
#include <opencv2/imgcodecs.hpp>

#include "core/CvUtil.h"
#include "core/FileIo.h"
// #include "x3d/lib/gl/GlTexture.h"
#include "DepthVideo.h"
// #include "GopCodec.h"

using namespace Eigen;
using namespace cv;

namespace fs = boost::filesystem;

namespace facebook {
namespace cp {

void QuantTable::fread(std::istream& is, const int /* format */) {
  readv(is, gops);
}

void QuantTable::fwrite(std::ostream& os) const {
  writev(os, gops);
}

Mat1f dequantize(const Mat1b& quantized, const QuantDescriptor& desc) {
  const int w = quantized.cols;
  const int h = quantized.rows;

  const float disparityRange = desc.maxDisparity - desc.minDisparity;

  Mat1f res(h, w);
  for (int y = 0; y < h; ++y) {
    const uint8_t* const src = quantized.ptr<const uint8_t>(y);
    float* const dst = res.ptr<float>(y);

    for (int x = 0; x < w; ++x) {
      const uint8_t val = src[x];
      const float disparity = desc.minDisparity + val * disparityRange / 255.f;
      dst[x] = 1.f / disparity;
    }
  }

  return res;
}

//**************************
//****                  ****
//****    DepthFrame    ****
//****                  ****
//**************************

DepthFrame::DepthFrame(
    DepthVideo& parentVideo, DepthStream& parentStream, const int index)
    : parentVideo_(parentVideo), parentStream_(parentStream), index_(index) {
  depthXform_ = createDepthXform(parentStream.depthXformDesc_);
  spatialXform_ = createSpatialXform(parentStream.spatialXformDesc_);
}

DepthFrame::~DepthFrame() {
}

const cv::Mat1f* DepthFrame::depth() const {
  return (const_cast<DepthFrame*>(this))->getDepth();
}

// const GlTexture* DepthFrame::glDepth() {
//   if (depthXform_->desc() != appliedDepthXformDesc_ ||
//       depthXform_->params() != appliedDepthXformParams_) {
//     glDepth_ = nullptr;
//   }

//   if (!glDepth_) {
//     const Mat1f* data = depth();
//     if (!data) {
//       return nullptr;
//     }

//     glDepth_ = std::make_unique<GlTexture>(*data);
//   }
//   return glDepth_.get();
// }

const Mat1f* DepthFrame::sourceDepth() const {
  return (const_cast<DepthFrame*>(this))->getSourceDepth();
}

const Mat* DepthFrame::depthXformParamMap() const {
  return (const_cast<DepthFrame*>(this))->getDepthXformParamMap();
}

Vector2f DepthFrame::sourceDepthMinMax() const {
  return (const_cast<DepthFrame*>(this))->getSourceDepthMinMax();
}

void DepthFrame::setDepth(const Mat1f& depth) {
  checkAndUpdateStreamDimensions(depth.cols, depth.rows);

  sourceDepth_ = std::make_unique<Mat1f>();
  *sourceDepth_ = depth;
  sourceDepthMinMax_ = Vector2f::Zero();
  xformedDepth_ = nullptr;
  depthXformParamMap_ = nullptr;
  appliedDepthXformDesc_.reset();
  appliedDepthXformParams_.clear();
  warp_ = nullptr;
  appliedSpatialXformDesc_.reset();
  appliedSpatialXformParams_.clear();
  // glDepth_ = nullptr;
}

const Mat2f* DepthFrame::warp() const {
  return (const_cast<DepthFrame*>(this))->getWarp();
}

void DepthFrame::clearCache() {
  clearXformedCache();
  sourceDepth_ = nullptr;
  sourceDepthMinMax_ = Vector2f::Zero();
}

void DepthFrame::clearXformedCache() {
  xformedDepth_ = nullptr;
  depthXformParamMap_ = nullptr;
  appliedDepthXformDesc_.reset();
  appliedDepthXformParams_.clear();
  warp_ = nullptr;
  appliedSpatialXformDesc_.reset();
  appliedSpatialXformParams_.clear();
  // glDepth_ = nullptr;
}

void DepthFrame::clear() {
  clearCache();
  intrinsics = DepthPhoto::Intrinsics();
  extrinsics = DepthPhoto::Extrinsics();
}

DepthXform& DepthFrame::depthXform() {
  if (!depthXform_) {
    throw std::runtime_error("Depth transform not initialized.");
  }
  return *depthXform_;
}

const DepthXform& DepthFrame::depthXform() const {
  return (const_cast<DepthFrame*>(this))->depthXform();
}

void DepthFrame::resetDepthXform() {
  depthXform_ = createDepthXform(parentStream_.depthXformDesc_);
  clearXformedCache();
}

SpatialXform& DepthFrame::spatialXform() {
  if (!spatialXform_) {
    throw std::runtime_error("Spatial transform not initialized.");
  }
  return *spatialXform_;
}

const SpatialXform& DepthFrame::spatialXform() const {
  return (const_cast<DepthFrame*>(this))->spatialXform();
}

void DepthFrame::resetSpatialXform() {
  spatialXform_ = createSpatialXform(parentStream_.spatialXformDesc_);
}

const Mat1f* DepthFrame::getSourceDepth() {
  if (!sourceDepth_) {
    sourceDepth_ = std::make_unique<Mat1f>();
    // if (parentStream_.gopTable_) {
    //   loadSourceDepthFromGop(*sourceDepth_);
    // } else {
      loadSourceDepthFromFile(*sourceDepth_);
    // }
  }

  if (sourceDepth_->empty()) {
    return nullptr;
  }

  return sourceDepth_.get();
}

void DepthFrame::loadSourceDepthFromFile(cv::Mat1f& sourceDepth) {
  const std::string fileName = fmt::format(
      "{:s}/depth/frame_{:06d}.raw", parentStream_.path(), index_);
  if (fs::exists(fileName)) {
    freadim(fileName, sourceDepth);

    // Convert from disparity to depth.
    for (int y = 0; y < sourceDepth.rows; ++y) {
      float* depthPtr = sourceDepth.ptr<float>(y);
      for (int x = 0; x < sourceDepth.cols; ++x) {
        float& d = depthPtr[x];

        // Set invalid depth (NaN, infinite, negative, etc.) to zero.
        if (std::isfinite(d) && d > 0.f) {
          d = 1.f / d;
        } else {
          d = 0.f;
        }
      }
    }

    checkAndUpdateStreamDimensions(sourceDepth.cols, sourceDepth.rows);
  }
}

// void DepthFrame::loadSourceDepthFromGop(cv::Mat1f& sourceDepth) {
//   auto& gopDecoder = parentStream_.gopDecoder_;
//   if (!gopDecoder) {
//     gopDecoder = std::make_unique<GopDecoder>(
//         parentStream_.path_,
//         parentStream_.width_, parentStream_.height_, *parentStream_.gopTable_);
//   }

//   Mat1b quantized(parentStream_.height_, parentStream_.width_);
//   gopDecoder->frameY(index_, quantized);

//   const int gopIndex = parentStream_.gopDecoder_->gopIndexFromFrame(index_);
//   sourceDepth =
//       dequantize(quantized, parentStream_.quantTable_->gops[gopIndex]);
// }

Vector2f DepthFrame::getSourceDepthMinMax() {
  if (sourceDepthMinMax_ == Vector2f::Zero()) {
    const Mat1f* depth = sourceDepth();
    if (!depth) {
      throw std::runtime_error("Could not get source depth map.");
    }

    // Writing a custom min-max computation function, because we need to ignore
    // zero depth values.
    float minValue = FLT_MAX;
    float maxValue = 0.f;
    for (int y = 0; y < depth->rows; ++y) {
      const float* depthPtr = depth->ptr<float>(y);
      for (int x = 0; x < depth->cols; ++x) {
        if (*depthPtr > 0.f) {
          if (*depthPtr < minValue) {
            minValue = *depthPtr;
          }
          if (*depthPtr > maxValue) {
            maxValue = *depthPtr;
          }
          ++depthPtr;
        }
      }
    }

    sourceDepthMinMax_ = Vector2f(minValue, maxValue);
  }

  return sourceDepthMinMax_;
}

const Mat* DepthFrame::getDepthXformParamMap() {
  if (!depthXformParamMap_) {
    depthXformParamMap_ = std::make_unique<Mat>();
    *depthXformParamMap_ = depthXform().paramMap(*this);
  }

  return depthXformParamMap_.get();
}

const Mat1f* DepthFrame::getDepth() {
  const Mat1f* src = sourceDepth();
  if (!src) {
    return nullptr;
  }

  if (!xformedDepth_ ||
      depthXform_->desc() != appliedDepthXformDesc_ ||
      depthXform_->params() != appliedDepthXformParams_) {
    appliedDepthXformDesc_ = depthXform_->desc();
    appliedDepthXformParams_ = depthXform_->params();
    xformedDepth_ = depthXform_->apply(*src);
  }

  return xformedDepth_.get();
}

const Mat2f* DepthFrame::getWarp() {
  if (!warp_ ||
      spatialXform_->desc() != appliedSpatialXformDesc_ ||
      spatialXform_->params() != appliedSpatialXformParams_) {
    appliedSpatialXformDesc_ = spatialXform_->desc();
    appliedSpatialXformParams_ = spatialXform_->params();
    warp_ = std::make_unique<Mat2f>(
        spatialXform_->warp(parentStream_.height(), parentStream_.width()));
  }

  return warp_.get();
}

void DepthFrame::checkAndUpdateStreamDimensions(
    const int width, const int height) {
  int& streamWidth = parentStream_.width_;
  int& streamHeight = parentStream_.height_;
  if (streamWidth <= 0) {
    streamWidth = width;
    streamHeight = height;
  } else {
    if (width != streamWidth || height != streamHeight) {
      throw std::runtime_error(
          "Depth frame dimensions do not match stream dimensions.");
    }
  }
}

int DepthFrame::width() const {
  return parentStream_.width();
}

int DepthFrame::height() const {
  return parentStream_.height();
}

float DepthFrame::aspect() const {
  return parentVideo_.aspect();
}

float DepthFrame::invAspect() const {
  return parentVideo_.invAspect();
}

//***************************
//****                   ****
//****    DepthStream    ****
//****                   ****
//***************************

DepthStream::DepthStream(DepthVideo& parentVideo)
    : parentVideo_(parentVideo) {
}

DepthStream::~DepthStream() {
}

int DepthStream::width() const {
  if (width_ < 0) {
    const_cast<DepthStream*>(this)->initDimensions();
  }
  return width_;
}

int DepthStream::height() const {
  if (height_ < 0) {
    const_cast<DepthStream*>(this)->initDimensions();
  }
  return height_;
}

void DepthStream::setDir(const std::string& dir) {
  dir_ = dir;
  path_ = parentVideo_.path() + "/" + dir_;
}

void DepthStream::resetDepthXforms(const XformDescriptor& desc) {
  depthXformDesc_ = desc;
  for (int frame = 0; frame < frames_.size(); ++frame) {
    DepthFrame& df = *frames_[frame];
    df.resetDepthXform();
  }
}

void DepthStream::resetSpatialXforms(const XformDescriptor& desc) {
  spatialXformDesc_ = desc;
  for (int frame = 0; frame < frames_.size(); ++frame) {
    DepthFrame& df = *frames_[frame];
    df.resetSpatialXform();
  }
}

void DepthStream::initDimensions() {
  if (width_ >= 0) {
    return;
  }

  for (auto& f : frames_) {
    const Mat1f* depth = f->depth();
    if (depth) {
      // If depth has been loaded, dimensions will have been initialized.
      return;
    }
  }

  width_ = height_ = 0;
}

void DepthStream::clearCache() {
  for (auto& f : frames_) {
    f->clearCache();
  }
}

}} // namespace facebook::cp
