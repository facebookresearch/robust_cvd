// Copyright 2004-present Facebook. All Rights Reserved.

#include "DepthPhoto.h"

#include <boost/filesystem.hpp>
#include <fmt/format.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/CvUtil.h"
#include "core/Enum.h"
#include "core/Enum_impl.h"
#include "core/FileIo.h"
#include "core/MathUtil.h"
#include "core/Misc.h"
#include "core/Projection.h"
// #include "x3d/photo3d/core/DepthMapConverter.h"

using namespace Eigen;

namespace facebook {
namespace cp {

// The default orientation is: right = +X, up = +Y, forward = -Z.
const Quaternionf DepthPhoto::Extrinsics::kDefaultOrientation =
    Quaternionf(1.f, 0.f, 0.f, 0.f);

const uint32_t DepthPhoto::kMagic = 0xDEADBEEF;
const uint32_t DepthPhoto::kFileFormatVersion = 3;
const uint32_t DepthPhoto::kMinFileFormatVersionSupported = 0;

const EnumStrings<DepthPhoto::Intrinsics::Projection> projectionStrs = {
    {DepthPhoto::Intrinsics::Projection::Perspective, "Perspective"},
    {DepthPhoto::Intrinsics::Projection::Cylindrical, "Cylindrical"},
    {DepthPhoto::Intrinsics::Projection::Equirectangular, "Equirectangular"}};

MAKE_VALIDATOR(DepthPhoto::Intrinsics::Projection, projectionStrs);

Vector3f DepthPhoto::Extrinsics::left() const {
  return orientation * -Vector3f::UnitX();
}

Vector3f DepthPhoto::Extrinsics::right() const {
  return orientation * Vector3f::UnitX();
}

Vector3f DepthPhoto::Extrinsics::down() const {
  return orientation * -Vector3f::UnitY();
}

Vector3f DepthPhoto::Extrinsics::up() const {
  return orientation * Vector3f::UnitY();
}

Vector3f DepthPhoto::Extrinsics::forward() const {
  return orientation * -Vector3f::UnitZ();
}

Vector3f DepthPhoto::Extrinsics::backward() const {
  return orientation * Vector3f::UnitZ();
}

Matrix4f DepthPhoto::Extrinsics::worldToCamera() const {
  Matrix4f translate, rotate;
  translate << 1.f, 0.f, 0.f, -position.x(), 0.f, 1.f, 0.f, -position.y(), 0.f,
      0.f, 1.f, -position.z(), 0.f, 0.f, 0.f, 1.f;

  Vector3f right = orientation * Vector3f::UnitX();
  Vector3f up = orientation * Vector3f::UnitY();
  Vector3f front = orientation * Vector3f::UnitZ();
  rotate << right.x(), right.y(), right.z(), 0.f, up.x(), up.y(), up.z(), 0.f,
      front.x(), front.y(), front.z(), 0.f, 0.f, 0.f, 0.f, 1.f;

  Matrix4f worldToCamera = rotate * translate;

  return worldToCamera;
}

DepthPhoto::Extrinsics DepthPhoto::Extrinsics::fromWorldToCamera(
    const Matrix4f& worldToCamera) {
  const Matrix4f& W = worldToCamera;
  Vector3f right(W(0, 0), W(0, 1), W(0, 2));
  Vector3f up(W(1, 0), W(1, 1), W(1, 2));
  Vector3f front(W(2, 0), W(2, 1), W(2, 2));

  Matrix4f rotate;
  rotate << right.x(), right.y(), right.z(), 0.f, up.x(), up.y(), up.z(), 0.f,
      front.x(), front.y(), front.z(), 0.f, 0.f, 0.f, 0.f, 1.f;

  Matrix4f U = rotate.transpose() * worldToCamera;

  Matrix3f R = rotate.block(0, 0, 3, 3);

  Extrinsics extr;
  extr.position = Vector3f(-U(0, 3), -U(1, 3), -U(2, 3));
  extr.orientation = Quaternionf(R.transpose());

  return extr;
}

void DepthPhoto::Extrinsics::fread(std::istream& is, const int /*format*/) {
  readEigen(is, position);
  readEigen(is, orientation);
}

void DepthPhoto::Extrinsics::fwrite(std::ostream& os) const {
  writeEigen(os, position);
  writeEigen(os, orientation);
}

// Default field-of-view values: 29.107 x 38.187 degrees.
const float DepthPhoto::Intrinsics::kDefaultHFov = 0.508015513f;
const float DepthPhoto::Intrinsics::kDefaultVFov = 0.666488587f;

void DepthPhoto::Intrinsics::resolveMissingFov(
    const float aspect,
    const bool silent) {
  bool isVerticalFovSet = vFov > 0;
  bool isHorizontalFovSet = hFov > 0;
  if (isVerticalFovSet && isHorizontalFovSet) {
    return;
  }

  if (aspect == 0) {
    throw std::runtime_error("Aspect ratio must be non-zero.");
  }

  const float defaultHalfWidth = tanf(kDefaultHFov / 2.f);
  const float defaultHalfHeight = tanf(kDefaultVFov / 2.f);
  const float defaultAspect = defaultHalfWidth / defaultHalfHeight;

  if (!isVerticalFovSet && !isHorizontalFovSet) {
    if (!silent) {
      LOG(INFO) << "Field of view is not set, using default values.";
    }
    // If neither value is set, set the fov of the shorter side to the
    // default value of the viewing virtual camera.
    if (aspect > defaultAspect) {
      vFov = kDefaultVFov;
      isVerticalFovSet = true;
    } else {
      hFov = kDefaultHFov;
      isHorizontalFovSet = true;
    }
  }

  // Calculate the missing value from the provided value
  if (isVerticalFovSet) {
    float halfHeight = tan(vFov / 2.0f);
    float halfWidth = halfHeight * aspect;
    hFov = atan(halfWidth) * 2.0f;
  } else if (isHorizontalFovSet) {
    float halfWidth = tan(hFov / 2.0f);
    float halfHeight = halfWidth / aspect;
    vFov = atan(halfHeight) * 2.0f;
  }
}

float DepthPhoto::Intrinsics::aspect() const {
  return std::tan(hFov / 2.f) / std::tan(vFov / 2.f);
}

Matrix4f DepthPhoto::Intrinsics::cameraToClip(
    const float zNear,
    const float zFar) const {
  float aspect = tan(hFov / 2.f) / tan(vFov / 2.f);
  return perspectiveProjection(zNear, zFar, rad2deg(vFov), aspect);
}

DepthPhoto::Intrinsics DepthPhoto::Intrinsics::fromCameraToClip(
    const Matrix4f& cameraToClip) {
  Intrinsics intr;
  intr.hFov = std::atan(1.f / cameraToClip(0, 0)) * 2.f;
  intr.vFov = std::atan(1.f / cameraToClip(1, 1)) * 2.f;
  return intr;
}

void DepthPhoto::Intrinsics::fread(std::istream& is, const int format) {
  *this = Intrinsics();

  if (format < 2) {
    is >> vFov;
    is >> hFov;
    return;
  }

  if (format >= 3) {
    projection = read<Projection>(is);
  }

  vFov = read<float>(is);
  hFov = read<float>(is);

  if (format >= 3) {
    centerLat = read<float>(is);
    centerLon = read<float>(is);
  }
}

void DepthPhoto::Intrinsics::fwrite(std::ostream& os) const {
  write(os, projection);
  write(os, vFov);
  write(os, hFov);
  write(os, centerLat);
  write(os, centerLon);
}

void DepthPhoto::Intrinsics::addCommandLineOptions() {
  addOption("projection", &projection);
  addOption("vFov", &vFov);
  addOption("hFov", &hFov);
  addOption("centerLat", &centerLat);
  addOption("centerLon", &centerLon);
}

void DepthPhoto::Intrinsics::printParams() const {
  printParam("projection", projection, projectionStrs);
  printParam("vFov", vFov);
  printParam("hFov", hFov);
  printParam("centerLat", centerLat);
  printParam("centerLon", centerLon);
}

DepthPhoto::DepthPhoto() {
  images_ = std::make_unique<ImageStore>();
}

DepthPhoto::~DepthPhoto() {}

DepthPhoto::DepthPhoto(const DepthPhoto& other) noexcept : DepthPhoto() {
  *this = other;
}

DepthPhoto& DepthPhoto::operator=(const DepthPhoto& other) noexcept {
  if (this != &other) {
    images_ = std::make_unique<ImageStore>();
    images_->colorImg_ = other.images_->colorImg_;
    images_->depthImg_ = other.images_->depthImg_;
    images_->matteImg_ = other.images_->matteImg_;
    isBGRA_ = other.isBGRA_;
    width_ = other.width_;
    height_ = other.height_;
    aspect_ = other.aspect_;
    extrinsics_ = other.extrinsics_;
    intrinsics_ = other.intrinsics_;
    time_ = other.time_;
  }
  return *this;
}

DepthPhoto::DepthPhoto(DepthPhoto&& other) noexcept {
  *this = std::move(other);
}

DepthPhoto& DepthPhoto::operator=(DepthPhoto&& other) noexcept {
  if (this != &other) {
    *this = other;
    other.clear();
  }
  return *this;
}

void DepthPhoto::clear() {
  images_->colorImg_ = cv::Mat4b();
  images_->depthImg_ = cv::Mat1f();
  images_->matteImg_ = cv::Mat1b();

  width_ = 0;
  height_ = 0;
  aspect_ = 0.f;

  extrinsics_ = Extrinsics();
  intrinsics_ = Intrinsics();
  time_ = 0.f;
}

void DepthPhoto::updateColor(const std::string& colorImgPath) {
  if (!boost::filesystem::is_regular_file(colorImgPath)) {
    throw std::runtime_error("Cannot find color image.");
  }

  cv::Mat3b imageBGR = imread(colorImgPath, cv::IMREAD_COLOR);
  cvtColor(imageBGR, images_->colorImg_, cv::COLOR_BGR2BGRA);
  isBGRA_ = true;
  if (images_->colorImg_.empty()) {
    throw std::runtime_error("Error loading color image.");
  }

  finishUpdateColor();
}

void DepthPhoto::updateColor(const cv::Mat4b& colorImg, const bool isBGRA) {
  if (colorImg.empty()) {
    throw std::runtime_error("Color image cannot be empty.");
  }
  images_->colorImg_ = colorImg;
  isBGRA_ = isBGRA;

  finishUpdateColor();
}

void DepthPhoto::finishUpdateColor() {
  width_ = images_->colorImg_.cols;
  height_ = images_->colorImg_.rows;
  aspect_ = width_ / float(height_);
}

// void DepthPhoto::updateDepth(
//     const std::string& depthImgPath,
//     DepthEncoding encoding,
//     DepthConversionParams conversionParams) {
//   if (!boost::filesystem::is_regular_file(depthImgPath)) {
//     throw std::runtime_error("Cannot find depth image.");
//   }

//   cv::Mat1b sourceDepthImg = imread(depthImgPath, cv::IMREAD_GRAYSCALE);
//   if (sourceDepthImg.empty()) {
//     throw std::runtime_error("Error loading depth image.");
//   }

//   updateDepth(sourceDepthImg, encoding, conversionParams);
// }

// void DepthPhoto::updateDepth(
//     const cv::Mat& depthImg,
//     DepthEncoding encoding,
//     DepthConversionParams conversionParams) {
//   conversionParams_ = conversionParams;
//   DepthMapConverter::toLinearDepth(
//       depthImg, images_->depthImg_, encoding, conversionParams);
// }

// float DepthPhoto::depth(const Eigen::Vector2f& imagePosition) const {
//   const int dw = images_->depthImg_.cols, dh = images_->depthImg_.rows;
//   int dx = clamp(imagePosition.x() * dw, 0, dw - 1);
//   int dy = clamp(imagePosition.y() * dh * aspect(), 0, dh - 1);
//   return images_->depthImg_(dy, dx);
// }

// void DepthPhoto::updateMatte(const cv::Mat1b& matteImg) {
//   if (matteImg.empty()) {
//     throw std::runtime_error("Matte image cannot be empty.");
//   }
//   images_->matteImg_ = matteImg;
// }

// Vector3f DepthPhoto::worldPosition(const Vector2f& imagePosition) const {
//   return worldPosition(imagePosition, depth(imagePosition));
// }

Eigen::Vector3f DepthPhoto::worldPosition(
    const Eigen::Vector2f& imagePosition,
    const float depth) const {
  const float x =
      (-1.f + 2.f * imagePosition.x()) * std::tan(intrinsics_.hFov / 2.f);
  const float y = (-1.f + 2.f * imagePosition.y() * aspect_) *
      std::tan(intrinsics_.vFov / 2.f);
  const Vector3f ray =
      extrinsics_.right() * x + extrinsics_.down() * y + extrinsics_.forward();
  return extrinsics_.position + ray * depth;
}

void DepthPhoto::save(const std::string& fileName) {
  std::ofstream os(fileName, std::ios::binary | std::ios::trunc);
  fwrite(os);
}

void DepthPhoto::load(const std::string& fileName) {
  std::ifstream is(fileName, std::ios::binary);
  fread(is);
}

void DepthPhoto::fwrite(std::ostream& os) {
  write(os, kMagic);
  write(os, kFileFormatVersion);

  writeim(os, images_->colorImg_);
  write(os, isBGRA_);

  writeim(os, images_->depthImg_);
  extrinsics_.fwrite(os);
  intrinsics_.fwrite(os);
  write(os, time_);
}

void DepthPhoto::fread(std::istream& is) {
  clear();

  uint32_t version = 0;

  uint32_t checkMagic = read<uint32_t>(is);

  if (checkMagic == kMagic) {
    version = read<uint32_t>(is);
  } else {
    // Did not see magic, assume file format version is 0, rewind file pointer.
    is.clear();
    is.seekg(0, std::ios::beg);
  }

  images_->colorImg_ = cv::Mat4b();
  readim(is, images_->colorImg_);

  if (version >= 1) {
    isBGRA_ = read<bool>(is);
  } else {
    isBGRA_ = false;
  }

  images_->depthImg_ = cv::Mat1f();
  readim(is, images_->depthImg_);

  width_ = images_->colorImg_.cols;
  height_ = images_->colorImg_.rows;
  aspect_ = width_ / float(height_);

  extrinsics_.fread(is, version);
  intrinsics_.fread(is, version);
  time_ = read<float>(is);
}
} // namespace cp
} // namespace facebook
