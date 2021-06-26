// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <memory>

#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include "core/Logging.h"
#include "core/ParamsBase.h"
// #include "x3d/photo3d/core/DepthConversionParams.h"
// #include "x3d/photo3d/core/DepthEncoding.h"

namespace facebook {
namespace cp {

class DepthPhoto {
 public:
  struct Extrinsics {
    static const Eigen::Quaternionf kDefaultOrientation;

    Eigen::Vector3f position = Eigen::Vector3f::Zero();

    // We are using right-handed coordinate systems throughout x3d, i.e., like
    // the default OpenGL world- and object-space coordinate systems.
    //
    // Camera orientation is represented as a quaternion. It can be constructed
    // from orthonormal right, up, forward vectors as follows. Note the negative
    // sign on the 3rd column, which is necessary to make this matrix a rotation
    // matrix (determinant = 1) and not a rotation-reflection matrix
    // (determinant = -1).
    //   Eigen::Matrix3f rotation;
    //   rotation.col(0) = right;
    //   rotation.col(1) = up;
    //   rotation.col(2) = -forward;
    //   orientation = Quaternionf(rotation);
    //
    // The right, up, forward can be obtained back as follows:
    //   Vector3f right = orientation * Vector3f::UnitX();
    //   Vector3f up = orientation * Vector3f::UnitY();
    //   Vector3f front = orientation * -Vector3f::UnitZ();
    Eigen::Quaternionf orientation = kDefaultOrientation;

    // Obtain orthogonal vectors from quaternion.
    Eigen::Vector3f left() const; // -X
    Eigen::Vector3f right() const; // +X
    Eigen::Vector3f down() const; // -Y
    Eigen::Vector3f up() const; // +Y
    Eigen::Vector3f forward() const; // -Z
    Eigen::Vector3f backward() const; // +Z

    // Construct a 'modelview' matrix.
    Eigen::Matrix4f worldToCamera() const;

    // Extract extrinsics from a 'modelview' matrix.
    static Extrinsics fromWorldToCamera(const Eigen::Matrix4f& worldToCamera);

    void fread(std::istream& is, const int format);
    void fwrite(std::ostream& os) const;
  };

  // Represents the projection of the image. When using equirect projection the
  // image contains a lat-lon crop, whose angular extents are determined by the
  // fov parameters, and which is centered around the centerLat/Lon parameters.
  // When using perspective projection, the centerLat/Lon parameters are
  // ignored.
  struct Intrinsics : public ParamsBase {
    enum class Projection {
      Perspective,
      Equirectangular,
      Cylindrical,
    };

    const static float kDefaultHFov;
    const static float kDefaultVFov;

    Projection projection = Projection::Perspective;
    float vFov = 0.f; // camera that took the photo's FOV
    float hFov = 0.f;
    float centerLat = 0.f; // Vertical center
    float centerLon = 0.f; // Horizontal center

    // If one of vFov/hFov is set, it will fill in the other. If both are set,
    // it doesn't do anything. If neither is set, it will "scale to fit" and
    // won't show any of the out of bounds pixels.
    void resolveMissingFov(const float aspect, const bool silent = false);

    bool isLandscape() const {
      return !isPortrait();
    }
    bool isPortrait() const {
      return hFov <= vFov;
    }

    float aspect() const;

    // Construct a projection matrix.
    Eigen::Matrix4f cameraToClip(
        const float zNear = 0.01f,
        const float zFar = 1000.f) const;

    // Extract intrinsics from a projection matrix (assumes perspective
    // projection).
    static Intrinsics fromCameraToClip(const Eigen::Matrix4f& cameraToClip);

    void fread(std::istream& is, const int format);
    void fwrite(std::ostream& os) const;

    void addCommandLineOptions() override;
    void printParams() const override;
  };

  // File format version written by save().
  const static uint32_t kFileFormatVersion;

  DepthPhoto();
  virtual ~DepthPhoto();

  // Copy constructors.
  DepthPhoto(const DepthPhoto&) noexcept;
  DepthPhoto& operator=(const DepthPhoto&) noexcept;

  // Move constructors.
  DepthPhoto(DepthPhoto&& other) noexcept;
  DepthPhoto& operator=(DepthPhoto&& other) noexcept;

  bool empty() const {
    return images_->depthImg_.empty() || images_->colorImg_.empty();
  }

  virtual void clear();

  void updateColor(const std::string& colorImgPath);
  void updateColor(const cv::Mat4b& colorImg, const bool isBGRA);

  // // The depth may have lower resolution than the color image.
  // virtual void updateDepth(
  //     const std::string& depthImgPath,
  //     DepthEncoding encoding,
  //     DepthConversionParams conversionParams = DepthConversionParams());
  // virtual void updateDepth(
  //     const cv::Mat& depthImg,
  //     DepthEncoding encoding,
  //     DepthConversionParams conversionParams = DepthConversionParams());

  // void updateMatte(const cv::Mat1b& matteImg);

  // Update the camera pose and time.
  void updateExtrinsics(const Extrinsics& extrinsics) {
    extrinsics_ = extrinsics;
  }

  void updateIntrinsics(const Intrinsics& intrinsics) {
    intrinsics_ = intrinsics;
    intrinsics_.resolveMissingFov(aspect_);
  }

  void updateTime(const float time) {
    time_ = time;
  }

  // Basic getters
  int width() const {
    return width_;
  }
  int height() const {
    return height_;
  }
  bool isBGRA() const {
    return isBGRA_;
  }
  float aspect() const {
    return width_ / float(height_);
  }
  float invAspect() const {
    return height_ / float(width_);
  }
  const cv::Mat4b& colorImg() const {
    return images_->colorImg_;
  }
  const cv::Mat1f& depthImg() const {
    return images_->depthImg_;
  }
  const cv::Mat1b& matteImg() const {
    return images_->matteImg_;
  }
  const Extrinsics& extrinsics() const {
    return extrinsics_;
  }
  const Intrinsics& intrinsics() const {
    return intrinsics_;
  }
  // const DepthConversionParams& depthConversionParams() const {
  //   return conversionParams_;
  // }

  // Compute the depth / world position for a given image position. The image
  // position is normalized to [0, 1] x [0, invAspect].
  // virtual float depth(const Eigen::Vector2f& imagePosition) const;
  Eigen::Vector3f worldPosition(const Eigen::Vector2f& imagePosition) const;
  Eigen::Vector3f worldPosition(
      const Eigen::Vector2f& imagePosition,
      const float depth) const;

  void save(const std::string& fileName);
  void load(const std::string& fileName);

  void fread(std::istream& is);
  void fwrite(std::ostream& os);

 protected:
  virtual void finishUpdateColor();

  // Magic number to recognize our binary file format.
  const static uint32_t kMagic;

  // Minimum file format version supported by load()
  const static uint32_t kMinFileFormatVersionSupported;

  // We're storing the images in a pointer, so we can manipulate them in const
  // functions. This is necessary, for example, in the derived DepthPhotoGl,
  // when using the const depth() access and when the OpenGL version is < 4.5:
  // in that case we need to read the whole depth texture back into the image.
  struct ImageStore {
    cv::Mat4b colorImg_;
    cv::Mat1f depthImg_;
    cv::Mat1b matteImg_;
  };

  std::unique_ptr<ImageStore> images_;

  // true if the color image is stored in BGRA, false if RGBA.
  bool isBGRA_ = false;

  int width_ = 0;
  int height_ = 0;
  float aspect_ = 0.f;

  Extrinsics extrinsics_;
  Intrinsics intrinsics_;
  // DepthConversionParams conversionParams_;
  float time_ = 0.f;
};

// extern const EnumStrings<DepthEncoding::Type> depthEncodingStrs;
// DECLARE_VALIDATOR(DepthEncoding::Type, depthEncodingStrs);

extern const EnumStrings<DepthPhoto::Intrinsics::Projection> projectionStrs;
DECLARE_VALIDATOR(DepthPhoto::Intrinsics::Projection, projectionStrs);

// // Enables using DepthEncoding::Flags as bitfield flags
// inline DepthEncoding::Flags operator|(
//     DepthEncoding::Flags a,
//     DepthEncoding::Flags b) {
//   return static_cast<DepthEncoding::Flags>(
//       static_cast<int>(a) | static_cast<int>(b));
// }

// inline DepthEncoding::Flags operator|=(
//     DepthEncoding::Flags& a,
//     DepthEncoding::Flags b) {
//   return a = a | b;
// }

// inline DepthEncoding::Flags operator&(
//     DepthEncoding::Flags a,
//     DepthEncoding::Flags b) {
//   return static_cast<DepthEncoding::Flags>(
//       static_cast<int>(a) & static_cast<int>(b));
// }

// inline DepthEncoding::Flags operator&=(
//     DepthEncoding::Flags& a,
//     DepthEncoding::Flags b) {
//   return a = a & b;
// }

} // namespace cp
} // namespace facebook
