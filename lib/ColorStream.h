// Copyright 2004-present Facebook. All Rights Reserved.

// This file provides a color stream for depth videos. Refer to `DepthVideo.h`
// for a usage example.

#pragma once

#include <memory>

#include <opencv2/core.hpp>

namespace facebook {
namespace cp {

class ColorStream;
class DepthVideo;
// class GlTexture;
// class GopDecoder;
// struct GopTable;
template <typename ValueType> class PoolCache;

//**************************
//****                  ****
//****    ColorFrame    ****
//****                  ****
//**************************

class ColorFrame {
  friend class DepthVideo;

 public:
  ~ColorFrame();

  // Disable copy and assignment
  ColorFrame(const ColorFrame&) = delete;
  ColorFrame& operator=(const ColorFrame&) = delete;

  // Return a typed image.
  const cv::Mat1b* image1b() const;
  const cv::Mat3b* image3b() const;
  const cv::Mat1f* image1f() const;
  const cv::Mat3f* image3f() const;

  // Return an untyped image.
  const cv::Mat* image() const;

  // const GlTexture* glImage();

 private:
  // Can't be constructed directly, only by friends.
  ColorFrame(ColorStream& parentStream, const int index);

  // This is called by the public const image() function. It is non-const,
  // because it is modifying the image cache.
  const cv::Mat* getImage();
  void loadImageFromFile(cv::Mat& image);
  // void loadImageFromGop(cv::Mat& image);

  ColorStream& parentStream_;
  int index_ = -1;
};

//***************************
//****                   ****
//****    ColorStream    ****
//****                   ****
//***************************

class ColorStream {
  // We're friends with our family---children and parents.
  friend class ColorFrame;
  friend class DepthVideo;

 public:
  ~ColorStream();

  // Disable copy and assignment
  ColorStream(const ColorStream&) = delete;
  ColorStream& operator=(const ColorStream&) = delete;

  // Accessors
  ColorFrame& frame(const int index);
  const ColorFrame& frame(const int index) const;

  const std::string& name() const { return name_; }
  const std::string& path() const { return path_; }
  const std::string& extension() const { return extension_; }
  int type() const { return type_; }
  int channels() const;
  int depth() const;

  int width() const;
  int height() const;

  void setDir(const std::string& dir);

 private:
  // Can't be constructed directly, only by friends.
  explicit ColorStream(DepthVideo& parentVideo);

  DepthVideo& parentVideo_;

  void initDimensions();

  std::string name_;
  std::string dir_;
  std::string path_;
  std::string extension_;
  int type_ = 0;

  int width_ = -1;
  int height_ = -1;

  // std::unique_ptr<GopTable> gopTable_;
  // std::unique_ptr<GopDecoder> gopDecoder_;

  std::vector<std::unique_ptr<ColorFrame>> frames_;

  static constexpr int cacheSize_ = 10;
  std::unique_ptr<PoolCache<cv::Mat>> imageCache_;
  // std::unique_ptr<PoolCache<GlTexture>> texCache_;
};


}} // namespace facebook::cp
