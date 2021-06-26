// Copyright 2004-present Facebook. All Rights Reserved.

#include "ColorStream.h"

#include <boost/filesystem.hpp>
#include <fmt/format.h>
#include <opencv2/imgcodecs.hpp>

#include "core/Cache.h"
#include "core/CvUtil.h"
// #include "x3d/lib/gl/GlTexture.h"
#include "DepthVideo.h"
// #include "GopCodec.h"
#include "PoolCache.h"

using namespace cv;

namespace fs = boost::filesystem;

namespace facebook {
namespace cp {

// Throw if mat is not null and does not match the specified type.
static void checkImageType(const Mat* const mat, const int type) {
  if (mat) {
    const int imageType = mat->type();
    if (imageType != type) {
      throw std::runtime_error("Image has incorrect type.");
    }
  }
}

//**************************
//****                  ****
//****    ColorFrame    ****
//****                  ****
//**************************

ColorFrame::ColorFrame(ColorStream& parentStream, const int index)
    : parentStream_(parentStream), index_(index) {
}

ColorFrame::~ColorFrame() {
}

const Mat1b* ColorFrame::image1b() const {
  const Mat* mat = (const_cast<ColorFrame*>(this))->getImage();
  checkImageType(mat, CV_8UC1);
  return (Mat1b*)mat;
}

const Mat3b* ColorFrame::image3b() const {
  const Mat* mat = (const_cast<ColorFrame*>(this))->getImage();
  checkImageType(mat, CV_8UC3);
  return (Mat3b*)mat;
}

const Mat1f* ColorFrame::image1f() const {
  const Mat* mat = (const_cast<ColorFrame*>(this))->getImage();
  checkImageType(mat, CV_32FC1);
  return (Mat1f*)mat;
}

const Mat3f* ColorFrame::image3f() const {
  const Mat* mat = (const_cast<ColorFrame*>(this))->getImage();
  checkImageType(mat, CV_32FC3);
  return (Mat3f*)mat;
}

const Mat* ColorFrame::image() const {
  const Mat* mat = (const_cast<ColorFrame*>(this))->getImage();
  checkImageType(mat, parentStream_.type());
  return mat;
}

// const GlTexture* ColorFrame::glImage() {
//   GlTexture* tex = nullptr;
//   if (!parentStream_.texCache_->get(index_, tex)) {
//     const Mat* image = getImage();
//     *tex = GlTexture(*image);
//   }

//   return tex;
// }

const Mat* ColorFrame::getImage() {
  Mat* image = nullptr;
  if (!parentStream_.imageCache_->get(index_, image)) {
    // if (parentStream_.gopTable_) {
    //   loadImageFromGop(*image);
    // } else {
      loadImageFromFile(*image);
    // }
  }

  return image;
}

void ColorFrame::loadImageFromFile(Mat& image) {
  const std::string& extension = parentStream_.extension();
  const std::string fileName = fmt::format(
      "{:s}/frame_{:06d}{:s}", parentStream_.path(), index_, extension);
  if (fs::exists(fileName)) {
    if (extension == ".raw") {
      freadim(fileName, image);
    } else {
      int flag = 0;
      if (parentStream_.channels() == 1) {
        flag = IMREAD_GRAYSCALE;
      } else if (parentStream_.channels() == 3) {
        flag = IMREAD_COLOR;
      } else {
        throw std::runtime_error(
            "Only 1 and 3 channel color streams supported.");
      }
      image = imread(fileName, flag);

      const int streamDepth = parentStream_.depth();
      const int loadedDepth = image.depth();
      if (streamDepth != image.depth()) {
        float scale = 1.f;
        if (streamDepth == CV_8U && loadedDepth == CV_32F) {
          // No scale adjustment needed.
        } else if (streamDepth == CV_32F && loadedDepth == CV_8U) {
          scale = 1.f / 256.f;
        } else {
          throw std::runtime_error("Unsupported type conversion.");
        }
        image.convertTo(image, parentStream_.depth(), scale);
      }

      if (image.type() != parentStream_.type()) {
        throw std::runtime_error("Loaded image has incorrect type.");
      }
    }

    // Check or update stream dimensions
    int& streamWidth = parentStream_.width_;
    int& streamHeight = parentStream_.height_;
    if (streamWidth <= 0) {
      streamWidth = image.cols;
      streamHeight = image.rows;
    } else {
      if (image.cols != streamWidth || image.rows != streamHeight) {
        throw std::runtime_error(
            "Color frame dimensions do not match stream dimensions.");
      }
    }
  }
}

// void ColorFrame::loadImageFromGop(Mat& image) {
//   auto& gopDecoder = parentStream_.gopDecoder_;
//   if (!gopDecoder) {
//     gopDecoder = std::make_unique<GopDecoder>(
//         parentStream_.path_,
//         parentStream_.width_, parentStream_.height_, *parentStream_.gopTable_);
//   }

//   if (image.type() != CV_8UC3 ||
//       image.cols != parentStream_.width() ||
//       image.rows != parentStream_.height()) {
//     image.create(parentStream_.height(), parentStream_.width(), CV_8UC3);
//   }

//   Mat3b wrapped(image);
//   gopDecoder->frameBgr(index_, wrapped);
// }

//***************************
//****                   ****
//****    ColorStream    ****
//****                   ****
//***************************

ColorStream::ColorStream(DepthVideo& parentVideo)
    : parentVideo_(parentVideo) {
  imageCache_ = std::make_unique<PoolCache<Mat>>(cacheSize_);
  // texCache_ = std::make_unique<PoolCache<GlTexture>>(cacheSize_);
}

ColorStream::~ColorStream() {
}

ColorFrame& ColorStream::frame(const int index) {
  return *frames_[index];
}

const ColorFrame& ColorStream::frame(const int index) const {
  return *frames_[index];
}

int ColorStream::channels() const {
  return CV_MAT_CN(type_);
}

int ColorStream::depth() const {
  return CV_MAT_DEPTH(type_);
}

int ColorStream::width() const {
  if (width_ < 0) {
    const_cast<ColorStream*>(this)->initDimensions();
  }
  return width_;
}

int ColorStream::height() const {
  if (height_ < 0) {
    const_cast<ColorStream*>(this)->initDimensions();
  }
  return height_;
}

void ColorStream::setDir(const std::string& dir) {
  dir_ = dir;
  path_ = parentVideo_.path() + "/" + dir_;
}

void ColorStream::initDimensions() {
  if (width_ >= 0) {
    return;
  }

  for (auto& f : frames_) {
    const Mat* image = f->image();
    if (image) {
      // If image has been loaded, dimensions will have been initialized.
      return;
    }
  }

  width_ = height_ = 0;
}

}} // namespace facebook::cp
