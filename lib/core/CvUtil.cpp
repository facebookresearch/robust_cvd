// Copyright 2004-present Facebook. All Rights Reserved.
//
// Author: Johannes Kopf (jkopf@fb.com)

#include "CvUtil.h"
#include "FileIo.h"

#include <stdexcept>

namespace facebook {
namespace cp {

const char* szDepth(const int depth) {
    const char* depthStr[] = {"8U", "8S", "16U", "16S", "32S", "32F", "64F"};

    if (depth >= 0 && depth <= 6) {
      return depthStr[depth];
    }

    error("invalid depth");

    return nullptr;
}

void freadim(FILE * fin, cv::Mat& dst) {
  int rows = fread<int>(fin);
  int cols = fread<int>(fin);
  int type = fread<int>(fin);
  size_t elemSize = fread<size_t>(fin);

  dst.create(rows, cols, type);

  for (int y = 0; y < rows; ++y) {
    ensure(fread(dst.ptr(y), elemSize, cols, fin) == (size_t)cols);
  }
}

void freadim(const std::string& fileName, cv::Mat& dst) {
  FILE * fin = fopen(fileName.c_str(), "rb");
  freadim(fin, dst);
  fclose(fin);
}

void readim(std::istream& is, cv::Mat& dst) {
  int rows = read<int>(is);
  int cols = read<int>(is);
  int type = read<int>(is);
  size_t elemSize = read<size_t>(is);
  if (elemSize > 4096) {
    throw std::runtime_error("Invalid element size.");
  }

  dst.create(rows, cols, type);

  // Stop here if the image is empty.
  if (rows == 0 || cols == 0) {
    return;
  }

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar numChannels = 1 + (type >> CV_CN_SHIFT);

  int sizePerChannel = 0;
  switch (depth) {
  case CV_8U:
  case CV_8S:
    sizePerChannel = 1;
    break;
  case CV_16U:
  case CV_16S:
    sizePerChannel = 2;
    break;
  case CV_32S:
  case CV_32F:
    sizePerChannel = 4;
    break;
  case CV_64F:
    sizePerChannel = 8;
    break;
  default:
    throw std::runtime_error("Unsupported image depth.");
  }

  if (sizePerChannel * numChannels != elemSize) {
    throw std::runtime_error("type / element size inconsistency.");
  }

  int checkType = ((numChannels - 1) << CV_CN_SHIFT) + depth;
  if (type != checkType) {
    throw std::runtime_error("Invalid image type.");
  }

  for (int y = 0; y < rows; ++y) {
    is.read(reinterpret_cast<char*>(dst.ptr(y)), elemSize * cols);
  }
}

void fwriteim(FILE * fout, const cv::Mat& src) {
  fwrite(fout, src.rows);
  fwrite(fout, src.cols);
  fwrite(fout, src.type());
  fwrite(fout, src.elemSize());

  for (int y = 0; y < src.rows; ++y) {
    ensure(fwrite(src.ptr(y), src.elemSize(), src.cols, fout) == (size_t)src.cols);
  }
}

void fwriteim(const std::string& fileName, const cv::Mat& src) {
  FILE * fout = fopen(fileName.c_str(), "wb");
  fwriteim(fout, src);
  fclose(fout);
}

void writeim(std::ostream& os, const cv::Mat& src) {
  write(os, src.rows);
  write(os, src.cols);
  write(os, src.type());
  write(os, src.elemSize());

  for (int y = 0; y < src.rows; ++y) {
    os.write(
        reinterpret_cast<const char*>(src.ptr(y)), src.elemSize() * src.cols);
  }
}

}} // namespace facebook::cp
