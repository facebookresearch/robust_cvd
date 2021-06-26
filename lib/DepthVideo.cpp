// Copyright 2004-present Facebook. All Rights Reserved.

#include "DepthVideo.h"

#include <fstream>

#include <boost/filesystem.hpp>

#include <fmt/format.h>

#include "core/CvUtil.h"
#include "core/FileIo.h"
#include "ColorStream.h"
#include "DepthStream.h"
// #include "GopCodec.h"

using namespace cv;
using namespace Eigen;

namespace fs = boost::filesystem;

namespace facebook {
namespace cp {

//**************************
//****                  ****
//****    DepthVideo    ****
//****                  ****
//**************************

DepthVideo::DepthVideo() {
  reset();
}

DepthVideo::~DepthVideo() {
}

void DepthVideo::printInfo() const {
  LOG(INFO) << "Path: " << path();
  LOG(INFO) << fmt::format(
      "Dimensions: {:d} x {:d} ({:f} aspect ratio)",
      width_, height_, aspect_);
  LOG(INFO) << fmt::format(
      "Frame count: {:d} ({:.2f}s duration)",
      metaFrames_.size(), duration_);
  LOG(INFO) << "Color streams: " << numColorStreams();
  for (int i = 0; i < numColorStreams(); ++i) {
    const ColorStream& cs = colorStream(i);

    std::string depth;
    switch (cs.depth()) {
    case CV_8U:
      depth = "unsigned byte";
      break;
    case CV_8S:
      depth = "signed byte";
      break;
    case CV_16U:
      depth = "unsigned short";
      break;
    case CV_16S:
      depth = "signed short";
      break;
    case CV_32S:
      depth = "signed int";
      break;
    case CV_32F:
      depth = "float";
      break;
    case CV_64F:
      depth = "double";
      break;
    default:
      throw std::runtime_error("Unsupported image depth.");
    }

    LOG(INFO) << fmt::format(
        "  {:2d}: {:s} ({:d} x {:d}, {:s}, {:d} channels)",
        i, cs.name(), cs.width(), cs.height(), depth, cs.channels());
    LOG(INFO) << "      Path: " << cs.path() << " (" << cs.extension() << ")";
  }
  LOG(INFO) << "Depth streams: " << numDepthStreams();
  for (int i = 0; i < numDepthStreams(); ++i) {
    const DepthStream& ds = depthStream(i);
    LOG(INFO) << fmt::format("  {:2d}: {:s} ({:d} x {:d})",
        i, ds.name(), ds.width(), ds.height());
    LOG(INFO) << "      Path: " << ds.path();
  }
}

void DepthVideo::reset() {
  path_ = "";

  metaFrames_.clear();
  colorStreams_.clear();
  depthStreams_.clear();

  duration_ = 0.f;
  aspect_ = 0.f;
  invAspect_ = 0.f;
}

void DepthVideo::init(
    const std::string& path, const int width, const int height,
    std::vector<std::unique_ptr<MetaFrame>>&& metaFrames) {
  reset();

  path_ = path;

  metaFrames_ = std::move(metaFrames);
  width_ = width;
  height_ = height;
  aspect_ = width / float(height);
  invAspect_ = 1.f / aspect_;

  duration_ =
      metaFrames_.back()->pts() * metaFrames_.size() /
      float(metaFrames_.size() - 1);
}

void DepthVideo::load(const std::string& path) {
  reset();

  LOG(INFO) << "Loading depth video '" << path << "'...";

  path_ = path;

  std::string binaryFile = path + "/video.dat";
  if (!fs::exists(binaryFile)) {
    throw std::runtime_error("Could not find 'video.dat'.");
  }

  std::ifstream is(binaryFile, std::ios::binary);
  if (!is) {
    throw std::runtime_error("Could not open 'video.dat'.");
  }

  // Header
  uint32_t magic = read<uint32_t>(is);
  if (magic != 0xDEADBEEF) {
    throw std::runtime_error(
        "Did not see magic marker at beginning of file.");
  }

  uint32_t fileFormat = read<uint32_t>(is);
  uint32_t dpFormat = read<uint32_t>(is);

  if (fileFormat > kFileFormatVersion) {
    throw std::runtime_error("File format too new.");
  }

  if (fileFormat < kMinSupportedFileFormat) {
    throw std::runtime_error("File format too old.");
  }

  int numFrames = read<int>(is);

  // Read frames.
  metaFrames_.resize(numFrames);
  for (int frame = 0; frame < metaFrames_.size(); ++frame) {
    float pts = read<float>(is);
    metaFrames_[frame] = std::make_unique<MetaFrame>(pts);
    if (frame > 0) {
      metaFrames_[frame - 1]->duration_ = pts - metaFrames_[frame - 1]->pts_;
    }
  }

  // Read color streams.
  int numColorStreams = read<int>(is);
  colorStreams_.resize(numColorStreams);
  for (int csi = 0; csi < colorStreams_.size(); ++csi) {
    // Using 'new' to access ColorStream's private constructor.
    colorStreams_[csi] = std::unique_ptr<ColorStream>(new ColorStream(*this));
    ColorStream& cs = *colorStreams_[csi];

    cs.name_ = readstr(is);
    cs.setDir(readstr(is));
    cs.extension_ = readstr(is);
    if (fileFormat >= 7) {
      read(is, cs.type_);
    } else {
      if (cs.name_ == "dynamic_mask") {
        cs.type_ = CV_8UC1;
      } else {
        cs.type_ = CV_32FC3;
      }
    }

    cs.width_ = read<int>(is);
    cs.height_ = read<int>(is);

    // if (fileFormat >= 12) {
    //   bool hasGopTable = read<bool>(is);
    //   if (hasGopTable) {
    //     cs.gopTable_ = std::make_unique<GopTable>();
    //     cs.gopTable_->fread(is, fileFormat);
    //   }
    // }

    // Read color frames.
    cs.frames_.resize(numFrames);
    for (int frame = 0; frame < cs.frames_.size(); ++frame) {
      // Using 'new' to access ColorFrame's private constructor.
      cs.frames_[frame] =
          std::unique_ptr<ColorFrame>(new ColorFrame(cs, frame));
    }
  }

  // Read depth streams.
  int numDepthStreams = read<int>(is);
  depthStreams_.resize(numDepthStreams);
  for (int dsi = 0; dsi < depthStreams_.size(); ++dsi) {
    // Using 'new' to access DepthStream's private constructor.
    depthStreams_[dsi] = std::unique_ptr<DepthStream>(new DepthStream(*this));
    DepthStream& ds = *depthStreams_[dsi];

    ds.name_ = readstr(is);
    ds.setDir(readstr(is));

    if (fileFormat < 10) {
      // Backward-compatible reading code: implicit depth xform type.
      std::string str = readstr(is);
      ds.depthXformDesc_.parse(str);

      ds.spatialXformDesc_.type = XformType::Spatial;
      ds.spatialXformDesc_.spatialType = SpatialXformType::Identity;
    } else {
      ds.depthXformDesc_.fread(is);
      ds.spatialXformDesc_.fread(is);
    }

    ds.width_ = read<int>(is);
    ds.height_ = read<int>(is);

    // if (fileFormat >= 13) {
    //   bool hasGopTable = read<bool>(is);
    //   if (hasGopTable) {
    //     ds.gopTable_ = std::make_unique<GopTable>();
    //     ds.gopTable_->fread(is, fileFormat);
    //     ds.quantTable_ = std::make_unique<QuantTable>();
    //     ds.quantTable_->fread(is, fileFormat);
    //   }
    // }

    // Read depth frames.
    ds.frames_.resize(numFrames);
    for (int frame = 0; frame < ds.frames_.size(); ++frame) {
      // Using 'new' to access DepthFrame's private constructor.
      ds.frames_[frame] =
          std::unique_ptr<DepthFrame>(new DepthFrame(*this, ds, frame));
      DepthFrame& f = *ds.frames_[frame];
      f.intrinsics.fread(is, dpFormat);
      f.extrinsics.fread(is, dpFormat);

      if (fileFormat >= 11) {
        read(is, f.enabled);
      }

      if (fileFormat < 10) {
        // Backward-compatible reading code: implicit depth xform type.
        XformDescriptor desc;
        desc.type = XformType::Depth;
        std::string str = readstr(is);
        desc.parse(str);
        f.depthXform_ = createDepthXform(desc);
        is.read(
          reinterpret_cast<char*>(f.depthXform_->data()),
          sizeof(double) * f.depthXform_->numParams());

        f.spatialXform_ = createSpatialXform(ds.spatialXformDesc_);
      } else {
        DepthXform* depthPtr =
            static_cast<DepthXform*>(readXform(is).release());
        f.depthXform_ = std::unique_ptr<DepthXform>(depthPtr);

        SpatialXform* spatialPtr =
            static_cast<SpatialXform*>(readXform(is).release());
        f.spatialXform_ = std::unique_ptr<SpatialXform>(spatialPtr);
      }
      if (f.depthXform_->desc() != ds.depthXformDesc_) {
        throw std::runtime_error("Inconsistent depth transform.");
      }
    }
  }

  read(is, duration_);
  read(is, width_);
  read(is, height_);
  read(is, aspect_);
  read(is, invAspect_);

  metaFrames_.back()->duration_ = duration_ - metaFrames_.back()->pts_;

  magic = read<uint32_t>(is);
  if (magic != 0xDEADBEEF) {
    throw std::runtime_error("Did not see magic marker at end of file.");
  }
}

void DepthVideo::save() {
  std::string fileName = path_ + "/video.dat";
  std::ofstream os(fileName, std::ios::binary);

  // Header
  write<uint32_t>(os, 0xDEADBEEF);
  write<uint32_t>(os, kFileFormatVersion);
  write<uint32_t>(os, DepthPhoto::kFileFormatVersion);

  // Write frames.
  write<int>(os, metaFrames_.size());
  for (int frame = 0; frame < metaFrames_.size(); ++frame) {
    const MetaFrame& f = *metaFrames_[frame];
    write<float>(os, f.pts_);
  }

  // Write color streams.
  write<int>(os, colorStreams_.size());
  for (const auto& cs : colorStreams_) {
    writestr(os, cs->name_);
    writestr(os, cs->dir_);
    writestr(os, cs->extension_);
    write(os, cs->type_);

    write<int>(os, cs->width_);
    write<int>(os, cs->height_);

    // if (cs->gopTable_) {
    //   write<bool>(os, true);
    //   cs->gopTable_->fwrite(os);
    // } else {
      write<bool>(os, false);
    // }

    // Write color frames.
    CHECK_EQ(cs->frames_.size(), metaFrames_.size());
    for (int fi = 0; fi < cs->frames_.size(); ++fi) {
      const ColorFrame& f = *cs->frames_[fi];
      CHECK_EQ(f.index_, fi);
    }
  }

  // Write depth streams.
  write<int>(os, depthStreams_.size());
  for (auto& ds : depthStreams_) {
    writestr(os, ds->name_);
    writestr(os, ds->dir_);
    ds->depthXformDesc().fwrite(os);
    ds->spatialXformDesc().fwrite(os);

    write<int>(os, ds->width_);
    write<int>(os, ds->height_);

    // if (ds->gopTable_) {
    //   write<bool>(os, true);
    //   ds->gopTable_->fwrite(os);
    //   ds->quantTable_->fwrite(os);
    // } else {
      write<bool>(os, false);
    // }

    // Write depth frames.
    CHECK_EQ(ds->frames_.size(), metaFrames_.size());
    for (int fi = 0; fi < ds->frames_.size(); ++fi) {
      const DepthFrame& f = *ds->frames_[fi];
      CHECK_EQ(f.index_, fi);

      f.intrinsics.fwrite(os);
      f.extrinsics.fwrite(os);
      write(os, f.enabled);
      writeXform(os, *f.depthXform_);
      writeXform(os, *f.spatialXform_);
    }
  }

  // Write remaining fields.
  write(os, duration_);
  write(os, width_);
  write(os, height_);
  write(os, aspect_);
  write(os, invAspect_);

  // Write a magic marker at the end of the file to detect errors when reading
  // it back it.
  write<uint32_t>(os, 0xDEADBEEF);
}

int DepthVideo::timeToFrame(const float time) const {
  if (metaFrames_.empty()) {
    throw std::runtime_error("Video has no frames.");
  }

  if (time < metaFrames_[0]->pts()) {
    throw std::runtime_error("Query time before first frame's time.");
  }

  if (time > duration_) {
    throw std::runtime_error("Query time after video duration.");
  }

  for (int i = 0; i < metaFrames_.size() - 1; ++i) {
    if (time >= metaFrames_[i]->pts() && time < metaFrames_[i + 1]->pts()) {
      return i;
    }
  }

  return metaFrames_.size() - 1;
}

bool DepthVideo::hasColorStream(const std::string& name) const {
  for (int i = 0; i < colorStreams_.size(); ++i) {
    if (colorStreams_[i]->name() == name) {
      return true;
    }
  }
  return false;
}

int DepthVideo::colorStreamIndex(const std::string& name) const {
  for (int i = 0; i < colorStreams_.size(); ++i) {
    if (colorStreams_[i]->name() == name) {
      return i;
    }
  }

  throw std::runtime_error("Could not find named color stream.");
}

const ColorStream& DepthVideo::colorStream(const int stream) const {
  return *colorStreams_[stream];
}

ColorStream& DepthVideo::colorStream(const int stream) {
  return *colorStreams_[stream];
}

ColorStream& DepthVideo::colorStream(const std::string& name) {
  return *colorStreams_[colorStreamIndex(name)];
}

const ColorStream& DepthVideo::colorStream(const std::string& name) const {
  return *colorStreams_[colorStreamIndex(name)];
}

void DepthVideo::createColorStream(
    const std::string& name,
    const std::string& dir,
    const std::string& extension,
    const int type,
    const std::pair<int, int>& size) {
  // Using 'new' to access ColorStream's private constructor.
  colorStreams_.push_back(std::unique_ptr<ColorStream>(new ColorStream(*this)));
  ColorStream& cs = *colorStreams_.back();
  cs.name_ = name;
  cs.setDir(dir);
  cs.extension_ = extension;
  cs.type_ = type;

  if (type != CV_8UC1 && type != CV_8UC3 &&
      type != CV_32FC1 && type != CV_32FC3) {
    throw std::runtime_error(
        "Color streams only support 1 or 3 channels and byte or float depth.");
  }

  cs.width_ = size.first;
  cs.height_ = size.second;

  cs.frames_.resize(metaFrames_.size());
  for (int frame = 0; frame < metaFrames_.size(); ++frame) {
    // Using 'new' to access ColorFrame's private constructor.
    cs.frames_[frame] = std::unique_ptr<ColorFrame>(new ColorFrame(cs, frame));
  }
}

// void DepthVideo::createColorStream(
//     const std::string& name,
//     const std::string& dir,
//     const GopTable& gopTable,
//     const std::pair<int, int>& size) {
//   // Using 'new' to access ColorStream's private constructor.
//   colorStreams_.push_back(std::unique_ptr<ColorStream>(new ColorStream(*this)));
//   ColorStream& cs = *colorStreams_.back();
//   cs.name_ = name;
//   cs.setDir(dir);
//   cs.type_ = CV_8UC3;
//   cs.gopTable_ = std::make_unique<GopTable>(gopTable);

//   cs.width_ = size.first;
//   cs.height_ = size.second;

//   cs.frames_.resize(metaFrames_.size());
//   for (int frame = 0; frame < metaFrames_.size(); ++frame) {
//     // Using 'new' to access ColorFrame's private constructor.
//     cs.frames_[frame] = std::unique_ptr<ColorFrame>(new ColorFrame(cs, frame));
//   }
// }

ColorFrame& DepthVideo::colorFrame(const int stream, const int frame) {
  return *colorStreams_.at(stream)->frames_.at(frame);
}

bool DepthVideo::hasDepthStream(const std::string& name) const {
  for (int i = 0; i < depthStreams_.size(); ++i) {
    if (depthStreams_[i]->name() == name) {
      return true;
    }
  }
  return false;
}

int DepthVideo::depthStreamIndex(const std::string& name) const {
  for (int i = 0; i < depthStreams_.size(); ++i) {
    if (depthStreams_[i]->name() == name) {
      return i;
    }
  }

  throw std::runtime_error("Could not find named depth stream.");
}

DepthStream& DepthVideo::depthStream(const int stream) {
  return *depthStreams_[stream];
}

const DepthStream& DepthVideo::depthStream(const int stream) const {
  return *depthStreams_[stream];
}

DepthStream& DepthVideo::depthStream(const std::string& name) {
  return *depthStreams_[depthStreamIndex(name)];
}

const DepthStream& DepthVideo::depthStream(const std::string& name) const {
  return *depthStreams_[depthStreamIndex(name)];
}

void DepthVideo::createDepthStream(
    const std::string& name,
    const std::string& dir,
    const std::pair<int, int>& size) {
  // Using 'new' to access DepthStream's private constructor.
  depthStreams_.push_back(std::unique_ptr<DepthStream>(new DepthStream(*this)));
  DepthStream& ds = *depthStreams_.back();
  ds.name_ = name;
  ds.setDir(dir);
  ds.depthXformDesc_.reset();
  ds.spatialXformDesc_.reset(XformType::Spatial);
  ds.width_ = size.first;
  ds.height_ = size.second;

  ds.frames_.resize(metaFrames_.size());
  for (int frame = 0; frame < metaFrames_.size(); ++frame) {
    // Using 'new' to access DepthFrame's private constructor.
    ds.frames_[frame] =
        std::unique_ptr<DepthFrame>(new DepthFrame(*this, ds, frame));
    const bool silent = true;
    ds.frames_[frame]->intrinsics.resolveMissingFov(aspect_, silent);
  }
}

// void DepthVideo::createDepthStream(
//     const std::string& name,
//     const std::string& dir,
//     const GopTable& gopTable,
//     const QuantTable& quantTable,
//     const std::pair<int, int>& size) {
//   createDepthStream(name, dir, size);

//   DepthStream& ds = *depthStreams_.back();
//   ds.gopTable_ = std::make_unique<GopTable>(gopTable);
//   ds.quantTable_ = std::make_unique<QuantTable>(quantTable);
// }

DepthFrame& DepthVideo::depthFrame(const int stream, const int frame) {
  return *depthStreams_.at(stream)->frames_.at(frame);
}

const DepthFrame& DepthVideo::depthFrame(
    const int stream, const int frame) const {
  return *depthStreams_.at(stream)->frames_.at(frame);
}

void DepthVideo::clearDepthCaches() {
  for (auto& ds : depthStreams_) {
    ds->clearCache();
  }
}

void DepthVideo::saveDepth(const int stream) {
  const DepthStream& ds = depthStream(stream);

  for (int frame = 0; frame < metaFrames_.size(); ++frame) {
    const fs::path fileName =
        fmt::format("{:s}/depth/frame_{:06d}.raw", ds.path(), frame);
    const Mat1f* depthImg = ds.frame(frame).depth();

    if (depthImg) {
      const int w = depthImg->cols;
      const int h = depthImg->rows;

      // Convert to disparity for saving.
      Mat1f disparity(h, w);
      for (int y = 0; y < h; ++y) {
        const float* srcPtr = depthImg->ptr<const float>(y);
        float* dstPtr = disparity.ptr<float>(y);
        for (int  x = 0; x < w; ++x) {
          const float& sd = srcPtr[x];
          float& dd = dstPtr[x];

          // Set invalid depth (NaN, infinite, negative, etc.) to zero.
          if (std::isfinite(sd) && sd > 0.f) {
            dd = 1.f / sd;
          } else {
            dd = 0.f;
          }
        }
      }

      const fs::path path = fileName.parent_path();
      if (!fs::exists(path)) {
        const fs::path parentPath = path.parent_path();
        if (!fs::exists(parentPath)) {
          fs::create_directory(parentPath);
        }
        fs::create_directory(path);
      }

      fwriteim(fileName.string(), disparity);
    } else {
      if (fs::exists(fileName)) {
        fs::remove(fileName);
      }
      CHECK(!fs::exists(fileName));
    }
  }
}

Vector3f DepthVideo::project(
    const DepthPhoto::Extrinsics& extr, const DepthPhoto::Intrinsics& intr,
    const Vector2f& loc, const float depth) const {
  const float tanHFov2 = tan(intr.hFov / 2.f);
  const float tanVFov2 = tan(intr.vFov / 2.f);

  const Vector3f right = extr.orientation * Vector3f::UnitX();
  const Vector3f up = extr.orientation * Vector3f::UnitY();
  const Vector3f front = extr.orientation * -Vector3f::UnitZ();

  const float rx = -1.f + 2.f * loc.x();
  const float ry = 1.f - 2.f * loc.y() / invAspect_;
  const Vector3f ray =
      front + right * (rx * tanHFov2) + up * (ry * tanVFov2);
  const Vector3f position = extr.position + ray * depth;

  return position;
}

Vector3f DepthVideo::project(
    const int stream, const int frame,
    const Vector2f& loc, const bool useWarp) const {
  const DepthStream& ds = depthStream(stream);
  const DepthFrame& df = ds.frame(frame);
  const Mat1f& depthImg = *df.depth();

  const int w = depthImg.cols;
  const int h = depthImg.rows;
  const int x = std::min(w - 1, int(loc.x() * w + 0.5f));
  const int y = std::min(h - 1, int(loc.y() / invAspect_ * h + 0.5f));
  const float depth = depthImg(y, x);

  if (useWarp) {
    const Mat2f warpImg = *df.warp();
    const Vec2f warp = warpImg(y, x);

    Vector2f warpedLoc(
        loc.x() + warp(0) / 2.f,
        loc.y() - warp(1) / 2.f * invAspect_);

    return project(df.extrinsics, df.intrinsics, warpedLoc, depth);
  } else {
    return project(df.extrinsics, df.intrinsics, loc, depth);
  }
}

}} // namespace facebook::cp
