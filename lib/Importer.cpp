// Copyright 2004-present Facebook. All Rights Reserved.

#include "Importer.h"

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <opencv2/imgproc.hpp>

#include <fmt/format.h>

#include "core/CvUtil.h"
#include "core/Misc.h"
#include "core/cnpy.h"
#include "ColorStream.h"
#include "DepthStream.h"
#include "Processor.h"

using namespace cv;
using namespace Eigen;
namespace fs = boost::filesystem;

namespace facebook {
namespace cp {

void DepthVideoImporter::importVideo(
    DepthVideo& video, const std::string& path, const bool discoverStreams) {
  LOG(INFO) << "Importing 3D video '" << path << "'...";

  // Import frame metadata
  std::vector<std::unique_ptr<MetaFrame>> frames;
  int width = -1;
  int height = -1;
  importMetaFrames(path, width, height, frames);
  video.init(path, width, height, std::move(frames));

  if (!discoverStreams) {
    return;
  }

  LOG(INFO) << "  Discovering color streams...";
  using ColorStreamDesc =
      std::tuple<std::string, std::string, std::string, int>;
  std::vector<ColorStreamDesc> colorStreams {
      {"color_full", "full", ".png", CV_32FC3},
      {"color_down", "down", ".raw", CV_32FC3},
      {"color_down_png", "down_png", ".png", CV_32FC3},
      {"dynamic_mask", "dynamic_mask", ".png", CV_8UC1},
  };

  for (const ColorStreamDesc& colorStreamDesc : colorStreams) {
    const std::string& dir = std::get<0>(colorStreamDesc);
    const std::string& name = std::get<1>(colorStreamDesc);
    const std::string& extension = std::get<2>(colorStreamDesc);
    const int type = std::get<3>(colorStreamDesc);

    const std::string path = video.path() + "/" + dir;
    if (fs::is_directory(path)) {
      LOG(INFO) << "    Found color stream '" << dir << "'.";
      video.createColorStream(name, dir, extension, type);
    }
  }

  std::vector<std::string> colorStreamCheckPaths;
  for (auto& entry :
      boost::make_iterator_range(fs::directory_iterator(video.path()), {})) {
    if (fs::is_directory(entry)) {
      colorStreamCheckPaths.push_back(entry.path().string());
    }
  }

  std::sort(colorStreamCheckPaths.begin(), colorStreamCheckPaths.end());

  for (const std::string& colorPath : colorStreamCheckPaths) {
    LOG(INFO) << "    Checking '" << colorPath << "'.";

    std::string relativePath = fs::relative(colorPath, video.path()).string();

    // Skip hardcoded streams.
    bool skip = false;
    for (const ColorStreamDesc& colorStreamDesc : colorStreams) {
      if (std::get<1>(colorStreamDesc) == relativePath) {
        skip = true;
        break;
      }
    }

    if (skip) {
      continue;
    }

    std::string infoFile = colorPath + "/stream_info.txt";
    if (!fs::exists(infoFile)) {
      continue;
    }

    std::ifstream is(infoFile, std::ios::binary);
    if (is.fail()) {
      throw std::runtime_error("Could not open frame file.");
    }

    std::string type;
    is >> type;
    if (type != "color") {
      continue;
    }

    std::string extension;
    is >> extension;

    std::string formatStr;
    is >> formatStr;

    int format = 0;
    if (formatStr == "32FC3") {
      format = CV_32FC3;
    } else if (formatStr == "8UC1") {
      format = CV_8UC1;
    } else {
      throw std::runtime_error("Invalid format string.");
    }

    LOG(INFO) << "    Found color stream '" << relativePath <<
        "' (" << extension << ", " << formatStr << ").";
    video.createColorStream(relativePath, relativePath, extension, format);
  }

  LOG(INFO) << "  Discovering depth streams...";

  std::vector<std::string> depthStreams;

  std::function<void (const std::string&)> discoverDepthStreams;

  discoverDepthStreams = [&](const std::string& path) {
    for (auto& entry :
        boost::make_iterator_range(fs::directory_iterator(path), {})) {
      if (fs::is_directory(entry)) {
        LOG(INFO) << "    Checking '" << entry.path().string() << "'.";
        if (fs::is_directory(entry.path().string() + "/depth")) {
          std::string relativePath =
              fs::relative(entry.path(), video.path()).string();
          depthStreams.push_back(relativePath);
        } else {
          discoverDepthStreams(entry.path().string());
        }
      }
    }
  };

  discoverDepthStreams(video.path());

  std::sort(depthStreams.begin(), depthStreams.end());

  for (int i = 0; i < depthStreams.size(); ++i) {
    LOG(INFO) << "    Found depth stream '" << depthStreams[i] << "'.";
    if (depthStreams[i] == "depth_colmap_dense") {
      importColmapDepth(video);
      video.createDepthStream(
          depthStreams[i], "depth_colmap_dense_imported");
    } else if (depthStreams[i] == "depth_colmap_dense_imported") {
      // Skip.
    } else {
      video.createDepthStream(
          depthStreams[i], depthStreams[i]);

      const std::string posesFile =
          video.path() + "/" + depthStreams[i] + "/poses.txt";
      if (fs::exists(posesFile)) {
        importPoses(video, posesFile, video.numDepthStreams() - 1);
      }
    }
  }

  std::vector<std::string> colmapFiles {
      video.path() + "/metadata.npz",
      video.path() + "/colmap_dense/metadata.npz",
  };

  for (const std::string& colmapFile : colmapFiles) {
    if (fs::is_regular_file(colmapFile)) {
      LOG(INFO) <<
          "  Importing COLMAP reconstruction from '" << colmapFile << "'...";
      for (int stream = 0; stream < video.numDepthStreams(); ++stream) {
        importColmapRecon(video, colmapFile, stream, false);
      }
      break;
    }
  }

  const std::string trackFile = video.path() + "/track2d.csv";
  if (fs::is_regular_file(trackFile)) {
    LOG(INFO) <<
        "  Importing tracks from '" << trackFile << "'...";
    importTracks(video, trackFile);
  }
}

void DepthVideoImporter::importMetaFrames(
    const std::string& path, int& width, int& height,
    std::vector<std::unique_ptr<MetaFrame>>& frames) {
  frames.clear();

  std::string frameFile = path + "/frames.txt";
  std::ifstream is(frameFile, std::ios::binary);
  if (is.fail()) {
    throw std::runtime_error("Could not open frame file.");
  }

  int numFrames = -1;
  is >> numFrames;
  is >> width;
  is >> height;
  LOG(INFO) << "  " << numFrames << " frames @ " << width << " x " << height;

  frames.resize(numFrames);

  float minPts = 0.f;

  for (int i = 0; i < numFrames; ++i) {
    float pts;
    is >> pts;

    // Remap pts so they start at 0.f.
    if (i == 0) {
      minPts = pts;
    }
    pts -= minPts;

    if (i > 0) {
      if (pts <= frames[i - 1]->pts()) {
        LOG(INFO) << "Frame " << (i - 1) << ", PTS = " << frames[i -1]->pts();
        LOG(INFO) << "Frame " << i << ", PTS = " << pts;
        throw std::runtime_error("Non-monotonic PTS detected.");
      }
    }

    frames[i] = std::make_unique<MetaFrame>(pts);
  }
}

float DepthVideoImporter::loadScale(const std::string& path) {
  LOG(INFO) << "Searching 'scales.csv'...";
  std::string scalesCsvPath;
  std::function<void (const std::string&)> discoverScalesCsv;
  discoverScalesCsv = [&](const std::string& path) {
    for (auto& entry :
        boost::make_iterator_range(fs::directory_iterator(path), {})) {
      if (entry.path().filename().string() == "scales.csv") {
        scalesCsvPath = entry.path().string();
      } else if (fs::is_directory(entry)) {
        LOG(INFO) << "    Checking '" << entry.path().string() << "'.";
        discoverScalesCsv(entry.path().string());
      }
    }
  };

  discoverScalesCsv(path);

  float scale = 1.f;

  if (scalesCsvPath.empty()) {
    LOG(INFO) << "Could not find 'scales.csv'. Using default scale.";
  } else {
    std::ifstream f(scalesCsvPath);
    if (f.fail()) {
      throw std::runtime_error("Could not open 'scales.csv'.");
    }

    int count = 0;
    for (std::string line; std::getline(f, line); ) {
      std::vector<std::string> parts = explode(line, ',');
      if (parts.size() != 2) {
        LOG(INFO) << "ERROR: invalid line '" << line << "'.";
        continue;
      }
      float s = atof(parts[1].c_str());
      scale += s;
      ++count;
    }

    if (count > 0) {
      scale /= count;
    }
  }

  LOG(INFO) << "  Scale = " << scale << ".";

  return scale;
}

void DepthVideoImporter::importColmapRecon(
    DepthVideo& video, const std::string &colmapFile,
    const int stream, const bool silent) {
  DepthStream& ds = video.depthStream(stream);

  float scale = loadScale(video.path());

  // Load indices of frames that have been reconstructed.
  std::vector<int> frameIndices;
  const std::string depthPath = ds.path() + "/depth";
  for (auto& entry :
      boost::make_iterator_range(fs::directory_iterator(depthPath), {})) {
    std::string name = entry.path().stem().string();
    if (name.length() != 12 || name.substr(0, 6) != "frame_") {
      throw std::runtime_error(
          "Depth file name does not have expected format.");
    }

    const int index = std::stoi(name.substr(6));
    frameIndices.push_back(index);
  }
  std::sort(frameIndices.begin(), frameIndices.end());

  // Initially disable all frames, later enable the ones that are reconstructed.
  for (int i = 0; i < video.numFrames(); ++i) {
    ds.frame(i).enabled = false;
  }

  // Load metadata.
  cnpy::npz_t meta = cnpy::npz_load(colmapFile);
  cnpy::NpyArray extrArr = meta["extrinsics"];
  cnpy::NpyArray intrArr = meta["intrinsics"];
  const std::vector<size_t>& extrShape = extrArr.shape;
  const std::vector<size_t>& intrShape = intrArr.shape;
  constexpr int DIM = 3;
  CHECK_EQ(extrShape[0], intrShape[0]);
  CHECK_EQ(extrShape[0], frameIndices.size());
  CHECK_EQ(extrShape.size(), 3);
  CHECK_EQ(extrShape[1], DIM);
  CHECK_EQ(extrShape[2], DIM + 1);
  CHECK_EQ(intrShape.size(), 2);
  CHECK_EQ(intrShape[1], 4);

  // We only support row-major storage and double values for now.
  constexpr int DOUBLE_SIZE = 8;
  CHECK(!extrArr.fortran_order);
  CHECK(!intrArr.fortran_order);
  CHECK_EQ(extrArr.word_size, DOUBLE_SIZE);
  CHECK_EQ(intrArr.word_size, DOUBLE_SIZE);

  using dtype_t = double;
  using dst_dtype_t = float;

  // Load extrinsics. The coordinate system for both the bundles here and the
  // numpy metadata file is +x pointing to the right, +y pointing up and camera
  // facing at -z direction.
  if (!silent) {
    LOG(INFO) << "Loading Extrinsics...";
  }
  using Pose = Matrix<dtype_t, DIM, DIM + 1, RowMajor>;
  const dtype_t* extrPtr = extrArr.data<dtype_t>();
  int extrElementSize = extrShape[1] * extrShape[2];
  for (int i = 0; i < extrShape[0]; ++i, extrPtr += extrElementSize) {
    const Pose pose = Map<const Pose>(extrPtr);
    DepthPhoto::Extrinsics extr;
    extr.position = pose.col(DIM).cast<dst_dtype_t>() / scale;
    extr.orientation = Quaternionf(
      pose.block<DIM, DIM>(0, 0).cast<dst_dtype_t>());
    DepthFrame& df = video.depthFrame(stream, frameIndices[i]);
    df.enabled = true;
    df.extrinsics = extr;
    if (!silent) {
      LOG(INFO) << "  Frame" << i;
      LOG(INFO) << "    " << extr.position;
      LOG(INFO) << "    " << pose.block<DIM, DIM>(0, 0).cast<dst_dtype_t>();
      LOG(INFO) << "    " << extr.orientation.coeffs();
    }
  }

  // set intrinsics
  if (!silent) {
    LOG(INFO) << "Loading Intrinsics...";
  }
  const dtype_t* intrPtr = intrArr.data<dtype_t>();
  cv::Size imSize(ds.width(), ds.height());
  for (int i = 0; i < intrShape[0]; ++i, intrPtr += intrShape[1]) {
    const Vector2d fxy = Map<const Vector2d>(intrPtr);
    DepthPhoto::Intrinsics intr;
    intr.hFov = 2 * atan2(imSize.width / 2.0, fxy.x());
    intr.vFov = 2 * atan2(imSize.height / 2.0, fxy.y());
    video.depthFrame(stream, frameIndices[i]).intrinsics = intr;

    if (!silent) {
      LOG(INFO) << "  Frame " << i;
      LOG(INFO) << "    Fxy " << fxy;
      intr.printParams();
    }
  }
}

void DepthVideoImporter::importColmapDepth(DepthVideo& video) {
  LOG(INFO) << "Importing COLMAP depth maps...";

  const std::string srcPath =
      video.path() + "/depth_colmap_dense/depth";
  const std::string dstPath =
      video.path() + "/depth_colmap_dense_imported/depth";

  if (fs::exists(dstPath)) {
    LOG(INFO) << "  Destination directory already exists, skipping.";
    return;
  }

  fs::create_directories(dstPath);

  float scale = loadScale(video.path());

  // Resize and adjust scale...
  for(auto& entry :
      boost::make_iterator_range(fs::directory_iterator(srcPath), {})) {
    const std::string name = entry.path().stem().string();
    const std::string srcFile = srcPath + "/" + name + ".raw";
    const std::string dstFile = dstPath + "/" + name + ".raw";

    Mat1f depth;
    freadim(srcFile, depth);

    const int w = depth.cols;
    const int h = depth.rows;

    for (int y = 0; y < h; ++y) {
      float* depthPtr = depth.ptr<float>(y);
      for (int x = 0; x < w; ++x) {
        float& d = depthPtr[x];
        if (!std::isfinite(d) || d < 0.f) {
          d = 0.f;
        } else {
          d *= scale;
        }
      }
    }

    fwriteim(dstFile, depth);
  }

  LOG(INFO) << "  Done.";
}

void DepthVideoImporter::importPoses(
    DepthVideo& video, const std::string& posesFile, const int stream) {
  LOG(INFO) << "Importing poses from '" << posesFile <<
      "' to stream " << stream << ".";

  DepthStream& ds = video.depthStream(stream);

  std::ifstream is(posesFile, std::ios::binary);
  if (is.fail()) {
    throw std::runtime_error("Could not open poses file.");
  }

  int numFrames = -1;
  is >> numFrames;

  if (numFrames > video.numFrames()) {
    throw std::runtime_error("Poses file has more frames than the video.");
  }

  for (int i = 0; i < numFrames; ++i) {
    Vector3f position;
    Quaternionf orientation;
    Vector2f fov;

    is >> position.x() >> position.y() >> position.z() >>
        orientation.x() >> orientation.y() >>
        orientation.z() >> orientation.w() >>
        fov.x() >> fov.y();

    DepthFrame& f = ds.frame(i);
    f.enabled = true;
    f.extrinsics.position = position;
    f.extrinsics.orientation = orientation;
    f.intrinsics.hFov = fov.x();
    f.intrinsics.vFov = fov.y();
  }

  for (int i = numFrames; i < video.numFrames(); ++i) {
    DepthFrame& f = ds.frame(i);
    f.enabled = false;
  }
}

void DepthVideoImporter::importTracks(
    DepthVideo& video, const std::string& trackFile) {
  std::ifstream f(trackFile);
  if (f.fail()) {
    throw std::runtime_error("Cannot open track file.");
  }

  const ColorStream& cs = video.colorStream("full");
  const Mat* colorImg = cs.frame(0).image();
  const float w = colorImg->cols;

  DepthVideoTrackTable tt;

  int lastFrame = -1;
  std::map<int, int> trackIdMap;

  for (std::string line; std::getline(f, line); ) {
    std::vector<std::string> parts = explode(line, ',');
    if (parts.size() != 4) {
      LOG(INFO) << "ERROR: invalid line '" << line << "'.";
      continue;
    }

    for (int i = 0; i < parts.size(); ++i) {
      trim(parts[i]);
    }

    int frame = atoi(parts[0].c_str());
    int trackId = atoi(parts[1].c_str());
    float x = atof(parts[2].c_str());
    float y = atof(parts[3].c_str());

    if (frame < lastFrame) {
      throw std::runtime_error("ERROR: Frames not in consecutive order.");
    }

    while (frame > lastFrame) {
      tt.addFrame();
      lastFrame++;
    }

    DepthVideoObs obs(x / w, y / w);

    if (contains(trackIdMap, trackId)) {
      tt.addObs(trackIdMap[trackId], frame, obs);
    } else {
      trackIdMap[trackId] = tt.createTrack(frame, obs);
    }
  }

  std::string outFile = video.path() + "/long_tracks.tracktable";
  std::ofstream os(outFile, std::ios::binary);
  tt.serialize(os);
}

}} // namespace facebook::cp
