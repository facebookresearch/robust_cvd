// Copyright 2004-present Facebook. All Rights Reserved.

#include "FlowConstraints.h"

#include <boost/filesystem.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/CvUtil.h"
#include "core/FileIo.h"
#include "core/MathUtil.h"

#include "ColorStream.h"
#include "DepthVideo.h"

using namespace Eigen;
using namespace cv;

namespace fs = boost::filesystem;

namespace facebook {
namespace cp {

void FlowConstraintsParams::save(std::ostream& os) const {
  write(os, matchSeparation);
}

void FlowConstraintsParams::load(std::istream& is) {
  read(is, matchSeparation);
}

bool FlowConstraintsParams::operator==(
    const FlowConstraintsParams& other) const {
  return matchSeparation == other.matchSeparation;
}

bool FlowConstraintsParams::operator!=(
    const FlowConstraintsParams& other) const {
  return !(*this == other);
}

FlowConstraintsCollection::FlowConstraintsCollection(
    const DepthVideo& video, const FlowConstraintsParams& params)
    : video_(&video), path_(video.path()), params_(params) {
  LOG(INFO) << "Setting up flow constraints...";

  std::string listFile = path_ + "/" + "flow_list.json";
  LOG(INFO) << "Loading flow list '" << listFile << "'...";
  if (!fs::exists(listFile)) {
    throw std::runtime_error("Flow list file does not exist.");
  }

  std::ifstream i(listFile);
  nlohmann::json list;
  i >> list;

  for (int i = 1; i < list.size(); ++i) {

  PairKey pair = std::make_pair<int, int>(
      static_cast<int>(list[i][0]), static_cast<int>(list[i][1]));

  if (!params.frameRange.inRange(pair.first) ||
      !params.frameRange.inRange(pair.second)) {
    continue;
    }

    // Using 'new' to access FlowConstraints's private constructor.
    pairs_.emplace(pair, std::unique_ptr<PairFlowConstraints>(
        new PairFlowConstraints(pair)));
  }

  for (int triplet = params.frameRange.firstFrame() + 1;
      triplet <= params.frameRange.lastFrame() - 1; ++triplet) {
    if (!params.frameRange.inRange(triplet - 1) ||
        !params.frameRange.inRange(triplet) ||
        !params.frameRange.inRange(triplet + 1)) {
      continue;
    }

    triplets_.emplace(triplet, std::unique_ptr<TripletFlowConstraints>(
        new TripletFlowConstraints(triplet)));
  }

  if (params.doNotUseCache) {
    compute();
  } else {
    if (!load()) {
      compute();
      save();
    }
  }
}

FlowConstraintsCollection::PairIterator
    FlowConstraintsCollection::pairBegin() const {
  return PairIterator(pairs_.begin());
}

FlowConstraintsCollection::PairIterator
    FlowConstraintsCollection::pairEnd() const {
  return PairIterator(pairs_.end());
}

FlowConstraintsCollection::TripletIterator
    FlowConstraintsCollection::tripletBegin() const {
  return TripletIterator(triplets_.begin());
}

FlowConstraintsCollection::TripletIterator
    FlowConstraintsCollection::tripletEnd() const {
  return TripletIterator(triplets_.end());
}

bool FlowConstraintsCollection::load() {
  std::string fileName = path_ + "/" + "flow_constraints.dat";
  if (!fs::exists(fileName)) {
    LOG(INFO) << "Constraints cache file '" << fileName << "' does not exist.";
    return false;
  }

  LOG(INFO) << "Loading cached constraints from '" << fileName << "'...";

  std::ifstream is(fileName, std::ios::binary);

  // Header
  uint32_t magic = read<uint32_t>(is);
  if (magic != 0xDEADBEEF) {
    throw std::runtime_error(
        "Did not see magic marker at beginning of file.");
  }

  uint32_t fileFormat = read<uint32_t>(is);

  if (fileFormat > kFileFormatVersion) {
    throw std::runtime_error("File format too new.");
  }

  if (fileFormat < kMinSupportedFileFormat) {
    throw std::runtime_error("File format too old.");
  }

  FlowConstraintsParams checkParams;
  checkParams.load(is);
  if (checkParams != params_) {
    LOG(INFO) << "Cache file has the wrong parameters... Not loading.";
    return false;
  }

  auto readConstraintsContainer = [&](std::ifstream& is, auto& container) {
    size_t len = read<size_t>(is);
    container.resize(len);
    for (int i = 0; i < len; ++i) {
      container[i].read(is);
    }
  };

  for (auto it = this->pairBegin(); it != this->pairEnd(); it++) {
    const PairKey& key = *it;
    PairKey checkKey;
    read(is, checkKey);
    if (checkKey != key) {
      throw std::runtime_error("Read incorrect pair from file.");
    }

    PairFlowConstraints& c = *pairs_.at(key);
    readConstraintsContainer(is, c.constraints_);
  }

  for (auto it = this->tripletBegin(); it != this->tripletEnd(); it++) {
    const TripletKey& key = *it;
    TripletKey checkKey;
    read(is, checkKey);
    if (checkKey != key) {
      throw std::runtime_error("Read incorrect triplet from file.");
    }

    TripletFlowConstraints& c = *triplets_.at(key);
    readConstraintsContainer(is, c.constraints_);
  }

  magic = read<uint32_t>(is);
  if (magic != 0xDEADBEEF) {
    throw std::runtime_error("Did not see magic marker at end of file.");
  }

  return true;
}

void FlowConstraintsCollection::save() {
  std::string fileName = path_ + "/" + "flow_constraints.dat";
  LOG(INFO) << "Writing constraints to '" << fileName << "'...";
  std::ofstream os(fileName, std::ios::binary);

  // Header
  write<uint32_t>(os, 0xDEADBEEF);
  write<uint32_t>(os, kFileFormatVersion);

  params_.save(os);

  auto writeConstraintsContainer = [&](std::ofstream& is, auto& container) {
    write<size_t>(os, container.size());
    for (int i = 0; i < container.size(); ++i) {
      container[i].write(os);
    }
  };

  for (auto it = this->pairBegin(); it != this->pairEnd(); it++) {
    const PairKey& key = *it;
    write(os, key);
    const PairFlowConstraints& c = at(key);
    writeConstraintsContainer(os, c.constraints_);
  }

  for (auto it = this->tripletBegin(); it != this->tripletEnd(); it++) {
    const TripletKey& key = *it;
    write(os, key);
    const TripletFlowConstraints& c = at(key);
    writeConstraintsContainer(os, c.constraints_);
  }

  write<uint32_t>(os, 0xDEADBEEF);
}

std::pair<cv::Mat2f, cv::Mat1b> FlowConstraintsCollection::loadFlowAndMask(
    const int srcFrame, const int dstFrame) const {
  const ColorStream& cs = video_->colorStream("down");
  const int w = cs.width();
  const int h = cs.height();

  std::string flowFile = fmt::format(
      "{:s}/flow/flow_{:06d}_{:06d}.raw", path_, srcFrame, dstFrame);
  if (!fs::exists(flowFile)) {
    throw std::runtime_error("Flow file does not exist.");
  }
  Mat2f flow;
  freadim(flowFile, flow);
  if (flow.cols != w || flow.rows != h) {
    throw std::runtime_error("Flow has the wrong size.");
  }

  std::string maskFile = fmt::format(
       "{:s}/flow_mask/mask_{:06d}_{:06d}.png",
       path_, srcFrame, dstFrame);
  if (!fs::exists(maskFile)) {
    throw std::runtime_error("Mask file does not exist.");
  }
  Mat1b mask = imread(maskFile, IMREAD_GRAYSCALE);
  if (mask.cols != w || mask.rows != h) {
    throw std::runtime_error("Mask has the wrong size.");
  }

  return std::make_pair(flow, mask);
}

Mat1f FlowConstraintsCollection::dynamicDistance(
    const int frame) const {
  if (video_->hasColorStream("dynamic_mask")) {
    const ColorStream& dynamicMaskStream = video_->colorStream("dynamic_mask");
    const Mat1b* dynamicMask = dynamicMaskStream.frame(frame).image1b();
    if (!dynamicMask) {
      throw std::runtime_error("Dynamic mask stream is missing a frame.");
    }

    const int w = dynamicMask->cols;
    const int h = dynamicMask->rows;

    Mat1b binarized(h, w);
    for (int y = 0; y < h; ++y) {
      const uint8_t* srcPtr = dynamicMask->ptr<const uint8_t>(y);
      uint8_t* dstPtr = binarized.ptr<uint8_t>(y);
      for (int x = 0; x < w; ++x) {
        dstPtr[x] = (srcPtr[x] < 127 ? 0 : 255);
      }
    }

    Mat1f dist;
    distanceTransform(binarized, dist, DIST_L2, DIST_MASK_5);

    return dist;
  } else {
    const ColorStream& cs = video_->colorStream("down");
    return Mat1f(cs.height(), cs.width(), FLT_MAX);
  }
}

void FlowConstraintsCollection::compute() {
  LOG(INFO) << "Computing constraints...";

  for (auto it = this->pairBegin(); it != this->pairEnd(); it++) {
    const PairKey& key = *it;
    compute(key);
  }

  for (auto it = this->tripletBegin(); it != this->tripletEnd(); it++) {
    const TripletKey& key = *it;
    compute(key);
  }
}

namespace {
template <typename DataType>
struct Pixel {
  float cornerStrength;
  DataType data;
  Pixel(const float c, const DataType& data)
      : cornerStrength(c), data(data) {
  }
  bool operator<(const Pixel& other) const {
    return cornerStrength > other.cornerStrength;
  }
};

Mat1b buildDiskMask(const int radius) {
  const int size = 2 * radius + 1;

  Mat1b mask(size, size);
  for (int y = 0; y < size; ++y) {
    uint8_t* ptr = mask.ptr<uint8_t>(y);
    for (int x = 0; x < size; ++x) {
      const int rx = x - radius;
      const int ry = y - radius;
      ptr[x] = (sqr(rx) + sqr(ry) <= sqr(radius) ? 255 : 0);
    }
  }

  return mask;
}

PairConstraint scaleConstraint(const PairConstraint& c, const Vector2f& scale) {
  return PairConstraint(
      {c[0].array() * scale.array(), c[1].array() * scale.array()});
}

TripletConstraint scaleConstraint(
    const TripletConstraint& c, const Vector2f& scale) {
  return TripletConstraint(
      {c[0].array() * scale.array(),
       c[1].array() * scale.array(),
       c[2].array() * scale.array()});
}

Vector2i referencePixel(const PairConstraint& c) {
  return Vector2i(c[0].x(), c[0].y());
}

Vector2i referencePixel(const TripletConstraint& c) {
  return Vector2i(c[1].x(), c[1].y());
}

template <typename Constraint, typename Container>
void sampleConstraints(
    const DepthVideo& video,
    const int constraintSeparation,
    std::vector<Pixel<Constraint>>& pixels,
    Container& output) {
  const ColorStream& cs = video.colorStream("down");
  const int w = cs.width();
  const int h = cs.height();

  // Sort pixels so that the most corner-like ones come first.
  std::sort(pixels.begin(), pixels.end());

  // Marking pixels that are too close to previously selected pixels.
  Mat1b invalid(h, w, uint8_t(0));

  // Build a round mask, used to prevent placing features too close together.
  Mat1b diskMask = buildDiskMask(constraintSeparation);

  Vector2f scale(1.f / w, video.invAspect() / h);

  for (const auto& p : pixels) {
    const Vector2i r = referencePixel(p.data);
    if (invalid(r.y(), r.x())) {
      continue;
    }

    output.push_back(scaleConstraint(p.data, scale));

    const int mx0 = std::max(0, r.x() - constraintSeparation);
    const int mx1 = std::min(w - 1, r.x() + constraintSeparation);
    const int my0 = std::max(0, r.y() - constraintSeparation);
    const int my1 = std::min(h - 1, r.y() + constraintSeparation);
    for (int my = my0; my <= my1; ++my) {
      const int dy = my - (r.y() - constraintSeparation);
      const uint8_t* diskPtr = diskMask.ptr<const uint8_t>(dy);
      uint8_t* invalidPtr = invalid.ptr<uint8_t>(my);
      for (int mx = mx0; mx <= mx1; ++mx) {
        const int dx = mx - (r.x() - constraintSeparation);
        if (diskPtr[dx]) {
          invalidPtr[mx] = 255;
        }
      }
    }
  }
}

} // anonymous namespace

void FlowConstraintsCollection::compute(const PairKey& pair) {
  LOG(INFO) << "  Pair (" << pair.first << ", " << pair.second << ")...";

  const ColorStream& cs = video_->colorStream("down");
  const Mat* color = cs.frame(pair.first).image();
  const int w = color->cols;
  const int h = color->rows;

  auto [flow, mask] = loadFlowAndMask(pair.first, pair.second);

  Mat1f dynamicDistance0 = dynamicDistance(pair.first);
  Mat1f dynamicDistance1 = dynamicDistance(pair.second);

  Vector2f dynamicMaskScale = Vector2f(
        dynamicDistance0.cols / float(cs.width()),
        dynamicDistance0.rows / float(cs.height()));

  Mat1f gray;
  cvtColor(*color, gray, COLOR_BGR2GRAY);

  const int blockSize = 3;
  Mat1f cornerResponse;
  cornerMinEigenVal(gray, cornerResponse, blockSize);

  using Pixel = Pixel<PairConstraint>;
  std::vector<Pixel> pixels;
  pixels.reserve(w * h);

  for (int iy0 = 0; iy0 < h; ++iy0) {
    const float* const cornerPtr = cornerResponse.ptr<const float>(iy0);
    const Vec2f* const flowPtr = flow.ptr<const Vec2f>(iy0);
    const uint8_t* const maskPtr = mask.ptr<const uint8_t>(iy0);

    const int iy0s = iy0 * dynamicMaskScale.y() + 0.5f;
    const float* const dynamicDistance0Ptr =
        dynamicDistance0.ptr<const float>(iy0s);

    for (int ix0 = 0; ix0 < w; ++ix0) {
      const int ix0s = ix0 * dynamicMaskScale.x() + 0.5f;
      if (maskPtr[ix0] &&
          dynamicDistance0Ptr[ix0s] > params_.minDynamicDistance) {
        const Vec2f& ff = flowPtr[ix0];
        const float fx1 = ix0 + ff(0);
        const float fy1 = iy0 + ff(1);
        const int ix1 = fx1 + 0.5f;
        const int iy1 = fy1 + 0.5f;

        if (ix1 >= 0 && ix1 < w && iy1 >= 0 && iy1 < h) {
          const int ix1s = fx1 * dynamicMaskScale.x() + 0.5f;
          const int iy1s = fy1 * dynamicMaskScale.y() + 0.5f;

          if (dynamicDistance1(iy1s, ix1s) > params_.minDynamicDistance) {
            pixels.emplace_back(
                cornerPtr[ix0],
                PairConstraint({Vector2fna {ix0, iy0}, {fx1, fy1}}));
          }
        }
      }
    }
  }

  PairFlowConstraints::ConstraintContainer& constraints =
      pairs_.at(pair)->constraints_;
  sampleConstraints(*video_, params_.matchSeparation, pixels, constraints);
}

void FlowConstraintsCollection::compute(const TripletKey& triplet) {
LOG(INFO) << "  Triplet (" <<
    triplet - 1 << ", " << triplet << ", " << triplet + 1 << ")...";

const ColorStream& cs = video_->colorStream("down");
const Mat* color = cs.frame(triplet).image();
const int w = color->cols;
const int h = color->rows;

auto [flow10, mask10] = loadFlowAndMask(triplet, triplet - 1);
auto [flow12, mask12] = loadFlowAndMask(triplet, triplet + 1);

Mat1f dynamicDistance0 = dynamicDistance(triplet - 1);
Mat1f dynamicDistance1 = dynamicDistance(triplet);
Mat1f dynamicDistance2 = dynamicDistance(triplet + 1);

Vector2f dynamicMaskScale = Vector2f(
      dynamicDistance0.cols / float(cs.width()),
      dynamicDistance0.rows / float(cs.height()));

Mat1f gray;
cvtColor(*color, gray, COLOR_BGR2GRAY);

const int blockSize = 3;
Mat1f cornerResponse;
cornerMinEigenVal(gray, cornerResponse, blockSize);

using Pixel = Pixel<TripletConstraint>;
std::vector<Pixel> pixels;
pixels.reserve(w * h);

for (int iy1 = 0; iy1 < h; ++iy1) {
  const float* const cornerPtr = cornerResponse.ptr<const float>(iy1);

  const Vec2f* const flow10Ptr = flow10.ptr<const Vec2f>(iy1);
  const uint8_t* const mask10Ptr = mask10.ptr<const uint8_t>(iy1);

  const Vec2f* const flow12Ptr = flow12.ptr<const Vec2f>(iy1);
  const uint8_t* const mask12Ptr = mask12.ptr<const uint8_t>(iy1);

  const int iy1s = iy1 * dynamicMaskScale.y() + 0.5f;
  const float* const dynamicDistance1Ptr =
      dynamicDistance1.ptr<const float>(iy1s);

  for (int ix1 = 0; ix1 < w; ++ix1) {
    const int ix1s = ix1 * dynamicMaskScale.x() + 0.5f;
    if (mask10Ptr[ix1] && mask12Ptr[ix1] &&
        dynamicDistance1Ptr[ix1s] > params_.minDynamicDistance) {
      const Vec2f& ff10 = flow10Ptr[ix1];
      const float fx0 = ix1 + ff10(0);
      const float fy0 = iy1 + ff10(1);
      const int ix0 = fx0 + 0.5f;
      const int iy0 = fy0 + 0.5f;

      const Vec2f& ff12 = flow12Ptr[ix1];
      const float fx2 = ix1 + ff12(0);
      const float fy2 = iy1 + ff12(1);
      const int ix2 = fx2 + 0.5f;
      const int iy2 = fy2 + 0.5f;

      if (ix0 >= 0 && ix0 < w && iy0 >= 0 && iy0 < h &&
          ix2 >= 0 && ix2 < w && iy2 >= 0 && iy2 < h) {
        const int ix0s = fx0 * dynamicMaskScale.x() + 0.5f;
        const int iy0s = fy0 * dynamicMaskScale.y() + 0.5f;
        const int ix2s = fx2 * dynamicMaskScale.x() + 0.5f;
        const int iy2s = fy2 * dynamicMaskScale.y() + 0.5f;

        if (dynamicDistance0(iy0s, ix0s) > params_.minDynamicDistance &&
            dynamicDistance1(iy2s, ix2s) > params_.minDynamicDistance) {

          TripletConstraint constraint(
              {Vector2f {fx0, fy0}, {ix1, iy1}, {fx2, fy2}});

          pixels.emplace_back(cornerPtr[ix0], constraint);
        }
      }
    }
  }
}

TripletFlowConstraints::ConstraintContainer& constraints =
    triplets_.at(triplet)->constraints_;
sampleConstraints(*video_, params_.matchSeparation, pixels, constraints);
}

void FlowConstraintsCollection::resetStaticFlag() {
  LOG(INFO) << "Resetting static flag...";

  for (auto it = pairBegin(); it != pairEnd(); it++) {
    const PairKey& key = *it;
    std::unique_ptr<PairFlowConstraints>& pairConstraints = pairs_.at(key);
    for (PairConstraint& c : pairConstraints->constraints_) {
      c.isStatic = true;
    }
  }

  for (auto it = tripletBegin(); it != tripletEnd(); it++) {
    const TripletKey& key = *it;
    std::unique_ptr<TripletFlowConstraints>& tripletConstraints =
        triplets_.at(key);
    for (TripletConstraint& c : tripletConstraints->constraints_) {
      c.isStatic = true;
    }
  }
}

void FlowConstraintsCollection::setStaticFlagFromDynamicMask(
    const int distance) {
  if (!video_->hasColorStream("dynamic_mask")) {
    resetStaticFlag();
    return;
  }

  LOG(INFO) << "Setting static flag from dynamic masks...";

  const ColorStream& dynamicMaskStream = video_->colorStream("dynamic_mask");
  const int w = dynamicMaskStream.width();
  const int h = dynamicMaskStream.height();

  std::vector<Mat1b> dynamicMasks(video_->numFrames());

  auto getDynamicMask = [&](const int frame) -> const Mat1b& {
    if (dynamicMasks[frame].empty()) {
      const Mat1f dd = dynamicDistance(frame);

      dynamicMasks[frame].create(dd.size());
      for (int y = 0; y < h; ++y) {
        const float* ddPtr = dd.ptr<const float>(y);
        uint8_t* maskPtr = dynamicMasks[frame].ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
          maskPtr[x] = (ddPtr[x] > distance ? 255 : 0);
        }
      }
    }

    return dynamicMasks[frame];
  };

  int numPairsStatic = 0;
  int numPairsDynamic = 0;

  for (auto it = pairBegin(); it != pairEnd(); it++) {
    const PairKey& key = *it;
    const int frame0 = key.first;
    const int frame1 = key.second;

    const Mat1b& mask0 = getDynamicMask(frame0);
    const Mat1b& mask1 = getDynamicMask(frame1);

    std::unique_ptr<PairFlowConstraints>& pairConstraints = pairs_.at(key);
    for (PairConstraint& c : pairConstraints->constraints_) {
      int ix0 = c[0].x() * w;
      int iy0 = c[0].y() * w;
      int ix1 = c[1].x() * w;
      int iy1 = c[1].y() * w;

      c.isStatic = (mask0(iy0, ix0) && mask1(iy1, ix1));
      (c.isStatic ? numPairsStatic : numPairsDynamic)++;
    }
  }

  int numTripletsStatic = 0;
  int numTripletsDynamic = 0;

  for (auto it = tripletBegin(); it != tripletEnd(); it++) {
    const TripletKey& key = *it;
    const int frame0 = key - 1;
    const int frame1 = key;
    const int frame2 = key + 1;

    const Mat1b& mask0 = getDynamicMask(frame0);
    const Mat1b& mask1 = getDynamicMask(frame1);
    const Mat1b& mask2 = getDynamicMask(frame2);

    std::unique_ptr<TripletFlowConstraints>& tripletConstraints =
        triplets_.at(key);
    for (TripletConstraint& c : tripletConstraints->constraints_) {
      int ix0 = c[0].x() * w;
      int iy0 = c[0].y() * w;
      int ix1 = c[1].x() * w;
      int iy1 = c[1].y() * w;
      int ix2 = c[2].x() * w;
      int iy2 = c[2].y() * w;

      c.isStatic = (mask0(iy0, ix0) && mask1(iy1, ix1) && mask2(iy2, ix2));
      (c.isStatic ? numTripletsStatic : numTripletsDynamic)++;
    }
  }

  LOG(INFO) << fmt::format("    {:d} static pairs, {:d} dynamic pairs.",
      numPairsStatic, numPairsDynamic);
  LOG(INFO) << fmt::format("    {:d} static triplets, {:d} dynamic triplets.",
      numTripletsStatic, numTripletsDynamic);
}

void FlowConstraintsCollection::pruneStaticFlag(const int distance) {
  const ColorStream& downStream = video_->colorStream("down");
  const int w = downStream.width();
  const int h = downStream.height();

  std::vector<Mat1b> masks(video_->numFrames());

  Mat1b diskMask = buildDiskMask(distance);

  for (int frame = 0; frame < video_->numFrames(); ++frame) {
    Mat1b& mask = masks[frame];
    mask.create(h, w);
    mask = 0;

    for (auto it = pairBegin(); it != pairEnd(); it++) {
      const PairKey& key = *it;
      if (key.first != frame && key.second != frame) {
        continue;
      }

      std::unique_ptr<PairFlowConstraints>& pairConstraints = pairs_.at(key);
      for (PairConstraint& c : pairConstraints->constraints_) {
        if (c.isStatic) {
          continue;
        }

        Vector2fna loc = (key.first == frame ? c[0] : c[1]);
        int x = loc.x() * w;
        int y = loc.y() * w;

        const int mx0 = std::max(0, x - distance);
        const int mx1 = std::min(w - 1, x + distance);
        const int my0 = std::max(0, y - distance);
        const int my1 = std::min(h - 1, y + distance);
        for (int my = my0; my <= my1; ++my) {
          const int dy = my - (y - distance);
          const uint8_t* diskPtr = diskMask.ptr<const uint8_t>(dy);
          uint8_t* maskPtr = mask.ptr<uint8_t>(my);
          for (int mx = mx0; mx <= mx1; ++mx) {
            const int dx = mx - (x - distance);
            if (diskPtr[dx]) {
              maskPtr[mx] = 255;
            }
          }
        }
      }
    }
  }

  for (auto it = pairBegin(); it != pairEnd(); it++) {
    const PairKey& key = *it;

    const Mat1b& mask0 = masks[key.first];
    const Mat1b& mask1 = masks[key.second];

    std::unique_ptr<PairFlowConstraints>& pairConstraints = pairs_.at(key);
    for (PairConstraint& c : pairConstraints->constraints_) {
      Vector2i p0(c[0].x() * w, c[0].y() * w);
      Vector2i p1(c[1].x() * w, c[1].y() * w);
      if (mask0(p0.y(), p0.x()) || mask1(p1.y(), p1.x())) {
        c.isStatic = false;
      }
    }
  }

  for (auto it = tripletBegin(); it != tripletEnd(); it++) {
    const TripletKey& key = *it;

    const Mat1b& mask0 = masks[key - 1];
    const Mat1b& mask1 = masks[key];
    const Mat1b& mask2 = masks[key + 1];

    std::unique_ptr<TripletFlowConstraints>& tripletConstraints =
        triplets_.at(key);
    for (TripletConstraint& c : tripletConstraints->constraints_) {
      Vector2i p0(c[0].x() * w, c[0].y() * w);
      Vector2i p1(c[1].x() * w, c[1].y() * w);
      Vector2i p2(c[2].x() * w, c[2].y() * w);

      if (mask0(p0.y(), p0.x()) ||
          mask1(p1.y(), p1.x()) ||
          mask2(p2.y(), p2.x())) {
        c.isStatic = false;
      }
    }
  }
}

}} // namespace facebook::cp
