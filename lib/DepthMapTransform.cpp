// Copyright 2004-present Facebook. All Rights Reserved.

#include "DepthMapTransform.h"

#include <fmt/format.h>

#include "core/Enum_impl.h"
#include "core/FileIo.h"
#include "core/Misc.h"

#include "DepthStream.h"
#include "DepthVideo.h"

using namespace Eigen;
using namespace cv;

namespace facebook {
namespace cp {

std::tuple<float, float> computeDepthRange(const Mat1f& depth) {
  float minDepth = std::numeric_limits<float>::max();
  float maxDepth = std::numeric_limits<float>::min();
  for (int y = 0; y < depth.rows; ++y) {
    const float* depthPtr = depth.ptr<float>(y);
    for (int x = 0; x < depth.cols; ++x) {
      const float d = depthPtr[x];
      if (std::isfinite(d) && d > 0) {
        minDepth = std::min(d, minDepth);
        maxDepth = std::max(d, maxDepth);
      }
    }
  }
  return {minDepth, maxDepth};
}

static std::string stringFromParams(const double* params, const int num) {
  if (num <= 0) {
    return "";
  }
  std::string res = fmt::format("{:.4f}", params[0]);
  for (int i = 1; i < num; ++i) {
    res += fmt::format(", {:.4f}", params[i]);
  }
  return res;
}

static std::string stringFromParams(
    const double *const * blocks, const int numBlocks, const int blockSize) {
  std::string res;
  for (int i = 0; i < numBlocks; ++i) {
    if (i > 0) {
      res += ", ";
    }
    res += stringFromParams(blocks[i], blockSize);
  }
  return res;
}

// This function simply stacks all parameter blocks into the residual vector.
template <typename T>
void paramsToResiduals(
    const T *const *params, T *residuals,
    const int numBlocks, const int blockSize) {
  int count = 0;
  for (int block = 0; block < numBlocks; ++block) {
    for (int i = 0; i < blockSize; ++i) {
      residuals[count++] = params[block][i];
    }
  }
}

const EnumStrings<XformType> xformTypeStrs = {
    {XformType::Depth, "Depth"},
    {XformType::Spatial, "Spatial"},
};
MAKE_VALIDATOR(XformType, xformTypeStrs);

const EnumStrings<DepthXformType> depthXformTypeStrs = {
    {DepthXformType::None, "None"},
    {DepthXformType::Identity, "Identity"},
    {DepthXformType::Global, "Global"},
    {DepthXformType::Grid, "Grid"},
};
MAKE_VALIDATOR(DepthXformType, depthXformTypeStrs);

const EnumStrings<SpatialXformType> spatialXformTypeStrs = {
    {SpatialXformType::None, "None"},
    {SpatialXformType::Identity, "Identity"},
    {SpatialXformType::VerticalLinear, "VerticalLinear"},
    {SpatialXformType::CornersBilinear, "CornersBilinear"},
    {SpatialXformType::BilinearGrid, "BilinearGrid"},
    {SpatialXformType::BicubicGrid, "BicubicGrid"},
};
MAKE_VALIDATOR(SpatialXformType, spatialXformTypeStrs);

//*******************************
//****                       ****
//****    XformDescriptor    ****
//****                       ****
//*******************************

XformDescriptor::XformDescriptor( std::istream& is) {
  fread(is);
}

void XformDescriptor::reset(const XformType type) {
  *this = XformDescriptor();

  if (type == XformType::Spatial) {
    this->type = XformType::Spatial;
    depthType = DepthXformType::None;
    spatialType = SpatialXformType::Identity;
  }
}

std::string XformDescriptor::str() const {
  std::string res;
  if (type == XformType::Depth) {
    const std::string typeStr = enumToString(depthType, depthXformTypeStrs);
    const std::string valueXformStr = enumToString(valueXform, valueXformStrs);

    res = typeStr + "(";

    switch (depthType) {
    case DepthXformType::Identity:
      break;
    case DepthXformType::Global:
      res += valueXformStr;
      break;
    case DepthXformType::Grid:
      if (gridSize.z() > 1) {
        res += fmt::format(
            "{:s}, {:s}, {:d}, {:d}, {:d}, {:f}, {:f}",
            valueXformStr, (cubicInterpolation ? "Cubic" : "Linear"),
            gridSize.x(), gridSize.y(), gridSize.z(),
            depthMinMax.x(), depthMinMax.y());
      } else {
        res += fmt::format(
            "{:s}, {:s}, {:d}, {:d}, {:d}",
            valueXformStr, (cubicInterpolation ? "Cubic" : "Linear"),
            gridSize.x(), gridSize.y(), gridSize.z());
      }
      break;
    default:
      throw std::runtime_error("Invalid depth transform type.");
    }

    res += ")";
  } else if (type == XformType::Spatial) {
    res = enumToString(spatialType, spatialXformTypeStrs);

    switch (spatialType) {
    case SpatialXformType::BilinearGrid:
    case SpatialXformType::BicubicGrid:
      res += fmt::format("({:d}, {:d})", gridSize.x(), gridSize.y());
      break;
    default:
      break;
    }
  } else {
    throw std::runtime_error("Unsupported transform type.");
  }

  return res;
}

void XformDescriptor::parse(const std::string& str) {
  depthType = DepthXformType::None;
  spatialType = SpatialXformType::None;

  const size_t pos = str.find('(');
  const std::string typeStr = str.substr(0, pos);

  std::vector<std::string> args;

  auto getArgs = [&]() {
    if (pos == str.npos || str[str.length() - 1] != ')') {
      throw std::runtime_error("Malformed descriptor string.");
    }

    std::string argsStr = str.substr(pos + 1, str.size() - 1 - (pos + 1));
    args = explode(argsStr, ',');
    for (std::string& s : args) {
      trim(s);
    }
  };

  auto checkNumArgs = [&](const int num) {
    if (args.size() != num) {
      throw std::runtime_error("Incorrect number of parameters.");
    }
  };

  if (type == XformType::Depth) {
    getArgs();

    // Backwards-compatibility hack to load old files using 2D grids.
    if (typeStr == "BicubicGrid" || typeStr == "BilinearGrid") {
      args = {
        args[0], // Value xform
        (typeStr == "BicubicGrid" ? "Cubic" : "Linear"),
        args[1], // Grid cols
        args[2], // Grid rows
        "1"
      };
      depthType = DepthXformType::Grid;
    } else {
      parseEnum(depthType, typeStr, depthXformTypeStrs);
    }

    switch (depthType) {
    case DepthXformType::Identity:
      checkNumArgs(0);
      break;
    case DepthXformType::Global:
      checkNumArgs(1);
      assert(!args.empty()); // For Lint.
      parseEnum(valueXform, args[0], valueXformStrs);
      break;
    case DepthXformType::Grid:
      if (args.size() < 5) {
        throw std::runtime_error("Incorrect number of parameters.");
      }
      parseEnum(valueXform, args[0], valueXformStrs);
      if (args[1] == "Cubic") {
        cubicInterpolation = true;
      } else if (args[1] == "Linear") {
        cubicInterpolation = false;
      } else {
        throw std::runtime_error("Invalid interpolation mode.");
      }

      gridSize.x() = std::stoi(args[2]);
      gridSize.y() = std::stoi(args[3]);
      gridSize.z() = std::stoi(args[4]);

      if (gridSize.z() <= 1) {
        checkNumArgs(5);
      } else {
        checkNumArgs(7);
        depthMinMax.x() = std::stof(args[5]);
        depthMinMax.y() = std::stof(args[6]);
      }
      break;
    default:
      throw std::runtime_error("Invalid depth transform type.");
    }
  } else if (type == XformType::Spatial) {
    parseEnum(spatialType, typeStr, spatialXformTypeStrs);

    switch(spatialType) {
    case SpatialXformType::BilinearGrid:
    case SpatialXformType::BicubicGrid:
      getArgs();
      checkNumArgs(2);
      assert(!args.empty()); // For Lint.
      gridSize.x() = std::stoi(args[0]);
      gridSize.y() = std::stoi(args[1]);
    default:
      break;
    }
  } else {
    throw std::runtime_error("Invalid transform type.");
  }
}

// We read and write string representations because it is easier to maintain
// backward compatibility if we don't have to worry about the layout of this
// struct.
void XformDescriptor::fread(std::istream& is) {
  read(is, type);
  std::string str = readstr(is);
  parse(str);
}

void XformDescriptor::fwrite(std::ostream& os) const {
  write(os, type);
  writestr(os, str());
}

void XformDescriptor::addCommandLineOptions() {
  addOption("type", &type);
  addOption("depthType", &depthType);
  addOption("spatialType", &spatialType);
  addOption("valueXform", &valueXform);
  addOption("cubicInterpolation", &cubicInterpolation);
  addOption("gridSizeX", &gridSize.x());
  addOption("gridSizeY", &gridSize.y());
  addOption("gridSizeZ", &gridSize.z());
  addOption("depthMin", &depthMinMax.x());
  addOption("depthMax", &depthMinMax.y());
}

void XformDescriptor::printParams() const {
  XformDescriptor defaultParams;

  printParamIfNeq("type", type, defaultParams.type, xformTypeStrs);
  printParamIfNeq(
      "depthType", depthType, defaultParams.depthType, depthXformTypeStrs);
  printParamIfNeq(
      "spatialType", spatialType, defaultParams.spatialType,
      spatialXformTypeStrs);
  printParamIfNeq(
      "valueXform", valueXform, defaultParams.valueXform, valueXformStrs);
  PRINT_PARAM_IF_NEQ(cubicInterpolation)
  printParamIfNeq("gridSizeX", gridSize.x(), defaultParams.gridSize.x());
  printParamIfNeq("gridSizeY", gridSize.y(), defaultParams.gridSize.y());
  printParamIfNeq("gridSizeZ", gridSize.z(), defaultParams.gridSize.z());
  printParamIfNeq("depthMin", depthMinMax.x(), defaultParams.depthMinMax.x());
  printParamIfNeq("depthMax", depthMinMax.y(), defaultParams.depthMinMax.y());
}

bool XformDescriptor::operator==(const XformDescriptor& other) const {
  if (type == other.type &&
      depthType == other.depthType &&
      spatialType == other.spatialType &&
      valueXform == other.valueXform &&
      gridSize == other.gridSize) {
    return true;
  }

  return false;
}

bool XformDescriptor::operator!=(const XformDescriptor& other) const {
  return !(*this == other);
}

//****************************
//****                    ****
//****    XformFunctor    ****
//****                    ****
//****************************

const std::vector<double*>& XformFunctor::paramBlocks() const {
  return paramBlocks_;
}

const std::vector<int>& XformFunctor::paramBlockSizes() const {
  return paramBlockSizes_;
}

int XformFunctor::numParamBlocks() const {
  return paramBlocks_.size();
}

//*********************
//****             ****
//****    Xform    ****
//****             ****
//*********************

std::unique_ptr<Xform> Xform::clone() const {
  auto res = createXform(desc_);
  res->params_ = params_;
  return res;
}

void Xform::copyFrom(const Xform& other) {
  if (other.desc_ != desc_) {
    throw std::runtime_error(
        "Can only copy parameters from same type of transform.");
  }

  params_ = other.params_;
}

std::string Xform::str() const {
  std::string res = desc_.str() + " [";
  for (int i = 0; i < params_.size(); ++i) {
    if (i > 0) {
      res += ", ";
    }
    res += fmt::format("{:.2f}", params_[i]);
  }
  res += "]";
  return res;
}

bool Xform::operator==(const Xform& other) const {
  return (desc_ == other.desc_ && params_ == other.params_);
}

bool Xform::operator!=(const Xform& other) const {
  return !(*this == other);
}

//**************************
//****                  ****
//****    DepthXform    ****
//****                  ****
//**************************

std::unique_ptr<Mat1f> DepthXform::apply(const Mat1f& src) const {
  std::unique_ptr<Mat1f> dst = std::make_unique<Mat1f>(src.size());

  const float xScale = 2.f / (src.cols - 1.f);
  const float yScale = 2.f / (src.rows - 1.f);
  Vector2f loc;

  for (int y = 0; y < src.rows; ++y) {
    const float* srcPtr = src.ptr<float>(y);
    float* dstPtr = dst->ptr<float>(y);
    loc.y() = 1.f - y * yScale;

    for (int x = 0; x < src.cols; ++x) {
      loc.x() = -1.f + x * xScale;
      std::unique_ptr<DepthFunctor> fn = createFunctor(srcPtr[x], loc);
      double const* const* params = fn->paramBlocks().data();
      dstPtr[x] = (*fn)(params);
    }
  }

  return dst;
}

Mat DepthXform::paramMap(const DepthFrame& /* df */) const {
  throw std::runtime_error(
      "Parameter map not implemented for this transform type.");
}

//****************************
//****                    ****
//****    SpatialXform    ****
//****                    ****
//****************************

Mat2f SpatialXform::warp(const int h, const int w) const {
  Mat2f dst(h, w);

  const float xScale = 2.f / (w - 1.f);
  const float yScale = 2.f / (h - 1.f);
  Vector2f loc;

  for (int y = 0; y < h; ++y) {
    Vec2f* dstPtr = dst.ptr<Vec2f>(y);
    loc.y() = 1.f - y * yScale;

    for (int x = 0; x < w; ++x) {
      loc.x() = -1.f + x * xScale;
      std::unique_ptr<SpatialFunctor> fn = createFunctor(loc);
      double const* const* params = fn->paramBlocks().data();
      Vector2d w = (*fn)(params);
      dstPtr[x] = Vec2f(w.x(), w.y());
    }
  }

  return dst;
}

//**********************************
//****                          ****
//****    IdentityDepthXform    ****
//****                          ****
//**********************************

namespace {
class IdentityDepthFunctor : public DepthFunctor {
 public:
  explicit IdentityDepthFunctor(const double srcDepth)
    : srcDepth_(srcDepth), srcDepthJet_(srcDepth) {
  }

  double operator()(double const* const* /* params */) const override {
    return srcDepth_;
  }

  Jet operator()(Jet const* const* /* params */) const override {
    return srcDepthJet_;
  }

 private:
  double srcDepth_ = 0.0;
  Jet srcDepthJet_ = Jet(0.0);
};
} // anonymous namespace

IdentityDepthXform::IdentityDepthXform() {
  desc_.type = XformType::Depth;
  desc_.depthType = DepthXformType::Identity;
}

std::unique_ptr<DepthFunctor> IdentityDepthXform::createFunctor(
    const float srcDepth, const Vector2f& /* loc */) const {
  return std::make_unique<IdentityDepthFunctor>(srcDepth);
}

//********************************
//****                        ****
//****    GlobalDepthXform    ****
//****                        ****
//********************************

namespace {
class GlobalDepthFunctor : public DepthFunctor {
 public:
  GlobalDepthFunctor(
      const double srcDepth, double* params, const ValueXform& valueXform)
    : valueXform_(valueXform), srcDepth_(srcDepth), srcDepthJet_(srcDepth) {
    paramBlocks_ = { params };
    paramBlockSizes_ = { valueXform_.numParams() };
  }

  double operator()(double const* const* params) const override {
    return valueXform_(srcDepth_, params[0]);
  }

  Jet operator()(Jet const* const* params) const override {
    return valueXform_(srcDepthJet_, params[0]);
  }

  std::string info() const override {
    assert(!paramBlocks_.empty()); // For Lint.
    return std::string("Params: (") +
        stringFromParams(paramBlocks_[0], valueXform_.numParams()) + ")";
  }

 private:
  const ValueXform& valueXform_;

  double srcDepth_ = 0.0;
  Jet srcDepthJet_ = Jet(0.0);
};
} // anonymous namespace

GlobalDepthXform::GlobalDepthXform(const ValueXformType valueXform)
    : valueXform_(ValueXform::getInstance(valueXform)) {
  desc_.type = XformType::Depth;
  desc_.depthType = DepthXformType::Global;
  desc_.valueXform = valueXform;
  params_.resize(valueXform_.numParams(), 1.0);
  paramBlocks_ = { params_.data() };
  paramBlockSizes_ = { valueXform_.numParams() };
}

std::unique_ptr<DepthFunctor> GlobalDepthXform::createFunctor(
    const float srcDepth, const Vector2f& /* loc */) const {
  double* params = const_cast<double*>(params_.data());
  return std::make_unique<GlobalDepthFunctor>(srcDepth, params, valueXform_);
}

//******************************
//****                      ****
//****    GridDepthXform    ****
//****                      ****
//******************************

namespace {
// This generates a short info string for a depth/spatial grid functor.
std::string gridFunctorInfo(
    const XformDescriptor& desc, const int N,
    const double* paramsBasePtr, const std::vector<double*>& paramBlocks,
    const std::vector<double>& weights) {
  const int K = paramBlocks.size();

  std::string res;

  for (int i = 0; i < K; ++i) {
    const int offset = (paramBlocks[i] - paramsBasePtr) / N;
    assert(paramBlocks[i] - paramsBasePtr == offset * N);

    const Vector3i gs = desc.gridSize;
    const int zStride = (gs.x() > 0 ? gs.x() * gs.y() : 1);

    res += "(";

    if (gs.x() > 1) {
      const int x = offset % gs.x();
      const int y = (offset % zStride) / gs.x();
      res += fmt::format("X{:d}, Y{:d}, ", x, y);
    }

    if (gs.z() > 1) {
      const int z = offset / zStride;
      res += fmt::format("Z{:d}, ", z);
    }

    res += fmt::format("W{:.2f}, P{:s}) ",
        weights[i], stringFromParams(paramBlocks[i], N));
  }

  return res;
}

class GridDepthFunctor : public DepthFunctor {
 public:
  GridDepthFunctor(
      const DepthXform& parent,
      const double srcDepth, const ValueXform& valueXform,
      std::vector<double*> paramBlocks, std::vector<double> weights)
    : parent_(parent), valueXform_(valueXform), srcDepth_(srcDepth),
      srcDepthJet_(srcDepth), weights_(weights) {
    paramBlocks_ = paramBlocks;
    paramBlockSizes_.resize(paramBlocks_.size(), valueXform_.numParams());
  }

  template <typename T>
  T eval(const T& srcDepth, T const* const* params) const {
    // Compute a linear combination of the surrounding vertices' transforms.
    T res = T(0.0);
    for (int i = 0; i < paramBlocks_.size(); ++i) {
      res += valueXform_(srcDepth, params[i]) * T(weights_[i]);
    }

    return res;
  }

  double operator()(double const* const* params) const override {
    return eval(srcDepth_, params);
  }

  Jet operator()(Jet const* const* params) const override {
    return eval(srcDepthJet_, params);
  }

  std::string info() const override {
    return gridFunctorInfo(
        parent_.desc(), valueXform_.numParams(),
        parent_.data(), paramBlocks_, weights_);
  }

 private:
  const DepthXform& parent_;
  const ValueXform& valueXform_;

  double srcDepth_ = 0.0;
  Jet srcDepthJet_ = Jet(0.0);
  std::vector<double> weights_;
};

template <typename T>
void computeGridDeformationCost(
    const Vector3i gridSize, const int dims,
    T const* const* params, T* residuals) {
  T* outPtr = residuals;

  const int yStride = gridSize.x();
  const int zStride = gridSize.x() * gridSize.y();

  for (int z = 0; z < gridSize.z(); ++z) {
    for (int y = 0; y < gridSize.y(); ++y) {
      for (int x = 0; x < gridSize.x(); ++x) {
        T const* thisBlock = params[x + y * yStride + z * zStride];

        auto addResidual = [&](T const* thatBlock) {
          for (int i = 0; i < dims; ++i) {
            T scale = min(abs(thisBlock[i]), abs(thatBlock[i]));
            *(outPtr++) = (thisBlock[i] - thatBlock[i]) / scale;
          }
        };

        if (x > 0) {
          T const* thatBlock = params[(x - 1) + y * yStride + z * zStride];
          addResidual(thatBlock);
        }
        if (y > 0) {
          T const* thatBlock = params[x + (y - 1) * yStride + z * zStride];
          addResidual(thatBlock);
        }
        if (z > 0) {
          T const* thatBlock = params[x + y * yStride + (z - 1) * zStride];
          addResidual(thatBlock);
        }
      }
    }
  }
}

// Compute cubic hermite spline coefficients, see
// https://en.wikipedia.org/wiki/Cubic_Hermite_spline.
void cubicSpline(std::array<double, 4>& w, const double t) {
  const double t2 = t * t;
  const double t3 = t2 * t;
  w[0] = -0.5 * t3 + t2 - 0.5 * t;
  w[1] = 1.5 * t3 - 2.5 * t2 + 1.0;
  w[2] = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
  w[3] = 0.5 * t3 - 0.5 * t2;
};

} // anonymous namespace

GridDepthXform::GridDepthXform(
    const ValueXformType valueXform, const bool cubicInterpolation,
    const Vector3i gridSize, const Vector2d depthMinMax)
    : valueXform_(ValueXform::getInstance(valueXform)),
      cubicInterpolation_(cubicInterpolation), gridSize_(gridSize),
      depthMinMax_(depthMinMax) {
  desc_.type = XformType::Depth;
  desc_.depthType = DepthXformType::Grid;
  desc_.valueXform = valueXform;
  desc_.cubicInterpolation = cubicInterpolation;
  desc_.gridSize = gridSize;
  desc_.depthMinMax = depthMinMax;

  if (gridSize.x() > 1 || gridSize.y() > 1) {
    if (gridSize.x() < 2 || gridSize.y() < 2) {
      throw std::runtime_error("Spatial grid transforms must have "
          "at least two rows and columns, respectively.");
    }
  }

  const int numParams =
      valueXform_.numParams() * gridSize.x() * gridSize.y() * gridSize.z();
  if (numParams <= 1) {
    throw std::runtime_error("Grid transform cannot have an empty grid.");
  }
  params_.resize(numParams, 1.0);

  // Store and verify depth range
  if (gridSize.z() > 1) {
    depthRange_ = depthMinMax_.y() - depthMinMax_.x();
    if (depthMinMax_.x() <= 0.0 || depthMinMax_.y() <= 0.0) {
      throw std::runtime_error("Depth values must be positive.");
    }
    if (depthRange_ <= 0.0) {
      throw std::runtime_error("Depth range must be positive.");
    }

    disparityMinMax_ = Vector2d(1.0 / depthMinMax_.y(), 1.0 / depthMinMax_.x());
    disparityRange_ = disparityMinMax_.y() - disparityMinMax_.x();
    disparityRangeInv_ = 1.0 / disparityRange_;

    // Precompute source depth values for handles
    handleSrcDisparityInterval_ = disparityRange_ / (gridSize.z() - 1);
    handleSrcDisparity_.resize(gridSize.z());

    for (int i = 0; i < gridSize.z(); ++i) {
      handleSrcDisparity_[i] =
          disparityMinMax_.x() + handleSrcDisparityInterval_ * i;
    }
  }

  for (int i = 0; i < gridSize.x() * gridSize.y() * gridSize.z(); ++i) {
    paramBlocks_.push_back(&params_[i * valueXform_.numParams()]);
    paramBlockSizes_.push_back(valueXform_.numParams());
  }
}

void GridDepthXform::linearGather(
    const float srcDepth, const Vector2f& loc,
    std::vector<double*>& paramBlocks, std::vector<double>& weights) const {
  const bool spatial = (desc_.gridSize.x() > 1);
  const bool depthWise = (desc_.gridSize.z() > 1);

  Vector3d maxCoord = Vector3d::Zero();
  Vector3d scaledCoord = Vector3d::Zero();
  Vector3i index = Vector3i::Zero();
  Vector3d relCoord = Vector3d::Zero();

  if (spatial) {
    maxCoord.x() = std::nextafter(desc_.gridSize.x() - 1, 0.0);
    maxCoord.y() = std::nextafter(desc_.gridSize.y() - 1, 0.0);

    scaledCoord.x() = std::clamp(
        (loc.x() + 1.0) * (desc_.gridSize.x() - 1) / 2.0, 0.0, maxCoord.x());
    scaledCoord.y() = std::clamp(
        (loc.y() + 1.0) * (desc_.gridSize.y() - 1) / 2.0, 0.0, maxCoord.y());
    index.x() = static_cast<int>(scaledCoord.x());
    index.y() = static_cast<int>(scaledCoord.y());
    assert(index.x() >= 0 && index.x() < desc_.gridSize.x() - 1);
    assert(index.y() >= 0 && index.y() < desc_.gridSize.y() - 1);

    relCoord.x() = scaledCoord.x() - index.x();
    relCoord.y() = scaledCoord.y() - index.y();
  }

  if (depthWise) {
    maxCoord.z() = std::nextafter(desc_.gridSize.z() - 1, 0.0);
    double srcDisparity = 1.0 / double(srcDepth);
    scaledCoord.z() = std::clamp(
        (srcDisparity - disparityMinMax_.x()) / handleSrcDisparityInterval_,
        0.0, maxCoord.z());
    index.z() = static_cast<int>(scaledCoord.z());
    assert(index.z() >= 0 && index.z() < desc_.gridSize.z() - 1);
    relCoord.z() = scaledCoord.z() - index.z();
  }

  paramBlocks.clear();
  weights.clear();

  if (spatial && depthWise) {
    const int yStride = desc_.gridSize.x();
    const int zStride = desc_.gridSize.x() * desc_.gridSize.y();
    const int i0 =
        index.x() + index.y() * yStride + index.z() * zStride;
    const int i1 =
        (index.x() + 1) + index.y() * yStride + index.z() * zStride;
    const int i2 =
        index.x() + (index.y() + 1) * yStride + index.z() * zStride;
    const int i3 =
        (index.x() + 1) + (index.y() + 1) * yStride + index.z() * zStride;
    const int i4 =
        index.x() + index.y() * yStride + (index.z() + 1) * zStride;
    const int i5 =
        (index.x() + 1) + index.y() * yStride + (index.z() + 1) * zStride;
    const int i6 =
        index.x() + (index.y() + 1) * yStride + (index.z() + 1) * zStride;
    const int i7 =
        (index.x() + 1) + (index.y() + 1) * yStride + (index.z() + 1) * zStride;

    double* b0 = const_cast<double*>(&params_[i0]);
    double* b1 = const_cast<double*>(&params_[i1]);
    double* b2 = const_cast<double*>(&params_[i2]);
    double* b3 = const_cast<double*>(&params_[i3]);
    double* b4 = const_cast<double*>(&params_[i4]);
    double* b5 = const_cast<double*>(&params_[i5]);
    double* b6 = const_cast<double*>(&params_[i6]);
    double* b7 = const_cast<double*>(&params_[i7]);

    double w0 = (1.0-relCoord.x()) * (1.0-relCoord.y()) * (1.0-relCoord.z());
    double w1 = relCoord.x() * (1.0 - relCoord.y()) * (1.0 - relCoord.z());
    double w2 = (1.0 - relCoord.x()) * relCoord.y() * (1.0 - relCoord.z());
    double w3 = relCoord.x() * relCoord.y() * (1.0 - relCoord.z());
    double w4 = (1.0 - relCoord.x()) * (1.0 - relCoord.y()) * relCoord.z();
    double w5 = relCoord.x() * (1.0 - relCoord.y()) * relCoord.z();
    double w6 = (1.0 - relCoord.x()) * relCoord.y() * relCoord.z();
    double w7 = relCoord.x() * relCoord.y() * relCoord.z();

    paramBlocks = {b0, b1, b2, b3, b4, b5, b6, b7};
    weights = {w0, w1, w2, w3, w4, w5, w6, w7};

  } else if (spatial) {
    const int stride = desc_.gridSize.x();
    const int i0 = index.x() + index.y() * stride;
    const int i1 = (index.x() + 1) + index.y() * stride;
    const int i2 = index.x() + (index.y() + 1) * stride;
    const int i3 = (index.x() + 1) + (index.y() + 1) * stride;

    double* b0 = const_cast<double*>(&params_[i0]);
    double* b1 = const_cast<double*>(&params_[i1]);
    double* b2 = const_cast<double*>(&params_[i2]);
    double* b3 = const_cast<double*>(&params_[i3]);

    const double w0 = (1.0 - relCoord.x()) * (1.0 - relCoord.y());
    const double w1 = relCoord.x() * (1.0 - relCoord.y());
    const double w2 = (1.0 - relCoord.x()) * relCoord.y();
    const double w3 = relCoord.x() * relCoord.y();

    paramBlocks = {b0, b1, b2, b3};
    weights = {w0, w1, w2, w3};
  } else if (depthWise) {
    double* b0 = const_cast<double*>(&params_[index.z()]);
    double* b1 = const_cast<double*>(&params_[index.z() + 1]);

    const double w0 = (1.0 - relCoord.z());
    const double w1 = relCoord.z();

    paramBlocks = {b0, b1};
    weights = {w0, w1};
  }
}

void GridDepthXform::cubicGather(
    const float srcDepth, const Vector2f& loc,
    std::vector<double*>& paramBlocks, std::vector<double>& weights) const {
  const bool spatial = (desc_.gridSize.x() > 1);
  const bool depthWise = (desc_.gridSize.z() > 1);

  const int yStride = desc_.gridSize.x();
  const int zStride = desc_.gridSize.x() * desc_.gridSize.y();

  Vector3d maxCoord = Vector3d::Zero();
  Vector3d scaledCoord = Vector3d::Zero();
  Vector3i index = Vector3i::Zero();
  Vector3d relCoord = Vector3d::Zero();

  if (spatial) {
    maxCoord.x() = std::nextafter(desc_.gridSize.x() - 1, 0.0);
    maxCoord.y() = std::nextafter(desc_.gridSize.y() - 1, 0.0);

    scaledCoord.x() = std::clamp(
        (loc.x() + 1.0) * (desc_.gridSize.x() - 1) / 2.0, 0.0, maxCoord.x());
    scaledCoord.y() = std::clamp(
        (loc.y() + 1.0) * (desc_.gridSize.y() - 1) / 2.0, 0.0, maxCoord.y());
    index.x() = static_cast<int>(scaledCoord.x());
    index.y() = static_cast<int>(scaledCoord.y());
    assert(index.x() >= 0 && index.x() < desc_.gridSize.x() - 1);
    assert(index.y() >= 0 && index.y() < desc_.gridSize.y() - 1);

    relCoord.x() = scaledCoord.x() - index.x();
    relCoord.y() = scaledCoord.y() - index.y();
  }

  if (depthWise) {
    maxCoord.z() = std::nextafter(desc_.gridSize.z() - 1, 0.0);
    double srcDisparity = 1.0 / double(srcDepth);
    scaledCoord.z() = std::clamp(
        (srcDisparity - disparityMinMax_.x()) / handleSrcDisparityInterval_,
        0.0, maxCoord.z());
    index.z() = static_cast<int>(scaledCoord.z());
    assert(index.z() >= 0 && index.z() < desc_.gridSize.z() - 1);
    relCoord.z() = scaledCoord.z() - index.z();
  }

  // Combine list of parameters and weights
  paramBlocks.clear();
  weights.clear();

  std::array<double, 4> wx;
  std::array<double, 4> wy;
  std::array<double, 4> wz;

  if (spatial) {
    cubicSpline(wx, relCoord.x());
    cubicSpline(wy, relCoord.y());
  }
  if (depthWise) {
    cubicSpline(wz, relCoord.z());
  }

  Vector3i offset0(
      (index.x() == 0 ? 1 : 0),
      (index.y() == 0 ? 1 : 0),
      (index.z() == 0 ? 1 : 0));

  Vector3i offset1(
      (spatial ? (index.x() == desc_.gridSize.x() - 2 ? 3 : 4) : 2),
      (spatial ? (index.y() == desc_.gridSize.y() - 2 ? 3 : 4) : 2),
      (depthWise ? (index.z() == desc_.gridSize.z() - 2 ? 3 : 4) : 2));

  for (int z = offset0.z(); z < offset1.z(); ++z) {
    const int pz = index.z() - 1 + z;
    for (int y = offset0.y(); y < offset1.y(); ++y) {
      const int py = index.y() - 1 + y;
      for (int x = offset0.x(); x < offset1.x(); ++x) {
        const int px = index.x() - 1 + x;
        const double* ptr = paramBlocks_[px + py * yStride + pz * zStride];
        paramBlocks.push_back(const_cast<double*>(ptr));
        weights.push_back(0.0);
      }
    }
  }

  Vector3i outStride = offset1 - offset0;

  for (int z = 0; z < (depthWise ? 4 : 1); ++z) {
    for (int y = 0; y < (spatial ? 4 : 1); ++y) {
      for (int x = 0; x < (spatial ? 4 : 1); ++x) {
        const int cx = std::clamp(x - offset0.x(), 0, outStride.x() - 1);
        const int cy = std::clamp(y - offset0.y(), 0, outStride.y() - 1);
        const int cz = std::clamp(z - offset0.z(), 0, outStride.z() - 1);
        const int outIdx =
            cx + cy * outStride.x() + cz * outStride.x() * outStride.y();
        weights[outIdx] += wx[x] * wy[y];
      }
    }
  }
}

Mat GridDepthXform::paramMap(const DepthFrame& df) const {
  const int w = df.width();
  const int h = df.height();
  const Mat1f* sourceDepth = df.sourceDepth();

  const int N = valueXform_.numParams();
  int type = CV_MAKE_TYPE(CV_64F, N);
  Mat map(h, w, type);

  std::vector<double*> params;
  std::vector<double> weights;

  const float xScale = 2.f / (w - 1.f);
  const float yScale = 2.f / (h - 1.f);
  Vector2f loc;

  for (int y = 0; y < h; ++y) {
    loc.y() = 1.f - y * yScale;
    const float* srcDepthPtr = sourceDepth->ptr<const float>(y);

    for (int x = 0; x < w; ++x) {
      loc.x() = -1.f + x * xScale;

      if (desc_.cubicInterpolation) {
        cubicGather(srcDepthPtr[x], loc, params, weights);
      } else {
        linearGather(srcDepthPtr[x], loc, params, weights);
      }

      double* const dst = map.ptr<double>(y, x);

      for (int d = 0; d < N; ++d) {
        dst[d] = 0.0;
      }

      for (int i = 0; i < params.size(); ++i) {
        for (int d = 0; d < N; ++d) {
          dst[d] += params[i][d] * weights[i];
        }
      }
    }
  }

  return map;
}

int GridDepthXform::numDeformationCostResiduals() const {
  const int X = desc_.gridSize.x();
  const int Y = desc_.gridSize.y();
  const int Z = desc_.gridSize.z();
  const int numEdges = (X - 1) * Y * Z + X * (Y - 1) * Z + X * Y * (Z - 1);
  return numEdges * valueXform_.numParams();
}

void GridDepthXform::computeDeformationCost(
    double const* const* params, double* residuals) const {
  computeGridDeformationCost(
      desc_.gridSize, valueXform_.numParams(), params, residuals);
}

void GridDepthXform::computeDeformationCost(
    Jet const* const* params, Jet* residuals) const {
  computeGridDeformationCost(
      desc_.gridSize, valueXform_.numParams(), params, residuals);
}

std::unique_ptr<DepthFunctor> GridDepthXform::createFunctor(
    const float srcDepth, const Eigen::Vector2f& loc) const {
  std::vector<double*> params;
  std::vector<double> weights;
  if (cubicInterpolation_) {
    cubicGather(srcDepth, loc, params, weights);
  } else {
    linearGather(srcDepth, loc, params, weights);
  }
  return std::make_unique<GridDepthFunctor>(
      *this, srcDepth, valueXform_, params, weights);
};

//************************************
//****                            ****
//****    IdentitySpatialXform    ****
//****                            ****
//************************************

namespace {
class IdentitySpatialFunctor : public SpatialFunctor {
 public:
  Vector2d operator()(double const* const* /* params */) const override {
    return Vector2d::Zero();
  }

  Matrix<Jet,2,1> operator()(Jet const* const* /* params */) const override {
    return Matrix<Jet,2,1>::Zero();
  }
};
} // anonymous namespace

IdentitySpatialXform::IdentitySpatialXform() {
  desc_.type = XformType::Spatial;
  desc_.spatialType = SpatialXformType::Identity;
}

std::unique_ptr<SpatialFunctor> IdentitySpatialXform::createFunctor(
    const Vector2f& /* loc */) const {
  return std::make_unique<IdentitySpatialFunctor>();
}

//******************************************
//****                                  ****
//****    VerticalLinearSpatialXform    ****
//****                                  ****
//******************************************

namespace {
class VerticalLinearSpatialFunctor : public SpatialFunctor {
 public:
  explicit VerticalLinearSpatialFunctor(
      double* params0, const double weight0,
      double* params1, const double weight1)
      : weight0_(weight0), weight1_(weight1) {
    paramBlocks_ = { params0, params1 };
    paramBlockSizes_ = { 2, 2 };
  }

  Vector2d operator()(double const* const* params) const override {
    Vector2d top(params[0][0], params[0][1]);
    Vector2d btm(params[1][0], params[1][1]);
    return top * weight0_ + btm * weight1_;
  }

  Vector2Jet operator()(Jet const* const* params) const override {
    Vector2Jet top(params[0][0], params[0][1]);
    Vector2Jet btm(params[1][0], params[1][1]);
    return top * weight0_ + btm * weight1_;
  }

  std::string info() const override {
    return std::string("Params: (") +
        stringFromParams(paramBlocks_.data(), 2, 2) + ")";
  }

 private:
  double weight0_ = 0.0;
  double weight1_ = 0.0;
};
} // anonymous namespace

VerticalLinearSpatialXform::VerticalLinearSpatialXform() {
  desc_.type = XformType::Spatial;
  desc_.spatialType = SpatialXformType::VerticalLinear;

  params_.resize(4, 0.0);
  paramBlocks_ = { &params_[0], &params_[2] };
  paramBlockSizes_ = { 2, 2 };
}

std::unique_ptr<SpatialFunctor> VerticalLinearSpatialXform::createFunctor(
    const Vector2f& loc) const {
  double weight0 = 0.5 + 0.5 * loc.y();
  double weight1 = 1.0 - weight0;

  return std::make_unique<VerticalLinearSpatialFunctor>(
      paramBlocks_[0], weight0, paramBlocks_[1], weight1);
}

int VerticalLinearSpatialXform::numDeformationCostResiduals() const {
  return 4;
}

void VerticalLinearSpatialXform::computeDeformationCost(
    const double *const *params, double *residuals) const {
  paramsToResiduals(params, residuals, 2, 2);
}

void VerticalLinearSpatialXform::computeDeformationCost(
    const Jet *const *params, Jet *residuals) const {
  paramsToResiduals(params, residuals, 2, 2);
}

//*******************************************
//****                                   ****
//****    CornersBilinearSpatialXform    ****
//****                                   ****
//*******************************************

namespace {
class CornersBilinearSpatialFunctor : public SpatialFunctor {
 public:
  explicit CornersBilinearSpatialFunctor(
      std::array<double*, 4> params, std::array<double, 4> weights)
      : weights_(weights) {
    paramBlocks_ = { params[0], params[1], params[2], params[3] };
    paramBlockSizes_ = { 2, 2, 2, 2 };
  }

  Vector2d operator()(double const* const* params) const override {
    Vector2d res = Vector2d::Zero();
    for (int i = 0; i < 4; ++i) {
      res += Vector2d(params[i][0], params[i][1]) * weights_[i];
    }
    return res;
  }

  Vector2Jet operator()(Jet const* const* params) const override {
    Vector2Jet res = Vector2Jet::Zero();
    for (int i = 0; i < 4; ++i) {
      res += Vector2Jet(params[i][0], params[i][1]) * weights_[i];
    }
    return res;
  }

  std::string info() const override {
    return std::string("Params: (") +
        stringFromParams(paramBlocks_.data(), 2, 2) + ")";
  }

 private:
  std::array<double, 4> weights_ = {0.0, 0.0, 0.0, 0.0};
};
} // anonymous namespace

CornersBilinearSpatialXform::CornersBilinearSpatialXform() {
  desc_.type = XformType::Spatial;
  desc_.spatialType = SpatialXformType::CornersBilinear;

  params_.resize(8, 0.0);
  paramBlocks_ = { &params_[0], &params_[2], &params_[4], &params_[6] };
  paramBlockSizes_ = { 2, 2, 2, 2 };
}

std::unique_ptr<SpatialFunctor> CornersBilinearSpatialXform::createFunctor(
    const Vector2f& loc) const {
  const double wx = 0.5 + 0.5 * loc.x();
  const double wy = 0.5 + 0.5 * loc.y();

  std::array<double*, 4> params {
      paramBlocks_[0], paramBlocks_[1], paramBlocks_[2], paramBlocks_[3]};
  std::array<double, 4> weights {
      wx * wy, (1.0 - wx) * wy, wx * (1.0 - wy), (1.0 - wx) * (1.0 - wy)};
  return std::make_unique<CornersBilinearSpatialFunctor>(params, weights);
}

int CornersBilinearSpatialXform::numDeformationCostResiduals() const {
  return 8;
}

void CornersBilinearSpatialXform::computeDeformationCost(
    const double *const *params, double *residuals) const {
  paramsToResiduals(params, residuals, 4, 2);
}

void CornersBilinearSpatialXform::computeDeformationCost(
    const Jet *const *params, Jet *residuals) const {
  paramsToResiduals(params, residuals, 4, 2);
}

//********************************
//****                        ****
//****    GridSpatialXform    ****
//****                        ****
//********************************

namespace {
class GridSpatialFunctor : public SpatialFunctor {
 public:
  GridSpatialFunctor(
      const GridSpatialXform& parent,
      std::vector<double*> paramBlocks, std::vector<double> weights)
      : parent_(parent) {
    paramBlocks_ = paramBlocks;
    paramBlockSizes_.resize(paramBlocks_.size(), 2);
    weights_ = weights;
  }

  template <typename T>
  Matrix<T, 2, 1> eval(T const* const* params) const {
    // Compute a linear combination of the surrounding vertices' warps.
    Matrix<T, 2, 1> res = Matrix<T, 2, 1>::Zero();
    for (int i = 0; i < paramBlocks_.size(); ++i) {
      res += Matrix<T, 2, 1>(params[i][0], params[i][1]) * T(weights_[i]);
    }
    return res;
  }

  Vector2d operator()(double const* const* params) const override {
    return eval(params);
  }

  Vector2Jet operator()(Jet const* const* params) const override {
    return eval(params);
  }

  std::string info() const override {
    return gridFunctorInfo(
        parent_.desc(), 2, parent_.data(), paramBlocks_, weights_);
  }

 private:
  const GridSpatialXform& parent_;
  std::vector<double> weights_;
};

void bilinearSpatialGridGather(
    const XformDescriptor& desc, const Vector2f& loc,
    const std::vector<double*>& gridParamBlocks,
    std::vector<double*>& fnParamBlocks, std::vector<double>& fnWeights) {
  const double maxx = std::nextafter(desc.gridSize.x() - 1, 0.0);
  const double maxy = std::nextafter(desc.gridSize.y() - 1, 0.0);

  const double sx =
      std::clamp((loc.x() + 1.0) * (desc.gridSize.x() - 1) / 2.0, 0.0, maxx);
  const double sy =
      std::clamp((loc.y() + 1.0) * (desc.gridSize.y() - 1) / 2.0, 0.0, maxy);

  const int ix = static_cast<int>(sx);
  const int iy = static_cast<int>(sy);
  assert(ix >= 0 && ix < desc.gridSize.x() - 1);
  assert(iy >= 0 && iy < desc.gridSize.y() - 1);

  const double rx = sx - ix;
  const double ry = sy - iy;

  const int C = desc.gridSize.x();
  double* b0 = const_cast<double*>(gridParamBlocks[ix + iy * C]);
  double* b1 = const_cast<double*>(gridParamBlocks[(ix + 1) + iy * C]);
  double* b2 = const_cast<double*>(gridParamBlocks[ix + (iy + 1) * C]);
  double* b3 = const_cast<double*>(gridParamBlocks[(ix + 1) + (iy + 1) * C]);

  const double w0 = (1.0 - rx) * (1.0 - ry);
  const double w1 = rx * (1.0 - ry);
  const double w2 = (1.0 - rx) * ry;
  const double w3 = rx * ry;

  fnParamBlocks = {b0, b1, b2, b3};
  fnWeights = {w0, w1, w2, w3};
}

void bicubicSpatialGridGather(
    const XformDescriptor& desc, const Vector2f& loc,
    const std::vector<double*>& gridParamBlocks,
    std::vector<double*>& fnParamBlocks, std::vector<double>& fnWeights) {
  // Compute grid cell (ix, iy) and relative position in cell (rx, ry).
  const double maxx = std::nextafter(desc.gridSize.x() - 1, 0.0);
  const double maxy = std::nextafter(desc.gridSize.y() - 1, 0.0);

  const double sx =
      std::clamp((loc.x() + 1.0) * (desc.gridSize.x() - 1) / 2.0, 0.0, maxx);
  const double sy =
      std::clamp((loc.y() + 1.0) * (desc.gridSize.y() - 1) / 2.0, 0.0, maxy);

  const int ix = static_cast<int>(sx);
  const int iy = static_cast<int>(sy);

  assert(ix >= 0 && ix < desc.gridSize.x() - 1);
  assert(iy >= 0 && iy < desc.gridSize.y() - 1);

  const double rx = sx - ix;
  const double ry = sy - iy;

  std::array<double, 4> wx;
  std::array<double, 4> wy;
  cubicSpline(wx, rx);
  cubicSpline(wy, ry);

  // Combine list of parameters and weights
  fnParamBlocks.clear();
  fnWeights.clear();

  const int x0 = (ix == 0 ? 1 : 0);
  const int x1 = (ix == desc.gridSize.x() - 2 ? 3 : 4);
  const int y0 = (iy == 0 ? 1 : 0);
  const int y1 = (iy == desc.gridSize.y() - 2 ? 3 : 4);
  const int xstride = x1 - x0;
  const int ystride = y1 - y0;

  for (int y = y0; y < y1; ++y) {
    const int py = iy - 1 + y;
    for (int x = x0; x < x1; ++x) {
      const int px = ix - 1 + x;
      const double* ptr = gridParamBlocks[px + py * desc.gridSize.x()];
      fnParamBlocks.push_back(const_cast<double*>(ptr));
      fnWeights.push_back(0.0);
    }
  }

  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      const int cx = std::clamp(x - x0, 0, xstride - 1);
      const int cy = std::clamp(y - y0, 0, ystride - 1);
      fnWeights[cx + cy * xstride] += wx[x] * wy[y];
    }
  }
}
} // anonymous namespace

GridSpatialXform::GridSpatialXform(const int rows, const int cols) {
  if (rows < 2 || cols < 2) {
    throw std::logic_error(
        "Need at least two rows and columns in depth transform grid.");
  }

  desc_.type = XformType::Spatial;
  desc_.gridSize.y() = rows;
  desc_.gridSize.x() = cols;

  const int numParams = desc_.gridSize.x() * desc_.gridSize.y() * 2;
  params_.resize(numParams, 0.0);

  for (int i = 0; i < desc_.gridSize.x() * desc_.gridSize.y(); ++i) {
    paramBlocks_.push_back(&params_[i * 2]);
    paramBlockSizes_.push_back(2);
  }
}

int GridSpatialXform::numDeformationCostResiduals() const {
  return desc_.gridSize.x() * desc_.gridSize.y() * 2;
}

void GridSpatialXform::computeDeformationCost(
    double const* const* params, double* residuals) const {
  paramsToResiduals(params, residuals, paramBlocks_.size(), 2);
}

void GridSpatialXform::computeDeformationCost(
    Jet const* const* params, Jet* residuals) const {
  paramsToResiduals(params, residuals, paramBlocks_.size(), 2);
}

//****************************************
//****                                ****
//****    BilinearGridSpatialXform    ****
//****                                ****
//****************************************

BilinearGridSpatialXform::BilinearGridSpatialXform(
    const int rows, const int cols)
    : GridSpatialXform(rows, cols) {
  desc_.spatialType = SpatialXformType::BilinearGrid;
}

std::unique_ptr<SpatialFunctor> BilinearGridSpatialXform::createFunctor(
    const Eigen::Vector2f& loc) const {
  std::vector<double*> params;
  std::vector<double> weights;
  bilinearSpatialGridGather(desc_, loc, paramBlocks_, params, weights);
  return std::make_unique<GridSpatialFunctor>(*this, params, weights);
};

//***************************************
//****                               ****
//****    BicubicGridSpatialXform    ****
//****                               ****
//***************************************

BicubicGridSpatialXform::BicubicGridSpatialXform(
    const int rows, const int cols)
    : GridSpatialXform(rows, cols) {
  desc_.spatialType = SpatialXformType::BicubicGrid;
}

std::unique_ptr<SpatialFunctor> BicubicGridSpatialXform::createFunctor(
    const Eigen::Vector2f& loc) const {
  std::vector<double*> params;
  std::vector<double> weights;
  bicubicSpatialGridGather(desc_, loc, paramBlocks_, params, weights);
  return std::make_unique<GridSpatialFunctor>(*this, params, weights);
};

//*********************************
//****                         ****
//****    General functions    ****
//****                         ****
//*********************************

std::unique_ptr<Xform> createXform(const XformDescriptor& desc) {
  if (desc.type == XformType::Depth) {
    return createDepthXform(desc);
  } else if (desc.type == XformType::Spatial) {
    return createSpatialXform(desc);
  } else {
    throw std::runtime_error("Invalid transform type.");
  }
}

std::unique_ptr<DepthXform> createDepthXform(const XformDescriptor& desc) {
  std::unique_ptr<DepthXform> xform;

  if (desc.type != XformType::Depth) {
    throw std::runtime_error("Transform has the wrong type.");
  }

  switch (desc.depthType) {
  case DepthXformType::Identity:
    xform = std::make_unique<IdentityDepthXform>();
    break;
  case DepthXformType::Global:
    xform = std::make_unique<GlobalDepthXform>(desc.valueXform);
    break;
  case DepthXformType::Grid:
    xform = std::make_unique<GridDepthXform>(
        desc.valueXform, desc.cubicInterpolation,
        desc.gridSize, desc.depthMinMax);
    break;
  default:
    throw std::runtime_error("Invalid depth transform type.");
  }

  return xform;
}

std::unique_ptr<SpatialXform> createSpatialXform(const XformDescriptor& desc) {
  std::unique_ptr<SpatialXform> xform;

  if (desc.type != XformType::Spatial) {
    throw std::runtime_error("Transform has the wrong type.");
  }

  switch (desc.spatialType) {
  case SpatialXformType::Identity:
    xform = std::make_unique<IdentitySpatialXform>();
    break;
  case SpatialXformType::VerticalLinear:
    xform = std::make_unique<VerticalLinearSpatialXform>();
    break;
  case SpatialXformType::CornersBilinear:
    xform = std::make_unique<CornersBilinearSpatialXform>();
    break;
  case SpatialXformType::BilinearGrid:
    xform = std::make_unique<BilinearGridSpatialXform>(
        desc.gridSize.y(), desc.gridSize.x());
    break;
  case SpatialXformType::BicubicGrid:
    xform = std::make_unique<BicubicGridSpatialXform>(
        desc.gridSize.y(), desc.gridSize.x());
    break;
  default:
    throw std::runtime_error("Invalid spatial transform type.");
  }

  return xform;
}

std::unique_ptr<Xform> readXform(std::istream& is) {
  XformDescriptor desc(is);
  std::unique_ptr<Xform> res = createXform(desc);
  is.read(
      reinterpret_cast<char*>(res->data()), sizeof(double) * res->numParams());
  return res;
}

void writeXform(std::ostream& os, const Xform& xform) {
  xform.desc().fwrite(os);
  os.write(
      reinterpret_cast<const char*>(xform.data()),
      sizeof(double) * xform.numParams());
}

}} // namespace facebook::cp
