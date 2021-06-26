// Copyright 2004-present Facebook. All Rights Reserved.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DepthMapTransform.h"
#include "ColorStream.h"
#include "DepthStream.h"
#include "DepthVideo.h"
#include "FlowConstraints.h"
#include "Importer.h"
#include "PoseOptimizer.h"
#include "Processor.h"
#include "core/Logging.h"


namespace py = pybind11;

// Custom type caster for converting numpy.ndarray (python) <-> cv::Mat (C++).
namespace pybind11 {
namespace detail{

template<>
struct type_caster<cv::Mat> {
 public:
  PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

  // Python -> C++: convert numpy.ndarray to cv::Mat.
  bool load(handle src, bool) {
    array b = reinterpret_borrow<array>(src);
    buffer_info info = b.request();

    int numChannels = 1;
    if (info.ndim == 2) {
    } else if (info.ndim == 3) {
      numChannels = info.shape[2];
    } else {
      throw std::runtime_error(
          "Can only convert 2D or 3D numpy.ndarray to cv::Mat.");
    }

    int dtype;
    if (info.format == format_descriptor<unsigned char>::format()) {
      dtype = CV_8UC(numChannels);
    } else if (info.format == format_descriptor<int>::format()) {
      dtype = CV_32SC(numChannels);
    } else if (info.format == format_descriptor<float>::format()) {
      dtype = CV_32FC(numChannels);
    } else if (info.format == format_descriptor<double>::format()) {
      dtype = CV_64FC(numChannels);
    } else {
      throw std::runtime_error(
          "Can only convert byte, int, float, and double numpy.ndarray"
          " to cv::Mat.");
    }

    value = cv::Mat(info.shape[1], info.shape[0], dtype, info.ptr);
    return true;
  }

  // C++ -> Python: convert cv::Mat to numpy.ndarray.
  static handle cast(const cv::Mat& mat, return_value_policy, handle) {
    std::string format = format_descriptor<unsigned char>::format();
    size_t elemSize = sizeof(unsigned char);
    int dim = (mat.depth() == mat.type())? 2 : 3;

    switch (mat.depth()) {
    case CV_8U:
      format = format_descriptor<unsigned char>::format();
      elemSize = sizeof(unsigned char);
      break;
    case CV_32S:
      format = format_descriptor<int>::format();
      elemSize = sizeof(int);
      break;
    case CV_32F:
      format = format_descriptor<float>::format();
      elemSize = sizeof(float);
      break;
    case CV_64F:
      format = format_descriptor<double>::format();
      elemSize = sizeof(double);
      break;
    default:
      throw std::runtime_error(
          "Can only convert byte, int, float, double cv::Mat"
          " to numpy.ndarray.");
    }

    std::vector<size_t> bufferDim;
    std::vector<size_t> strides;
    if (dim == 2) {
      bufferDim = {(size_t)mat.rows, (size_t)mat.cols};
      strides = {elemSize * (size_t)mat.cols, elemSize};
    } else if (dim == 3) {
      bufferDim = {(size_t)mat.rows, (size_t)mat.cols, (size_t)mat.channels()};
      strides = {
          (size_t)elemSize * mat.cols * mat.channels(),
          (size_t)elemSize * mat.channels(), (size_t)elemSize
      };
    }
    return array(buffer_info(
        mat.data, elemSize, format, dim, bufferDim, strides)).release();
  }
};

template<>
struct type_caster<cv::Mat1f> {
 public:
  PYBIND11_TYPE_CASTER(cv::Mat1f, _("numpy.ndarray"));

  // Python -> C++: convert numpy.ndarray to cv::Mat.
  bool load(handle src, bool) {
    array b = reinterpret_borrow<array>(src);
    buffer_info info = b.request();
    value = cv::Mat(info.shape[1], info.shape[0], CV_32FC(1), info.ptr);
    return true;
  }

  // C++ -> Python: convert cv::Mat to numpy.ndarray.
  static handle cast(const cv::Mat1f& mat, return_value_policy, handle) {
    std::string format = format_descriptor<float>::format();
    const size_t elemSize = sizeof(float);
    const int dim = 2;
    std::vector<size_t> bufferdim {(size_t)mat.rows, (size_t)mat.cols};
    std::vector<size_t> strides {elemSize * (size_t)mat.cols, elemSize};
    return array(buffer_info(
        mat.data, elemSize, format, dim, bufferdim, strides)).release();
  }
};

template<>
struct type_caster<cv::Mat2f> {
 public:
  PYBIND11_TYPE_CASTER(cv::Mat2f, _("numpy.ndarray"));

  // Python -> C++: convert numpy.ndarray to cv::Mat.
  bool load(handle src, bool) {
    array b = reinterpret_borrow<array>(src);
    buffer_info info = b.request();
    value = cv::Mat(info.shape[1], info.shape[0], CV_32FC(2), info.ptr);
    return true;
  }

  // C++ -> Python: convert cv::Mat to numpy.ndarray.
  static handle cast(const cv::Mat2f& mat, return_value_policy, handle) {
    std::string format = format_descriptor<float>::format();
    const size_t elemSize = sizeof(float);
    const int dim = 3;
    std::vector<size_t> bufferdim {
        (size_t)mat.rows, (size_t)mat.cols, (size_t)mat.channels()};
    std::vector<size_t> strides = {
          (size_t)elemSize * mat.cols * mat.channels(),
          (size_t)elemSize * mat.channels(), (size_t)elemSize};
    return array(buffer_info(
        mat.data, elemSize, format, dim, bufferdim, strides)).release();
  }
};

}} // pybind11::detail

namespace facebook {
namespace cp {

void initLib() {
  initLogging("");
}

PYBIND11_MODULE(lib_python, m) {
    m.def("initLib", &initLib);
    m.def("logToStdout", &logToStdout);

    py::class_<Eigen::Quaternionf>(m, "Quaternionf")
        .def("x", (float& (Eigen::Quaternionf::*) ()) &Eigen::Quaternionf::x)
        .def("y", (float& (Eigen::Quaternionf::*) ()) &Eigen::Quaternionf::y)
        .def("z", (float& (Eigen::Quaternionf::*) ()) &Eigen::Quaternionf::z)
        .def("w", (float& (Eigen::Quaternionf::*) ()) &Eigen::Quaternionf::w);

    py::class_<DepthPhoto::Extrinsics>(m, "Extrinsics")
        .def_readwrite("position", &DepthPhoto::Extrinsics::position)
        .def_readwrite("orientation", &DepthPhoto::Extrinsics::orientation)
        .def("left", &DepthPhoto::Extrinsics::left)
        .def("right", &DepthPhoto::Extrinsics::right)
        .def("down", &DepthPhoto::Extrinsics::down)
        .def("up", &DepthPhoto::Extrinsics::up)
        .def("forward", &DepthPhoto::Extrinsics::forward)
        .def("backward", &DepthPhoto::Extrinsics::backward)
        .def("worldToCamera", &DepthPhoto::Extrinsics::worldToCamera)
        .def("fromWorldToCamera", &DepthPhoto::Extrinsics::fromWorldToCamera);

    py::class_<DepthPhoto::Intrinsics>(m, "Intrinsics")
        .def_readwrite("vFov", &DepthPhoto::Intrinsics::vFov)
        .def_readwrite("hFov", &DepthPhoto::Intrinsics::hFov)
        .def_readwrite("centerLat", &DepthPhoto::Intrinsics::centerLat)
        .def_readwrite("centerLon", &DepthPhoto::Intrinsics::centerLon);

    py::class_<DepthVideoTrackTable>(m, "DepthVideoTrackTable")
        .def(py::init())
        .def("save", &DepthVideoTrackTable::save)
        .def("load", &DepthVideoTrackTable::load);

    /****************
    ****  Xform  ****
    ****************/

    m.def("computeDepthRange", &computeDepthRange);

    py::enum_<ValueXformType>(m, "ValueXformType")
        .value("None", ValueXformType::None)
        .value("Scale", ValueXformType::Scale)
        .value("ScaleShift", ValueXformType::ScaleShift);

    py::enum_<XformType>(m, "XformType")
        .value("Depth", XformType::Depth)
        .value("Spatial", XformType::Spatial);

    py::enum_<DepthXformType>(m, "DepthXformType")
        .value("None", DepthXformType::None)
        .value("Identity", DepthXformType::Identity)
        .value("Global", DepthXformType::Global)
        .value("Grid", DepthXformType::Grid);

    py::enum_<SpatialXformType>(m, "SpatialXformType")
        .value("None", SpatialXformType::None)
        .value("Identity", SpatialXformType::Identity)
        .value("VerticalLinear", SpatialXformType::VerticalLinear)
        .value("CornersBilinear", SpatialXformType::CornersBilinear)
        .value("BilinearGrid", SpatialXformType::BilinearGrid)
        .value("BicubicGrid", SpatialXformType::BicubicGrid);

    py::class_<XformDescriptor>(m, "XformDescriptor")
        .def_readwrite("type", &XformDescriptor::type)
        .def_readwrite("depthType", &XformDescriptor::depthType)
        .def_readwrite("spatialType", &XformDescriptor::spatialType)
        .def_readwrite("valueXform", &XformDescriptor::valueXform)
        .def_readwrite("gridSize", &XformDescriptor::gridSize)
        .def_readwrite("depthMinMax", &XformDescriptor::depthMinMax)
        .def("reset", &XformDescriptor::reset)
        .def("str", &XformDescriptor::str)
        .def("parse", &XformDescriptor::parse)
        .def("fread", &XformDescriptor::fread)
        .def("fwrite", &XformDescriptor::fwrite);

    py::class_<Xform>(m, "Xform")
        .def("clone", &Xform::clone)
        .def("copyFrom", &Xform::copyFrom)
        .def("desc", &Xform::desc)
        .def("str", &Xform::str)
        .def("params",
            (const std::vector<double>& (Xform::*)() const)&Xform::params)
        .def("numParams", &Xform::numParams);

    py::class_<DepthXform, Xform>(m, "DepthXform")
        .def("paramMap", &DepthXform::paramMap)
        .def("desc", &Xform::desc);

    py::class_<SpatialXform, Xform>(m, "SpatialXform")
        .def("warp", &SpatialXform::warp);

    /*********************
    ****  DepthVideo  ****
    *********************/

    py::class_<MetaFrame>(m, "MetaFrame")
        .def("pts", &MetaFrame::pts);

    py::class_<ColorFrame>(m, "ColorFrame")
        .def("image", &ColorFrame::image);

    py::class_<ColorStream>(m, "ColorStream")
        .def("frame",
            (ColorFrame& (ColorStream::*)(int))&ColorStream::frame,
            py::return_value_policy::reference)
        .def("name", &ColorStream::name)
        .def("path", &ColorStream::path)
        .def("extension", &ColorStream::extension)
        .def("width", &ColorStream::width)
        .def("height", &ColorStream::height)
        .def("setDir", &ColorStream::setDir);

    py::class_<DepthFrame>(m, "DepthFrame")
        .def("depth", &DepthFrame::depth)
        .def("sourceDepth", &DepthFrame::sourceDepth)
        .def("setDepth", &DepthFrame::setDepth)
        .def("warp", &DepthFrame::warp)
        .def("clear", &DepthFrame::clear)
        .def("clearCache", &DepthFrame::clearCache)
        .def("clearXformedCache", &DepthFrame::clearXformedCache)
        .def("depthXform",
            (DepthXform& (DepthFrame::*)())&DepthFrame::depthXform,
            py::return_value_policy::reference)
        .def("resetDepthXform", &DepthFrame::resetDepthXform)
        .def("spatialXform",
            (SpatialXform& (DepthFrame::*)())&DepthFrame::spatialXform,
            py::return_value_policy::reference)
        .def("resetSpatialXform", &DepthFrame::resetSpatialXform)
        .def_readwrite("intrinsics", &DepthFrame::intrinsics)
        .def_readwrite("extrinsics", &DepthFrame::extrinsics);

    py::class_<DepthStream>(m, "DepthStream")
        .def("frame",
            (DepthFrame& (DepthStream::*)(int))&DepthStream::frame,
            py::return_value_policy::reference)
        .def("name", &DepthStream::name)
        .def("path", &DepthStream::path)
        .def("depthXformDesc", &DepthStream::depthXformDesc)
        .def("spatialXformDesc", &DepthStream::spatialXformDesc)
        .def("width", &DepthStream::width)
        .def("height", &DepthStream::height)
        .def("setDir", &DepthStream::setDir)
        .def("resetDepthXforms", &DepthStream::resetDepthXforms)
        .def("resetSpatialXforms", &DepthStream::resetSpatialXforms)
        .def("clearCache", &DepthStream::clearCache);

    py::class_<DepthVideo>(m, "DepthVideo")
        .def(py::init())
        .def("printInfo", &DepthVideo::printInfo)
        .def("reset", &DepthVideo::reset)
        .def("load", &DepthVideo::load)
        .def("save", &DepthVideo::save)
        .def("width", &DepthVideo::width)
        .def("height", &DepthVideo::height)
        .def("aspect", &DepthVideo::aspect)
        .def("invAspect", &DepthVideo::invAspect)
        .def("path", &DepthVideo::path)
        .def("frame", &DepthVideo::frame)
        .def("numFrames", &DepthVideo::numFrames)
        .def("duration", &DepthVideo::duration)
        .def("timeToFrame", &DepthVideo::timeToFrame)
        .def("time", &DepthVideo::time)
        .def("numColorStreams", &DepthVideo::numColorStreams)
        .def("hasColorStream", &DepthVideo::hasColorStream)
        .def("colorStreamIndex", &DepthVideo::colorStreamIndex)
        .def("colorStream",
            (ColorStream& (DepthVideo::*)(int))&DepthVideo::colorStream,
            py::return_value_policy::reference)
        .def("colorStream",
            (ColorStream& (DepthVideo::*)(const std::string&))
                &DepthVideo::colorStream,
            py::return_value_policy::reference)
        .def("createColorStream", py::overload_cast<const std::string&, const std::string&, const std::string&, const int, const std::pair<int, int>&>(&DepthVideo::createColorStream), "",
            py::arg("name"), py::arg("dir"), py::arg("extension"), py::arg("type"), py::arg("size") = std::pair<int, int>{-1, -1})
        // .def("createColorStream", py::overload_cast<const std::string&, const std::string&, const GopTable&, const std::pair<int, int>&>(&DepthVideo::createColorStream), "",
        //     py::arg("name"), py::arg("dir"), py::arg("gopTable"), py::arg("size") = std::pair<int, int>{-1, -1})
        .def("colorFrame", &DepthVideo::colorFrame)
        .def("numDepthStreams", &DepthVideo::numDepthStreams)
        .def("hasDepthStream", &DepthVideo::hasDepthStream)
        .def("depthStreamIndex", &DepthVideo::depthStreamIndex)
        .def("depthStream",
            (DepthStream& (DepthVideo::*)(int))&DepthVideo::depthStream,
            py::return_value_policy::reference)
        .def("depthStream",
            (DepthStream& (DepthVideo::*)(const std::string&))
                &DepthVideo::depthStream,
            py::return_value_policy::reference)
        .def("createDepthStream", py::overload_cast<
            const std::string&, const std::string&, const std::pair<int, int>&>
            (&DepthVideo::createDepthStream), "",
            py::arg("name"), py::arg("dir"),
            py::arg("size") = std::pair<int, int>{-1, -1})
        .def("depthFrame", static_cast<
            const DepthFrame& (DepthVideo::*)(const int, const int) const>(
                &DepthVideo::depthFrame))
        .def("clearDepthCaches", &DepthVideo::clearDepthCaches)
        .def("saveDepth", &DepthVideo::saveDepth);

    /********************************
    ****  FlowConstraintsParams  ****
    ********************************/

    py::class_<FlowConstraintsParams>(m, "FlowConstraintsParams")
        .def(py::init())
        .def_readwrite("matchSeparation",
            &FlowConstraintsParams::matchSeparation)
        .def_readwrite("minDynamicDistance",
            &FlowConstraintsParams::minDynamicDistance)
        .def_readwrite("frameRange",
            &FlowConstraintsParams::frameRange)
        .def_readwrite("doNotUseCache",
            &FlowConstraintsParams::doNotUseCache);

    /************************************
    ****  FlowConstraintsCollection  ****
    ************************************/

    py::class_<FlowConstraintsCollection>(m, "FlowConstraintsCollection")
        .def(py::init<const DepthVideo&, const FlowConstraintsParams&>())
        .def("load", &FlowConstraintsCollection::load)
        .def("save", &FlowConstraintsCollection::save)
        .def("resetStaticFlag", &FlowConstraintsCollection::resetStaticFlag)
        .def("setStaticFlagFromDynamicMask",
            &FlowConstraintsCollection::setStaticFlagFromDynamicMask)
        .def("pruneStaticFlag", &FlowConstraintsCollection::pruneStaticFlag);

    /*****************************
    ****  DepthVideoImporter  ****
    *****************************/

    py::class_<DepthVideoImporter>(m, "DepthVideoImporter")
        .def("importVideo", &DepthVideoImporter::importVideo)
        .def("importColmapRecon", &DepthVideoImporter::importColmapRecon)
        .def("importColmapDepth", &DepthVideoImporter::importColmapDepth)
        .def("importPoses", &DepthVideoImporter::importPoses)
        .def("importTracks", &DepthVideoImporter::importTracks)
        .def("loadScale", &DepthVideoImporter::loadScale);

    /*********************
    ****  FrameRange  ****
    *********************/

    py::class_<FrameRange>(m, "FrameRange")
        .def(py::init())
        .def("fromString", &FrameRange::fromString)
        .def("toString", &FrameRange::toString)
        .def("resolve", &FrameRange::resolve)
        .def("isEmpty", &FrameRange::isEmpty)
        .def("firstFrame", &FrameRange::firstFrame)
        .def("lastFrame", &FrameRange::lastFrame)
        .def("count", &FrameRange::count)
        .def("isConsecutive", &FrameRange::isConsecutive)
        .def("inRange", &FrameRange::inRange)
        .def("checkEmpty", &FrameRange::checkEmpty);

    /**********************************
    ****  DepthVideoPoseOptimizer  ****
    **********************************/

    py::enum_<StaticLossType>(m, "StaticLossType")
        .value("Euclidean", StaticLossType::Euclidean)
        .value("ReproDisparity", StaticLossType::ReproDisparity)
        .value("ReproDepthRatio", StaticLossType::ReproDepthRatio)
        .value("ReproLogDepth", StaticLossType::ReproLogDepth);

    py::enum_<SmoothLossType>(m, "SmoothLossType")
        .value("EuclideanLaplacian", SmoothLossType::EuclideanLaplacian)
        .value("ReproDisparityLaplacian",
            SmoothLossType::ReproDisparityLaplacian)
        .value("ReproDepthRatioConsistency",
            SmoothLossType::ReproDepthRatioConsistency)
        .value("ReproLogDepthConsistency",
            SmoothLossType::ReproLogDepthConsistency);

    py::enum_<IntrinsicsOptimization>(m, "IntrinsicsOptimization")
        .value("Fixed", IntrinsicsOptimization::Fixed)
        .value("Shared", IntrinsicsOptimization::Shared)
        .value("PerFrame", IntrinsicsOptimization::PerFrame);

    py::class_<DepthVideoPoseOptimizer> dvpo(m, "DepthVideoPoseOptimizer");

    using DvpoParams = DepthVideoPoseOptimizer::Params;
    py::class_<DvpoParams>(dvpo, "Params")
        .def(py::init())
        .def_readwrite("frameRange", &DvpoParams::frameRange)
        .def_readwrite("maxIterations", &DvpoParams::maxIterations)
        .def_readwrite("numThreads", &DvpoParams::numThreads)
        .def_readwrite("numSteps", &DvpoParams::numSteps)
        .def_readwrite("robustness", &DvpoParams::robustness)
        .def_readwrite("staticLossType", &DvpoParams::staticLossType)
        .def_readwrite("staticSpatialWeight", &DvpoParams::staticSpatialWeight)
        .def_readwrite("staticDepthWeight", &DvpoParams::staticDepthWeight)
        .def_readwrite("smoothLossType", &DvpoParams::smoothLossType)
        .def_readwrite("smoothStaticWeight", &DvpoParams::smoothStaticWeight)
        .def_readwrite("smoothDynamicWeight", &DvpoParams::smoothDynamicWeight)
        .def_readwrite("positionReg", &DvpoParams::positionReg)
        .def_readwrite("scaleReg", &DvpoParams::scaleReg)
        .def_readwrite("scaleRegGridSize", &DvpoParams::scaleRegGridSize)
        .def_readwrite(
            "depthDeformRegInitial", &DvpoParams::depthDeformRegInitial)
        .def_readwrite(
            "depthDeformRegFinal", &DvpoParams::depthDeformRegFinal)
        .def_readwrite(
            "adaptiveDeformationCost", &DvpoParams::adaptiveDeformationCost)
        .def_readwrite("spatialDeformReg", &DvpoParams::spatialDeformReg)
        .def_readwrite(
            "graduateDepthDeformReg", &DvpoParams::graduateDepthDeformReg)
        .def_readwrite("focalReg", &DvpoParams::focalReg)
        .def_readwrite("coarseToFine", &DvpoParams::coarseToFine)
        .def_readwrite("ctfLong", &DvpoParams::ctfLong)
        .def_readwrite("ctfShort", &DvpoParams::ctfShort)
        .def_readwrite("deferredSpatialOpt", &DvpoParams::deferredSpatialOpt)
        .def_readwrite("dsoLong", &DvpoParams::dsoLong)
        .def_readwrite("dsoShort", &DvpoParams::dsoShort)
        .def_readwrite("focalLong", &DvpoParams::focalLong)
        .def_readwrite("intrOpt", &DvpoParams::intrOpt)
        .def_readwrite("fixPoses", &DvpoParams::fixPoses)
        .def_readwrite("fixDepthXforms", &DvpoParams::fixDepthXforms)
        .def_readwrite("fixSpatialXforms", &DvpoParams::fixSpatialXforms);

    /******************************
    ****  DepthVideoProcessor  ****
    ******************************/

    py::class_<DepthVideoProcessor> dvp(m, "DepthVideoProcessor");

    // DepthVideoProcessor::Params
    using DvpParams = DepthVideoProcessor::Params;
    py::class_<DvpParams>(dvp, "Params")
        .def(py::init())
        .def_readwrite("op", &DvpParams::op)
        .def_readwrite("frameRange", &DvpParams::frameRange)
        .def_readwrite("colorStream", &DvpParams::colorStream)
        .def_readwrite("depthStream", &DvpParams::depthStream)
        .def_readwrite("sourceDepthStream", &DvpParams::sourceDepthStream)
        .def_readwrite("spatialRadius", &DvpParams::spatialRadius)
        .def_readwrite("frameRadius", &DvpParams::frameRadius)
        .def_readwrite("depthSigma", &DvpParams::depthSigma)
        .def_readwrite("colorSigma", &DvpParams::colorSigma)
        .def_readwrite("median", &DvpParams::median)
        .def_readwrite("farConnections", &DvpParams::farConnections)
        .def_readwrite("matchSeparation", &DvpParams::matchSeparation)
        .def_readwrite("flowConsistancyThresh",
            &DvpParams::flowConsistancyThresh)
        .def_readwrite("trackSpawnDistance", &DvpParams::trackSpawnDistance)
        .def_readwrite("trackPruneDistance", &DvpParams::trackPruneDistance)
        .def_readwrite("minDynamicDistance", &DvpParams::minDynamicDistance)
        .def_readwrite("minTrackLength", &DvpParams::minTrackLength)
        .def_readwrite("depthXformDesc", &DvpParams::depthXformDesc)
        .def_readwrite("spatialXformDesc", &DvpParams::spatialXformDesc)
        .def_readwrite("poseOptimizer", &DvpParams::poseOptimizer);

    // DepthVideoProcessor::Op
    using Op = DepthVideoProcessor::Op;
    py::enum_<Op>(dvp, "Op")
        .value("None", Op::None)
        .value("Reset", Op::Reset)
        .value("Copy", Op::Copy)
        .value("BilateralFilter", Op::BilateralFilter)
        .value("FlowGuidedFilter", Op::FlowGuidedFilter)
        .value("ComputeConstraints", Op::ComputeConstraints)
        .value("ResetConstraintStaticFlag", Op::ResetConstraintStaticFlag)
        .value("SetConstraintStaticFlagFromDynamicMask",
            Op::SetConstraintStaticFlagFromDynamicMask)
        .value("ComputeTracks", Op::ComputeTracks)
        .value("GridXformSplit", Op::GridXformSplit)
        .value("ResetPoses", Op::ResetPoses)
        .value("ResetDepthXforms", Op::ResetDepthXforms)
        .value("ResetSpatialXforms", Op::ResetSpatialXforms)
        .value("NormalizeDepth", Op::NormalizeDepth)
        .value("OptimizePoses", Op::OptimizePoses)
        .value("ResetNormalizeOptimize", Op::ResetNormalizeOptimize);

    // DepthVideoProcessor
    dvp.def(py::init<DepthVideo* const>())
        .def("process", &DepthVideoProcessor::process)
        .def("reset", &DepthVideoProcessor::reset)
        .def("copy", &DepthVideoProcessor::copy)
        .def("bilateralFilter", &DepthVideoProcessor::bilateralFilter)
        .def("gridXformSplit", &DepthVideoProcessor::gridXformSplit)
        .def("resetPoses", &DepthVideoProcessor::resetPoses)
        .def("resetDepthXforms", &DepthVideoProcessor::resetDepthXforms)
        .def("resetSpatialXforms", &DepthVideoProcessor::resetSpatialXforms)
        .def("normalizeDepth", &DepthVideoProcessor::normalizeDepth)
        .def("optimizePoses", &DepthVideoProcessor::optimizePoses);
}

}} // namespace facebook::cp
