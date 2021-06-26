# @nolint
cmake_minimum_required(VERSION 3.3)

message(STATUS "Robust Consistent Video Depth Estimation")

project (lib_python)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_CPP_STANDARD -std=c++17)
set(PYBIND11_PYTHON_VERSION 3.6)
set(ENABLE_PRECOMPILED_HEADERS OFF)

find_path(GLOG_INCLUDE_DIR NAMES glog/logging.h)
find_library(GLOG_LIBRARY NAMES glog)
if(GLOG_INCLUDE_DIR AND GLOG_LIBRARY)
  message(STATUS "Found glog:")
  message(STATUS "  (Headers)       ${GLOG_INCLUDE_DIR}")
  message(STATUS "  (Library)       ${GLOG_LIBRARY}")
else()
  message(FATAL_ERROR "glog not found")
endif()

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(Ceres REQUIRED)
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
find_package(fmt REQUIRED)

message(STATUS "Pybind11 version: ${pybind11_VERSION}")
message(STATUS "Python   version: ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

include_directories(core ${GLOG_INCLUDE_DIR} ${GFLAGS_INCLUDE_DIR} ${CERES_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_library(lib_cvd STATIC)
file(GLOB SOURCES "*.cpp" "core/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "PythonBindings.cpp")
target_sources(lib_cvd PUBLIC ${SOURCES})
target_include_directories(lib_cvd
                       PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                       PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core
                       )

pybind11_add_module(lib_python PythonBindings.cpp)
target_link_libraries(lib_python PRIVATE lib_cvd fmt::fmt Ceres::ceres ${GLOG_LIBRARY} ${GFLAGS_LIBRARY} ${OpenCV_LIBS} Boost::filesystem Boost::system Boost::program_options Eigen3::Eigen pybind11::embed)
