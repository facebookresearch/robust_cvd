// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <fstream>
#include <string>

#include <Eigen/Core>

#include "Platform.h"

#if !PLATFORM_MOBILE
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#endif

#include "Enum.h"
#include "Logging.h"

// Macros for adding options and printing parameters, to avoid some
// redundancies. For example, instead of writing
//    printParamIfNeq("frequency", frequency, defaultParams.frequency);
// you can simply write
//    PRINT_PARAM_IF_NEQ(frequency)
#define ADD_OPTION(PARAM) addOption(#PARAM, &PARAM);
#define PRINT_PARAM(PARAM) printParam(#PARAM, PARAM);
#define PRINT_PARAM_IF_NEQ(PARAM) \
    printParamIfNeq(#PARAM, PARAM, defaultParams.PARAM);

namespace facebook {
namespace cp {

// This is a base class for parameter structs that can be parsed from command
// line arguments and/or config files. It builds on boost::program_options.
//
// This example shows how to use the class:
//
// class MyParams : public ParamsBase {
//  public:
//
//   int someParam;
//   std::string anotherParam;
//   NestedParams nestedParams;
//
//   void addCommandLineOptions() override;
//   void printParams() const override;
// };
//
// void MyParams::addCommandLineOptions() {
//   addOption("someParam", &someParam);
//   addOption("anotherParam", &anotherParam);
//
//   // You can also parse nested parameter structs. You can optionally specify
//   // a prefix here for the parameters in the nested struct.
//   addSubParamsOptions(nested, "prefix");
// }
//
// void MyParams::printParams() const {
//   printParam("someParam", someParam);
//   printParam("anotherParam", anotherParam);
//   nested.printParams();
// }
//
// To read in parameter values from a config file and command-like arguments,
// use
//    DerivedParams params;
//    params.parseCommandLine(argc, argv);
//
// To print a set of parameters, use
//    params.printParams()
// or to print to a file, use
//    params.openStream(sparseRecon.outputPath("paramsLog.txt"));
//    params.printParams();
//    params.closeStream();

class ParamsBase {
 public:
  virtual ~ParamsBase();

  virtual void addCommandLineOptions() = 0;

  virtual void printParams() const = 0;

  virtual void parseCommandLine(
      int argc,
      char* argv[],
      const std::string& defaultConfigFile = "");

  void parseFile(const std::string& configFile);

  void setPrefix(const std::string& prefix) {
    if (prefix.empty()) {
      prefix_.clear();
    } else {
      prefix_ = prefix + ".";
    }
  }

  std::string configFile() const {
    return configFile_;
  }

  void openStream(const std::string& dumpFile);

  void closeStream();

protected:
  template <typename T>
  void addOption(const std::string& key, T * value) {
#if !PLATFORM_MOBILE
    desc_->add_options()((prefix_ + key).c_str(),
                         boost::program_options::value<T>(value));
#endif
  }

  template <typename T>
  void printParam(
      const std::string& key,
      const T& value,
      const EnumStrings<T>& strs) const {
    if (fileStream_ && fileStream_->is_open()) {
      *fileStream_ << prefix_ + key << ": " << enumToString(value, strs)
        << "\n";
    } else {
      LOG(INFO) << prefix_ + key << ": " << enumToString(value, strs);
    }
  }

  template <typename T>
  void printParamIfNeq(
      const std::string& key,
      const T& value, const T& compare,
      const EnumStrings<T>& strs) const {
    if (value != compare) {
      printParam(key, value, strs);
    }
  }

  template <typename T>
  void printParam(const std::string& key, const T& value) const {
    if (fileStream_ && fileStream_->is_open()) {
      *fileStream_ << prefix_ + key << ": " << value << "\n";
    } else {
      LOG(INFO) << prefix_ + key << ": " << value;
    }
  }

  template <typename T>
  void printParamIfNeq(
      const std::string& key, const T& value, const T& compare) const {
    if (value != compare) {
      printParam(key, value);
    }
  }

  template <typename T>
  void printParam(const std::string& key, const std::vector<T>& vec) const {
    for (size_t i = 0; i < vec.size(); ++i) {
      if (fileStream_ && fileStream_->is_open()) {
        *fileStream_ << prefix_ + key << "[" << i << "]: " << vec[i] << "\n";
      } else {
        LOG(INFO) << prefix_ + key << "[" << i << "]: " << vec[i];
      }
    }
  }

  template <typename T, int M, int N>
  void printParam(
      const std::string& key, const Eigen::Matrix<T, M, N>& mat) const {
    std::string firstRowPrefix = prefix_ + key + ": (";
    std::string otherRowPrefix = std::string(firstRowPrefix.length(), ' ');

    std::stringstream ss;
    if (M == 1 || N == 1) {
      ss << firstRowPrefix;
      for (int i = 0; i < std::max(M, N); ++i) {
        if (i > 0) {
          ss << ", ";
        }
        ss << mat(i);
      }
      ss << ")";
    } else {
      for (int m = 0; m < M; ++m) {
        ss << (m == 0 ? firstRowPrefix : otherRowPrefix);
        for (int n = 0; n < N; ++n) {
          if (n > 0) {
            ss << ", ";
          }
          ss << mat(m, n);
        }
        ss << (m < M - 1 ? ",\n" : ")");
      }
    }

    LOG(INFO) << ss.str();
  }

  void addSubParamsOptions(
      ParamsBase& subParams,
      const std::string& prefix = "") {
    subParams.setPrefix(prefix_ + prefix);

#if !PLATFORM_MOBILE
    CHECK(subParams.desc_ == nullptr);
    subParams.desc_ = desc_;
    subParams.addCommandLineOptions();
    subParams.desc_ = nullptr;
#endif
  }

 private:
  std::string prefix_;
#if !PLATFORM_MOBILE
  boost::program_options::options_description* desc_ = nullptr;
#endif
  std::string configFile_;
  static std::ofstream* fileStream_; // shared by all instances
};

#if !PLATFORM_MOBILE
// This macro returns a default location for a configuration file: a file named
// 'config.txt' in the directory of the file the macro is expanded in.
// Typically, this is used in a line in Main.cpp such as
//     params.parseCommandLine(argc, argv, DEFAULT_CONFIG_FILE);
#define DEFAULT_CONFIG_FILE                                                    \
    (boost::filesystem::path(__FILE__).parent_path().append("config.txt")).    \
        string()
#else
#define DEFAULT_CONFIG_FILE std::string()
#endif

}} // namespace facebook::cp
