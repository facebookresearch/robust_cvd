// Copyright 2004-present Facebook. All Rights Reserved.

#include "ParamsBase.h"

#include <iostream> // NOLINT
#include <fstream>

#include "Platform.h"

#if !PLATFORM_MOBILE
namespace po = boost::program_options;
#endif

namespace facebook {
namespace cp {

ParamsBase::~ParamsBase() {
}

void ParamsBase::parseCommandLine(
    int argc, char* argv[], const std::string& defaultConfigFile) {
#if !PLATFORM_MOBILE
  try {
    desc_ = new po::options_description();
    addCommandLineOptions();

    std::string configFile;
    desc_->add_options()("config", po::value<std::string>(&configFile));

#ifdef __APPLE__
    // On Mac we swallow this default option added by Xcode.
    desc_->add_options()("NSDocumentRevisionsDebugMode", po::bool_switch());
#endif

    po::variables_map variableMap;
    int style =
        po::command_line_style::unix_style |
        po::command_line_style::allow_long_disguise;
    po::store(po::parse_command_line(argc, argv, *desc_, style), variableMap);

    po::notify(variableMap);

    if (configFile.empty()) {
      configFile = defaultConfigFile;
    }
    configFile_ = "";

    if (!configFile.empty()) {
      std::ifstream configStream(configFile);
      if (!configStream) {
        std::stringstream errorStream;
        errorStream << "Could not open config file '" << configFile << "'";
        throw std::runtime_error(errorStream.str());
      }

      std::cout << "Parsing config file '" << configFile << "'" << std::endl;
      po::store(po::parse_config_file(configStream, *desc_, true), variableMap);
      po::notify(variableMap);
      configFile_ = configFile; // save for later use, if desired (e.g., copy)
    }

    delete desc_;
    desc_ = nullptr;
  } catch (std::exception & e) {
    std::cerr << e.what() << std::endl;
  }
#endif
}

void ParamsBase::parseFile(const std::string& configFile) {
#if !PLATFORM_MOBILE
  try {
    desc_ = new po::options_description();
    addCommandLineOptions();

    po::variables_map variableMap;
    std::ifstream configStream(configFile);
    if (!configStream) {
      std::stringstream errorStream;
      errorStream << "Could not open config file '" << configFile << "'";
      throw std::runtime_error(errorStream.str());
    }

    std::cout << "Parsing config file '" << configFile << "'" << std::endl;
    po::store(po::parse_config_file(configStream, *desc_, true), variableMap);
    po::notify(variableMap);

    delete desc_;
    desc_ = nullptr;
  } catch (std::exception & e) {
    std::cerr << e.what() << std::endl;
  }
#endif
}

void ParamsBase::openStream(const std::string& dumpFile) {
  fileStream_ = new std::ofstream(dumpFile);
}

void ParamsBase::closeStream() {
  if (fileStream_ != nullptr) {
    fileStream_->close();
    delete fileStream_;
    fileStream_ = nullptr;
  }
}

std::ofstream* ParamsBase::fileStream_ = nullptr; // shared by all instances


}} // namespace facebook::cp
