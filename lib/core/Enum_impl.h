// Copyright 2004-present Facebook. All Rights Reserved.
//
// Author: Johannes Kopf (jkopf@fb.com)

// This header file is only supposed to be included in the translation units
// that

#pragma once

#include "Platform.h"

#if !PLATFORM_MOBILE
#include <boost/program_options.hpp>
#endif

#include "Enum.h"

namespace facebook {
namespace cp {

#if !PLATFORM_MOBILE
// Validate helper function for boost::program_options
template <typename T>
void validatex(boost::any& v, const std::vector<std::string>& values,
               const EnumStrings<T>& strs) {
  namespace po = boost::program_options;
  if (values.size() != 1) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  T value;
  if (!parseEnum(value, values[0], strs)) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  v = boost::any(value);
}

#define MAKE_VALIDATOR(enumType, enumStrs)                                     \
void validate(                                                                 \
    boost::any& v, const std::vector<std::string>& values, enumType*, int) {   \
  validatex(v, values, enumStrs);                                              \
}
#else
#define MAKE_VALIDATOR(enumType, enumStrs)
#endif

}} // namespace facebook::cp
