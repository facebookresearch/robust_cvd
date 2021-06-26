// Copyright 2004-present Facebook. All Rights Reserved.
//
// Author: Johannes Kopf (jkopf@fb.com)

#pragma once

#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// #include <folly/lang/Exception.h>

#include "Debug.h"
#include "Logging.h"
#include "StringUtil.h"

namespace boost {
class any;
}

namespace facebook {
namespace cp {

// This macro can be used in enum string definitions. For example, instead of
//
// const EnumStrings<StaticLossType> staticLossStrs = {
//     {StaticLossType::Euclidean, "Euclidean"},
//     {StaticLossType::ReprojectionDisparity, "ReprojectionDisparity"},
//     {StaticLossType::ReprojectionLogDepth, "ReprojectionLogDepth"},
//     {StaticLossType::ReprojectionDepthRatio, "ReprojectionDepthRatio"},
// };
//
// you can write shorter
//
// const EnumStrings<StaticLossType> staticLossStrs = {
//     ENUM_STR(StaticLossType, Euclidean),
//     ENUM_STR(StaticLossType, ReprojectionDisparity),
//     ENUM_STR(StaticLossType, ReprojectionLogDepth),
//     ENUM_STR(StaticLossType, ReprojectionDepthRatio),
// };
#define ENUM_STR(name, value) {name::value, #value}


// This macro automatically creates an enum, in particular one that can be
// used by ParamsBase and parsed from the command line.
// It also extends from EnumBase and has additional helper methods.
// Usage: GENERATE_ENUM(EnumName, A, B, C, D)
#define GENERATE_ENUM(EnumName, ...) \
  enum class EnumName : unsigned int { __VA_ARGS__ };
// TODO: Auto generate the enumStrs

// Type for a pair of an enum value and its string representation
template <typename T>
using EnumStrings = std::vector<std::pair<T, const char*>>;

// Return a string representation of an enum value
template <typename T>
std::string enumToString(const T val, const EnumStrings<T>& strs) {
  for (auto& pair : strs) {
    if (val == pair.first) {
      return pair.second;
    }
  }

  // // Enum value was not found, throw an exception
  // folly::throw_exception<std::runtime_error>(
  //     "Value not found in enum string representation.");
  std::cerr << "Value not found in enum string representation.";

  return "";
}

// Parse enum string represetation, returns true and enum value if found, and
// false otherwise
template <typename T>
bool parseEnum(T& res, const std::string& s, const EnumStrings<T>& strs) {
  for (auto& pair : strs) {
    if (s == pair.second) {
      // Found the enum value
      res = pair.first;
      return true;
    }
  }

  // Did not find the enum value
  return false;
}

// Parse enum string representation, throws if value not found in list.
template <typename T>
T parseEnum(const std::string& s, const EnumStrings<T>& strs) {
  T res;
  if (!parseEnum(res, s, strs)) {
    // folly::throw_exception<std::runtime_error>("Invalid enum value.");
    std::cerr << "Invalid enum value.";
  }
  return res;
}

#if !PLATFORM_MOBILE
#define DECLARE_VALIDATOR(enumType, enumStrs)                                  \
  void validate(                                                               \
      boost::any& v, const std::vector<std::string>& values, enumType*, int);
#else
#define DECLARE_VALIDATOR(enumType, enumStrs)
#endif

// Used to parse arbitrary enums with unsigned int as underlying type from
// strings.
class EnumDescription {
 public:
  typedef std::map<std::string, unsigned int> EnumMapping;
  typedef std::map<unsigned int, std::string> ReverseEnumMapping;

  EnumDescription(unsigned int* valuePtr, const EnumMapping& enumMapping)
      : valuePtr_(valuePtr), enumMapping_(enumMapping) {
    for (std::pair<std::string, int> element : enumMapping) {
      reverseEnumMapping_[element.second] = element.first;
    }
  };

  bool fromString(const std::string& strValue) {
    auto it = enumMapping_.find(strValue);
    if (it == enumMapping_.end()) {
      return false;
    }
    *valuePtr_ = enumMapping_[strValue];
    return true;
  }

  std::string toString() const {
    return reverseEnumMapping_.at(*valuePtr_);
  }

  unsigned int underlyingValue() {
    return *valuePtr_;
  }

 private:
  unsigned int* valuePtr_;
  EnumMapping enumMapping_;
  ReverseEnumMapping reverseEnumMapping_;
};

inline std::istream& operator>>(std::istream& stream, EnumDescription& e) {
  std::string str;
  if (!(stream >> str)) {
    return stream;
  }
  if (!e.fromString(str)) {
    stream.setstate(stream.rdstate() | std::ios::failbit);
  }
  return stream;
}

inline std::ostream& operator<<(std::ostream& os, const EnumDescription& e) {
  os << e.toString();
  return os;
}
} // namespace cp
} // namespace facebook
