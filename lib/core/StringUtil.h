// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <algorithm>
#include <cctype>
#include <locale>

namespace facebook {
namespace cp {

// Trims whitespace from start in place.
static inline void ltrim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
      return !std::isspace(ch);
  }));
}

// Trims whitespace from end in place.
static inline void rtrim(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
      return !std::isspace(ch);
  }).base(), s.end());
}

// Trims whitespace from both ends in place.
static inline void trim(std::string& s) {
  ltrim(s);
  rtrim(s);
}

// Converts string to lower case using ::tolower on each character in place.
static inline void toLowerCase(std::string& s) {
  std::for_each(s.begin(), s.end(), [](char& c) { c = ::tolower(c); });
}

} // namespace cp
} // namespace facebook
