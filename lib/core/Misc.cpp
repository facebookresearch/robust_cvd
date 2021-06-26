// Copyright 2004-present Facebook. All Rights Reserved.
//
// Author: Johannes Kopf (jkopf@fb.com)

#include "Misc.h"

#include <cassert>
// #include <re2/re2.h>
#include <sstream>

namespace facebook {
namespace cp {

// static const re2::RE2 expandEnvironmentVariablesRegex("\\$([^\\s]+)");

const char* yesno(const bool cond) {
  return (cond ? "yes" : "no");
}

const char* endisabled(const bool cond) {
  return (cond ? "enabled" : "disabled");
}

// Split a string using a character delimiter
std::vector<std::string> explode(const std::string& s, const char delimiter) {
  std::vector<std::string> res;
  std::istringstream is(s);

  for (std::string token; std::getline(is, token, delimiter);) {
    res.push_back(std::move(token));
  }

  return res;
}

// void expandEnvironmentVariables(std::string& s) {
//   // matches '$NAME', matches 0 captures '$NAME' and match 1 captures 'NAME'.
//   // avoid ASAN issue
//   std::string input = s;
//   re2::StringPiece piece(input);

//   std::string match;
//   while (re2::RE2::FindAndConsume(&piece, expandEnvironmentVariablesRegex, &match)) {
//     const char* env = getenv(match.c_str());
//     re2::RE2::GlobalReplace(&s, "\\$" + match, env);
//   }
// }

int argcXcodeUnmodified(int argc, char* argv[]) {
  assert(argc >= 0);

  if (argc == 0) {
    return argc;
  }

  int argcNew = 0;
  while (argcNew < argc) {
    if (strstr(argv[argcNew], "-NSDocumentRevisionsDebugMode")) {
      break;
    }
    ++argcNew;
  }

  assert(argcNew <= argc);

  return argcNew;
}

}} // namespace facebook::cp
