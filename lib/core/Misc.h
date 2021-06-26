// Copyright 2004-present Facebook. All Rights Reserved.
//
// Author: Johannes Kopf (jkopf@fb.com)

#pragma once

#include <algorithm>
#include <atomic>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "DetectStl.h"

namespace facebook {
namespace cp {

// Returns a string with either "yes" or "no"
const char* yesno(const bool cond);

// Returns a string with either "enabled" or "disabled"
const char* endisabled(const bool cond);

// Split a string using a character delimiter
std::vector<std::string> explode(const std::string& s, const char delimiter);

// // Expand environment variables in a string (using '$NAME' syntax)
// void expandEnvironmentVariables(std::string& s);

// XCode adds extra an command line option "-NSDocumentRevisionsDebugMode" by
// default. This returns a corrected argc: the number of arguments before this
// additional one.
int argcXcodeUnmodified(int argc, char* argv[]);

// Checks if a std::vector contains a certain element
#ifdef HAS_VECTOR
template<typename T>
bool contains(const std::vector<T>& vector, const T& el) {
  return std::find(vector.begin(), vector.end(), el) != vector.end();
}
#endif

// Checks if a std::set contains a certain element
#ifdef HAS_SET
template<typename T>
bool contains(const std::set<T>& set, const T& el) {
  return set.find(el) != set.end();
}
#endif

// Checks if a std::string contains a certain substring
#ifdef HAS_STRING
template<typename T>
bool contains(const std::string& str, const T& sub) {
  return str.find(sub) != str.npos;
}
#endif

// Checks if a std::map contains a certain key
#ifdef HAS_MAP
template<typename T1, typename T2>
bool contains(const std::map<T1, T2>& map, const T1& key) {
  return map.find(key) != map.end();
}
#endif

// Checks if a std::unordered_map contains a certain key
#ifdef HAS_UNORDERED_MAP
template<typename T1, typename T2, typename T3>
bool contains(const std::unordered_map<T1, T2, T3>& map, const T1& key) {
  return map.find(key) != map.end();
}
#endif

// Checks if a std::unordered_set contains a certain key
#ifdef HAS_UNORDERED_SET
template<typename T1, typename T2>
bool contains(const std::unordered_set<T1, T2>& set, const T1& key) {
  return set.find(key) != set.end();
}
#endif

// Returns true if the type of obj is Base or derived from Base.
template<typename Base, typename T>
inline bool instanceof(const T& obj) {
  return dynamic_cast<const Base*>(&obj) != nullptr;
}

// Determine whether two sets have a common element.
template <typename T>
bool hasCommonElement(const std::set<T>& s0, const std::set<T>& s1) {
  auto i0 = s0.begin();
  auto i1 = s1.begin();
  if (i0 == s0.end() || i1 == s1.end()) {
    return false;
  }

  for (;;) {
    const T& v0 = *i0;
    const T& v1 = *i1;

    if (v0 < v1) {
      ++i0;
      if (i0 == s0.end()) {
        return false;
      }
    } else if (v0 > v1) {
      ++i1;
      if (i1 == s1.end()) {
        return false;
      }
    } else {
      return true;
    }
  }
}

//
// Helper class for distributing tasks to multiple threads. It is
// effectively a thread-safe counter. Useful when worker threads are processing
// an indexed list of tasks. The thread that creates the task also would create
// an instance of WorkCounter constructed with the 'total' number of work items.
// The worker threads would receive a pointer to WorkCounter and as they process
// items they call 'Next()' for the next work item's 'index'. When no work is
// left to do 'Next()' will return false and the 'index' should be ignored.
//
// An optional feature is the EnterWorker method, this is used when a
// worker thread starts-up. It can query EnterWorker to obtain a unique
// 'worker index' for that thread. This should only be called once by
// and given worker thread.
//
class WorkCounter {
 public:
  explicit WorkCounter(int total)
      : total_(total), count_(0), workerIndex_(0), callback_(nullptr) {}

  WorkCounter(int total, void (*ondone)())
      : total_(total), count_(0), workerIndex_(0), callback_(ondone) {}

  int EnterWorker() {
    return workerIndex_.fetch_add(1);
  }

  bool Next(int& index) {
    index = count_.fetch_add(1);
    if (index == total_ && callback_ != nullptr) {
      callback_();
    }
    return index < total_;
  }

  int Total() const {
    return total_;
  }

 protected:
  int total_;
  std::atomic<int> count_;
  std::atomic<int> workerIndex_;
  void (*callback_)();
};
}
}
