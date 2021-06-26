// Copyright 2004-present Facebook. All Rights Reserved.

#include "Logging.h"

// Google glog's api does not have an external function that allows one to check
// if glog is initialized or not. It does have an internal function - so we are
// declaring it here. This is a hack but has been used by a bunch of others too
// (e.g. https://fburl.com/579584489).
namespace google {
namespace glog_internal_namespace_ {
bool IsGoogleLoggingInitialized();
} // namespace glog_internal_namespace_
} // namespace google

namespace facebook {
namespace cp {

void initLogging(const char* argv0) {
#ifdef WIN32
  google::InitGoogleLogging(argv0);
#else
  if (!google::glog_internal_namespace_::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(argv0);
  }
#endif
  FLAGS_minloglevel = 0;
  FLAGS_log_prefix = 0;

  logToStderr();
}

void logToStdout() {
  FLAGS_logtostderr = 0;
}

void logToStderr() {
  FLAGS_logtostderr = 1;
}

}
}
