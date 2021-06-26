// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#ifdef __APPLE__
    #include "TargetConditionals.h"
    #ifdef TARGET_OS_MAC
      #undef  GOOGLE_STRIP_LOG
      #define GOOGLE_STRIP_LOG 0
    #endif
#endif

#include <glog/logging.h>

namespace facebook {
namespace cp {

void initLogging(const char* argv0);
void logToStdout();
void logToStderr();

}
}
