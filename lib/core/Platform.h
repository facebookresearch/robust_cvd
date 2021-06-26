// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

// This header defines preprocessor variables depending on which platform it is
// compiled for. We currently support these defines:
//   PLATFORM_IPHONE   ---   iPhone
//   PLATFORM_ANDROID  ---   Android
//   PLATFORM_MAC      ---   Mac
//   PLATFORM_WINDOWS  ---   Windows
//   PLATFORM_LINUX    ---   Linux (e.g., devserver)
//   PLATFORM_MOBILE   ---   iPhone or Android

#ifdef __APPLE__
  #include "TargetConditionals.h"
  #if TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE
    #define PLATFORM_IPHONE 1
  #endif
  #if TARGET_OS_OSX
    #define PLATFORM_MAC 1
  #endif
#endif

#ifdef __ANDROID__
  #ifndef PLATFORM_ANDROID
    #define PLATFORM_ANDROID 1
  #endif
#endif

#ifdef _WIN32
  #define PLATFORM_WINDOWS 1
#endif

#ifdef __linux__
  #define PLATFORM_LINUX 1
#endif

#if PLATFORM_ANDROID || PLATFORM_IPHONE
  #define PLATFORM_MOBILE 1
#endif
