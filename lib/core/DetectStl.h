// Copyright 2004-present Facebook. All Rights Reserved.
//
// Author: Johannes Kopf (jkopf@fb.com)

#pragma once

// Detect which STL headers have been included before this file by the user.
// This is useful for selectively defining some functionality.
#if defined(_LIBCPP_ARRAY) || defined (_GLIBCXX_ARRAY) || defined(_ARRAY_)
  #define HAS_ARRAY
#endif
#if defined(_LIBCPP_MAP) || defined (_GLIBCXX_MAP) || defined(_MAP_)
  #define HAS_MAP
#endif
#if defined(_LIBCPP_SET) || defined (_GLIBCXX_SET) || defined(_SET_)
  #define HAS_SET
#endif
#if defined(_LIBCPP_STRING) || defined (_GLIBCXX_STRING) || defined(_STRING_)
  #define HAS_STRING
#endif
#if defined(_LIBCPP_VECTOR) || defined (_GLIBCXX_VECTOR) || defined(_VECTOR_)
  #define HAS_VECTOR
#endif
#if (defined(_LIBCPP_UNORDERED_MAP) || defined (_GLIBCXX_UNORDERED_MAP) ||     \
     defined(_UNORDERED_MAP_))
  #define HAS_UNORDERED_MAP
#endif
#if (defined(_LIBCPP_UNORDERED_SET) || defined (_GLIBCXX_UNORDERED_SET) ||     \
     defined(_UNORDERED_SET_))
  #define HAS_UNORDERED_SET
#endif
