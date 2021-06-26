// Copyright 2004-present Facebook. All Rights Reserved.
//
// Author: Johannes Kopf (jkopf@fb.com)

#pragma once

// This macro is useful for swallowing "unused variable" warnings in production
// builds. Consider this example:
//
//   const float epsilon = 0.001;
//   assert(x < epsilon);
//   _unused(epsilon);
//
// The variable 'epsilon' is only used for checking in an assert statement.
// Without the _unused(epsilon) statement the compiler would warn about epsilon
// being unsused in release builds.
#define _unused(x) ((void)(x))

namespace facebook {
namespace cp {

// Prints an error message and halts the program
// TODO: create error functions with format string arguments
void error(const char* msg);
void error();

// Call error if the condition is true
void ensure(bool cond);
void ensure(const bool cond, const char* message);

}} // namespace facebook::cp
