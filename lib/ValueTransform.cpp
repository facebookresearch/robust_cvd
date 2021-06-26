// Copyright 2004-present Facebook. All Rights Reserved.

#include "ValueTransform.h"

#include "core/Enum_impl.h"

namespace facebook {
namespace cp {

const EnumStrings<ValueXformType> valueXformStrs = {
    {ValueXformType::None, "None"},
    {ValueXformType::Scale, "Scale"},
    {ValueXformType::ScaleShift, "ScaleShift"}};
MAKE_VALIDATOR(ValueXformType, valueXformStrs);

const ValueXform& ValueXform::getInstance(const ValueXformType& type) {
  if (type == ValueXformType::Scale) {
    static ScaleXform scaleXform;
    return scaleXform;
  } else if (type == ValueXformType::ScaleShift) {
    static ScaleShiftXform scaleShiftXform;
    return scaleShiftXform;
  } else {
    throw std::runtime_error("Invalid value transform.");
  }
}

}} // namespace facebook::cp
