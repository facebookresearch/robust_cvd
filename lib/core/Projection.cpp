// Copyright 2004-present Facebook. All Rights Reserved.

#include "Projection.h"

#include "MathUtil.h"

using namespace Eigen;

namespace facebook {
namespace cp {

Matrix4f perspectiveProjection(
    float nearPlane, float farPlane,
    float verticalFOVDegrees, float aspectRatio) {
  float verticalFOVRadians = verticalFOVDegrees  * M_PI_F / 180.0f;
  float nearPlaneHeight    = 2.0f * tanf(verticalFOVRadians * 0.5f) * nearPlane;

  float sx = 2.0f * nearPlane / (nearPlaneHeight * aspectRatio);
  float sy = 2.0f * nearPlane / (nearPlaneHeight);
  float sz = -(farPlane + nearPlane) / (farPlane - nearPlane);
  float pz = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);

  Matrix4f projection;
  projection <<   sx, 0.0f,  0.0f, 0.0f,
                0.0f,   sy,  0.0f, 0.0f,
                0.0f, 0.0f,    sz,   pz,
                0.0f, 0.0f, -1.0f, 0.0f;

  return projection;
}

Matrix4f orthographicProjection(
    float left, float right, float bottom, float top,
    float nearVal, float farVal) {
  float xdiff = right - left;
  float ydiff = top - bottom;
  float zdiff = farVal - nearVal;
  Matrix4f projection;
  projection << 2.f / xdiff, 0.f, 0.f, -(right + left) / xdiff,
                0.f, 2.f / ydiff, 0.f, -(top + bottom) / ydiff,
                0.f, 0.f, -2.f / zdiff, -(farVal + nearVal) / zdiff,
                0.f, 0.f, 0.f, 1.f;

  return projection;
}

Matrix4f lookAtMatrix(
    const Vector3f& eye, const Vector3f& up, const Vector3f& lookAt) {
    const Vector3f forward = (lookAt - eye).normalized();
    const Vector3f side = forward.cross(up).normalized();
  const Vector3f upCorr = side.cross(forward);

  Matrix4f res;
  res << side.x(), side.y(), side.z(), -side.dot(eye),
         upCorr.x(), upCorr.y(), upCorr.z(), -upCorr.dot(eye),
         -forward.x(), -forward.y(), -forward.z(), forward.dot(eye),
         0.f, 0.f, 0.f, 1.f;
  return res;
}

Matrix4f translationMatrix(const Vector3f& offset) {
  Matrix4f res;
  res << 1.f, 0.f, 0.f, offset.x(),
         0.f, 1.f, 0.f, offset.y(),
         0.f, 0.f, 1.f, offset.z(),
         0.f, 0.f, 0.f, 1.f;
  return res;
}
}} // namespace facebook::cp
