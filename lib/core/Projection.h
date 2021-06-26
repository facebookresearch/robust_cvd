// Copyright 2004-present Facebook. All Rights Reserved.
//

#pragma once

#include <Eigen/Core>

namespace facebook {
namespace cp {
// Creates an OpenGL perspective projection matrix.
Eigen::Matrix4f perspectiveProjection(float nearPlane,
                                      float farPlane,
                                      float verticalFOVDegrees,
                                      float aspectRatio);

// Creates an OpenGL orthographic projection matrix.
Eigen::Matrix4f orthographicProjection(float left, float right,
                                       float bottom, float top,
                                       float nearVal, float farVal);

// Creates a "look at" OpenGL modelview matrix, similar to gluLookAt().
Eigen::Matrix4f lookAtMatrix(
    const Eigen::Vector3f& eye,
    const Eigen::Vector3f& up,
    const Eigen::Vector3f& lookAt);

// Creates a translation OpenGL modelview matrix, similar to glTranslatef().
Eigen::Matrix4f translationMatrix(const Eigen::Vector3f& offset);
}
}
