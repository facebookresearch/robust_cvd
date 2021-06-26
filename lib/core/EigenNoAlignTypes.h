// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <Eigen/Core>

namespace facebook {
namespace cp {

using Vector2fna = Eigen::Matrix<float, 2, 1, Eigen::DontAlign>;
using Vector3fna = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Vector4fna = Eigen::Matrix<float, 4, 1, Eigen::DontAlign>;

using Vector2dna = Eigen::Matrix<double, 2, 1, Eigen::DontAlign>;
using Vector3dna = Eigen::Matrix<double, 3, 1, Eigen::DontAlign>;
using Vector4dna = Eigen::Matrix<double, 4, 1, Eigen::DontAlign>;

using Vector2ina = Eigen::Matrix<int, 2, 1, Eigen::DontAlign>;
using Vector3ina = Eigen::Matrix<int, 3, 1, Eigen::DontAlign>;
using Vector4ina = Eigen::Matrix<int, 4, 1, Eigen::DontAlign>;

using Vector2una = Eigen::Matrix<unsigned int, 2, 1, Eigen::DontAlign>;
using Vector3una = Eigen::Matrix<unsigned int, 3, 1, Eigen::DontAlign>;
using Vector4una = Eigen::Matrix<unsigned int, 4, 1, Eigen::DontAlign>;

using Vector2ucna = Eigen::Matrix<unsigned char, 2, 1, Eigen::DontAlign>;
using Vector3ucna = Eigen::Matrix<unsigned char, 3, 1, Eigen::DontAlign>;
using Vector4ucna = Eigen::Matrix<unsigned char, 4, 1, Eigen::DontAlign>;

using Matrix2fna = Eigen::Matrix<float, 2, 2, Eigen::DontAlign>;
using Matrix3fna = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>;
using Matrix4fna = Eigen::Matrix<float, 4, 4, Eigen::DontAlign>;

using Matrix2dna = Eigen::Matrix<double, 2, 2, Eigen::DontAlign>;
using Matrix3dna = Eigen::Matrix<double, 3, 3, Eigen::DontAlign>;
using Matrix4dna = Eigen::Matrix<double, 4, 4, Eigen::DontAlign>;

using Quaternionfna = Eigen::Quaternion<float, Eigen::DontAlign>;
using Quaterniondna = Eigen::Quaternion<double, Eigen::DontAlign>;

}} // namespace facebook::cp
