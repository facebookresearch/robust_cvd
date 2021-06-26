// Copyright 2004-present Facebook. All Rights Reserved.
//
// Author: Johannes Kopf (jkopf@fb.com)

#pragma once

#include <istream>
#include <string>
#include <vector>

#include "Debug.h"
#include "DetectStl.h"
#include "Platform.h"

#if defined (HAS_MAP) || defined (HAS_SET) || defined (HAS_UNORDERED_MAP)
#include <iterator>
#endif

namespace facebook {
namespace cp {

// Check for directory existence
bool dirExists(const char* path);
bool dirExists(const std::string& path);

// Checks for file existence
bool fileExists(const char* fileName);
bool fileExists(const std::string& fileName);

// Joins two paths which may or may not have trailing slashes '/'
std::string joinPaths(const std::string& path1, const std::string& path2);

// Extracts the file name from a path
std::string trimPath(const std::string& filePath);

// Extracts the base directory from a path
std::string trimFileName(const std::string& filePath);

// Removes the file extension from a path
std::string trimExtension(const std::string& filePath);

// Extracts the file extension from a path
std::string getExtension(const std::string& filePath);

// Create a directory.
// The perms parameter is an octal number representing the unix permissions
// mode, e.g. 774 = rwxrwxr--.
// Works even if multiple nested new directories need to be created.
// Does nothing if path already exists.
void createDirectory(const std::string& path, const int perms);

// Returns a list of all files in a directory.
void listDirectory(const std::string& path, std::vector<std::string>& files);

// Compares file names in format 'frame_%d.png'
bool compareFileNumbers(const std::string& first, const std::string& second);

// Read a value from a file
template<typename T> void fread(FILE* fin, T& value) {
  ensure(fread(&value, sizeof(T), 1, fin) == 1);
}

template<typename T> void read(std::istream& is, T& value) {
  is.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template<typename T> T fread(FILE* fin) {
  T temp;
  fread(fin, temp);
  return temp;
}

template<typename T> T read(std::istream& is) {
  T temp;
  read(is, temp);
  return temp;
}

// Read contents of a vector from a file
template<typename T> void freadv(FILE* fin, std::vector<T>& vec) {
  size_t len = fread<size_t>(fin);
  vec.resize(len);
  if (len > 0) {
    ensure(fread(&vec[0], sizeof(T), len, fin) == len);
  }
}

// Note: only use this if T is serializable as a binary chunk (no pointers)
template<typename T> void readv(std::istream& is, std::vector<T>& vec) {
  size_t len = read<size_t>(is);
  vec.resize(len);
  if (len > 0) {
    is.read(reinterpret_cast<char *>(&vec[0]), sizeof(T) * len);
  }
}

// Note: only use this if T is serializable as a binary chunk (no pointers)
template <typename T>
void readvv(std::istream& is, std::vector<std::vector<T>>& vecs) {
  size_t len = read<size_t>(is);
  vecs.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    std::vector<T> v;
    readv(is, v);
    vecs.push_back(std::move(v));
  }
}

// Read/write string from/to a file
std::string freadstr(FILE* fin);
std::string readstr(std::istream& is);
void fwritestr(FILE* fout, const std::string& s);
void writestr(std::ostream& os, const std::string& s);

// Write a value to a file
template<typename T> void fwrite(FILE* fout, const T& value) {
  ensure(fwrite(&value, sizeof(T), 1, fout) == 1);
}

template <typename T> void write(std::ostream& os, const T& value) {
  os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

// Write contents of a vector to a file
template<typename T> void fwritev(FILE* fout, const std::vector<T>& vec) {
  size_t len = vec.size();
  fwrite(fout, len);
  if (len > 0) {
    ensure(fwrite(&vec[0], sizeof(T), len, fout) == len);
  }
}

// Note: only use this if T is serializable as a binary chunk (no pointers)
template<typename T> void writev(std::ostream& os, const std::vector<T>& vec) {
  size_t len = vec.size();
  write(os, len);
  if (len > 0) {
    os.write(reinterpret_cast<const char*>(&vec[0]), sizeof(T) * len);
  }
}

// Note: only use this if T is serializable as a binary chunk (no pointers)
template <typename T>
void writevv(std::ostream& os, const std::vector<std::vector<T>>& vecs) {
  size_t len = vecs.size();
  write(os, len);
  for (const auto& vec : vecs) {
    writev(os, vec);
  }
}

// Write a sequence to a file, e.g., fwriteseq(fout, set.begin(), set.size()).
template<typename Iter>
void fwriteseq(FILE* fout, Iter it, const size_t size) {
  fwrite(fout, size);
  for (size_t i = 0; i < size; ++i) {
    fwrite(fout, *it++);
  }
}

// Read a sequence from a file, e.g., freadseq(fout, std::back_inserter(vec)).
template<typename Iter>
void freadseq(FILE* fin, Iter it) {
  size_t size = fread<size_t>(fin);
  for (size_t i = 0; i < size; ++i) {
    *it = fread<typename Iter::container_type::value_type>(fin);
    it++;
  }
}

#ifdef HAS_ARRAY
template<typename Type, size_t Count>
void fwriteseq(FILE* fout, const std::array<Type, Count>& a) {
  ensure(fwrite(a.data(), sizeof(Type), Count, fout) == Count);
}

template<typename Type, size_t Count>
void freadseq(FILE* fin, std::array<Type, Count>& a) {
  ensure(fread(a.data(), sizeof(Type), Count, fin) == Count);
}
#endif

#ifdef HAS_SET
template<typename T> void fwriteseq(FILE* fout, const std::set<T>& set) {
  fwriteseq(fout, set.begin(), set.size());
}

template<typename T> void freadseq(FILE* fin, std::set<T>& set) {
  set.clear();
  freadseq(fin, std::inserter(set, set.begin()));
}
#endif

#ifdef HAS_MAP
template<typename T1, typename T2>
void fwriteseq(FILE* fout, const std::map<T1, T2>& map) {
  fwriteseq(fout, map.begin(), map.size());
}

template<typename T1, typename T2>
void freadseq(FILE* fin, std::map<T1, T2>& map) {
  freadseq(fin, std::inserter(map, map.begin()));
}
#endif

#ifdef HAS_UNORDERED_MAP
template<typename T1, typename T2>
void fwriteseq(FILE* fout, const std::unordered_map<T1, T2>& map) {
  fwriteseq(fout, map.begin(), map.size());
}

template<typename T1, typename T2>
void freadseq(FILE* fin, std::unordered_map<T1, T2>& map) {
  freadseq(fin, std::inserter(map, map.begin()));
}
#endif

#ifdef EIGEN_CORE_H
template <typename T, int M, int N, int O>
void fwriteEigen(FILE* fout, const Eigen::Array<T, M, N, O>& a) {
  int count = a.rows() * a.cols();
  ensure(fwrite(a.data(), sizeof(T), count, fout) == count);
}

template <typename T, int M, int N, int O>
void writeEigen(std::ostream& os, const Eigen::Array<T, M, N, O>& a) {
  size_t size = sizeof(T) * a.rows() * a.cols();
  os.write(reinterpret_cast<const char*>(a.data()), size);
}

template <typename T, int M, int N, int O>
void fwriteEigen(FILE* fout, const Eigen::Matrix<T, M, N, O>& m) {
  int count = m.rows() * m.cols();
  ensure(fwrite(m.data(), sizeof(T), count, fout) == count);
}

template <typename T, int M, int N, int O>
void writeEigen(std::ostream& os, const Eigen::Matrix<T, M, N, O>& m) {
  size_t size = sizeof(T) * m.rows() * m.cols();
  os.write(reinterpret_cast<const char*>(m.data()), size);
}

template <typename T, int O>
void fwriteEigen(FILE* fout, const Eigen::Quaternion<T, O>& q) {
  fwriteEigen(fout, q.coeffs());
}

template <typename T, int O>
void writeEigen(std::ostream& os, const Eigen::Quaternion<T, O>& q) {
  writeEigen(os, q.coeffs());
}

template <typename T, int M, int N, int O>
void freadEigen(FILE* fin, Eigen::Array<T, M, N, O>& a) {
  int count = a.rows() * a.cols();
  ensure(fread(a.data(), sizeof(T), count, fin) == count);
}

template <typename T, int M, int N, int O>
void readEigen(std::istream& is, Eigen::Array<T, M, N, O>& a) {
  is.read(reinterpret_cast<char*>(a.data()), sizeof(T) * a.rows() * a.cols());
}

template <typename T, int M, int N, int O>
void freadEigen(FILE* fin, Eigen::Matrix<T, M, N, O>& m) {
  int count = m.rows() * m.cols();
  ensure(fread(m.data(), sizeof(T), count, fin) == count);
}

template <typename T, int M, int N, int O>
void readEigen(std::istream& is, Eigen::Matrix<T, M, N, O>& m) {
  is.read(reinterpret_cast<char*>(m.data()), sizeof(T) * m.rows() * m.cols());
}

template <typename T, int O>
void freadEigen(FILE* fin, Eigen::Quaternion<T, O>& q) {
  freadEigen(fin, q.coeffs());
}

template <typename T, int O>
void readEigen(std::istream& is, Eigen::Quaternion<T, O>& q) {
  readEigen(is, q.coeffs());
}
#endif

// Read all characters in a file.
bool freadAll(std::string& content, const char* fileName);
bool freadAll(std::string& content, const std::string& fileName);

}} // namespace facebook::cp
