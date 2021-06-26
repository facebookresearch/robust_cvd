// Copyright 2004-present Facebook. All Rights Reserved.

#include <functional>
#include <fstream>
#include <vector>

namespace facebook {
namespace cp {

class Video3d;

class Photo3dReader {
 public:
  using ReadFn = std::function<void(const int, std::istream&)>;

  Photo3dReader(const std::string& fileName, ReadFn readFn);
  bool read(const int frame);

 private:
  ReadFn readFn_;
  std::ifstream stream_;
  std::vector<size_t> frameOffset_;
};

class Photo3dWriter {
 public:
  using WriteFn = std::function<void(const int, std::ostream&)>;

  Photo3dWriter(
      const std::string& fileName, WriteFn writeFn, const int lastFrame);
};

}} // namespace facebook::cp
