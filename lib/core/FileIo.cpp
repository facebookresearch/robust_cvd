// Copyright 2004-present Facebook. All Rights Reserved.

#include "FileIo.h"

#include <algorithm>
#include <string>
#if PLATFORM_WINDOWS
#include <direct.h>
#endif
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <boost/format.hpp>

namespace facebook {
namespace cp {

bool dirExists(const char* path) {
  struct stat info;

  if (stat(path, &info)>=0 && info.st_mode & S_IFDIR) {
    return true;
  } else {
    return false;
  }
}

bool dirExists(const std::string& path) {
  return dirExists(path.c_str());
}

bool fileExists(const char* fileName) {
  FILE * f = fopen(fileName, "r");
  if (!f) {
    return false;
  } else {
    fclose(f);
    return true;
  }
}

bool fileExists(const std::string& fileName) {
  return fileExists(fileName.c_str());
}

std::string joinPaths(const std::string& path1, const std::string& path2) {
  return path1 + (path1[path1.length() - 1] == '/' ? "" : "/") + path2;
}

std::string trimPath(const std::string& filePath) {
  size_t ofs = 0;

  size_t pos = filePath.find_last_of('/');
  if (pos != filePath.npos) { ofs = pos+1; }

  pos = filePath.find_last_of('\\');
  if (pos != filePath.npos) { ofs = std::max(ofs, pos+1); }

  if (ofs == 0) {
    return filePath;
  } else {
    return filePath.substr(ofs);
  }
}

std::string trimFileName(const std::string& filePath) {
  size_t ofs = 0;

  size_t pos = filePath.find_last_of('/');
  if (pos != filePath.npos) {
    ofs = std::max(ofs, pos);
  }

  pos = filePath.find_last_of('\\');
  if (pos != filePath.npos) {
    ofs = std::max(ofs, pos);
  }

  return filePath.substr(0, ofs);
}

std::string trimExtension(const std::string& filePath) {
  size_t pos = filePath.find_last_of('.');
  if (pos == filePath.npos) {
    return filePath;
  } else {
    return filePath.substr(0, pos);
  }
}

std::string getExtension(const std::string& filePath) {
  size_t pos = filePath.find_last_of('.');
  if (pos == filePath.npos) {
    return "";
  } else {
    return filePath.substr(pos+1);
  }
}

void createDirectory(const std::string& path, const int perms) {
#if PLATFORM_WINDOWS
  for (int i = 0; i < (int)path.length(); ++i) {
    if (path[i] == '/' || path[i] == '\\') {
      _mkdir(path.substr(0, i).c_str());
    }
  }

  _mkdir(path.c_str());
#else
  if (perms < 0 || perms > 0777) {
    error("invalid permissions");
  }

  mode_t mode =
    (perms&0400?S_IRUSR:0) | (perms&0200?S_IWUSR:0) | (perms&0100?S_IXUSR:0) |
    (perms&0040?S_IRGRP:0) | (perms&0020?S_IWGRP:0) | (perms&0010?S_IXGRP:0) |
    (perms&0004?S_IROTH:0) | (perms&0002?S_IWOTH:0) | (perms&0001?S_IXOTH:0);

  for (int i = 0; i < (int)path.length(); ++i) {
    if (path[i] == '/') {
      mkdir(path.substr(0, i).c_str(), mode);
    }
  }

  mkdir(path.c_str(), mode);
#endif
}

//#if !PLATFORM_WINDOWS
void listDirectory(const std::string& path, std::vector<std::string>& files) {
  DIR * dir = opendir(path.c_str());

  while (dir) {
    dirent * ent;
    if ((ent = readdir(dir)) != nullptr) {
      std::string fileName = ent->d_name;
      if (fileName != "." && fileName != "..") {
        files.push_back(fileName);
      }
    } else {
      closedir(dir);
      dir = nullptr;
    }
  }
}
//#endif

std::string freadstr(FILE * fin) {
  size_t len = fread<size_t>(fin);
  std::string res(len, char(0));
  if (len > 0) {
    ensure(fread(&res[0], 1, len, fin) == len);
  }
  return res;
}

std::string readstr(std::istream& is) {
  size_t len = read<size_t>(is);
  std::string res(len, char(0));
  if (len > 0) {
    is.read(&res[0], len);
  }
  return res;
}

void fwritestr(FILE * fout, const std::string& s){
  size_t len = s.size();
  fwrite(fout, len);
  if (len > 0) {
    ensure(fwrite(&s[0], 1, len, fout) == len);
  }
}

void writestr(std::ostream& os, const std::string& s) {
  size_t len = s.size();
  write(os, len);
  if (len > 0) {
    os.write(&s[0], len);
  }
}

bool freadAll(std::string& str, const char* fileName) {
  FILE * fin = fopen(fileName, "rb");
  if (!fin) { return false; }

  if (fseek(fin, 0, SEEK_END) != 0) { error(); }

  size_t len = ftell(fin);
  str.resize(len+1);

  if (fseek(fin, 0, SEEK_SET) != 0) { error(); }

  if (fread(&str[0], 1, len, fin) != len) { error(); }
  str[len] = 0;

  fclose(fin);

  return true;
}

bool freadAll(std::string& str, const std::string& fileName) {
  return freadAll(str, fileName.c_str());
}

bool compareFileNumbers(const std::string& first, const std::string& second) {
  // Supports file names that look like:
  //   SomeRandomTextWithUnderscores_[SequenceNumber]
  // where sequence numbers are numberic digits
  size_t firstIndex = first.find_last_of("_");
  size_t secondIndex = second.find_last_of("_");

  if (firstIndex == std::string::npos || secondIndex == std::string::npos) {
    error((boost::format("Unable to figure ordering of '%s' and '%s'") %
      first % second).str().c_str());
  }

  int first_frame_number = std::atoi(first.substr(firstIndex + 1).c_str());
  int second_frame_number = std::atoi(second.substr(secondIndex + 1).c_str());

  return first_frame_number < second_frame_number;
}

}} // namespace facebook::cp
