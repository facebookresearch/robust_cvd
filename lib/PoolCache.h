// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <list>
#include <unordered_map>

// This class provides a fixed-capacity cache with least-recently-used eviction
// scheme. Internally, the cache stores a fixed-size pool of objects. If a
// requested index is in the cache the corresponding pool element is returned.
// If it is _not_ in the cache the least recently used pool element is returned,
// and the user needs to update the object.
//
// Simple usage example:
//
// PoolCache<Mat> cache;
//
// ...
//
// Mat* image = nullptr;
// if (!cache.get(index, image)) {
//   std::string fileName = folly::format("frame_{:06d}.png", index).str();
//   cv::imread(fileName, *image);
// }

namespace facebook {
namespace cp {

template <typename ValueType>
class PoolCache {
 public:
  using Index = int;
  using Slot = int;
  using IndexSlot = std::pair<Index, Slot>;
  using ListType = std::list<IndexSlot>;
  using MapType = std::unordered_map<Index, ListType::iterator>;

  explicit PoolCache(const int size)
      : size_(size) {
    pool_ = new ValueType[size];
  }

  ~PoolCache() {
    delete[] pool_;
  }

  bool get(const int index, ValueType*& result) {
    Slot slot;
    auto it = map_.find(index);
    if (it == map_.end()) {
      // Index not in pool.
      if (numEntries_ < size_) {
        // Not all pool slots are used, yet.
        slot = numEntries_++;
        list_.emplace_front(IndexSlot(index, slot));
      } else {
        IndexSlot& lru = list_.back();
        slot = lru.second;

        map_.erase(lru.first);

        list_.pop_back();
        list_.emplace_front(IndexSlot(index, slot));
      }
      map_[index] = list_.begin();

      // The previously least-recently used slot, has now been made the most
      // recently used one.
      result = &pool_[slot];

      // Return false to signal to caller that they need to update the pool
      // entry.
      return false;
    } else {
      // Index is in pool.
      slot = it->second->second;
      if (it->second != list_.begin()) {
        list_.splice(list_.begin(), list_, it->second);
        map_[index] = list_.begin();
      }

      result = &pool_[slot];
      return true;
    }
  }

 private:
  const int size_;
  ValueType* pool_ = nullptr;
  MapType map_;
  ListType list_;
  int numEntries_ = 0;
};

}} // namespace facebook::cp
