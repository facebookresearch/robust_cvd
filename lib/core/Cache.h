// Copyright 2004-present Facebook. All Rights Reserved.
//
// Author: Johannes Kopf (jkopf@fb.com)

#pragma once

#include <stddef.h>
#include <algorithm>
#include <list>
#include <unordered_map>

namespace facebook {
namespace cp {

/**
  * This class provides a fixed-capacity cache with a least-recently-used
  * eviction scheme. It maintains internally a list of (key,value) pairs sorted
  * by recency of access (the most-recently used element is at the front, the
  * least-recently used element is at the back). In addition it maintains an
  * unordered_map to provide O(1) access to elements.
  */

template <typename TKey, typename TValue>
class LruCache {
 public:
  typedef std::pair<const TKey, TValue> KeyValuePair;
  typedef std::list<KeyValuePair> ListType;
  typedef typename ListType::iterator iterator;
  typedef typename ListType::const_iterator const_iterator;
  typedef std::unordered_map<TKey, iterator> MapType;

  // No default constructor, need to specify size of cache
  LruCache() = delete;

  // Constructs a cache with a specified capacity
  explicit LruCache(const size_t maxEntries)
    : maxEntries_(maxEntries) {
  }

  // Disable copying and assignment
  LruCache(const LruCache&) = delete;
  LruCache& operator=(const LruCache&) = delete;

  // Returns the number of elements in the cache
  size_t size() const {
    return numEntries_;
  }

  // Returns whether the cache is empty (i.e. whether its size is 0.)
  bool empty() const {
    return (size() == 0);
  }

  // Returns an iterator referring to the first (most recently used) element in
  // the cache.
  iterator begin() {
    return cacheList_.begin();
  }

  // Returns a const-iterator referring to the first (most recently used)
  // element in the cache.
  const_iterator begin() const {
    return cacheList_.begin();
  }

  const_iterator cbegin() const {
    return cacheList_.cbegin();
  }

  // Returns an iterator referring to the last (least recently used) element in
  // the cache.
  iterator end() {
    return cacheList_.end();
  }

  // Returns a const-iterator referring to the last (least recently used)
  // element in the cache.
  const_iterator end() const {
    return cacheList_.cend();
  }

  const_iterator cend() const {
    return cacheList_.cend();
  }

  // Sets a key-value pair in the cache.
  void set(const TKey& key, TValue&& value) {
    // Is element with key already in cache?
    typename MapType::iterator it = cacheMap_.find(key);
    if (it != cacheMap_.end()) {
      // Element found, remove key from list. The map will be set below.
      cacheList_.erase(it->second);
      --numEntries_;
    }

    // Add element to front of list and update map
    cacheList_.emplace_front(key, std::move(value));
    cacheMap_[key] = cacheList_.begin();
    ++numEntries_;

    // Too many elements?
    if (numEntries_ > maxEntries_) {
      // Remove element at back of list first from map, then from list
      cacheMap_.erase(cacheList_.back().first);
      cacheList_.pop_back();
      --numEntries_;
    }
  }

  // If the element with specified key is in the cache an iterator referring to
  // the element is returned, also the element is moved to the front of the
  // list. If the element is not in the cache an iterator to LruCache::end is
  // returned.
  iterator get(const TKey& key) {
    // Is element with key in cache?
    auto it = cacheMap_.find(key);
    if (it == cacheMap_.end()) {
      // Not in cache, return past-the-end iterator
      return end();
    } else {
      // Is element not at front of list?
      if (it->second != cacheList_.begin()) {
        // Element not at front of list, move to front of list
        cacheList_.splice(cacheList_.begin(), cacheList_, it->second);
        cacheMap_[key] = cacheList_.begin();
      }
      return cacheList_.begin();
    }
  }

 private:
  size_t numEntries_ = 0;
  size_t maxEntries_ = 0;
  ListType cacheList_;
  MapType cacheMap_;
};

}} // namespace facebook::cp
