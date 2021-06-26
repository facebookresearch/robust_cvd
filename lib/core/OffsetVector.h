// Copyright 2004-present Facebook. All Rights Reserved.
//
// Author: Johannes Kopf (jkopf@fb.com)

// This is a container similar to std::vector that is accessed by an offset
// index, i.e. OffsetVector<T>(5, 3) constructs a container with the following
// valid indices: 5, 6, 7. Like a std::vector, OffsetVector can grow and
// shrink dynamically.

#pragma once

#include <stdlib.h>
#include <cassert>
#include <vector>

namespace facebook {
namespace cp {

template <typename T>
class OffsetVector {
public:
    OffsetVector() {}

    OffsetVector(const size_t offset, const size_t size)
        : offset_(offset) {
      allocate(size);
    }

    OffsetVector(const size_t offset, const size_t size, const T& value)
        : OffsetVector(offset, size) {
      std::fill(data_.begin(), data_.end(), value);
    }

    ~OffsetVector() {
      deallocate();
    }

    // Disable copying and assignment
    OffsetVector(const OffsetVector&) = delete;
    OffsetVector& operator=(const OffsetVector&) = delete;

    // Move constructor
    OffsetVector(OffsetVector&& other) {
      data_ = std::move(other.data_);

      offsetData_ = other.offsetData_;
      other.offsetData_ = nullptr;
      
      offset_ = other.offset_;
      other.offset_ = 0;
    }
  
    // Move assignment
    OffsetVector& operator=(OffsetVector&& other) {
      if (this != &other) {
        data_ = std::move(other.data_);

        offsetData_ = other.offsetData_;
        other.offsetData_ = nullptr;
        
        offset_ = other.offset_;
        other.offset_ = 0;
      }
      
      return *this;
    }

    // Returns the offset of the vector (i.e., the index of the first element.)
    size_t offset() const {
      return offset_;
    }

    // Returns the number of elements in the vector.
    size_t size() const {
      return data_.size();
    }

    // Returns whether the array is empty (i.e. whether its size is 0.)
    bool empty() const {
      return data_.empty();
    }

    // Returns the index of the first element in the vector. If the vector is
    // empty the index of the first element that would be inserted is returned.
    size_t firstIndex() const {
      // NOTE: we're not checking for emptyness here.
      return offset_;
    }

    // Returns the index of the first element in the vector.
    size_t lastIndex() const {
      checkNotEmpty();
      return offset_ + size() - 1;
    }

    // Returns the index of the first element in the vector.
    int firstIndexI() const {
      return (int)firstIndex();
    }

    // Returns the index of the last element in the vector.
    int lastIndexI() const {
      return (int)lastIndex();
    }

    // Return index from start
    int indexFromFirst(const size_t index) const {
      return (int)(index - offset_);
    }

    // Return true if the index is in-bounds (i.e.,
    // firstIndex <= index <= lastIndex).
    bool hasIndex(const size_t index) const {
      return (index >= offset_ && index < offset_ + size());
    }

    // Removes all elements from the vector (which are destroyed), leaving the
    // container with a size of 0.
    void clear() {
      data_.clear();
    }

    // Resizes the container so that it contains size elements, starting at
    // index offset.
    void resize(const size_t offset, const size_t size) {
      offset_ = offset;
      allocate(size);
    }

    // Returns a const iterator referring to the first element in the vector.
    typename std::vector<T>::const_iterator begin() const {
      checkNotEmpty();
      return data_.begin();
    }

    // Returns a iterator referring to the first element in the vector.
    typename std::vector<T>::iterator begin() {
      checkNotEmpty();
      return data_.begin();
    }

    // Returns a const iterator referring to the last element in the vector.
    typename std::vector<T>::const_iterator end() const {
      checkNotEmpty();
      return data_.end();
    }

    // Returns a iterator referring to the last element in the vector.
    typename std::vector<T>::iterator end() {
      checkNotEmpty();
      return data_.end();
    }
  
    // Returns a const reference to first element in array.
    const T& front() const {
      checkNotEmpty();
      return data_.front();
    }

    // Returns a reference to first element in array.
    T& front() {
      checkNotEmpty();
      return data_.front();
    }
  
    // Returns a const reference to last element in array.
    const T& back() const {
      checkNotEmpty();
      return data_.back();
    }

    // Returns a reference to last element in array.
    T& back() {
      checkNotEmpty();
      return data_.back();
    }

    // Returns a const reference to an element.
    const T& operator[](const size_t index) const {
      checkBounds(index);
      return offsetData_[index];
    }

    // Returns a reference to an element.
    T& operator[](const size_t index) {
      checkBounds(index);
      return offsetData_[index];
    }

    // Returns a const reference to an element.
    const T& at(const size_t index) const {
      checkBounds(index);
      return offsetData_[index];
    }

    // Returns a reference to an element.
    T& at(const size_t index) {
      checkBounds(index);
      return offsetData_[index];
    }

    // Returns a const reference to an absolute indexed (non-offset!) element.
    const T& atAbs(const size_t index) const {
      checkAbsBounds(index);
      return data_[index];
    }

    // Returns a reference to an absolute indexed (non-offset!) element.
    T& atAbs(const size_t index) {
      checkAbsBounds(index);
      return data_[index];
    }

    // Adds a new element at the end of the vector, after its current last
    // element. The content of val is **copied** to the new element.
    void push_back(const T& el) {
      data_.push_back(el);

      // raw data inside the vector might have moved, so we need to update the
      // offset address
      offsetData_ = data_.data() - offset_;
    }

    // Adds a new element at the end of the vector, after its current last
    // element. The content of val is **moved** to the new element.
    void push_back(T&& el) {
      data_.push_back(std::move(el));

      // raw data inside the vector might have moved, so we need to update the
      // offset address
      offsetData_ = data_.data() - offset_;
    }

private:
    void allocate(const size_t size) {
      if (size > 0) {
        data_.resize(size);
        offsetData_ = data_.data() - offset_;
      } else {
        data_.clear();
        offsetData_ = nullptr;
      }
    }

    void deallocate() {
      data_.clear();
      offsetData_ = nullptr;
    }

    // Check if index is in bounds (only in debug builds!)
    void checkBounds(const size_t index) const {
      assert(index >= offset_);
      assert(index < offset_ + data_.size());
    }

    // Check if absolute (non-offset) index is in bounds (only in debug
    // builds!)
    void checkAbsBounds(const size_t index) const {
      assert(index < data_.size());
    }

    // Check if the vector is non-empty (only in debug builds!)
    void checkNotEmpty() const {
      assert(data_.size() > 0);
    }

    std::vector<T> data_;
    T * offsetData_ = nullptr;
    size_t offset_ = 0;
};

}}
