// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <string>
#include <map>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "core/EigenNoAlignTypes.h"
#include "core/FileIo.h"
#include "FrameRange.h"

namespace facebook {
namespace cp {

class DepthVideo;

struct FlowConstraintsParams {
  FrameRange frameRange;

  // Minimum distance between two matches (in pixels).
  int matchSeparation = 10;

  int minDynamicDistance = -1;

  // Always recompute the result if enabled.
  bool doNotUseCache = false;

  void save(std::ostream& os) const;
  void load(std::istream& is);
  bool operator==(const FlowConstraintsParams& other) const;
  bool operator!=(const FlowConstraintsParams& other) const;
};

// Constraints for a single frame pair or (consecutive-frame) triplet. Pairs are
// indexed with a std::pair<int, int>, and triplets are indexed with an int
// referring to the center frame.
template <typename KeyType, typename ConstraintType>
class FlowConstraints {
 friend class FlowConstraintsCollection;

public:
  using Key = KeyType;
  using Constraint = ConstraintType;
  using ConstraintContainer = std::vector<Constraint>;

  const Key& key() const { return key_; }
  int count() const { return constraints_.size(); }

  // Iterators.
  typename ConstraintContainer::const_iterator begin() const {
    return constraints_.begin();
  }
  typename ConstraintContainer::const_iterator end() const {
    return constraints_.end();
  }

  const Constraint& operator[](const int& index) const {
    return constraints_[index];
  }

  const Constraint& at(const int& index) const {
    return constraints_.at(index);
  }

 private:
  explicit FlowConstraints(const Key& key)
      : key_(key) {
  }

  Key key_;
  ConstraintContainer constraints_;
};

template <int N>
struct FlowConstraint {
  using Type = std::array<Vector2fna, N>;

  Type loc;
  bool isStatic = true;

  FlowConstraint() {}
  explicit FlowConstraint(const Type& loc) : loc(loc) {}

  const Vector2fna& operator[](const int index) const {
    return loc[index];
  }

  Vector2fna& operator[](const int index) {
    return loc[index];
  }

  void read(std::ifstream& is) {
    facebook::cp::read(is, loc);
    // facebook::cp::read(is, isStatic);
  }

  void write(std::ofstream& os) const {
    facebook::cp::write(os, loc);
    // facebook::cp::write(os, isStatic);
  }
};

using PairKey = std::pair<int, int>;
using PairConstraint = FlowConstraint<2>;
using PairFlowConstraints = FlowConstraints<PairKey, PairConstraint>;

using TripletKey = int;
using TripletConstraint = FlowConstraint<3>;
using TripletFlowConstraints = FlowConstraints<TripletKey, TripletConstraint>;

// Collection of constraints for multiple frame pairs and triplets.
class FlowConstraintsCollection {
 public:
  static constexpr uint32_t kFileFormatVersion = 3;
  static constexpr uint32_t kMinSupportedFileFormat = 3;

  template <typename Container>
  class Iterator : public Container::const_iterator {
   private:
    using Base = typename Container::const_iterator;
    using Key = typename Container::key_type;

   public:
    Iterator() : Container::const_iterator() {
    }
    explicit Iterator(Base s)
        : Container::const_iterator(s) {
    }
    const Key* operator->() const {
      return (const Key* const)&(Base::operator->()->first);
    }
    const Key& operator*() const {
      return (const Key&)(Base::operator*().first);
    }
  };

  using PairContainer = std::map<PairKey, std::unique_ptr<PairFlowConstraints>>;
  using PairIterator = Iterator<PairContainer>;

  using TripletContainer =
      std::map<TripletKey, std::unique_ptr<TripletFlowConstraints>>;
  using TripletIterator = Iterator<TripletContainer>;

  // The constructor either loads the constraints from a cache file, or
  // computes (and saves) them if the cache file is not present.
  FlowConstraintsCollection(
      const DepthVideo& video, const FlowConstraintsParams& params);

  bool load();
  void save();

  // Iterators over frame pairs and triplets.
  PairIterator pairBegin() const;
  PairIterator pairEnd() const;
  TripletIterator tripletBegin() const;
  TripletIterator tripletEnd() const;

  bool hasPair(const PairKey& key) const {
    return (pairs_.find(key) != pairs_.end());
  }

  bool hasTriplet(const TripletKey& key) const {
    return (triplets_.find(key) != triplets_.end());
  }

  // Access constraints for a frame pair / triplet.
  const PairFlowConstraints& operator()(const PairKey& key) const {
    return at(key);
  }

  const PairFlowConstraints& at(const PairKey& key) const {
    return *pairs_.at(key);
  }

  const TripletFlowConstraints& operator()(const TripletKey& key) const {
    return at(key);
  }

  const TripletFlowConstraints& at(const TripletKey& key) const {
    return *triplets_.at(key);
  }

  void resetStaticFlag();
  void setStaticFlagFromDynamicMask(const int distance);
  void pruneStaticFlag(const int distance);

 private:
  std::pair<cv::Mat2f, cv::Mat1b> loadFlowAndMask(const int srcFrame, const int dstFrame) const;
  cv::Mat1f dynamicDistance(const int frame) const;
  void compute();
  void compute(const PairKey& pair);
  void compute(const TripletKey& triplet);

  const DepthVideo* video_ = nullptr;

  std::string path_;
  FlowConstraintsParams params_;

  PairContainer pairs_;
  TripletContainer triplets_;
};

}} // namespace facebook::cp
