// Copyright 2004-present Facebook. All Rights Reserved.

// Classes for storing tracks and observations in a "track table".
//
// An observation is a point location in a frame. It is a templated type, two
// example implementations are provided:
// * Observation2D: used in the "regular video" tracker. Stores the 2D location
//   of the point in the frame
// * Observation3dCube: used in the 360 video tracker. Stores two redundant
//   representation of a point: 3D unit vector and cube face + 2D location.
//
// A track is a collection of observations in consecutive frames. The track
// class has a templated variable data that can store any user-provided
// per-track data.
//
// This header file only contains the interface to the track table classes. The
// implementation is provided in TrackTable-impl.h. Since these are template
// classes the user need to explictly instantiate it as follows:
//
//   // Include the track table *impl* file here because the compiler needs to
//   // see the implementation to instantiate the template classes.
//   #include <compphotolib/tracking/TrackTable-inl.h>
//
//   // Example Explicit template instantiation
//   template class Track<Observation2d>;
//   template class Track<Observation3dCube, TrackExtraData>;
//   template class TrackTable<Observation3dCube, TrackExtraData>;

#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <set>
#include <unordered_map>

#include <opencv2/core/core.hpp>

#include "OffsetVector.h"

namespace facebook {
namespace cp {

// Observation structure for 2D video tracking
struct Observation2d {
  cv::Vec2f loc;  // location of matched point
#ifdef STORE_MATCH_ERROR
  float err;      // matching error
#endif
#ifdef VISUALIZE_JITTER
  cv::Vec2f jitter; // non-smooth velocity
#endif

  Observation2d() = default;

#ifdef STORE_MATCH_ERROR
  Observation2d(cv::Vec2f loc, float err = 0.0f) :
      loc(loc), err(err) {
  }
#else
  Observation2d(cv::Vec2f loc) :
    loc(loc) {
  }
#endif
};

// Observation structure for 360 video tracking. It contains a redundant
// representation of the observation: (1) a unit 3D vector, (2) a cube face
// index + 2D location within the cube face.
struct Observation3dCube {
  cv::Vec3f dir;
  unsigned char face;
  cv::Vec2f loc;

  Observation3dCube() = default;

  Observation3dCube(cv::Vec3f dir, unsigned char face, cv::Vec2f loc) :
      dir(dir), face(face), loc(loc) {
  }
};

template <typename Obs>
class TrackBaseSequential {
 public:
  using SaveFn = std::function<int(const Obs*, int, FILE*)>;
  using StreamSaveFn = std::function<int(const Obs*, int, std::ostream&)>;
  using LoadFn = std::function<void(Obs*, int, FILE*)>;
  using StreamLoadFn = std::function<void(Obs*, int, std::istream&)>;

  // Construct a track and deserialize it
  TrackBaseSequential(FILE* fin, LoadFn loadFn = nullptr);
  TrackBaseSequential(std::istream& is, StreamLoadFn loadFn = nullptr);

  // Construct a track from an initial observation
  TrackBaseSequential(const int frame, const Obs& obs);

  // Access to observations
  const Obs& obs(const int frame) const;
  Obs& obs(const int frame);

  // Is the track visible in a frame?
  bool inFrame(const int frame) const;

  // First / last frames the track is observed in.
  int firstFrame() const;
  int lastFrame() const;

  const std::vector<int>& getFrameIds() const;

  // Make the track shorter, so that it ends with the specified new last frame.
  void shorten(const int newLastFrame);

  void addObs(const int frame, const Obs& obs);

  // Undo the addition of the last observation
  void removeLastObs();

  // Length of the track
  int length() const;

  void serialize(FILE* fout, SaveFn saveFn = nullptr) const;
  void serialize(std::ostream& os, StreamSaveFn saveFn = nullptr) const;
  void deserialize(FILE* fin, LoadFn loadFn = nullptr);
  void deserialize(std::istream& is, StreamLoadFn loadFn= nullptr);

 private:
  OffsetVector<Obs> obs_;
  std::vector<int> visibleFrames_;
};

template <typename Obs>
class TrackBaseUnstructured {
 public:
  using SaveFn = std::function<int(const Obs*, int, FILE*)>;
  using StreamSaveFn = std::function<int(const Obs*, int, std::ostream&)>;
  using LoadFn = std::function<void(Obs*, int, FILE*)>;
  using StreamLoadFn = std::function<void(Obs*, int, std::istream&)>;

  TrackBaseUnstructured(FILE* fin, LoadFn loadFn = nullptr);
  TrackBaseUnstructured(std::istream& is, StreamLoadFn loadFn = nullptr);

  // Construct a track from an initial observation
  TrackBaseUnstructured(const int frame, const Obs& obs);

  // Returns the frame IDs, useful for iterating
  const std::vector<int>& getFrameIds() const;

  // First / last frames the track is observed in.
  int firstFrame() const;
  int lastFrame() const;

  // Access to observations
  const Obs& obs(const int frame) const;
  Obs& obs(const int frame);

  // Is the track visible in a frame?
  bool inFrame(const int frame) const;

  void addObs(const int frame, const Obs& obs);

  void shorten(const int newLastFrame);

  // Undo the addition of the last observation
  void removeLastObs();

  // Length of the track
  int length() const;

  void serialize(FILE* fout, SaveFn saveFn = nullptr) const;
  void serialize(std::ostream& os, StreamSaveFn saveFn = nullptr) const;
  void deserialize(FILE* fin, LoadFn loadFn = nullptr);
  void deserialize(std::istream& is, StreamLoadFn loadFn = nullptr);

 private:
  std::unordered_map<int, Obs> obs_;
  std::vector<int> visibleFrames_;
};

// class Frame
class FrameBase {
public:
  // Constructor
  FrameBase() = default;

  // Disable copy and assignment
  FrameBase(const FrameBase&) = delete;
  FrameBase& operator=(const FrameBase&) = delete;

  // Tracks observed in this frame. This field should *not* be serialized, as
  // it is reconstructed from the information stored in the tracks.
  std::set<size_t> tracks;

  void serialize(FILE*) const {};
  void serialize(std::ostream&) const {};
  void deserialize(FILE*) {};
  void deserialize(std::istream&) {};
};

// Trait class that identifies whether Frame stores time parameters. If a
// specialization exists that is derived from std::true_type, then Frame is
// expected to have a function with the following signature for setting the
// time stamp and PTS increment:
//   Frame::setTime(const double time, const int64_t ptsInc);
template <typename Frame> struct frame_has_time : std::false_type {};

// Trait class that identifies whether Frame stores keypoints. If a
// specialization exists that is derived from std::true_type, then Frame is
// expected to have the following two members:
//   std::vector<cv::KeyPoint> keyPoints;
//   cv::Mat descriptors;
template <typename Frame> struct frame_has_keypoints : std::false_type {};

// class TrackTable
template <typename Obs, typename Track, typename Frame>
class TrackTable {
public:
  // Types
  typedef Obs ObsType;
  typedef Track TrackType;
  typedef Frame FrameType;
  typedef std::unique_ptr<Track> PTrack;

  // Constructors
  TrackTable() = default;
  explicit TrackTable(const int startFrame);
  virtual ~TrackTable() {}

  // Disable copy and assignment
  TrackTable(const TrackTable&) = delete;
  TrackTable& operator=(const TrackTable&) = delete;

  // Reinitialize
  void init(const int startFrame);

  // Adds a new tracked frame. This has to be done before adding any
  // observations in this frame.
  void addFrame();

  // Return the start frame, last tracked frame, number of frames
  int startFrame() const;
  int lastFrame() const;
  int numFrames() const;

  // Is the frame index valid?
  bool hasFrame(const int frame) const;

  // Clear everything
  void clear();

  // Clear tracks in a frame
  void clearFrameTracks(const int frame);

  // Clear tracks in the last frame
  void clearLastFrameTracks();

  // Creates a new track, returns its ID. IDs are unique and assigned in a
  // running fashion.
  size_t createTrack(const int frame, const Obs& obs);

  // Delete a track
  void deleteTrack(const size_t trackId);

  // Make a track shorter
  void shortenTrack(const size_t trackId, const int lastFrame);

  // Return number of tracks. Some IDs might be "invalid" because tracks have
  // been deleted.
  size_t numTracks() const;

  // Check whether a track ID is valid. It can be invalid either because the ID
  // is out of bounds, or the track has been deleted.
  bool hasTrack(const size_t id) const;

  // Return a reference or const reference to a track
  const Track& track(const size_t id) const;
  Track& track(const size_t id);

  // Add an observation to a track
  void addObs(const size_t trackId, const int frame, const Obs& obs);

  // Remove an observation from a track
  void removeObs(const size_t trackId, const int frame);

  // Returns a sequence of track IDs observed in a frame
  const std::set<size_t>& frameTracks(const int frame) const;

  // Returns per-frame extra data
  const Frame& frame(const int frame) const;
  Frame& frame(const int frame);

  // Returns the number of tracks observed in a frame
  int frameTracksCount(const int frame) const;

  // Save / load track table from a file.
  void save(const std::string& fileName) const;
  void serialize(FILE* fout, typename Track::SaveFn saveFn = nullptr) const;
  void serialize(
      std::ostream& os, typename Track::StreamSaveFn saveFn = nullptr) const;
  void load(const std::string& fileName);
  void deserialize(FILE* fin, typename Track::LoadFn loadFn = nullptr);
  void deserialize(
      std::istream& is, typename Track::StreamLoadFn loadFn = nullptr);

  // Checks internal consistency of the track table
  bool checkSanity();

  // Predicts the location of a track in a given frame based on previous
  // frame(s)
  virtual void predictTrack(
      const int /*trackId*/,
      const int /*frameId*/,
      cv::Vec2f& /*loc*/) {}

  // Signals a new frame to initilize for track prediction
  virtual void prepareFrameForPrediction(
      const int /*frameId*/,
      const int /*width*/,
      const int /*height*/) {}

 private:
  // TODO:  this is declared as a deque,
  //  but a lot of code (e.g., serialize) assumes starts @ 0
  std::deque<PTrack> tracks_;
  OffsetVector<std::unique_ptr<Frame>> frames_;
};
}
} // namespace facebook::cp
