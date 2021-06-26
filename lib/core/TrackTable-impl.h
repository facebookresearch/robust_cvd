// Copyright 2004-present Facebook. All Rights Reserved.

// NOTE: This file should be included only in a ***single*** translation unit,
// where the template classes are explicitly instantiated.
// See fbcode/compphoto/projects/omnistab/Instantiation.cpp for an example.
// All other places should include the corresponding header file ***without***
// the -impl in the file name.

#include "TrackTable.h"

#include <fstream>

#include <FileIo.h>

#include "Logging.h"

namespace facebook {
namespace cp {

//
// Track implementation
//

template <typename Obs>
TrackBaseSequential<Obs>::TrackBaseSequential(const int frame, const Obs& obs) {
  obs_.resize(frame, 1);
  obs_[frame] = obs;
  visibleFrames_.push_back(frame);
}

template <typename Obs>
TrackBaseUnstructured<Obs>::TrackBaseUnstructured(
    const int frame,
    const Obs& obs) {
  obs_[frame] = obs;
  visibleFrames_.push_back(frame);
  CHECK_EQ(obs_.size(), visibleFrames_.size());
}

template <typename Obs>
TrackBaseSequential<Obs>::TrackBaseSequential(FILE* fin, LoadFn loadFn) {
  deserialize(fin, loadFn);
}

template <typename Obs>
TrackBaseSequential<Obs>::TrackBaseSequential(
    std::istream& is, StreamLoadFn loadFn) {
  deserialize(is, loadFn);
}

template <typename Obs>
TrackBaseUnstructured<Obs>::TrackBaseUnstructured(FILE* fin, LoadFn loadFn) {
  deserialize(fin, loadFn);
}

template <typename Obs>
TrackBaseUnstructured<Obs>::TrackBaseUnstructured(
    std::istream& is, StreamLoadFn loadFn) {
  deserialize(is, loadFn);
}

// Const reference to observation
template <typename Obs>
const Obs& TrackBaseSequential<Obs>::obs(const int frame) const {
  return obs_[frame];
}

// Const reference to observation
template <typename Obs>
const Obs& TrackBaseUnstructured<Obs>::obs(const int frame) const {
  return obs_.at(frame);
}

// Non-const reference to observation
template <typename Obs>
Obs& TrackBaseSequential<Obs>::obs(const int frame) {
  return obs_[frame];
}

// Non-const reference to observation
template <typename Obs>
Obs& TrackBaseUnstructured<Obs>::obs(const int frame) {
  CHECK(obs_.count(frame) > 0);
  return obs_[frame];
}

// Is the track visible in a frame?
template <typename Obs>
bool TrackBaseSequential<Obs>::inFrame(const int frame) const {
  return (frame >= firstFrame() && frame <= lastFrame());
}

// Is the track visible in a frame?
template <typename Obs>
bool TrackBaseUnstructured<Obs>::inFrame(const int frame) const {
  return obs_.count(frame) > 0;
}

// First frame the track is observed in.
template <typename Obs>
int TrackBaseSequential<Obs>::firstFrame() const {
  return obs_.firstIndexI();
}

// First frame the track is observed in.
template <typename Obs>
int TrackBaseUnstructured<Obs>::firstFrame() const {
  return visibleFrames_[0];
}

// Last frame the track is observed in.
template <typename Obs>
int TrackBaseUnstructured<Obs>::lastFrame() const {
  CHECK_EQ(visibleFrames_.size(), obs_.size());
  return visibleFrames_[visibleFrames_.size() - 1];
}

// Last frame the track is observed in.
template <typename Obs>
int TrackBaseSequential<Obs>::lastFrame() const {
  return obs_.lastIndexI();
}

template <typename Obs>
const std::vector<int>& TrackBaseUnstructured<Obs>::getFrameIds() const {
  return visibleFrames_;
}

template <typename Obs>
const std::vector<int>& TrackBaseSequential<Obs>::getFrameIds() const {
  return visibleFrames_;
}

// Make the track shorter, so that it ends with the specified new last frame.
template <typename Obs>
void TrackBaseSequential<Obs>::shorten(const int newLastFrame) {
  int firstFrame = obs_.firstIndexI();
  assert(newLastFrame >= firstFrame);
  assert(newLastFrame <= obs_.lastIndexI());
  size_t newSize = newLastFrame + 1 - firstFrame;
  visibleFrames_.resize(newSize);
  return obs_.resize(firstFrame, newSize);
}

// Make the track shorter, so that it ends with the specified new last frame.
template <typename Obs>
void TrackBaseUnstructured<Obs>::shorten(const int newLastFrame) {
  LOG(FATAL) << "This function is not defined for unstructured sequences.";
}

template <typename Obs>
void TrackBaseSequential<Obs>::addObs(const int frame, const Obs& obs) {
  assert(frame == obs_.lastIndexI() + 1);
  obs_.push_back(obs);
  visibleFrames_.push_back(frame);
}

template <typename Obs>
void TrackBaseUnstructured<Obs>::addObs(const int frame, const Obs& obs) {
  obs_[frame] = obs;
  visibleFrames_.push_back(frame);

  CHECK_EQ(visibleFrames_.size(), obs_.size());
}

template <typename Obs>
void TrackBaseSequential<Obs>::removeLastObs() {
  CHECK_EQ(length(), visibleFrames_.size());

  if (visibleFrames_.size() == 0u) {
    return;
  }

  const size_t newLength = visibleFrames_.size() - 1u;

  obs_.resize(obs_.offset(), newLength);
  visibleFrames_.resize(newLength);

  CHECK_EQ(length(), visibleFrames_.size());
}

template <typename Obs>
void TrackBaseUnstructured<Obs>::removeLastObs() {
  CHECK_EQ(obs_.size(), visibleFrames_.size());

  if (obs_.size() == 0) {
    return;
  }

  obs_.erase(lastFrame());
  visibleFrames_.pop_back();

  CHECK_EQ(visibleFrames_.size(), obs_.size());
}

// Length of the track
template <typename Obs>
int TrackBaseSequential<Obs>::length() const {
  return obs_.lastIndexI() - obs_.firstIndexI() + 1;
}

// Length of the track
template <typename Obs>
int TrackBaseUnstructured<Obs>::length() const {
  return static_cast<int>(obs_.size());
}

// Save track to a file.
template <typename Obs>
void TrackBaseSequential<Obs>::serialize(FILE* fout, SaveFn saveFn) const {
  fwrite(fout, obs_.offset());
  size_t size = obs_.size();
  fwrite(fout, size);
  if (saveFn == nullptr) {
    if (size > 0) {
      ensure(fwrite(&obs_.atAbs(0), sizeof(Obs), size, fout) == size);
    }
  } else {
    saveFn(&obs_.atAbs(0), static_cast<int>(size), fout);
  }
}

template <typename Obs>
void TrackBaseSequential<Obs>::serialize(
    std::ostream& os, StreamSaveFn saveFn) const {
  write(os, obs_.offset());
  size_t size = obs_.size();
  write(os, size);
  if (saveFn == nullptr) {
    if (size > 0) {
      os.write(
          reinterpret_cast<const char*>(&obs_.atAbs(0)), size * sizeof(Obs));
    }
  } else {
    saveFn(&obs_.atAbs(0), static_cast<int>(size), os);
  }
}

// Save track to a file.
template <typename Obs>
void TrackBaseUnstructured<Obs>::serialize(FILE* fout, SaveFn saveFn) const {
  fwrite(fout, obs_.size());
  if (saveFn == nullptr) {
    for (const auto& keyValue : obs_) {
      ensure(fwrite(&keyValue.first, sizeof(int), 1, fout) == 1);
      ensure(fwrite(&keyValue.second, sizeof(Obs), 1, fout) == 1);
    }
  } else {
    LOG(FATAL) << "Not supported.";
  }
}

template <typename Obs>
void TrackBaseUnstructured<Obs>::serialize(
    std::ostream& os, StreamSaveFn saveFn) const {
  write(os, obs_.size());
  if (saveFn == nullptr) {
    for (const auto& keyValue : obs_) {
      os.write(reinterpret_cast<const char*>(&keyValue.first), sizeof(int));
      os.write(reinterpret_cast<const char*>(&keyValue.second), sizeof(Obs));
    }
  } else {
    LOG(FATAL) << "Not supported.";
  }
}

// Load track from a file.
template <typename Obs>
void TrackBaseSequential<Obs>::deserialize(FILE* fin, LoadFn loadFn) {
  size_t offset = fread<size_t>(fin);
  size_t size = fread<size_t>(fin);
  obs_.resize(offset, size);
  if (loadFn == nullptr) {
    if (size > 0) {
      ensure(fread(&obs_.atAbs(0), sizeof(Obs), size, fin) == size);
    }
  } else {
    loadFn(&obs_.atAbs(0), static_cast<int>(size), fin);
  }
}

template <typename Obs>
void TrackBaseSequential<Obs>::deserialize(
    std::istream& is, StreamLoadFn loadFn) {
  size_t offset = read<size_t>(is);
  size_t size = read<size_t>(is);
  obs_.resize(offset, size);
  if (loadFn == nullptr) {
    if (size > 0) {
      is.read(reinterpret_cast<char*>(&obs_.atAbs(0)), sizeof(Obs) * size);
    }
  } else {
    loadFn(&obs_.atAbs(0), static_cast<int>(size), is);
  }
}

// Load track from a file.
template <typename Obs>
void TrackBaseUnstructured<Obs>::deserialize(FILE* fin, LoadFn loadFn) {
  size_t size = fread<size_t>(fin);
  if (loadFn == nullptr) {
    // Read in the track entries, which may not be sorted by index,
    //  since they were written out from an unordered_map.
    for (int i = 0; i < size; i ++) {
      int index;
      Obs value;
      ensure(fread(&index, sizeof(int), 1, fin) == 1);
      ensure(fread(&value, sizeof(Obs), 1, fin) == 1);
      obs_[index] = value;
      visibleFrames_.push_back(index);
    }
    std::sort(visibleFrames_.begin(), visibleFrames_.end());
  } else {
    LOG(FATAL) << "Not supported.";
  }
}

template <typename Obs>
void TrackBaseUnstructured<Obs>::deserialize(
    std::istream& is, StreamLoadFn loadFn) {
  size_t size = read<size_t>(is);
  if (loadFn == nullptr) {
    // Read in the track entries, which may not be sorted by index,
    //  since they were written out from an unordered_map.
    for (int i = 0; i < size; i ++) {
      int index;
      Obs value;
      is.read(reinterpret_cast<char*>(&index), sizeof(int));
      is.read(reinterpret_cast<char*>(&value), sizeof(Obs));
      obs_[index] = value;
      visibleFrames_.push_back(index);
    }
    std::sort(visibleFrames_.begin(), visibleFrames_.end());
  } else {
    LOG(FATAL) << "Not supported.";
  }
}

//
// TrackTable implementation
//

// Constructor
template <typename Obs, typename Track, typename Frame>
TrackTable<Obs, Track, Frame>::TrackTable(
    const int startFrame) {
  init(startFrame);
}

template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::init(
    const int startFrame) {
  frames_.resize(startFrame, 0);
  tracks_.clear();
}

// Adds a new tracked frame.
template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::addFrame() {
  auto pf = std::make_unique<Frame>();
  frames_.push_back(std::move(pf));
}

// Returns the start frame
template <typename Obs, typename Track, typename Frame>
int TrackTable<Obs, Track, Frame>::startFrame() const {
  return frames_.firstIndexI();
}

// Returns the last tracked frame
template <typename Obs, typename Track, typename Frame>
int TrackTable<Obs, Track, Frame>::lastFrame() const {
  return frames_.lastIndexI();
}

// Returns the number of frames
template <typename Obs, typename Track, typename Frame>
int TrackTable<Obs, Track, Frame>::numFrames() const {
  return static_cast<int>(frames_.size());
}

// Is the frame index valid?
template <typename Obs, typename Track, typename Frame>
bool TrackTable<Obs, Track, Frame>::hasFrame(
    const int frame) const {
  return frames_.hasIndex(frame);
}

// Clear everything
template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::clear() {
  tracks_.clear();
  frames_.clear();
}

// Clear tracks in a frame
template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::clearFrameTracks(
    const int frame) {
  const auto& frameTracks = frames_[frame]->tracks;
  for (auto it = frameTracks.begin(); it != frameTracks.end(); ++it) {
    auto& pt = tracks_[*it];
    assert(pt != nullptr);
  }
  frames_[frame]->tracks.clear();
}

// Clear tracks in the last frame
template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::clearLastFrameTracks() {
  if (frames_.empty()) {
    return;
  }

  const auto& frameTracks = frames_.back()->tracks;

  for (auto it = frameTracks.begin(); it != frameTracks.end(); ++it) {
    auto& pt = tracks_[*it];
    assert(pt != nullptr);
    pt->removeLastObs();
  }

  frames_.back()->tracks.clear();
}

// Creates a new track
template <typename Obs, typename Track, typename Frame>
size_t TrackTable<Obs, Track, Frame>::createTrack(
    const int frame, const Obs& obs) {
  // Create track struct
  auto pt = std::make_unique<Track>(frame, obs);

  // Add to tracks
  size_t trackId = tracks_.size();
  tracks_.push_back(std::move(pt));

  // Add to frame
  assert(frames_.hasIndex(frame));
  frames_[frame]->tracks.insert(trackId);

  return trackId;
}

// Delete a track
template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::deleteTrack(const size_t trackId) {
  auto& pt = tracks_[trackId];
  assert(pt != nullptr);

  // Remove from frames
  for (const int frame : pt->getFrameIds()) {
    assert(frames_.hasIndex(frame));
    frames_[frame]->tracks.erase(trackId);
  }

  // Delete track
  tracks_[trackId] = nullptr;
}

// Make a track shorter
template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::shortenTrack(
    const size_t trackId,
    const int lastFrame) {
  auto& pt = tracks_[trackId];
  assert(pt != nullptr);
  assert(frames_.lastIndex() >= pt->lastFrame());

  for (int frame = lastFrame + 1; frame <= pt->lastFrame(); ++frame) {
    frames_[frame]->tracks.erase(trackId);
  }

  pt->shorten(lastFrame);
}

// Return number of tracks.
template <typename Obs, typename Track, typename Frame>
size_t TrackTable<Obs, Track, Frame>::numTracks() const {
  return tracks_.size();
}

// Check whether a track ID is valid.
template <typename Obs, typename Track, typename Frame>
bool TrackTable<Obs, Track, Frame>::hasTrack(
    const size_t id) const {
  if (id >= tracks_.size()) {
    return false;
  }
  return (tracks_[id] != nullptr);
}

// Returns a const reference to a track
template <typename Obs, typename Track, typename Frame>
const Track& TrackTable<Obs, Track, Frame>::track(const size_t id) const {
  assert(tracks_[id] != nullptr);
  return *tracks_[id];
}

// Returns a non-const reference to a track
template <typename Obs, typename Track, typename Frame>
Track& TrackTable<Obs, Track, Frame>::track(const size_t id) {
  assert(tracks_[id] != nullptr);
  return *tracks_[id];
}

// Add an observation to a track
template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::addObs(
    const size_t trackId,
    const int frame,
    const Obs& obs) {
  auto& pt = tracks_[trackId];
  assert(pt != nullptr);
  assert(frames_.hasIndex(frame));

  pt->addObs(frame, obs);

  frames_[frame]->tracks.insert(trackId);
}

// Remove an observation from a track
template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::removeObs(
    const size_t trackId,
    const int frame) {
  assert(tracks_[trackId] != nullptr);

  Track& t = *tracks_[trackId];

  // Can only remove observation at end of track
  ensure(t.lastFrame() == frame);
}

// Returns a sequence of track IDs observed in a frame
template <typename Obs, typename Track, typename Frame>
const std::set<size_t>&
TrackTable<Obs, Track, Frame>::frameTracks(const int frame) const {
  assert(frames_.hasIndex(frame));
  return frames_[frame]->tracks;
}

// Returns a const reference to per-frame extra data
template <typename Obs, typename Track, typename Frame>
const Frame& TrackTable<Obs, Track, Frame>::frame(const int frame) const {
  assert(frames_.hasIndex(frame));
  return *frames_[frame];
}

// Returns a non-const reference to per-frame extra data
template <typename Obs, typename Track, typename Frame>
Frame& TrackTable<Obs, Track, Frame>::frame(const int frame) {
  assert(frames_.hasIndex(frame));
  return *frames_[frame];
}

// Returns the number of tracks observed in a frame
template <typename Obs, typename Track, typename Frame>
int TrackTable<Obs, Track, Frame>::frameTracksCount(
    const int frame) const {
  return static_cast<int>(frames_[frame]->tracks.size());
}

// Save track table to a file.
template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::save(const std::string& fileName) const {
  std::ofstream os(fileName, std::ios::binary);
  serialize(os);
}

template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::serialize(
    FILE* fout, typename Track::SaveFn saveFn) const {
  size_t numTracks = tracks_.size();
  fwrite(fout, numTracks);

  // Something to consider for the future - we are writing all tracks out
  // to stable storage without looking at how important these tracks are.
  // A caller can prune the track list ahead of time but a warning here
  // might avoid unfortunate mistakes.
  int numObs = 0;
  for (size_t trackId = 0; trackId < tracks_.size(); ++trackId) {
    const PTrack& pt = tracks_[trackId];
    if (pt == nullptr) {
      fwrite<bool>(fout, false);
    } else {
      fwrite<bool>(fout, true);
      pt->serialize(fout, saveFn);
      numObs += pt->length();
    }
  }
  VLOG(2) << "serialize; numTracks: " << numTracks << ", numObs: " << numObs;

  fwrite(fout, frames_.offset());
  fwrite(fout, frames_.size());

  for (auto& frame : frames_) {
    // We don't write the tracks structure since this can be reconstituted
    // from the frames structure written earlier.  We do write the frame
    // data though.
    frame->serialize(fout);
  }
}

template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::serialize(
    std::ostream& os, typename Track::StreamSaveFn saveFn) const {
  size_t numTracks = tracks_.size();
  write(os, numTracks);

  // Something to consider for the future - we are writing all tracks out
  // to stable storage without looking at how important these tracks are.
  // A caller can prune the track list ahead of time but a warning here
  // might avoid unfortunate mistakes.
  int numObs = 0;
  for (size_t trackId = 0; trackId < tracks_.size(); ++trackId) {
    auto& pt = tracks_[trackId];
    if (pt == nullptr) {
      write<bool>(os, false);
    } else {
      write<bool>(os, true);
      pt->serialize(os, saveFn);
      numObs += pt->length();
    }
  }
  VLOG(2) << "serialize; numTracks: " << numTracks << ", numObs: " << numObs;

  write(os, frames_.offset());
  write(os, frames_.size());

  for (auto& frame : frames_) {
    // We don't write the tracks structure since this can be reconstituted
    // from the frames structure written earlier.  We do write the frame
    // data though.
    frame->serialize(os);
  }
}

// Load track table from a file.
template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::load(const std::string& fileName) {
  std::ifstream is(fileName, std::ios::binary);
  if (!is.good()) {
    throw std::runtime_error("Could not open file.");
  }
  deserialize(is);
}

template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::deserialize(
    FILE* fin, typename Track::LoadFn loadFn) {
  size_t numTracks = fread<size_t>(fin);
  tracks_.resize(numTracks);

  for (size_t trackId = 0; trackId < numTracks; ++trackId) {
    bool valid = fread<bool>(fin);
    if (valid) {
      // Construct and deserialize a track
      tracks_[trackId] = std::make_unique<Track>(fin, loadFn);
    } else {
      tracks_[trackId] = nullptr;
    }
  }

  size_t offset = fread<size_t>(fin);
  size_t size = fread<size_t>(fin);
  frames_.resize(offset, size);
  for (auto& frame : frames_) {
    frame = std::make_unique<Frame>();
    frame->deserialize(fin);
  }

  // Go through the tracks and recreate the set of tracks in each frame
  // from the track structure.  This information, if saved to the serialized
  // format, would be redundant.
  for (size_t trackId = 0; trackId < numTracks; ++trackId) {
    const auto& t = tracks_[trackId];
    if (!t) {
      continue;
    }
    for (size_t frameId = offset; frameId < size + offset; ++frameId) {
      if (t->inFrame(static_cast<int>(frameId))) {
        frames_[frameId]->tracks.insert(trackId);
      }
    }
  }
}

template <typename Obs, typename Track, typename Frame>
void TrackTable<Obs, Track, Frame>::deserialize(
    std::istream& is, typename Track::StreamLoadFn loadFn) {
  size_t numTracks = read<size_t>(is);
  tracks_.resize(numTracks);

  for (size_t trackId = 0; trackId < numTracks; ++trackId) {
    bool valid = read<bool>(is);
    if (valid) {
      // Construct and deserialize a track
      tracks_[trackId] = std::make_unique<Track>(is, loadFn);
    } else {
      tracks_[trackId] = nullptr;
    }
  }

  size_t offset = read<size_t>(is);
  size_t size = read<size_t>(is);
  frames_.resize(offset, size);
  for (auto& frame : frames_) {
    frame = std::make_unique<Frame>();
    frame->deserialize(is);
  }

  // Go through the tracks and recreate the set of tracks in each frame
  // from the track structure.  This information, if saved to the serialized
  // format, would be redundant.
  for (size_t trackId = 0; trackId < numTracks; ++trackId) {
    const auto& t = tracks_[trackId];
    if (!t) {
      continue;
    }
    for (size_t frameId = offset; frameId < size + offset; ++frameId) {
      if (t->inFrame(static_cast<int>(frameId))) {
        frames_[frameId]->tracks.insert(trackId);
      }
    }
  }
}

// Checks internal consistency of the track table
template <typename Obs, typename Track, typename Frame>
bool TrackTable<Obs, Track, Frame>::checkSanity() {
  // Ensure that all tracks are reference in the frames sets
  for (size_t trackId = 0; trackId < tracks_.size(); ++trackId) {
    if (tracks_[trackId] == nullptr) {
      continue;
    }

    const auto& t = *tracks_[trackId];

    ensure(t.length() >= 1);

    for (int frame = t.firstFrame(); frame < t.lastFrame(); ++frame) {
      ensure(
          frames_[frame]->tracks.find(trackId) != frames_[frame]->tracks.end());
    }
  }

  // Ensure that all frames sets refer to actual tracks
  for (int frame = frames_.firstIndexI(); frame <= frames_.lastIndexI();
      ++frame) {
    for (const size_t trackId : frames_[frame]->tracks) {
      ensure(hasTrack(trackId));
      auto& t = track(trackId);
      ensure(t.firstFrame() <= frame && frame <= t.lastFrame());
    }
  }

  return true;
}

}} // namespace facebook::cp
