// Copyright 2004-present Facebook. All Rights Reserved.

// Include the track table *inl* file here because the compiler needs to see
// the implementation to instantiate the template classes.

#include "core/TrackTable-impl.h"

#include "Processor.h"

namespace facebook {
namespace cp {

template class TrackBaseSequential<DepthVideoObs>;
template class TrackTable<
    DepthVideoObs, DepthVideoTrack, DepthVideoTrackedFrame>;

}} // namespace facebook::cp
