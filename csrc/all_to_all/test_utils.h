test_utils.h

#pragma once

#include "core/buffer.h"
#include "core/utils.h"

#include <cstdint>

#include <algorithm>
#include <ostream>
#include <random>
#include <vector>

namespace pplx {

/// Test data for all-to-all dispatch/combine.
template <typename T> struct RankTestData {
  const size_t m;
  const size_t hiddenDim;
  const size_t hiddenDimScale;
  const size_t numExperts;
  const size_t expertsPerToken;
  HostBuffer<T> x;
  HostBuffer<float> xScale;
  HostBuffer<uint32_t> indices; // Indices is length NUM_TOKENS * NUM_EXPERTS_PER_TOKEN
  HostBuffer<float> weights;
  HostBuffer<uint32_t> numRouted;

  RankTestData(
      std::mt19937 &gen,
      size_t maxNumTokens,
      size_t numExperts,
      size_t expertsPerToken,
      size_t hiddenDim,
      size_t blockSize
  );

  std::ostream &print(std::ostream &os) const;
};

template <typename T>
RankTestData<T>::RankTestData(
    std::mt19937 &gen,
    size_t maxNumTokens,
    size_t numExperts,
    size_t expertsPerToken,
    size_t hiddenDim,
    size_t blockSize
)
    : m(std::uniform_int_distribution<>(1, maxNumTokens)(gen)),
      hiddenDim(hiddenDim),
      hiddenDimScale(ceil_div(hiddenDim, blockSize)),
      numExperts(numExperts),
      expertsPerToken(expertsPerToken),
      x(m * hiddenDim),
      xScale(m * hiddenDimScale),
      indices(m * expertsPerToken),
      weights(m * expertsPerToken),
      numRouted(numExperts) {
  // Initialize routing counts for each expert to zero
  for (size_t i = 0; i < numExperts; ++i) {
    numRouted[i] = 0;
  }

  // For each token
  for (size_t i = 0; i < m; ++i) {
    // Create a vector containing all exert IDs
    std::vector<uint32_t> experts(numExperts);
    for (size_t j = 0; j < numExperts; ++j) {
      experts[j] = j;
    }
    // Shuffle exper IDs randomly so each token gets a random ordering of experts
    std::shuffle(experts.begin(), experts.end(), gen);
    // Create a random weight for each assignment
    std::uniform_real_distribution<> weight(0.0f, 1.0f);
    // For current token, pick the first `expertsPerToken` experts
    for (size_t j = 0; j < expertsPerToken; ++j) {
      // 'expert' is the j-th candidate from the shuffled list.
      uint32_t expert = experts[j];
      // Update the routing count for that expert.
      // 'loc' is the current count of tokens already routed to this expert
      // (it gets incremented here; it could be used later to know which "slot" in a per-expert buffer to place data).
      uint32_t loc = numRouted[expert]++;
      
      // Record this expert's ID in the indices array for token i.
      // The index in the array for token 'i' and assignment 'j' is: i * expertsPerToken + j.
      indices[i * expertsPerToken + j] = expert;
      
      // Weights here represet routing / gating values which are later 
      // needed for combine phase to determine how much each expert will contribute 
      // to final token representation 
      weights[i * expertsPerToken + j] = weight(gen);
    }
  }

  // Populate the tokens.
  if constexpr (std::is_integral<T>::value) {
    std::uniform_int_distribution<> value(-256, 256);
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < hiddenDim; ++j) {
        x[i * hiddenDim + j] = value(gen);
      }
    }
  } else {
    std::uniform_real_distribution<> value(-10.0f, 10.0f);
    // For each token
    for (size_t i = 0; i < m; ++i) {
      // For each value in feature vector
      for (size_t j = 0; j < hiddenDim; ++j) {
        x[i * hiddenDim + j] = value(gen);
      }
    }
  }

  // Populate the scales.
  std::uniform_real_distribution<> value(0, 100.0f);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < hiddenDimScale; ++j) {
      xScale[i * hiddenDimScale + j] = value(gen);
    }
  }
}

template <typename T> std::ostream &RankTestData<T>::print(std::ostream &os) const {
  for (unsigned j = 0; j < m; ++j) {
    os << "#" << j << " ->";
    for (unsigned k = 0; k < expertsPerToken; ++k) {
      auto e = indices[j * expertsPerToken + k];
      auto w = weights[j * expertsPerToken + k];
      os << " " << e << ":" << w;
    }
    os << std::endl;
    os << "    ";
    for (unsigned k = 0; k < hiddenDim; ++k) {
      os << (float)x[j * hiddenDim + k] << " ";
    }
    os << std::endl;
    os << "    ";
    for (unsigned k = 0; k < hiddenDimScale; ++k) {
      os << xScale[j * hiddenDimScale + k] << " ";
    }
    os << std::endl;
  }
  for (unsigned j = 0; j < numExperts; ++j) {
    const size_t numTokens = numRouted[j];
    os << "Expert " << j << ": " << numTokens << " tokens" << std::endl;
  }
  os << std::flush << std::endl;
  return os;
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const RankTestData<T> &data) {
  return data.print(os);
}

} // namespace pplx