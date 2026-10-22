////////////////////////////////////////////////////////////////////////////////
/// DISCLAIMER
///
/// Copyright 2026 SereneDB GmbH, Berlin, Germany
///
/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
///
///     http://www.apache.org/licenses/LICENSE-2.0
///
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.
///
/// Copyright holder is SereneDB GmbH, Berlin, Germany
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <absl/container/inlined_vector.h>

#include <cstdint>
#include <cstring>
#include <duckdb/common/types/string_type.hpp>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "iresearch/analysis/text/dict/string_table.hpp"
#include "iresearch/analysis/text/term_view.hpp"
#include "iresearch/formats/term_reader.hpp"
#include "iresearch/index/index_reader.hpp"
#include "iresearch/search/detail/phrase_slop_matcher.hpp"
#include "iresearch/search/detail/term_predicate.hpp"
#include "iresearch/search/filters/phrase_filter.hpp"
#include "iresearch/utils/string.hpp"

namespace irs {

struct PhraseWordStat {
  uint64_t docs = 0;
  uint64_t freq = 0;
};

bool IsStandalone(const ByPhraseOptions& phrase) noexcept;
bool HasPatternParts(const ByPhraseOptions& phrase) noexcept;
bool HasOpenGap(const ByPhraseOptions& phrase) noexcept;
std::vector<PhraseWordStat> PhraseWordStats(const ByPhraseOptions& phrase,
                                            const TermReader& reader);
std::vector<PhraseWordStat> PhraseWordStats(const ByPhraseOptions& phrase,
                                            const IndexReader& index,
                                            field_id field);

struct CompiledPhrase {
  static constexpr uint32_t kMaxBits = 64;
  static constexpr uint32_t kMaxPlainSlots = 8;

  struct Slot {
    PosAttr::value_t offs_min = 0;
    PosAttr::value_t offs_max = 0;
    std::optional<duckdb::string_t> word;
    TermPredicate::ptr pattern;
    uint64_t bit = 0;
  };

  struct Accept {
    absl::InlinedVector<uint32_t, 4> slots;
    uint64_t mask = 0;
  };

  struct Sieve {
    static constexpr size_t kBitsPerTerm = 8;
    static constexpr size_t kMaxWords = 4096;

    std::vector<uint64_t> words;
    uint32_t shift = 0;
  };

  struct Slop {
    PosAttr::value_t max = 0;
    std::vector<int64_t> offsets;
    std::vector<detail::slop::GroupPair> pairs;
  };

  struct Anchor {
    uint32_t slot = 0;
    uint64_t left = 0;
    uint64_t right = 0;
  };

  struct Automaton {
    struct Extra {
      uint64_t range = 0;
      uint64_t bit = 0;
    };

    std::vector<Extra> extras;
    uint64_t wild = 0;
    uint64_t last = 0;
  };

  CompiledPhrase(const ByPhraseOptions& phrase,
                 std::vector<std::vector<bstring>> expanded,
                 std::span<const PhraseWordStat> words);

  CompiledPhrase(CompiledPhrase&&) = delete;
  CompiledPhrase& operator=(CompiledPhrase&&) = delete;

  std::vector<Slot> slots;
  std::vector<bstring> terms;
  Slop slop;
  bool open = false;
  bool plain = false;
  analysis::dict::StringMap<std::string_view, Accept> accept;
  Sieve sieve;
  std::vector<uint32_t> pattern_slots;
  std::optional<Anchor> anchor;
  std::optional<Automaton> automaton;
};

inline uint64_t SieveHash(const duckdb::string_t& term) noexcept {
  uint64_t head = 0;
  std::memcpy(&head, &term, sizeof head);
  head ^= head >> 33;
  head *= UINT64_C(0xff51afd7ed558ccd);
  head ^= head >> 33;
  head *= UINT64_C(0xc4ceb9fe1a85ec53);
  return head ^ (head >> 33);
}

inline uint64_t SieveBits(uint64_t hash) noexcept {
  return (uint64_t{1} << (hash & 63)) | (uint64_t{1} << ((hash >> 6) & 63));
}

inline bool MayAccept(const CompiledPhrase& phrase,
                      const duckdb::string_t& term) noexcept {
  const auto hash = SieveHash(term);
  const auto bits = SieveBits(hash);
  return (phrase.sieve.words[hash >> phrase.sieve.shift] & bits) == bits;
}

inline const CompiledPhrase::Accept* FindAccepted(
  const CompiledPhrase& phrase, const duckdb::string_t& term) noexcept {
  return MayAccept(phrase, term) ? phrase.accept.Find(term) : nullptr;
}

bool AcceptsTerm(const CompiledPhrase& phrase, const CompiledPhrase::Slot& slot,
                 const duckdb::string_t& term);

template<typename Visitor>
void ForEachPattern(const CompiledPhrase& phrase, const duckdb::string_t& term,
                    Visitor&& visit) {
  if (phrase.pattern_slots.empty()) {
    return;
  }
  const auto view = AsBytesView(term);
  for (const auto slot : phrase.pattern_slots) {
    if (phrase.slots[slot].pattern->Accepts(view)) {
      visit(slot);
    }
  }
}

template<typename Visitor>
void ForEachSlot(const CompiledPhrase& phrase, const duckdb::string_t& term,
                 Visitor&& visit) {
  if (const auto* found = FindAccepted(phrase, term)) {
    for (const auto slot : found->slots) {
      visit(slot);
    }
  }
  ForEachPattern(phrase, term, visit);
}

inline uint64_t MaskOf(const CompiledPhrase& phrase,
                       const duckdb::string_t& term) {
  uint64_t mask = 0;
  if (const auto* found = FindAccepted(phrase, term)) {
    mask = found->mask;
  }
  ForEachPattern(phrase, term,
                 [&](uint32_t slot) { mask |= phrase.slots[slot].bit; });
  return mask;
}

template<typename Visitor>
void ForEachAccepting(const CompiledPhrase& phrase,
                      const duckdb::string_t& term, Visitor&& visit) {
  if (!phrase.plain) {
    ForEachSlot(phrase, term, visit);
    return;
  }
  const auto n = static_cast<uint32_t>(phrase.slots.size());
  for (uint32_t slot = 0; slot != n; ++slot) {
    if (*phrase.slots[slot].word == term) {
      visit(slot);
    }
  }
}

}  // namespace irs
