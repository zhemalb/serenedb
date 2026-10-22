////////////////////////////////////////////////////////////////////////////////
/// DISCLAIMER
///
/// Copyright 2016 by EMC Corporation, All Rights Reserved
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
/// Copyright holder is EMC Corporation
///
/// @author Andrey Abramov
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <vector>

#include "iresearch/analysis/token_attributes.hpp"
#include "iresearch/formats/posting/common.hpp"
#include "iresearch/formats/posting/iterator_pos.hpp"
#include "iresearch/index/field_meta.hpp"
#include "iresearch/search/detail/column_collector.hpp"
#include "iresearch/search/scorers/score_function.hpp"
#include "iresearch/search/scorers/scorer.hpp"
#include "iresearch/utils/containers/fixed.hpp"
#include "iresearch/utils/empty.hpp"

namespace irs {

struct TermInterval {
  PosAttr::value_t offs_max{};
  PosAttr::value_t offs_min{};
  PosAttr::value_t lead_offset{};
  uint32_t term_group{};
};

template<bool Offs>
using FixedTermTraits = IteratorTraitsImpl<true, true, Offs>;

template<bool Offs>
using FixedTermPositionImpl = PositionImpl<FixedTermTraits<Offs>>;

template<bool Offs>
using FixedTermPosition = std::pair<FixedTermPositionImpl<Offs>*, TermInterval>;

template<typename T>
struct TermPositionTraits {
  using PositionImpl = PosAttr;

  static PosAttr::value_t Position(T& pos) {
    auto res = pos_limits::eof();
    pos.first->visit(&res, [](void* ctx, auto& it) {
      SDB_ASSERT(ctx);
      auto& position = *reinterpret_cast<PosAttr::value_t*>(ctx);
      if (pos_limits::valid(it.position->value())) {
        position = std::min(position, it.position->value());
      }
      return true;
    });
    return res;
  }

  static const TermInterval& Interval(const T& pos) noexcept {
    return pos.second;
  }

  static void ResetPos(const T&) {}
};

template<bool Offs>
struct TermPositionTraits<FixedTermPosition<Offs>> {
  using T = FixedTermPosition<Offs>;
  using PositionImpl = FixedTermPositionImpl<Offs>;

  static PosAttr::value_t Position(T& pos) noexcept {
    return pos.first->value();
  }

  static const TermInterval& Interval(const T& pos) noexcept {
    return pos.second;
  }

  static void ResetPos(const T& pos) { pos.first->reset(); }

  static uint32_t Read(T& pos, PosAttr::value_t* out, score_t*) {
    static_assert(!Offs);
    return pos.first->ReadPositions(out);
  }

  static const OffsAttr& Offsets(const T& pos) noexcept {
    static_assert(Offs);
    const auto* offs = irs::get<OffsAttr>(*pos.first);
    SDB_ASSERT(offs);
    return *offs;
  }
};

// clang-format off
// clang-format on

template<typename Iterator>
class SinglePositionStrategy {
 public:
  using Value = typename std::iterator_traits<Iterator>::value_type;
  using Traits = TermPositionTraits<Value>;
  using PositionImpl = Traits::PositionImpl;

  SinglePositionStrategy(Iterator& it, PositionImpl& lead_position,
                         bool = false)
    : _lead_it{it}, _lead_pos{lead_position} {}

  void NotifyNextLead(const Iterator&) noexcept {
    SDB_ASSERT(pos_limits::valid(_lead_pos.value()));
    SDB_ASSERT(!pos_limits::eof(_lead_pos.value()));
    _base_position = _lead_pos.value();
  }

  PosAttr::value_t NextPosition(const Iterator& it) noexcept {
    return _base_position + Traits::Interval(*it).offs_min;
  }

  static bool Match(PosAttr::value_t seek, PosAttr::value_t sought,
                    const Iterator&) noexcept {
    return seek == sought;
  }

  bool AdvanceIterators(bool match, PosAttr::value_t sought, const Iterator&,
                        Iterator& it) {
    if (!match) {
      SDB_ASSERT(sought > Traits::Interval(*it).lead_offset);
      _lead_pos.seek(sought - Traits::Interval(*it).lead_offset);
    }
    _base_position = sought;
    ++it;
    return match;
  }

 private:
  Iterator& _lead_it;
  PositionImpl& _lead_pos;
  PosAttr::value_t _base_position{pos_limits::eof()};
};

template<typename Iterator>
class IntervalPositionStrategy {
 public:
  using Value = typename std::iterator_traits<Iterator>::value_type;
  using Traits = TermPositionTraits<Value>;
  using PositionImpl = Traits::PositionImpl;

  IntervalPositionStrategy(Iterator& lead, PositionImpl& lead_position,
                           bool reversed = false)
    : _lead_it{lead}, _lead_pos{lead_position}, _reversed{reversed} {}

  void NotifyNextLead(const Iterator& end) noexcept {
    SDB_ASSERT(pos_limits::valid(_lead_pos.value()));
    SDB_ASSERT(!pos_limits::eof(_lead_pos.value()));
    _base_position = _lead_pos.value();
    _interval_delta = 0;
    if (_skipped) {
      for (auto reset_it = _lead_it + 1; reset_it != end; ++reset_it) {
        Traits::ResetPos(*reset_it);
      }
      _skipped = false;
    }
    _need_reset = false;
  }

  PosAttr::value_t NextPosition(const Iterator& it) {
    return Window(it).low + _interval_delta;
  }

  bool Match(PosAttr::value_t, PosAttr::value_t sought,
             const Iterator& it) const noexcept {
    return sought <= Window(it).high;
  }

  bool AdvanceIterators(bool match, PosAttr::value_t sought,
                        const Iterator& end, Iterator& it) {
    PosAttr::value_t lead_bound;
    if (AdvanceIterators(match, sought, end, it, lead_bound)) {
      return true;
    }
    _lead_pos.seek(lead_bound);
    return false;
  }

  bool AdvanceIterators(bool match, PosAttr::value_t sought,
                        const Iterator& end, Iterator& it,
                        PosAttr::value_t& lead_bound) {
    const auto fail_it = it;
    _interval_delta = 0;
    if (match) {
      ++it;
      if (_need_reset && it != end) {
        SDB_ASSERT(_skipped);
        for (auto reset_it = it; reset_it != end; ++reset_it) {
          Traits::ResetPos(*reset_it);
        }
        _need_reset = false;
      }
      SDB_ASSERT(!_need_reset || it == end);
      _base_position = sought;
      return true;
    }

    if (_skipped ? StepBack(it) : SkipBack(sought, fail_it, it)) {
      return true;
    }

    const auto bound = _skipped ? 0 : Reach(sought, _lead_it, fail_it);
    lead_bound = std::max(bound, _lead_pos.value() + 1);
    return false;
  }

 private:
  struct Range {
    PosAttr::value_t low;
    PosAttr::value_t high;
  };

  void Rebase(const Iterator& it) {
    const auto prev_base_it = it - 1;
    _base_position = prev_base_it == _lead_it ? _lead_pos.value()
                                              : Traits::Position(*prev_base_it);
  }

  bool SkipBack(PosAttr::value_t sought, const Iterator& fail_it,
                Iterator& it) {
    SDB_ASSERT(!_skipped);
    while (it != _lead_it + 1) {
      --it;
      Rebase(it);
      const auto window = Window(it);
      const auto want = Reach(sought, it, fail_it);
      if (want <= window.low || want > window.high) {
        continue;
      }
      _interval_delta = want - window.low;
      _skipped = true;
      return true;
    }
    return false;
  }

  bool StepBack(Iterator& it) {
    while (it != _lead_it + 1) {
      --it;
      const auto current_position = Traits::Position(*it);
      Rebase(it);
      const auto window = Window(it);
      if (current_position < window.high) {
        _need_reset = true;
        _interval_delta = current_position - window.low + 1;
        _skipped = true;
        return true;
      }
    }
    return false;
  }

  const TermInterval& Gap(const Iterator& it) const noexcept {
    return Traits::Interval(_reversed ? *(it - 1) : *it);
  }

  Range Window(const Iterator& it) const noexcept {
    const auto& gap = Gap(it);
    if (_reversed) {
      return {_base_position > gap.offs_max ? _base_position - gap.offs_max
                                            : pos_limits::min(),
              _base_position > gap.offs_min ? _base_position - gap.offs_min
                                            : pos_limits::invalid()};
    }
    return {_base_position + gap.offs_min, _base_position + gap.offs_max};
  }

  PosAttr::value_t Reach(PosAttr::value_t sought, Iterator from,
                         Iterator to) const noexcept {
    if (_reversed) {
      PosAttr::value_t span = 0;
      for (auto it = from + 1; it <= to; ++it) {
        span += Gap(it).offs_min;
      }
      return sought + span;
    }
    SDB_ASSERT(sought + Traits::Interval(*from).lead_offset >=
               Traits::Interval(*to).lead_offset);
    return sought - Traits::Interval(*to).lead_offset +
           Traits::Interval(*from).lead_offset;
  }

  Iterator& _lead_it;
  PositionImpl& _lead_pos;
  PosAttr::value_t _base_position{pos_limits::eof()};
  PosAttr::value_t _interval_delta{0};
  bool _reversed{false};
  bool _skipped{false};
  bool _need_reset{false};
};

template<typename TermPositionT, bool Offs, bool HasFreq, bool HasIntervals,
         bool HasBoost = false, size_t N = 0>
class PhraseMatcher {
 public:
  using TermPosition = TermPositionT;
  using Traits = TermPositionTraits<TermPosition>;
  using Positions = containers::Fixed<TermPosition, N>;
  using ExecutionStrategy =
    std::conditional_t<HasIntervals,
                       IntervalPositionStrategy<typename Positions::iterator>,
                       SinglePositionStrategy<typename Positions::iterator>>;

  static_assert(!HasBoost || HasFreq);

  static constexpr bool kHasScale = HasBoost;
  static constexpr bool kHasFreq = HasFreq;
  static constexpr bool kOffsets = Offs;
  static constexpr size_t kRarestLeadSlots = 4;
  static constexpr bool kRarestLead =
    HasIntervals && !Offs && (N == 0 || N >= kRarestLeadSlots);
  static constexpr bool kCounting = HasIntervals && HasFreq;
  static constexpr uint32_t kCheckWindow = 8;
  static constexpr uint32_t kCheckFails = 4;
  static constexpr uint32_t kMaxSkip = 1024;

  explicit PhraseMatcher(size_t size) : _pos{size}, _tuples{size} {}

  TermPosition& Position(size_t i) noexcept {
    SDB_ASSERT(i < _pos.size());
    return _pos[i];
  }

  void Finish() {
    SDB_ASSERT(!_pos.empty());
    SDB_ASSERT(_pos.front().second.offs_min == 0);
    SDB_ASSERT(_pos.front().second.offs_max == 0);
    if constexpr (HasIntervals) {
      uint64_t scale = 1;
      for (const auto& slot : _pos) {
        scale *= slot.second.offs_max - slot.second.offs_min + 1;
        if (scale >= kMaxFreq) {
          scale = kMaxFreq;
          break;
        }
      }
      _freq_scale = static_cast<uint32_t>(scale);
    }
  }

  template<bool Ordered = false>
  IRS_FORCE_INLINE bool Match() {
    _phrase_freq = NextPosition<Ordered>();
    return _phrase_freq != 0;
  }

  uint32_t GetFreq() const noexcept { return _phrase_freq; }

  score_t GetScale() const noexcept
    requires(kHasScale)
  {
    if (_phrase_freq == 0) {
      return kNoBoost;
    }
    return _phrase_boost;
  }

  uint32_t DocFreqBound() {
    const auto freq = LeadFreq();
    if constexpr (HasIntervals) {
      uint64_t by_window = uint64_t{freq} * _freq_scale;
      uint64_t by_occurrence = 1;
      for (const auto& slot : _pos) {
        by_occurrence *= slot.first->DocFreq();
        if (by_occurrence >= by_window) {
          return static_cast<uint32_t>(std::min<uint64_t>(by_window, kMaxFreq));
        }
      }
      return static_cast<uint32_t>(std::min<uint64_t>(by_occurrence, kMaxFreq));
    }
    return freq;
  }

  uint32_t FreqBoundOf(uint32_t freq) const noexcept {
    if constexpr (HasIntervals) {
      return static_cast<uint32_t>(
        std::min<uint64_t>(uint64_t{freq} * _freq_scale, kMaxFreq));
    }
    return freq;
  }

  std::pair<uint32_t, uint32_t> Offsets() const noexcept
    requires(Offs)
  {
    return {Traits::Offsets(_pos.front()).start,
            Traits::Offsets(_pos.back()).end};
  }

  bool NextAlignment()
    requires(Offs)
  {
    return NextPosition() != 0;
  }

 private:
  template<bool Ordered = false>
  IRS_FORCE_INLINE uint32_t NextPosition() {
    if constexpr (HasBoost) {
      _phrase_boost = 0.f;
    }
    if constexpr (kCounting) {
      return CountTuples();
    } else if constexpr (HasIntervals || Offs) {
      if constexpr (kRarestLead) {
        if (_pos.size() >= kRarestLeadSlots) {
          return NextPositionRarest();
        }
      }
      return NextPositionGeneric<Ordered>();
    } else {
      return NextPositionOptimized<Ordered>();
    }
  }

  uint32_t LeadFreq() {
    if constexpr (kRarestLead) {
      if (_pos.size() >= kRarestLeadSlots) {
        return Rarest()->first->DocFreq();
      }
    }
    OrderByDocFreq();
    return _pos.front().first->DocFreq();
  }

  IRS_FORCE_INLINE void TakeBoost() noexcept {
    if constexpr (HasBoost) {
      score_t match = kNoBoost;
      for (const auto& slot : _pos) {
        match = std::min(match, Traits::Boost(slot));
      }
      _phrase_boost = std::max(_phrase_boost, match);
    }
  }

  template<bool Ordered = false>
  uint32_t NextPositionGeneric() {
    if constexpr (!Ordered) {
      OrderByDocFreq();
    }
    auto& lead = *_pos.front().first;
    lead.next();
    auto lead_it = std::begin(_pos);
    ExecutionStrategy strategy{lead_it, lead, _reversed};
    SDB_ASSERT(_pos.size() > 1);

    for (auto end = std::end(_pos); !pos_limits::eof(lead.value());) {
      strategy.NotifyNextLead(end);
      bool match = true;
      for (auto it = lead_it + 1; it != end;) {
        auto& pos = *it->first;

        const auto term_position = strategy.NextPosition(it);
        if (!pos_limits::valid(term_position)) {
          return 0;
        }
        const auto sought = pos.seek(term_position);

        if (pos_limits::eof(sought)) {
          return 0;
        }
        match = strategy.AdvanceIterators(
          strategy.Match(term_position, sought, it), sought, end, it);
        if (!match) {
          break;
        }
      }
      if (match) {
        return 1;
      }
    }

    return 0;
  }

  struct Tuples {
    explicit Tuples(size_t slots) : starts(slots + 1) {}

    std::vector<uint32_t> starts;
    std::vector<PosAttr::value_t> positions;
    std::vector<uint64_t> counts;
    [[no_unique_address]] utils::Need<HasBoost, std::vector<score_t>> boosts;
    [[no_unique_address]] utils::Need<HasBoost, std::vector<score_t>> best;
    uint32_t checks = 0;
    uint32_t fails = 0;
    uint32_t skip = 0;
    uint32_t interval = 0;
  };

  TermPosition& Slot(size_t i) noexcept {
    return _pos[_reversed ? _pos.size() - 1 - i : i];
  }

  score_t* BoostsAt(uint32_t at) noexcept {
    if constexpr (HasBoost) {
      return _tuples.boosts.data() + at;
    } else {
      return nullptr;
    }
  }

  bool Exists() {
    if constexpr (kRarestLead) {
      if (_pos.size() >= kRarestLeadSlots) {
        return NextPositionRarest() != 0;
      }
    }
    return NextPositionGeneric() != 0;
  }

  uint32_t CountTuples() {
    static_assert(kCounting && !Offs);
    auto& t = _tuples;
    const size_t n = _pos.size();
    SDB_ASSERT(n > 1);
    if (t.skip != 0) {
      --t.skip;
    } else {
      const bool exists = Exists();
      t.fails += !exists;
      if (++t.checks == kCheckWindow) {
        t.interval = t.fails >= kCheckFails
                       ? 0
                       : std::clamp(2 * t.interval, kCheckWindow, kMaxSkip);
        t.skip = t.interval;
        t.checks = 0;
        t.fails = 0;
      }
      if (!exists) {
        return 0;
      }
      for (auto& slot : _pos) {
        Traits::ResetPos(slot);
      }
    }
    size_t total = 0;
    for (const auto& slot : _pos) {
      total += slot.first->DocFreq();
    }
    if (t.positions.size() < total) {
      t.positions.resize(total);
      t.counts.resize(total);
      if constexpr (HasBoost) {
        t.boosts.resize(total);
        t.best.resize(total);
      }
    }
    auto* const positions = t.positions.data();
    auto* const counts = t.counts.data();
    uint32_t at = 0;
    for (size_t i = 0; i != n; ++i) {
      t.starts[i] = at;
      at += Traits::Read(Slot(i), positions + at, BoostsAt(at));
    }
    t.starts[n] = at;

    for (uint32_t k = t.starts[n - 1]; k != at; ++k) {
      counts[k] = 1;
      if constexpr (HasBoost) {
        t.best[k] = t.boosts[k];
      }
    }
    for (size_t i = n - 1; i-- != 0;) {
      const auto& gap = Traits::Interval(Slot(i + 1));
      const auto* const next = positions + t.starts[i + 1];
      const auto* const next_counts = counts + t.starts[i + 1];
      const uint32_t m = t.starts[i + 2] - t.starts[i + 1];
      uint32_t lo = 0;
      uint32_t hi = 0;
      uint64_t sum = 0;
      for (uint32_t k = t.starts[i], end = t.starts[i + 1]; k != end; ++k) {
        const uint64_t base = positions[k];
        const uint64_t high = base + gap.offs_max;
        const uint64_t low = base + gap.offs_min;
        while (hi != m && next[hi] <= high) {
          sum += next_counts[hi];
          ++hi;
        }
        while (lo != hi && next[lo] < low) {
          sum -= next_counts[lo];
          ++lo;
        }
        counts[k] = sum;
        if constexpr (HasBoost) {
          const auto* const next_best = t.best.data() + t.starts[i + 1];
          score_t best = 0.f;
          for (uint32_t j = lo; j != hi; ++j) {
            best = std::max(best, next_best[j]);
          }
          t.best[k] = sum != 0 ? std::min(t.boosts[k], best) : 0.f;
        }
      }
    }
    uint64_t freq = 0;
    for (uint32_t k = 0, end = t.starts[1]; k != end; ++k) {
      freq += counts[k];
      if constexpr (HasBoost) {
        _phrase_boost = std::max(_phrase_boost, t.best[k]);
      }
    }
    return static_cast<uint32_t>(std::min<uint64_t>(freq, kMaxFreq));
  }

 private:
  struct Side {
    bool matched = false;
    bool exhausted = false;
    PosAttr::value_t lead_bound = 0;
  };

  uint32_t NextPositionRarest() {
    SDB_ASSERT(_pos.size() > 1);
    const auto lead_it = Rarest();
    if (lead_it == _pos.begin()) {
      return NextPositionFrom<true, false>(lead_it);
    }
    if (lead_it + 1 == _pos.end()) {
      return NextPositionFrom<false, true>(lead_it);
    }
    const auto rarest = [](auto first, auto last) {
      auto freq = first->first->DocFreq();
      for (++first; first != last; ++first) {
        freq = std::min(freq, first->first->DocFreq());
      }
      return freq;
    };
    if (rarest(_pos.begin(), lead_it) < rarest(lead_it + 1, _pos.end())) {
      return NextPositionFrom<true, true, true>(lead_it);
    }
    return NextPositionFrom<true, true>(lead_it);
  }

  template<bool Forward, bool Backward, bool BackwardFirst = false>
  uint32_t NextPositionFrom(typename Positions::iterator lead_it) {
    using Iterator = typename Positions::iterator;
    using ReverseIterator = std::reverse_iterator<Iterator>;
    auto& lead = *lead_it->first;
    const auto end = _pos.end();
    auto reverse_lead_it = ReverseIterator{lead_it + 1};
    const ReverseIterator reverse_end{_pos.begin()};
    IntervalPositionStrategy<Iterator> forward{lead_it, lead};
    IntervalPositionStrategy<ReverseIterator> backward{reverse_lead_it, lead,
                                                       true};
    for (lead.next(); !pos_limits::eof(lead.value());) {
      Side after{.matched = true};
      Side before{.matched = true};
      if constexpr (BackwardFirst) {
        before = Complete(backward, reverse_lead_it, reverse_end);
        if (before.matched) {
          after = Complete(forward, lead_it, end);
        }
      } else {
        if constexpr (Forward) {
          after = Complete(forward, lead_it, end);
        }
        if constexpr (Backward) {
          if (after.matched) {
            before = Complete(backward, reverse_lead_it, reverse_end);
          }
        }
      }
      if (after.matched && before.matched) {
        return 1;
      }
      if (after.exhausted || before.exhausted) {
        return 0;
      }
      lead.seek(
        std::max({after.lead_bound, before.lead_bound, lead.value() + 1}));
    }
    return 0;
  }

  template<typename Strategy, typename Iterator>
  IRS_FORCE_INLINE Side Complete(Strategy& strategy, const Iterator& lead_it,
                                 const Iterator& end) {
    Side side;
    strategy.NotifyNextLead(end);
    for (auto it = lead_it + 1; it != end;) {
      auto& pos = *it->first;

      const auto term_position = strategy.NextPosition(it);
      if (!pos_limits::valid(term_position)) {
        side.exhausted = true;
        return side;
      }
      const auto sought = pos.seek(term_position);

      if (pos_limits::eof(sought)) {
        side.exhausted = true;
        return side;
      }
      if (!strategy.AdvanceIterators(strategy.Match(term_position, sought, it),
                                     sought, end, it, side.lead_bound)) {
        return side;
      }
    }
    side.matched = true;
    return side;
  }

  auto Rarest() {
    return absl::c_min_element(_pos, [](const auto& l, const auto& r) {
      return l.first->DocFreq() < r.first->DocFreq();
    });
  }

  void OrderByDocFreq() {
    if constexpr (Offs) {
    } else if constexpr (HasIntervals) {
      if (_pos.back().first->DocFreq() < _pos.front().first->DocFreq()) {
        absl::c_reverse(_pos);
        _reversed = !_reversed;
      }
    } else {
      absl::c_sort(_pos, [](const auto& l, const auto& r) {
        return l.first->DocFreq() < r.first->DocFreq();
      });
    }
  }

  template<bool Ordered = false>
  uint32_t NextPositionOptimized() {
    if constexpr (!Ordered) {
      OrderByDocFreq();
    }
    auto begin = _pos.begin();
    auto end = _pos.end();

    const auto new_lead_offset = begin->second.lead_offset;
    auto& lead = *begin->first;
    ++begin;
    auto lead_pos = lead.seek(pos_limits::min() + new_lead_offset);

    uint32_t phrase_freq = 0;
    while (true) {
    restart:
      if (pos_limits::eof(lead_pos)) [[unlikely]] {
        return phrase_freq;
      }
      for (auto it = begin; it != end; ++it) {
        const auto target =
          (lead_pos - new_lead_offset) + it->second.lead_offset;
        const auto sought = it->first->seek(target);
        if (sought != target) {
          if (pos_limits::eof(sought)) [[unlikely]] {
            return phrase_freq;
          }
          lead_pos =
            lead.seek((sought - it->second.lead_offset) + new_lead_offset);
          goto restart;
        }
      }
      if constexpr (HasFreq) {
        ++phrase_freq;
        TakeBoost();
        lead.next();
        lead_pos = lead.value();
      } else {
        return 1;
      }
    }
  }

  Positions _pos;
  [[no_unique_address]] utils::Need<kCounting, Tuples> _tuples;
  uint32_t _phrase_freq = 0;
  [[no_unique_address]] utils::Need<HasBoost, score_t> _phrase_boost{};
  uint32_t _freq_scale = 1;
  bool _reversed = false;
};

template<bool Offs, bool HasFreq, bool HasIntervals, size_t N = 0>
using FixedPhraseMatcher =
  PhraseMatcher<FixedTermPosition<Offs>, Offs, HasFreq, HasIntervals, false, N>;

}  // namespace irs
