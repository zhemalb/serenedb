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

#include <absl/algorithm/container.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "iresearch/analysis/token_attributes.hpp"
#include "iresearch/formats/posting/iterator_pos.hpp"
#include "iresearch/index/iterators.hpp"
#include "iresearch/search/detail/phrase_matcher.hpp"
#include "iresearch/search/scorers/score_function.hpp"
#include "iresearch/utils/containers/fixed.hpp"
#include "iresearch/utils/empty.hpp"
#include "iresearch/utils/type_limits.hpp"

namespace irs {
namespace detail::ordered {

struct Gap {
  PosAttr::value_t lo;
  PosAttr::value_t hi;
};

struct Result {
  uint64_t freq = 0;
  double weight = 0.0;
  score_t boost = 0.f;
  bool any = false;
};

struct Entry {
  PosAttr::value_t pos = 0;
  PosAttr::value_t start = 0;
  uint32_t start_offset = 0;
  score_t boost = kNoBoost;
};

struct Level {
  PosAttr::value_t cur = pos_limits::eof();
  bool started = false;
  bool has = false;
  Entry best;
};

struct SpanOffsets {
  uint32_t start;
  uint32_t end;
};

inline bool Reaches(const Gap& gap, PosAttr::value_t from,
                    PosAttr::value_t to) noexcept {
  return gap.hi == pos_limits::kOpenGap ||
         static_cast<uint64_t>(from) + gap.hi >= to;
}

template<typename Source, bool Offs, bool HasBoost>
class Walker {
 public:
  Walker(Source& source, std::span<const Gap> gaps, std::vector<Level>& levels)
    : _source{source}, _gaps{gaps}, _levels{levels} {
    SDB_ASSERT(_levels.size() == gaps.size() - 1);
    for (auto& level : _levels) {
      level.started = false;
      level.has = false;
    }
  }

  const Entry* Query(size_t i, PosAttr::value_t x) {
    auto& level = _levels[i];
    if (!level.started) {
      level.started = true;
      level.cur = _source.Next(i);
    }
    while (!pos_limits::eof(level.cur) && level.cur <= x) {
      const auto cur = level.cur;
      Entry entry;
      bool defined = false;
      if (i == 0) {
        entry.pos = cur;
        entry.start = cur;
        if constexpr (Offs) {
          entry.start_offset = _source.StartOffset(0);
        }
        if constexpr (HasBoost) {
          entry.boost = _source.Boost(0);
        }
        defined = true;
      } else if (cur >= _gaps[i].lo) {
        const auto* prev = Query(i - 1, cur - _gaps[i].lo);
        if (prev && Reaches(_gaps[i], prev->pos, cur)) {
          entry.pos = cur;
          entry.start = prev->start;
          entry.start_offset = prev->start_offset;
          if constexpr (HasBoost) {
            entry.boost = std::min(prev->boost, _source.Boost(i));
          }
          defined = true;
        }
      }
      if (defined) {
        level.best = entry;
        level.has = true;
      }
      level.cur = _source.Next(i);
    }
    return level.has ? &level.best : nullptr;
  }

 private:
  Source& _source;
  std::span<const Gap> _gaps;
  std::vector<Level>& _levels;
};

inline bool AllOpen(std::span<const Gap> gaps) noexcept {
  return absl::c_all_of(gaps.subspan(1), [](const Gap& gap) {
    return gap.hi == pos_limits::kOpenGap;
  });
}

template<typename Source>
bool Exists(Source& source, std::span<const Gap> gaps) {
  auto pos = source.Next(0);
  for (size_t i = 1; i != gaps.size() && !pos_limits::eof(pos); ++i) {
    const auto target = uint64_t{pos} + gaps[i].lo;
    if (target >= pos_limits::eof()) {
      return false;
    }
    pos = source.Seek(i, static_cast<PosAttr::value_t>(target));
  }
  return !pos_limits::eof(pos);
}

template<bool Offs, bool HasBoost, typename Source, typename Collect>
Result Match(Source& source, std::span<const Gap> gaps,
             PosAttr::value_t min_extent, bool early_exit,
             std::vector<Level>& levels, Collect&& collect) {
  Result res;
  const size_t n = gaps.size();
  SDB_ASSERT(n >= 2);
  if (early_exit && AllOpen(gaps)) {
    res.any = Exists(source, gaps);
    res.freq = res.any ? 1 : 0;
    return res;
  }
  const auto& last = gaps[n - 1];
  uint64_t last_start = 0;
  bool emitted = false;
  double weight = 0.0;
  const auto emit = [&](const Entry& pred, PosAttr::value_t e) {
    if (emitted && pred.start <= last_start) {
      return false;
    }
    emitted = true;
    last_start = pred.start;
    const int64_t width = static_cast<int64_t>(e) - pred.start + 1;
    const int64_t extra =
      std::max<int64_t>(width - static_cast<int64_t>(min_extent) + 1, 1);
    ++res.freq;
    weight += 1.0 / static_cast<double>(extra);
    if constexpr (HasBoost) {
      res.boost =
        std::max(res.boost, std::min(pred.boost, source.Boost(n - 1)));
    }
    if constexpr (Offs) {
      collect(pred.start, e, pred.start_offset, source.EndOffset(n - 1));
    } else {
      collect(pred.start, e, 0u, 0u);
    }
    return early_exit;
  };

  if (n == 2) {
    Entry pred;
    bool has = false;
    auto cur = source.Next(0);
    for (auto e = source.Next(1); !pos_limits::eof(e); e = source.Next(1)) {
      if (e < last.lo) {
        continue;
      }
      const auto x = e - last.lo;
      for (; !pos_limits::eof(cur) && cur <= x; cur = source.Next(0)) {
        pred.pos = cur;
        pred.start = cur;
        if constexpr (Offs) {
          pred.start_offset = source.StartOffset(0);
        }
        if constexpr (HasBoost) {
          pred.boost = source.Boost(0);
        }
        has = true;
      }
      if (has && Reaches(last, pred.pos, e) && emit(pred, e)) {
        break;
      }
    }
  } else {
    Walker<Source, Offs, HasBoost> walker{source, gaps, levels};
    for (auto e = source.Next(n - 1); !pos_limits::eof(e);
         e = source.Next(n - 1)) {
      if (e < last.lo) {
        continue;
      }
      const auto* pred = walker.Query(n - 2, e - last.lo);
      if (pred && Reaches(last, pred->pos, e) && emit(*pred, e)) {
        break;
      }
    }
  }
  res.weight = weight;
  res.any = emitted;
  return res;
}

struct Scratch {
  std::vector<size_t> next;
  std::vector<Level> levels;
};

class VectorSource {
 public:
  VectorSource(const std::vector<std::vector<PosAttr::value_t>>& slots,
               std::vector<size_t>& next)
    : _slots{slots}, _next{next} {
    _next.assign(slots.size(), 0);
  }

  PosAttr::value_t Next(size_t i) {
    const auto& slot = _slots[i];
    return _next[i] < slot.size() ? slot[_next[i]++] : pos_limits::eof();
  }

  PosAttr::value_t Seek(size_t i, PosAttr::value_t target) {
    const auto& slot = _slots[i];
    auto& next = _next[i];
    while (next < slot.size() && slot[next] < target) {
      ++next;
    }
    return next < slot.size() ? slot[next++] : pos_limits::eof();
  }

  uint32_t StartOffset(size_t) const noexcept { return 0; }

  uint32_t EndOffset(size_t) const noexcept { return 0; }

  score_t Boost(size_t) const noexcept { return kNoBoost; }

 private:
  const std::vector<std::vector<PosAttr::value_t>>& _slots;
  std::vector<size_t>& _next;
};

inline Result Run(
  const std::vector<std::vector<PosAttr::value_t>>& slots,
  std::span<const Gap> gaps, bool early_exit, Scratch& scratch,
  std::vector<std::pair<PosAttr::value_t, PosAttr::value_t>>* spans = nullptr) {
  if (spans) {
    spans->clear();
  }
  SDB_ASSERT(gaps.size() == slots.size());
  if (slots.size() < 2) {
    return {};
  }
  PosAttr::value_t min_extent = 1;
  for (size_t i = 1; i != gaps.size(); ++i) {
    min_extent += gaps[i].lo;
  }
  VectorSource source{slots, scratch.next};
  scratch.levels.resize(gaps.size() - 1);
  auto res = Match<false, false>(
    source, gaps, min_extent, early_exit, scratch.levels,
    [&](PosAttr::value_t start, PosAttr::value_t end, uint32_t, uint32_t) {
      if (spans) {
        spans->emplace_back(start, end);
      }
    });
  if (early_exit) {
    res.freq = res.any ? 1 : 0;
  }
  return res;
}

template<typename Positions, bool Offs, bool HasBoost>
class IterSource {
 public:
  using Traits = TermPositionTraits<typename Positions::value_type>;

  explicit IterSource(Positions& pos) : _pos{pos} {}

  PosAttr::value_t Next(size_t i) {
    auto& it = *_pos[i].first;
    return it.next() ? it.value() : pos_limits::eof();
  }

  PosAttr::value_t Seek(size_t i, PosAttr::value_t target) {
    return _pos[i].first->seek(target);
  }

  uint32_t StartOffset(size_t i) const noexcept
    requires(Offs)
  {
    return Traits::Offsets(_pos[i]).start;
  }

  uint32_t EndOffset(size_t i) const noexcept
    requires(Offs)
  {
    return Traits::Offsets(_pos[i]).end;
  }

  score_t Boost(size_t i) const noexcept {
    if constexpr (HasBoost) {
      return Traits::Boost(_pos[i]);
    } else {
      return kNoBoost;
    }
  }

 private:
  Positions& _pos;
};

}  // namespace detail::ordered

template<typename TermPositionT, bool Offs, bool HasFreq, size_t N = 0,
         bool HasBoost = false>
class PhraseOrderedMatcher {
 public:
  using TermPosition = TermPositionT;
  using Traits = TermPositionTraits<TermPosition>;
  using Positions = containers::Fixed<TermPosition, N>;

  static constexpr bool kHasScale = HasFreq;
  static constexpr bool kHasFreq = HasFreq;
  static constexpr bool kOffsets = Offs;

  static_assert(HasFreq || !HasBoost);

  explicit PhraseOrderedMatcher(size_t size) : _pos{size}, _gaps{size} {}

  TermPosition& Position(size_t i) noexcept {
    SDB_ASSERT(i < _pos.size());
    return _pos[i];
  }

  void Finish() {
    SDB_ASSERT(_pos.size() >= 2);
    _min_extent = 1;
    for (size_t i = 0; i != _pos.size(); ++i) {
      const auto& interval = _pos[i].second;
      _gaps[i] = {interval.offs_min, interval.offs_max};
      if (i != 0) {
        _min_extent += interval.offs_min;
      }
    }
    _levels.resize(_pos.size() - 1);
  }

  template<bool Ordered = false>
  IRS_FORCE_INLINE bool Match() {
    _phrase_freq = 0;
    if constexpr (Offs) {
      _start_offset = 0;
      _end_offset = 0;
      _matches.clear();
      _match_idx = 0;
    }
    detail::ordered::IterSource<Positions, Offs, HasBoost> source{_pos};
    const auto res = detail::ordered::Match<Offs, HasBoost>(
      source, {_gaps.data(), _gaps.size()}, _min_extent, !HasFreq, _levels,
      [&]([[maybe_unused]] PosAttr::value_t start,
          [[maybe_unused]] PosAttr::value_t end,
          [[maybe_unused]] uint32_t start_offset,
          [[maybe_unused]] uint32_t end_offset) {
        if constexpr (Offs && HasFreq) {
          _matches.push_back({start_offset, end_offset});
        }
      });
    if (!res.any) {
      return false;
    }
    Take(res);
    return true;
  }

  uint32_t GetFreq() const noexcept { return _phrase_freq; }

  uint32_t DocFreqBound()
    requires(HasFreq)
  {
    uint32_t bound = std::numeric_limits<uint32_t>::max();
    for (const auto& slot : _pos) {
      bound = std::min(bound, slot.first->DocFreq());
    }
    return bound;
  }

  uint32_t FreqBoundOf(uint32_t freq) const noexcept
    requires(HasFreq)
  {
    return freq;
  }

  score_t GetScale() const noexcept
    requires(kHasScale)
  {
    if constexpr (HasBoost) {
      return _phrase_scale * _phrase_boost;
    } else {
      return _phrase_scale;
    }
  }

  std::pair<uint32_t, uint32_t> Offsets() const noexcept
    requires(Offs)
  {
    return {_start_offset, _end_offset};
  }

  bool NextAlignment()
    requires(Offs)
  {
    if constexpr (!HasFreq) {
      return false;
    } else {
      if (_match_idx >= _matches.size()) {
        return false;
      }
      _start_offset = _matches[_match_idx].start;
      _end_offset = _matches[_match_idx].end;
      ++_match_idx;
      return true;
    }
  }

 private:
  void Take(const detail::ordered::Result& res) {
    if constexpr (HasFreq) {
      _phrase_freq = static_cast<uint32_t>(res.freq);
      _phrase_scale =
        static_cast<score_t>(res.weight / static_cast<double>(res.freq));
      if constexpr (HasBoost) {
        _phrase_boost = res.boost;
      }
      if constexpr (Offs) {
        SDB_ASSERT(static_cast<uint32_t>(_matches.size()) == _phrase_freq);
        _start_offset = _matches[0].start;
        _end_offset = _matches[0].end;
        _match_idx = 1;
      }
    } else {
      _phrase_freq = 1;
      _phrase_scale = kNoBoost;
    }
  }

  Positions _pos;
  containers::Fixed<detail::ordered::Gap, N> _gaps;
  PosAttr::value_t _min_extent = 1;
  std::vector<detail::ordered::Level> _levels;
  std::vector<detail::ordered::SpanOffsets> _matches;
  size_t _match_idx = 0;

  uint32_t _phrase_freq = 0;
  score_t _phrase_scale = kNoBoost;
  [[no_unique_address]] utils::Need<HasBoost, score_t> _phrase_boost{};
  uint32_t _start_offset{0};
  uint32_t _end_offset{0};
};

template<bool Offs, bool HasFreq, size_t N = 0>
using FixedPhraseOrderedMatcher =
  PhraseOrderedMatcher<FixedTermPosition<Offs>, Offs, HasFreq, N>;

}  // namespace irs
