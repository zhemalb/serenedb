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

#include <algorithm>
#include <bit>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "iresearch/search/detail/phrase_matcher.hpp"
#include "iresearch/utils/containers/fixed.hpp"
#include "iresearch/utils/empty.hpp"
#include "iresearch/utils/type_limits.hpp"

namespace irs::detail {

template<typename Leaf, bool HasBoost = false>
class PhraseVariadicPositions {
 public:
  using Position =
    std::remove_reference_t<decltype(std::declval<Leaf&>().Positions())>;

  static constexpr bool kOffsets = Leaf::kOffsets;
  static constexpr bool kHasBoost = HasBoost;

  explicit PhraseVariadicPositions(uint32_t count)
    : _live{count}, _boosts{count}, _runs(size_t{count} + 1) {}

  void On(doc_id_t) noexcept {
    _value = pos_limits::invalid();
    _size = 0;
    _freq = 0;
  }

  void Add(Position& pos, [[maybe_unused]] score_t boost) {
    _freq += pos.DocFreq();
    if constexpr (HasBoost) {
      _boosts[_size] = boost;
    }
    _live[_size++] = &pos;
  }

  PosAttr::value_t value() const noexcept { return _value; }

  PosAttr::value_t seek(PosAttr::value_t target) {
    if (_size == 1) [[likely]] {
      _value = _live[0]->seek(target);
      Won(0);
      return _value;
    }
    auto min = pos_limits::eof();
    for (uint32_t i = 0; i != _size; ++i) {
      const auto pos = _live[i]->seek(target);
      if (pos == target) {
        Won(i);
        return _value = target;
      }
      if (pos < min) {
        min = pos;
        Won(i);
      }
    }
    return _value = min;
  }

  bool next() {
    if (_size == 1) [[likely]] {
      auto& pos = *_live[0];
      if (!pos.next()) {
        _value = pos_limits::eof();
        return false;
      }
      Won(0);
      _value = pos.value();
      return true;
    }
    if (pos_limits::eof(_value)) {
      return false;
    }
    auto min = pos_limits::eof();
    for (uint32_t i = 0; i != _size; ++i) {
      auto& pos = *_live[i];
      const auto at = pos.value();
      if (at == _value && !pos.next()) {
        continue;
      }
      if (at < _value && pos_limits::eof(pos.seek(_value + 1))) {
        continue;
      }
      if (pos.value() < min) {
        min = pos.value();
        Won(i);
      }
    }
    _value = min;
    return !pos_limits::eof(min);
  }

  const OffsAttr& Offsets() const noexcept
    requires(kOffsets)
  {
    return _offs;
  }

  score_t Boost() const noexcept
    requires(HasBoost)
  {
    return _boost;
  }

  void reset() {
    for (uint32_t i = 0; i != _size; ++i) {
      _live[i]->reset();
    }
    _value = pos_limits::invalid();
  }

  uint32_t Read(PosAttr::value_t* out, [[maybe_unused]] score_t* boosts) {
    _value = pos_limits::eof();
    if (_size == 1) [[likely]] {
      const auto count = _live[0]->ReadPositions(out);
      if constexpr (HasBoost) {
        std::fill_n(boosts, count, _boosts[0]);
      }
      return count;
    }
    uint32_t total = 0;
    PosAttr::value_t max = pos_limits::invalid();
    for (uint32_t i = 0; i != _size; ++i) {
      _runs[i] = total;
      const auto count = _live[i]->ReadPositions(out + total);
      if (count != 0) {
        max = std::max(max, out[total + count - 1]);
      }
      total += count;
    }
    _runs[_size] = total;
    const size_t words = max / kBits + 1;
    if (_bits.size() < words) {
      _bits.resize(words);
    }
    std::fill_n(_bits.data(), words, 0);
    if constexpr (HasBoost) {
      if (_boost_at.size() <= max) {
        _boost_at.resize(size_t{max} + 1);
      }
    }
    for (uint32_t i = 0; i != _size; ++i) {
      for (uint32_t j = _runs[i], end = _runs[i + 1]; j != end; ++j) {
        const auto pos = out[j];
        auto& word = _bits[pos / kBits];
        const auto mask = uint64_t{1} << (pos % kBits);
        if constexpr (HasBoost) {
          if ((word & mask) == 0) {
            _boost_at[pos] = _boosts[i];
          }
        }
        word |= mask;
      }
    }
    uint32_t count = 0;
    for (size_t w = 0; w != words; ++w) {
      for (auto bits = _bits[w]; bits != 0; bits &= bits - 1) {
        const auto pos = static_cast<PosAttr::value_t>(
          w * kBits + static_cast<size_t>(std::countr_zero(bits)));
        out[count] = pos;
        if constexpr (HasBoost) {
          boosts[count] = _boost_at[pos];
        }
        ++count;
      }
    }
    return count;
  }

  uint32_t DocFreq() { return _freq; }

 private:
  static constexpr size_t kBits = std::numeric_limits<uint64_t>::digits;

  IRS_FORCE_INLINE void Won([[maybe_unused]] uint32_t i) noexcept {
    if constexpr (kOffsets) {
      const auto* offs = irs::get<OffsAttr>(*_live[i]);
      SDB_ASSERT(offs);
      _offs = *offs;
    }
    if constexpr (HasBoost) {
      _boost = _boosts[i];
    }
  }

  containers::Fixed<Position*> _live;
  [[no_unique_address]] utils::Need<HasBoost, containers::Fixed<score_t>>
    _boosts;
  std::vector<uint64_t> _bits;
  std::vector<uint32_t> _runs;
  [[no_unique_address]] utils::Need<HasBoost, std::vector<score_t>> _boost_at;
  uint32_t _size = 0;
  uint32_t _freq = 0;
  PosAttr::value_t _value = pos_limits::invalid();
  [[no_unique_address]] utils::Need<kOffsets, OffsAttr> _offs;
  [[no_unique_address]] utils::Need<HasBoost, score_t> _boost;
};

template<typename Leaf, bool HasBoost = false>
class PhraseVariadicPos {
 public:
  using Position = PhraseVariadicPositions<Leaf, HasBoost>;

  PhraseVariadicPos(Leaf* begin, uint32_t count,
                    [[maybe_unused]] const score_t* boosts)
    : _positions{count},
      _begin{begin},
      _boosts{boosts},
      _heap{count == 1 ? 0 : count,
            [begin](Leaf*& slot, size_t i) noexcept { slot = begin + i; }},
      _stack{count == 1 ? 0 : 2 * size_t{count} + 1},
      _live_count{_heap.size()},
      _count{count} {
    SDB_ASSERT(_count != 0);
    std::make_heap(_heap.begin(), _heap.end(), Greater);
  }

  uint32_t Estimate() const noexcept {
    uint32_t estimate = 0;
    for (uint32_t i = 0; i != _count; ++i) {
      estimate += _begin[i].Estimate();
    }
    return estimate;
  }

  Position& Positions() noexcept { return _positions; }

  doc_id_t Next() {
    if (_count == 1) [[likely]] {
      return Settle(_begin[0].Next());
    }
    return Settle(Converge(_doc + 1));
  }

  doc_id_t Seek(doc_id_t target) {
    if (target <= _doc) {
      return _doc;
    }
    if (_count == 1) [[likely]] {
      return Settle(_begin[0].Seek(target));
    }
    return Settle(Converge(target));
  }

  doc_id_t Probe(doc_id_t target) {
    if (target <= _doc) {
      return _doc;
    }
    if (_count == 1) [[likely]] {
      return Settle(_begin[0].Probe(target));
    }
    return Settle(Converge(target));
  }

  void OnMatch() {
    if (!doc_limits::eof(_doc)) {
      CollectLive(_doc);
    }
  }

 private:
  static bool Greater(const Leaf* lhs, const Leaf* rhs) noexcept {
    return lhs->Value() > rhs->Value();
  }

  doc_id_t Converge(doc_id_t target) {
    SDB_ASSERT(_count > 1);
    while (_live_count != 0) {
      auto* top = _heap.front();
      if (top->Value() >= target) {
        return top->Value();
      }
      const auto doc = top->Probe(target);
      if (doc_limits::eof(doc)) [[unlikely]] {
        std::pop_heap(_heap.begin(), _heap.begin() + _live_count, Greater);
        --_live_count;
      } else {
        SinkTop();
      }
    }
    return doc_limits::eof();
  }

  void SinkTop() noexcept {
    const auto size = _live_count;
    auto* item = _heap.front();
    size_t i = 0;
    for (;;) {
      auto child = 2 * i + 1;
      if (child >= size) {
        break;
      }
      if (child + 1 != size &&
          _heap[child + 1]->Value() < _heap[child]->Value()) {
        ++child;
      }
      if (!(_heap[child]->Value() < item->Value())) {
        break;
      }
      _heap[i] = _heap[child];
      i = child;
    }
    _heap[i] = item;
  }

  doc_id_t Settle(doc_id_t doc) {
    _positions.On(doc);
    return _doc = doc;
  }

  void CollectLive(doc_id_t doc) {
    if (_count == 1) [[likely]] {
      _positions.Add(_begin[0].Positions(), BoostOf(_begin));
      return;
    }
    size_t depth = 0;
    _stack[depth++] = 0;
    while (depth != 0) {
      const auto i = _stack[--depth];
      if (i >= _live_count || _heap[i]->Value() != doc) {
        continue;
      }
      _positions.Add(_heap[i]->Positions(), BoostOf(_heap[i]));
      _stack[depth++] = 2 * i + 1;
      _stack[depth++] = 2 * i + 2;
    }
  }

  score_t BoostOf([[maybe_unused]] const Leaf* leaf) const noexcept {
    if constexpr (HasBoost) {
      return _boosts[leaf - _begin];
    } else {
      return kNoBoost;
    }
  }

  Position _positions;
  Leaf* _begin;
  [[no_unique_address]] utils::Need<HasBoost, const score_t*> _boosts;
  containers::Fixed<Leaf*> _heap;
  containers::Fixed<uint32_t> _stack;
  size_t _live_count = 0;
  uint32_t _count;
  doc_id_t _doc = doc_limits::invalid();
};

}  // namespace irs::detail
namespace irs {

template<typename Leaf, bool HasBoost>
struct TermPositionTraits<
  std::pair<detail::PhraseVariadicPositions<Leaf, HasBoost>*, TermInterval>> {
  using T =
    std::pair<detail::PhraseVariadicPositions<Leaf, HasBoost>*, TermInterval>;
  using PositionImpl = detail::PhraseVariadicPositions<Leaf, HasBoost>;

  static PosAttr::value_t Position(T& pos) noexcept {
    return pos.first->value();
  }

  static const TermInterval& Interval(const T& pos) noexcept {
    return pos.second;
  }

  static void ResetPos(const T& pos) { pos.first->reset(); }

  static uint32_t Read(T& pos, PosAttr::value_t* out, score_t* boosts) {
    return pos.first->Read(out, boosts);
  }

  static const OffsAttr& Offsets(const T& pos) noexcept {
    return pos.first->Offsets();
  }

  static score_t Boost(const T& pos) noexcept { return pos.first->Boost(); }
};

}  // namespace irs
