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

#include "iresearch/search/detail/token_phrase.hpp"

#include <absl/algorithm/container.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>
#include <utility>

namespace irs {

void PhraseCheck::AnchorMatch::Reset(uint32_t anchors) {
  _batch_base = 0;
  _end = 0;
  _carry_base = 0;
  _carry_terms.clear();
  _carry_pos.clear();
  _arena.Reset();
  _pending.clear();
  _last = 0;
  _anchors = anchors;
  _steps = 0;
  _overran = false;
}

void PhraseCheck::AnchorMatch::Carry(uint64_t left) {
  const uint64_t first = PosAt(_pending.empty() ? _end - 1 : _pending.front());
  const auto keep = first > left ? first - left : 0;
  auto from = _end;
  while (from > _carry_base && PosAt(from - 1) >= keep) {
    --from;
  }
  const auto drop =
    static_cast<ptrdiff_t>(std::min(from, _batch_base) - _carry_base);
  _carry_terms.erase(_carry_terms.begin(), _carry_terms.begin() + drop);
  _carry_pos.erase(_carry_pos.begin(), _carry_pos.begin() + drop);
  for (auto at = std::max(from, _batch_base); at != _end; ++at) {
    auto term = _batch_terms[at - _batch_base];
    if (!term.IsInlined()) {
      const auto size = static_cast<uint32_t>(term.GetSize());
      auto* data = _arena.Allocate(size);
      std::memcpy(data, term.GetData(), size);
      term = {reinterpret_cast<const char*>(data), size};
    }
    _carry_terms.emplace_back(term);
    _carry_pos.emplace_back(_batch_pos[at - _batch_base]);
  }
  _carry_base = from;
}

void PhraseCheck::AnchorMatch::Consumed(const TokenBatch& batch,
                                        const State& state) {
  if (!state.done && batch.count != 0) {
    Carry(state.phrase->anchor->left);
  }
}

void PhraseCheck::AutomatonMatch::Reset() {
  _bits = 0;
  _mask = 0;
  _at = 0;
}

void PhraseCheck::PositionsMatch::Reset(size_t n) {
  _slots.resize(n);
  for (auto& slot : _slots) {
    slot.clear();
  }
}

PhraseCheck::PhraseCheck(const CompiledPhrase& phrase,
                         const ByPhraseOptions::TokenizerFactory& tokenizer,
                         bool count)
  : _tokenizer{tokenizer()},
    _anchor{_analyzer.Batch()},
    _state{.phrase = &phrase, .count = count},
    _dense{!_tokenizer->Traits().explicit_pos} {
  const std::span slots{phrase.slots};
  _all_open = phrase.open && !slots.empty() &&
              absl::c_all_of(slots.subspan(1), [](const auto& slot) {
                return slot.offs_max == pos_limits::kOpenGap;
              });
}

bool PhraseCheck::Anchored(duckdb::idx_t row, uint32_t anchors) const {
  const auto& phrase = *_state.phrase;
  if (!phrase.anchor || (phrase.open && _state.count) ||
      (_all_open && phrase.anchor->right >= pos_limits::kOpenGap)) {
    return false;
  }
  if (!_state.count || anchors == 0) {
    return true;
  }
  const auto window = phrase.anchor->left + phrase.anchor->right;
  const auto tokens = _rows.Bytes(row) / kBytesPerToken;
  return window <= (kWalkPerToken * tokens + AnchorMatch::kStepSlack) / anchors;
}

PhraseVerdict PhraseCheck::Check(duckdb::idx_t row, uint32_t anchors) {
  Start(Anchored(row, anchors), anchors);
  Analyze(row);
  if (_route == Route::Anchor) {
    _anchor.Drain(_state);
    if (_anchor.Overran()) {
      Start(false);
      Analyze(row);
    }
  }
  return WithMatch([&](auto& match) { return match.Finish(_state); });
}

void PhraseCheck::Analyze(duckdb::idx_t row) {
  _rows.ForEach(row, [&](duckdb::string_t value) {
    _analyzer.Scan(*_tokenizer, value, *this);
    return !_state.done;
  });
}

void PhraseCheck::Start(bool anchored, uint32_t anchors) {
  _state.done = false;
  _state.freq = 0;
  _last_pos = 0;
  _value_base = 0;
  const auto& phrase = *_state.phrase;
  if (anchored) {
    _route = Route::Anchor;
    _anchor.Reset(anchors);
  } else if (phrase.automaton && !_state.count) {
    _route = Route::Automaton;
    _automaton.Reset();
  } else if (_all_open && !_state.count) {
    _route = Route::Earliest;
    _earliest.Reset();
  } else {
    _route = Route::Positions;
    _positions.Reset(phrase.slots.size());
  }
}

void PhraseCheck::Consume(TokenBatch& batch, DocRuns) {
  Take(batch);
  _fed = 0;
  if (_route == Route::Anchor) {
    _anchor.Consumed(batch, _state);
  }
}

bool PhraseCheck::OnProgress(TokenBatch& batch) {
  Take(batch);
  return !_state.done;
}

void PhraseCheck::Take(TokenBatch& batch) {
  const auto from = std::exchange(_fed, batch.count);
  const auto count = batch.count;
  if (_state.done || from == count) {
    return;
  }
  if (_dense) {
    std::span<uint32_t> fresh{batch.pos + from, count - from};
    absl::c_iota(fresh, _last_pos + 1);
    _last_pos += count - from;
  } else {
    for (auto i = from; i != count; ++i) {
      batch.pos[i] += _value_base;
    }
    _last_pos = std::max(_last_pos, batch.pos[count - 1]);
  }
  WithMatch([&](auto& match) { match.Feed(batch, from, _state); });
}

void PhraseCheck::AnchorMatch::Drain(State& state) {
  if (state.done) {
    return;
  }
  _batch_base = _carry_base + _carry_terms.size();
  _end = _batch_base;
  for (const auto at : _pending) {
    if (Hit(at, state)) {
      break;
    }
  }
  _pending.clear();
  state.done = true;
}

PhraseVerdict PhraseCheck::Verdict(const State& state) noexcept {
  if (state.freq == 0) {
    return {};
  }
  return {.freq = state.count ? std::saturate_cast<uint32_t>(state.freq) : 1};
}

void PhraseCheck::AnchorMatch::Feed(const TokenBatch& batch, uint32_t from,
                                    State& state) {
  SDB_ASSERT(batch.terms == _batch_terms.data());
  const auto& layout = *state.phrase->anchor;
  _batch_base = _carry_base + _carry_terms.size();
  _end = _batch_base + batch.count;
  const uint64_t last = batch.pos[batch.count - 1];
  size_t keep = 0;
  for (const auto at : _pending) {
    if (PosAt(at) + layout.right < last) {
      if (Hit(at, state)) {
        return;
      }
    } else {
      _pending[keep++] = at;
    }
  }
  _pending.resize(keep);
  const auto& word = *state.phrase->slots[layout.slot].word;
  for (auto i = from; i != batch.count; ++i) {
    if (batch.terms[i] != word) {
      continue;
    }
    const auto pos = batch.pos[i];
    if (pos == _last) {
      continue;
    }
    _last = pos;
    const auto at = _batch_base + i;
    if (pos + layout.right < last) {
      if (Hit(at, state)) {
        return;
      }
    } else {
      _pending.emplace_back(at);
    }
  }
}

PhraseVerdict PhraseCheck::AnchorMatch::Finish(const State& state) const {
  SDB_ASSERT(state.done && !_overran);
  return Verdict(state);
}

bool PhraseCheck::AnchorMatch::Hit(size_t at, State& state) {
  const auto slot = state.phrase->anchor->slot;
  if (const auto left = Ways<false>(slot, at, state)) {
    state.freq += left * Ways<true>(slot, at, state);
  }
  state.done |=
    (!state.count && state.freq != 0) || (_anchors != 0 && --_anchors == 0);
  return state.done;
}

template<bool Right>
uint64_t PhraseCheck::AnchorMatch::Ways(uint32_t slot, size_t at,
                                        State& state) {
  const auto edge =
    static_cast<uint32_t>(Right ? state.phrase->slots.size() - 1 : 0);
  if (slot == edge) {
    return 1;
  }
  const auto other = Right ? slot + 1 : slot - 1;
  return other == edge ? Scan<Right, true>(slot, at, state)
                       : Walk<Right>(slot, at, state);
}

template<bool Right>
uint64_t PhraseCheck::AnchorMatch::Walk(uint32_t slot, size_t at,
                                        State& state) {
  return Scan<Right, false>(slot, at, state);
}

template<bool Right>
uint64_t PhraseCheck::AnchorMatch::Deeper(uint32_t slot, size_t at,
                                          State& state) {
  return Ways<Right>(slot, at, state);
}

template<bool Right, bool Last>
uint64_t PhraseCheck::AnchorMatch::Scan(uint32_t slot, size_t at,
                                        State& state) {
  const auto other = Right ? slot + 1 : slot - 1;
  const auto& next = state.phrase->slots[other];
  const auto& gap = Right ? next : state.phrase->slots[slot];
  const uint64_t offs_min = gap.offs_min;
  const uint64_t offs_max = gap.offs_max;
  const auto* const word = next.word ? &*next.word : nullptr;
  const uint64_t pos = PosAt(at);
  const auto limit = kStepsPerToken * _end + kStepSlack;
  const auto carry_base = _carry_base;
  const auto batch_base = _batch_base;
  SDB_ASSERT(_steps <= limit);
  auto budget = limit - _steps;
  uint64_t ways = 0;
  uint64_t accepted = std::numeric_limits<uint64_t>::max();
  const auto walk = [&](const duckdb::string_t* terms,
                        const uint32_t* positions, size_t base,
                        size_t n) IRS_FORCE_INLINE {
    for (size_t k = 0; k != n; ++k) {
      const auto i = Right ? k : n - 1 - k;
      const uint64_t p = positions[i];
      const auto distance = Right ? p - pos : pos - p;
      if (distance > offs_max) {
        return false;
      }
      if (budget == 0) [[unlikely]] {
        _overran = true;
        state.done = true;
        return false;
      }
      --budget;
      if (distance < offs_min || p == accepted) {
        continue;
      }
      const auto& term = terms[i];
      if (word
            ? *word != term
            : !AcceptsTerm(*state.phrase, state.phrase->slots[other], term)) {
        continue;
      }
      accepted = p;
      if constexpr (Last) {
        ++ways;
      } else {
        _steps = limit - budget;
        ways += Deeper<Right>(other, base + i, state);
        budget = limit - _steps;
        if (_overran) {
          return false;
        }
      }
      if (!state.count && ways != 0) {
        return false;
      }
    }
    return true;
  };
  const duckdb::string_t* terms = _batch_terms.data();
  const uint32_t* positions = _batch_pos.data();
  size_t base = batch_base;
  size_t n = 0;
  bool more = false;
  if constexpr (Right) {
    more = at + 1 < batch_base;
    base = at + 1;
    if (more) {
      terms = _carry_terms.data() + (base - carry_base);
      positions = _carry_pos.data() + (base - carry_base);
      n = batch_base - base;
    } else {
      terms += base - batch_base;
      positions += base - batch_base;
      n = _end - base;
    }
  } else {
    more = at > batch_base;
    if (more) {
      n = at - batch_base;
    } else {
      terms = _carry_terms.data();
      positions = _carry_pos.data();
      base = carry_base;
      n = std::min(at, batch_base) - carry_base;
    }
  }
  while (walk(terms, positions, base, n) && more) {
    more = false;
    if constexpr (Right) {
      terms = _batch_terms.data();
      positions = _batch_pos.data();
      base = batch_base;
      n = _end - batch_base;
    } else {
      terms = _carry_terms.data();
      positions = _carry_pos.data();
      base = carry_base;
      n = batch_base - carry_base;
    }
  }
  _steps = limit - budget;
  if (_overran) {
    return 0;
  }
  return !state.count && ways != 0 ? 1 : ways;
}

void PhraseCheck::AutomatonMatch::Feed(const TokenBatch& batch, uint32_t from,
                                       State& state) {
  for (auto i = from; i != batch.count; ++i) {
    const auto pos = batch.pos[i];
    if (pos != _at) {
      Flush(state);
      if (state.done) {
        return;
      }
      if (_at != 0) {
        for (auto gap = _at + 1; gap < pos && _bits != 0; ++gap) {
          Step(0, state);
        }
      }
      _at = pos;
    }
    _mask |= MaskOf(*state.phrase, batch.terms[i]);
  }
}

PhraseVerdict PhraseCheck::AutomatonMatch::Finish(State& state) {
  if (!state.done) {
    Flush(state);
  }
  return Verdict(state);
}

void PhraseCheck::AutomatonMatch::Flush(State& state) {
  if (_at == 0 || (_bits == 0 && _mask == 0)) {
    _mask = 0;
    return;
  }
  Step(_mask, state);
  _mask = 0;
}

void PhraseCheck::AutomatonMatch::Step(uint64_t mask, State& state) {
  const auto& layout = *state.phrase->automaton;
  auto next = ((_bits << 1) | 1) & (mask | layout.wild);
  for (const auto& extra : layout.extras) {
    if (_bits & extra.range) {
      next |= extra.bit & mask;
    }
  }
  _bits = next;
  if (_bits & layout.last) {
    state.freq = 1;
    state.done = true;
  }
}

void PhraseCheck::EarliestMatch::Resolve(State& state) {
  const auto& slots = state.phrase->slots;
  const auto n = static_cast<uint32_t>(slots.size());
  while (_next < n && (_next == 0 || _at >= _last + slots[_next].offs_min) &&
         absl::c_linear_search(_accepted, _next)) {
    _last = _at;
    ++_next;
  }
  _accepted.clear();
  if (_next == n) {
    state.freq = 1;
    state.done = true;
  }
}

void PhraseCheck::EarliestMatch::Feed(const TokenBatch& batch, uint32_t from,
                                      State& state) {
  for (auto i = from; i != batch.count; ++i) {
    const auto pos = batch.pos[i];
    if (pos != _at) {
      Resolve(state);
      if (state.done) {
        return;
      }
      _at = pos;
    }
    ForEachAccepting(*state.phrase, batch.terms[i],
                     [&](uint32_t slot) { _accepted.emplace_back(slot); });
  }
}

PhraseVerdict PhraseCheck::EarliestMatch::Finish(State& state) {
  if (!state.done) {
    Resolve(state);
  }
  return Verdict(state);
}

void PhraseCheck::PositionsMatch::Feed(const TokenBatch& batch, uint32_t from,
                                       State& state) {
  for (auto i = from; i != batch.count; ++i) {
    const auto pos = batch.pos[i];
    ForEachAccepting(*state.phrase, batch.terms[i], [&](uint32_t slot) {
      auto& slot_positions = _slots[slot];
      if (slot_positions.empty() || slot_positions.back() != pos) {
        slot_positions.emplace_back(pos);
      }
    });
  }
}

PhraseVerdict PhraseCheck::PositionsMatch::Finish(State& state) {
  const auto& phrase = *state.phrase;
  const auto n = phrase.slots.size();
  if (absl::c_any_of(_slots, [](const auto& slot) { return slot.empty(); })) {
    return {};
  }
  if (phrase.open) {
    _gaps.clear();
    for (const auto& slot : phrase.slots) {
      _gaps.emplace_back(slot.offs_min, slot.offs_max);
    }
    const auto res =
      detail::ordered::Run(_slots, _gaps, !state.count, _ordered);
    if (!res.any) {
      return {};
    }
    if (!state.count) {
      return {.freq = 1};
    }
    return {
      .freq = std::saturate_cast<uint32_t>(res.freq),
      .scale = static_cast<score_t>(res.weight / static_cast<double>(res.freq)),
    };
  }
  if (n > 1 && phrase.slop.max != 0) {
    detail::slop::SpanCursors cursors{_slots, _scratch};
    const auto res =
      detail::slop::Sweep<0>(cursors, phrase.slop.offsets, phrase.slop.max,
                             phrase.slop.pairs, _scratch, !state.count, [] {});
    if (!res.any) {
      return {};
    }
    if (!state.count) {
      return {.freq = 1};
    }
    return {
      .freq = std::saturate_cast<uint32_t>(res.freq),
      .scale = static_cast<score_t>(res.weight / static_cast<double>(res.freq)),
    };
  }

  _valid.clear();
  for (const auto p : _slots.back()) {
    _valid.emplace_back(p, 1);
  }
  for (size_t i = n - 1; i != 0; --i) {
    const auto& slot = phrase.slots[i];
    _next.clear();
    size_t lo = 0;
    size_t hi = 0;
    uint64_t window = 0;
    for (const auto p : _slots[i - 1]) {
      const uint64_t min = uint64_t{p} + slot.offs_min;
      const uint64_t max = uint64_t{p} + slot.offs_max;
      for (; hi != _valid.size() && _valid[hi].pos <= max; ++hi) {
        window += _valid[hi].ways;
      }
      for (; lo != hi && _valid[lo].pos < min; ++lo) {
        window -= _valid[lo].ways;
      }
      if (window != 0) {
        _next.emplace_back(
          p, std::min<uint64_t>(window, std::numeric_limits<uint32_t>::max()));
      }
    }
    std::swap(_valid, _next);
    if (_valid.empty()) {
      return {};
    }
  }
  for (const auto& way : _valid) {
    state.freq += way.ways;
  }
  return Verdict(state);
}

}  // namespace irs
