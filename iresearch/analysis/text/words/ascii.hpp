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

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <duckdb/common/types/string_type.hpp>

#include "iresearch/analysis/text/words/masks.hpp"

namespace irs::analysis::words {

enum WbClass : uint8_t {
  kOT = 0,
  kAL,
  kNU,
  kEX,
  kML,
  kMN,
  kMNL,
  kSQ,
  kWS,
  kNL,
};

inline constexpr auto kWbClass = [] {
  std::array<uint8_t, 256> t{};
  for (int c = 'A'; c <= 'Z'; ++c) {
    t[c] = kAL;
  }
  for (int c = 'a'; c <= 'z'; ++c) {
    t[c] = kAL;
  }
  for (int c = '0'; c <= '9'; ++c) {
    t[c] = kNU;
  }
  t['_'] = kEX;
  t[':'] = kML;
  t[','] = kMN;
  t[';'] = kMN;
  t['.'] = kMNL;
  t['\''] = kSQ;
  t[' '] = kWS;
  t['\r'] = kNL;
  t['\n'] = kNL;
  t['\v'] = kNL;
  t['\f'] = kNL;
  return t;
}();

constexpr bool IsWordClass(uint8_t cls) noexcept {
  static_assert(kAL + 1 == kNU && kNU + 1 == kEX);
  return cls >= kAL && cls <= kEX;
}

struct Segment {
  uint32_t begin;
  uint32_t end;
  bool ascii_only;
  bool has_alpha;
  bool has_digit;
};

inline void AdvanceWordRun(const byte_type* b, size_t& i, size_t n,
                           bool& has_alpha, bool& has_digit) {
  while (n - i >= classify::kClassifyBlock) {
    const auto m = ClassifyWordBlock(b + i);
    const uint32_t nonword = ~m.word;
    if (nonword != 0) {
      const auto k = static_cast<uint32_t>(std::countr_zero(nonword));
      const uint32_t run = (uint32_t{1} << k) - 1;
      has_alpha |= (m.alpha & run) != 0;
      has_digit |= (m.digit & run) != 0;
      i += k;
      return;
    }
    has_alpha |= m.alpha != 0;
    has_digit |= m.digit != 0;
    i += classify::kClassifyBlock;
  }
  while (i < n) {
    const uint8_t cls = kWbClass[b[i]];
    if (!IsWordClass(cls)) {
      return;
    }
    has_alpha |= cls == kAL;
    has_digit |= cls == kNU;
    ++i;
  }
}

IRS_FORCE_INLINE inline size_t ExtendWordRun(const byte_type* b, size_t i,
                                             size_t n, bool& has_alpha,
                                             bool& has_digit) noexcept {
  while (i + 1 < n) {
    const uint8_t c = kWbClass[b[i]];
    const uint8_t last = kWbClass[b[i - 1]];
    const uint8_t next = kWbClass[b[i + 1]];
    if (last == kAL && next == kAL && (c == kML || c == kMNL || c == kSQ)) {
      ++i;
      AdvanceWordRun(b, i, n, has_alpha, has_digit);
      continue;
    }
    if (last == kNU && next == kNU && (c == kMN || c == kMNL || c == kSQ)) {
      ++i;
      AdvanceWordRun(b, i, n, has_alpha, has_digit);
      continue;
    }
    break;
  }
  return i;
}

IRS_FORCE_INLINE inline WordBridgeMasks ClassifyWordBridgeAt(
  const byte_type* b, size_t n, size_t at) noexcept {
  if (at >= n) {
    return {};
  }
  if (n - at >= classify::kClassifyBlock) {
    return ClassifyWordBridgeBlock(b + at);
  }
  return ClassifyWordBridge(classify::LoadPadded(b + at, n - at));
}

struct WordBridgeChunk {
  uint64_t word;
  uint64_t alpha;
  uint64_t digit;
  uint64_t mid_al;
  uint64_t mid_nu;
};

IRS_FORCE_INLINE inline WordBridgeChunk JoinWordBridge(
  const WordBridgeMasks& lo, const WordBridgeMasks& hi) noexcept {
  constexpr auto kShift = classify::kClassifyBlock;
  return {lo.word | (uint64_t{hi.word} << kShift),
          lo.alpha | (uint64_t{hi.alpha} << kShift),
          lo.digit | (uint64_t{hi.digit} << kShift),
          lo.mid_al | (uint64_t{hi.mid_al} << kShift),
          lo.mid_nu | (uint64_t{hi.mid_nu} << kShift)};
}

template<typename Emit, typename Progress = classify::NoProgress>
IRS_FORCE_INLINE void ScanAsciiRuns(duckdb::string_t value, Emit&& emit,
                                    Progress progress = {}) {
  constexpr size_t kBlock = classify::kClassifyBlock;
  constexpr size_t kChunk = 2 * kBlock;
  constexpr uint64_t kAll = ~uint64_t{0};
  const auto* b = reinterpret_cast<const byte_type*>(value.GetData());
  const size_t n = value.GetSize();
  bool open = false;
  size_t begin = 0;
  bool has_alpha = false;
  bool has_digit = false;
  uint64_t carry_run = 0;
  uint64_t carry_alpha = 0;
  uint64_t carry_digit = 0;
  size_t base = 0;
  auto lo = ClassifyWordBridgeAt(b, n, 0);
  for (;;) {
    const auto hi = ClassifyWordBridgeAt(b, n, base + kBlock);
    const auto ahead = ClassifyWordBridgeAt(b, n, base + kChunk);
    const auto c = JoinWordBridge(lo, hi);
    const uint64_t next_alpha = ahead.alpha & 1;
    const uint64_t next_digit = ahead.digit & 1;
    const uint64_t bridge = (c.mid_al & ((c.alpha << 1) | carry_alpha) &
                             ((c.alpha >> 1) | (next_alpha << 63))) |
                            (c.mid_nu & ((c.digit << 1) | carry_digit) &
                             ((c.digit >> 1) | (next_digit << 63)));
    const uint64_t run = c.word | bridge;
    const uint64_t next_run =
      (ahead.word & 1) |
      ((ahead.mid_al & 1) & (c.alpha >> 63) & ((ahead.alpha >> 1) & 1)) |
      ((ahead.mid_nu & 1) & (c.digit >> 63) & ((ahead.digit >> 1) & 1));
    uint64_t starts = run & ~((run << 1) | carry_run);
    uint64_t ends = run & ~((run >> 1) | (next_run << 63));
    if (open) {
      if (ends == 0) {
        has_alpha |= c.alpha != 0;
        has_digit |= c.digit != 0;
      } else {
        const auto e = static_cast<uint32_t>(std::countr_zero(ends));
        const uint64_t mask = kAll >> (63 - e);
        has_alpha |= (c.alpha & mask) != 0;
        has_digit |= (c.digit & mask) != 0;
        emit(Segment{static_cast<uint32_t>(begin),
                     static_cast<uint32_t>(base + e + 1), true, has_alpha,
                     has_digit});
        open = false;
        ends &= ends - 1;
      }
    }
    while (ends != 0) {
      const auto s = static_cast<uint32_t>(std::countr_zero(starts));
      const auto e = static_cast<uint32_t>(std::countr_zero(ends));
      const uint64_t mask = (kAll >> (63 - e)) & (kAll << s);
      emit(Segment{static_cast<uint32_t>(base + s),
                   static_cast<uint32_t>(base + e + 1), true,
                   (c.alpha & mask) != 0, (c.digit & mask) != 0});
      starts &= starts - 1;
      ends &= ends - 1;
    }
    if (starts != 0) {
      const auto s = static_cast<uint32_t>(std::countr_zero(starts));
      const uint64_t mask = kAll << s;
      begin = base + s;
      has_alpha = (c.alpha & mask) != 0;
      has_digit = (c.digit & mask) != 0;
      open = true;
    }
    if (base + kChunk >= n) {
      SDB_ASSERT(!open);
      return;
    }
    if (!progress()) {
      return;
    }
    carry_run = run >> 63;
    carry_alpha = c.alpha >> 63;
    carry_digit = c.digit >> 63;
    base += kChunk;
    lo = ahead;
  }
}

template<typename Emit, typename Progress = classify::NoProgress>
IRS_FORCE_INLINE void ScanAscii(duckdb::string_t value, Emit&& emit,
                                Progress progress = {}) {
  const auto* b = reinterpret_cast<const byte_type*>(value.GetData());
  const size_t n = value.GetSize();
  size_t i = 0;
  size_t reported = 0;
  while (i < n) {
    if (!classify::KeepScanning(progress, reported, i)) {
      return;
    }
    const size_t seg_begin = i;
    const uint8_t c0 = kWbClass[b[i]];
    bool has_alpha = false;
    bool has_digit = false;
    switch (c0) {
      case kWS:
        do {
          ++i;
        } while (i < n && kWbClass[b[i]] == kWS);
        break;
      case kNL:
        i += (b[i] == '\r' && i + 1 < n && b[i + 1] == '\n') ? 2 : 1;
        break;
      case kAL:
      case kNU:
      case kEX:
        AdvanceWordRun(b, i, n, has_alpha, has_digit);
        i = ExtendWordRun(b, i, n, has_alpha, has_digit);
        break;
      default:
        ++i;
        break;
    }
    emit(Segment{static_cast<uint32_t>(seg_begin), static_cast<uint32_t>(i),
                 true, has_alpha, has_digit});
  }
}

}  // namespace irs::analysis::words
