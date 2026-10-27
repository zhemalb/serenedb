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

#include "iresearch/analysis/token_attributes.hpp"
#include "iresearch/formats/posting/block_io.hpp"
#include "iresearch/formats/posting/common.hpp"
#include "iresearch/formats/posting_meta.hpp"
#include "iresearch/store/data_input.hpp"
#include "iresearch/store/store_utils.hpp"
#include "iresearch/utils/down_cast.hpp"
#include "iresearch/utils/empty.hpp"

namespace irs {

struct DocState {
  const IndexInput* pos_in;
  const IndexInput* pay_in;
  const PostingMeta* term_state;
  uint32_t* enc_buf;
};

template<typename IteratorTraits>
class PositionImpl final : public PosAttr {
 public:
  Attribute* GetMutable(TypeInfo::type_id type) noexcept final {
    if constexpr (IteratorTraits::Offset()) {
      return irs::Type<OffsAttr>::id() == type ? &_offs : nullptr;
    } else {
      return nullptr;
    }
  }

  value_t seek(value_t target) final {
    const uint32_t freq = _freq;
    if (_pend_pos > freq) {
      Skip(_pend_pos - freq);
      _pend_pos = freq;
    }
    while (_value < target && _pend_pos) {
      if (_buf_pos == pos_limits::kBlockSize) {
        ReadBlock();
        _buf_pos = 0;
      }
      _value += _pos_deltas[_buf_pos];
      SDB_ASSERT(pos_limits::valid(_value));
      ReadAttributes();

      ++_buf_pos;
      --_pend_pos;
    }
    if (0 == _pend_pos && _value < target) {
      _value = pos_limits::eof();
    }
    return _value;
  }

  bool next() final {
    if (0 == _pend_pos) {
      _value = pos_limits::eof();

      return false;
    }

    const uint32_t freq = _freq;

    if (_pend_pos > freq) {
      Skip(_pend_pos - freq);
      _pend_pos = freq;
    }

    if (_buf_pos == pos_limits::kBlockSize) {
      ReadBlock();
      _buf_pos = 0;
    }
    _value += _pos_deltas[_buf_pos];
    SDB_ASSERT(pos_limits::valid(_value));
    ReadAttributes();

    ++_buf_pos;
    --_pend_pos;
    return true;
  }

  void ReadDeltas(uint32_t* pos, uint32_t* offs_start, uint32_t* offs_len,
                  uint32_t n) {
    if (_pend_pos != 0) {
      Skip(_pend_pos);
      _pend_pos = 0;
    }
    Walk(n, [&](uint32_t at, uint32_t take) {
      std::copy_n(_pos_deltas + at, take, pos);
      pos += take;
      if constexpr (IteratorTraits::Offset()) {
        if (offs_start != nullptr) {
          std::copy_n(_offs_start_deltas + at, take, offs_start);
          std::copy_n(_offs_lengths + at, take, offs_len);
          offs_start += take;
          offs_len += take;
        }
      }
    });
  }

  uint32_t ReadPositions(uint32_t* out) {
    static_assert(!IteratorTraits::Offset());
    const uint32_t freq = _freq;
    if (_pend_pos > freq) {
      Skip(_pend_pos - freq);
      _pend_pos = freq;
    }
    const auto count = static_cast<uint32_t>(_pend_pos);
    auto value = _value;
    auto* it = out;
    Walk(count, [&](uint32_t at, uint32_t take) {
      const auto* const deltas = _pos_deltas + at;
      for (uint32_t i = 0; i != take; ++i) {
        value += deltas[i];
        it[i] = value;
      }
      it += take;
      _pend_pos -= take;
    });
    _value = value;
    return count;
  }

  void SkipDeltas(uint64_t n) noexcept { _pend_pos += n; }

  void reset() final {
    if (_pend_pos >= _freq) {
      return;
    }
    const auto consumed = static_cast<uint32_t>(_freq - _pend_pos);
    Clear();
    _pend_pos = _freq;
    if (consumed <= _buf_pos) {
      _buf_pos -= consumed;
      return;
    }
    Rewind();
  }

  // prepares iterator to work
  template<typename InputType>
  void Prepare(const DocState& state) {
    if (!_pos.in) {
      _pos.in = irs::utils::downCast<InputType>(*state.pos_in).Reopen();
      SDB_ASSERT(_pos.in);
      _pos.view = block_io::View(*_pos.in);
    }
    _pos.hint.Arm(state.term_state->pos_start,
                  state.term_state->pos_start + PosExtent(*state.term_state));
    _enc_buf = state.enc_buf;

    if constexpr (IteratorTraits::Offset()) {
      if (!_pay.in) {
        _pay.in = irs::utils::downCast<InputType>(*state.pay_in).Reopen();
        SDB_ASSERT(_pay.in);
        _pay.view = block_io::View(*_pay.in);
      }
      _pay.hint.Arm(state.term_state->pay_start,
                    state.term_state->pay_start + PayExtent(*state.term_state));
    }
    Land(state.term_state->pos_start, state.term_state->pay_start,
         state.term_state->pos_offset);
  }

  // notifies iterator that doc iterator has skipped to a new block
  void Prepare(const BlockLanding& state) noexcept {
    Land(state.pos_ptr, state.pay_ptr, state.pos_offset);
  }

  // notify the positions that the document stream has moved forward
  void Notify(uint32_t freq, uint32_t n) noexcept {
    _freq = freq;
    _pend_pos += n;
#ifdef SDB_DEV
    ++_doc_seq;
#endif
  }

  void Clear() noexcept {
    _value = pos_limits::invalid();
    ClearAttributes();
  }

  uint32_t DocFreq() const noexcept { return _freq; }

 private:
  struct Stream {
    void Load() {
      hint.Advance(*in, group);
      if (view != nullptr) {
        header = view->ReadStable(group, PosGroup::kHeaderBytes);
      } else {
        in->Seek(group);
        in->ReadData(copy, PosGroup::kHeaderBytes);
        header = copy;
      }
    }

    void Hop() {
      group = PosGroup::Next(group, header);
      Load();
    }

    void Step() {
      group = PosGroup::Next(group, header);
      if (view != nullptr) {
        SDB_ASSERT(view->Position() == group);
        header = view->ReadStable(PosGroup::kHeaderBytes);
      } else {
        SDB_ASSERT(in->Position() == group);
        in->ReadData(copy, PosGroup::kHeaderBytes);
        header = copy;
      }
    }

    void SeekBlock(uint32_t block) {
      const auto at =
        group + PosGroup::kHeaderBytes + PosGroup::Start(header, block);
      if (view != nullptr) {
        view->Seek(at);
      } else {
        in->Seek(at);
      }
    }

    void Account() {
      hint.Advance(*in, view != nullptr ? view->Position() : in->Position());
    }

    IndexInput::ptr in;
    BytesViewInput* view = nullptr;
    uint64_t group = 0;
    const byte_type* header = nullptr;
    byte_type copy[PosGroup::kHeaderBytes];
    GrowingHint hint;
  };

  struct DocStart {
    uint64_t pos_group = 0;
    [[no_unique_address]] utils::Need<IteratorTraits::Offset(), uint64_t>
      pay_group;
    uint32_t block = 0;
    uint32_t buf_pos = 0;
#ifdef SDB_DEV
    uint64_t seq = 0;
#endif
  };

  void Land(uint64_t pos_group, uint64_t pay_group, uint64_t pend) noexcept {
    _pos.group = pos_group;
    _pos.header = nullptr;
    if constexpr (IteratorTraits::Offset()) {
      _pay.group = pay_group;
    }
    _next = 0;
    _buf_pos = pos_limits::kBlockSize;
    _pend_pos = pend;
  }

  void MarkDocStart(uint32_t block, uint32_t buf_pos) noexcept {
    _doc_start.pos_group = _pos.group;
    if constexpr (IteratorTraits::Offset()) {
      _doc_start.pay_group = _pay.group;
    }
    _doc_start.block = block;
    _doc_start.buf_pos = buf_pos;
#ifdef SDB_DEV
    _doc_start.seq = _doc_seq;
#endif
  }

  IRS_NO_INLINE void Rewind() {
    const auto start = _doc_start;
#ifdef SDB_DEV
    SDB_ASSERT(start.seq == _doc_seq);
#endif
    if (_pos.group != start.pos_group) {
      _pos.group = start.pos_group;
      _pos.header = nullptr;
      if constexpr (IteratorTraits::Offset()) {
        _pay.group = start.pay_group;
      }
    } else {
      _pos.SeekBlock(start.block);
      if constexpr (IteratorTraits::Offset()) {
        _pay.SeekBlock(start.block);
      }
    }
    _next = start.block;
    ReadBlock();
    _doc_start = start;
    _buf_pos = start.buf_pos;
  }

  void Enter(uint64_t block) {
    if (_pos.header == nullptr) {
      _pos.Load();
      if constexpr (IteratorTraits::Offset()) {
        _pay.Load();
      }
    }
    for (; block >= PosGroup::kBlocks; block -= PosGroup::kBlocks) {
      _pos.Hop();
      if constexpr (IteratorTraits::Offset()) {
        _pay.Hop();
      }
    }
    _next = static_cast<uint32_t>(block);
    _pos.SeekBlock(_next);
    if constexpr (IteratorTraits::Offset()) {
      _pay.SeekBlock(_next);
    }
  }

  void Skip(uint64_t count) {
    SDB_ASSERT(count != 0);
    const uint64_t left = pos_limits::kBlockSize - _buf_pos;
    if (count > left) {
      count -= left;
      Enter(_next + count / pos_limits::kBlockSize);
      count %= pos_limits::kBlockSize;
      if (count == 0) {
        _buf_pos = pos_limits::kBlockSize;
      } else {
        ReadBlock();
        _buf_pos = 0;
      }
    }
    _buf_pos += count;
    SDB_ASSERT(_buf_pos <= pos_limits::kBlockSize);
    Clear();
  }

  template<typename Visit>
  IRS_FORCE_INLINE void Walk(uint32_t n, Visit&& visit) {
    while (n != 0) {
      if (_buf_pos == pos_limits::kBlockSize) {
        ReadBlock();
        _buf_pos = 0;
      }
      const auto take = static_cast<uint32_t>(
        std::min<uint64_t>(n, pos_limits::kBlockSize - _buf_pos));
      visit(static_cast<uint32_t>(_buf_pos), take);
      _buf_pos += take;
      n -= take;
    }
  }

  void ReadAttributes() noexcept {
    if constexpr (IteratorTraits::Offset()) {
      _offs.start += _offs_start_deltas[_buf_pos];
      _offs.end = _offs.start + _offs_lengths[_buf_pos];
    }
  }

  void ClearAttributes() noexcept {
    if constexpr (IteratorTraits::Offset()) {
      _offs.clear();
    }
  }

  template<typename Input>
  IRS_FORCE_INLINE void Decode(Input& pos, Input* pay) {
    block_io::ReadBlock(pos, _enc_buf, _pos_deltas);
    if constexpr (IteratorTraits::Offset()) {
      block_io::ReadBlock(*pay, _enc_buf, _offs_start_deltas);
      block_io::ReadBlock(*pay, _enc_buf, _offs_lengths);
    }
  }

  void ReadBlock() {
    if (_pend_pos < _freq && _next != 0) {
      const auto consumed = static_cast<uint32_t>(_freq - _pend_pos);
      if (consumed <= pos_limits::kBlockSize) {
        MarkDocStart(_next - 1, pos_limits::kBlockSize - consumed);
      }
    }
    if (_pos.header == nullptr) [[unlikely]] {
      Enter(_next);
    } else if (_next == PosGroup::kBlocks) [[unlikely]] {
      _pos.Step();
      if constexpr (IteratorTraits::Offset()) {
        _pay.Step();
      }
      _next = 0;
    }
    if (_pend_pos == _freq) {
      MarkDocStart(_next, 0);
    }
    _pos.Account();
    if constexpr (IteratorTraits::Offset()) {
      _pay.Account();
    }
    if (_pos.view != nullptr && (!IteratorTraits::Offset() || PayView()))
      [[likely]] {
      Decode<BytesViewInput>(*_pos.view, PayView());
    } else {
      Decode<IndexInput>(*_pos.in, PayIn());
    }
    ++_next;
  }

  IRS_FORCE_INLINE BytesViewInput* PayView() const noexcept {
    if constexpr (IteratorTraits::Offset()) {
      return _pay.view;
    } else {
      return nullptr;
    }
  }

  IRS_FORCE_INLINE IndexInput* PayIn() const noexcept {
    if constexpr (IteratorTraits::Offset()) {
      return _pay.in.get();
    } else {
      return nullptr;
    }
  }

  template<typename T>
  using ForOffset = utils::Need<IteratorTraits::Offset(), T>;

  uint32_t _pos_deltas[pos_limits::kBlockSize];
  [[no_unique_address]] ForOffset<uint32_t[pos_limits::kBlockSize]>
    _offs_start_deltas;
  [[no_unique_address]] ForOffset<uint32_t[pos_limits::kBlockSize]>
    _offs_lengths;
  uint32_t _freq = 0;  // length of the posting list for a document
  uint32_t _next = 0;
  uint32_t* _enc_buf;      // auxillary buffer to decode data
  uint64_t _pend_pos = 0;  // how many positions "behind" we are
  uint64_t _buf_pos = pos_limits::kBlockSize;  // position in pos_deltas_
  DocStart _doc_start;
#ifdef SDB_DEV
  uint64_t _doc_seq = 0;
#endif
  Stream _pos;
  [[no_unique_address]] ForOffset<Stream> _pay;
  [[no_unique_address]] ForOffset<OffsAttr> _offs;
};

// use base PosAttr type for ancestors
template<typename IteratorTraits>
struct Type<PositionImpl<IteratorTraits>> : Type<PosAttr> {};

}  // namespace irs
