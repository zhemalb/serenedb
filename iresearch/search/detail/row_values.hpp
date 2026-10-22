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

#include <duckdb/common/allocator.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/types/vector.hpp>
#include <duckdb/storage/arena_allocator.hpp>

#include "iresearch/analysis/tokenizer.hpp"
#include "iresearch/utils/json_leaf.hpp"

namespace irs {

class RowValues {
 public:
  void Bind(duckdb::Vector& values, duckdb::idx_t count);

  template<typename Visitor>
  void ForEach(duckdb::idx_t row, Visitor&& visit) {
    ForEachStored(row, [&](duckdb::string_t value) {
      if (!_json) {
        return visit(value);
      }
      const auto leaf = _reader.Read({value.GetData(), value.GetSize()});
      if (leaf.kind != JsonLeafKind::String) {
        return true;
      }
      return visit(duckdb::string_t{leaf.string.data(),
                                    static_cast<uint32_t>(leaf.string.size())});
    });
  }

  uint64_t Bytes(duckdb::idx_t row) const {
    uint64_t bytes = 0;
    ForEachStored(row, [&](duckdb::string_t value) {
      bytes += value.GetSize();
      return true;
    });
    return bytes;
  }

 private:
  template<typename Visitor>
  void ForEachStored(duckdb::idx_t row, Visitor&& visit) const {
    const auto idx = _format.sel->get_index(row);
    if (!_format.validity.RowIsValid(idx)) {
      return;
    }
    const auto children = [&](uint64_t begin, uint64_t size) {
      const auto* data =
        duckdb::UnifiedVectorFormat::GetData<duckdb::string_t>(_children);
      analysis::ForEachValidRow(
        _children, begin, static_cast<uint32_t>(size),
        [&](uint32_t, uint32_t child) { return visit(data[child]); });
    };
    if (_type == duckdb::LogicalTypeId::LIST) {
      const auto entry =
        duckdb::UnifiedVectorFormat::GetData<duckdb::list_entry_t>(
          _format)[idx];
      children(entry.offset, entry.length);
      return;
    }
    if (_type == duckdb::LogicalTypeId::ARRAY) {
      children(idx * _array_size, _array_size);
      return;
    }
    visit(duckdb::UnifiedVectorFormat::GetData<duckdb::string_t>(_format)[idx]);
  }

  duckdb::UnifiedVectorFormat _format;
  duckdb::UnifiedVectorFormat _children;
  duckdb::LogicalTypeId _type = duckdb::LogicalTypeId::VARCHAR;
  uint64_t _array_size = 0;
  bool _json = false;
  duckdb::ArenaAllocator _strings{duckdb::Allocator::DefaultAllocator()};
  JsonLeafReader _reader{_strings};
};

}  // namespace irs
