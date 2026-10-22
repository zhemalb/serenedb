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

#include <utility>

#include "iresearch/analysis/token_attributes.hpp"
#include "iresearch/formats/posting/block_index.hpp"
#include "iresearch/index/index_reader.hpp"
#include "iresearch/index/iterators.hpp"
#include "iresearch/search/detail/column_collector.hpp"
#include "iresearch/search/scorers/score_args.hpp"
#include "iresearch/search/scorers/score_provider.hpp"
#include "iresearch/search/scorers/scorer.hpp"
#include "iresearch/search/top/admit.hpp"
#include "iresearch/search/top/pruned_leaf.hpp"
#include "iresearch/search/top/root.hpp"
#include "iresearch/utils/attribute_provider.hpp"
#include "iresearch/utils/empty.hpp"
#include "iresearch/utils/type_limits.hpp"

namespace irs::top {

template<typename Slots, typename Table>
class PrunedPhrase : public Root {
  static_assert(Slots::kHasFreqBound,
                "a phrase with no bound cannot be pruned");

  static constexpr bool kHasScale = requires(const Slots& s) { s.Scale(); };

  struct Provider final : AttributeProvider {
    Attribute* GetMutable(TypeInfo::type_id type) noexcept final {
      if (type == irs::Type<FreqBlockAttr>::id()) {
        return &freq;
      }
      if constexpr (kHasScale) {
        if (type == irs::Type<ScaleBlockAttr>::id()) {
          return &scale;
        }
      }
      return nullptr;
    }

    FreqBlockAttr freq;
    [[no_unique_address]] utils::Need<kHasScale, ScaleBlockAttr> scale;
  };

  static constexpr uint32_t kWindow = 64;
  static constexpr uint32_t kMaxInterval = 64;
  static constexpr uint32_t kPassShare = 2;

 public:
  template<typename... Args>
  PrunedPhrase(Table table, const SubReader& segment, const TermReader& field,
               const irs::detail::ScoreArgs& args, Args&&... slots)
    : _slots{std::forward<Args>(slots)...},
      _fetcher{*args.fetcher},
      _admit{table} {
    _provider.freq.value = _freqs;
    if constexpr (kHasScale) {
      _provider.scale.value = _scales;
      absl::c_fill(_scales, kNoBoost);
    }
    SDB_ASSERT(args.scorer != nullptr);
    _score = args.scorer->PrepareScorer({
      .segment = segment,
      .field = field.meta(),
      .doc_attrs = _provider,
      .fetcher = *args.fetcher,
      .stats = args.stats,
      .boost = args.boost,
    });
    if (auto source = args.scorer->PrepareScoreBoundSource()) {
      _bound_func = args.scorer->PrepareScorer({
        .segment = segment,
        .field = field.meta(),
        .doc_attrs = *source,
        .fetcher = *args.fetcher,
        .stats = args.stats,
        .boost = args.boost,
      });
      _bound_source = std::move(source);
    }
  }

  void Run(doc_id_t min, doc_id_t max, LoserScoreCollector& collector) final {
    auto doc = _slots.Seek(min);
    while (doc < max) {
      const auto threshold = collector.ScoreThreshold();
      if (threshold == std::numeric_limits<score_t>::lowest()) {
        doc = MatchFirst(doc, max, collector);
        continue;
      }
      if (doc > _block_last) {
        doc = SkipBlocks(doc, threshold);
        continue;
      }
      const auto end = _block_last < max ? _block_last + 1 : max;
      if (_skip != 0) {
        --_skip;
        doc = MatchFirst(doc, end, collector);
      } else {
        doc = BoundFirst(doc, end, collector);
      }
    }
    FlushBatch(collector);
    _admit.Flush(collector);
  }

 private:
  score_t BoundScore(BoundPair bound) {
    _bound_source->Set(_slots.FreqBoundOf(bound.freq), bound.norm);
    return _bound_func.Score();
  }

  score_t RunScore(const BlockIndex& index, uint32_t r) {
    if (_run != r) {
      _run = r;
      _run_score = BoundScore(index.RunBound(r));
    }
    return _run_score;
  }

  doc_id_t SkipBlocks(doc_id_t doc, score_t threshold) {
    auto* const leaf = _bound_source ? _slots.BoundLeaf() : nullptr;
    const auto* const index = leaf ? leaf->Blocks() : nullptr;
    if (!index) {
      _block_last = doc_limits::eof();
      return doc;
    }
    const auto first = leaf->LeafBlock();
    const auto b = FirstCompetitive(
      *index, first, threshold, [&](uint32_t r) { return RunScore(*index, r); },
      [&](uint32_t k) { return BoundScore(index->Bound(k)); });
    if (b == first) {
      _block_last = index->Last(b);
      return doc;
    }
    if (b == index->Size()) {
      return doc_limits::eof();
    }
    return _slots.Seek(index->Last(b - 1) + 1);
  }

  doc_id_t MatchFirst(doc_id_t doc, doc_id_t end,
                      LoserScoreCollector& collector) {
    for (uint32_t seen = 0; doc < end && seen != kWindow;
         doc = _slots.Next(doc), ++seen) {
      if (!_slots.Match(doc)) {
        continue;
      }
      _docs[_batch] = doc;
      _freqs[_batch] = _slots.Freq();
      if constexpr (kHasScale) {
        _scales[_batch] = _slots.Scale();
      }
      if (++_batch == kScoreBlock) {
        _fetcher.FetchScoreBlock(std::span<const doc_id_t, kScoreBlock>{_docs});
        _score.ScoreBlock(_scores);
        _admit.AddDocs(collector, _docs, kScoreBlock, _scores);
        _batch = 0;
      }
    }
    return doc;
  }

  doc_id_t BoundFirst(doc_id_t doc, doc_id_t end,
                      LoserScoreCollector& collector) {
    FlushBatch(collector);
    if constexpr (kHasScale) {
      _scales[0] = kNoBoost;
    }
    for (; doc < end && _seen != kWindow; doc = _slots.Next(doc)) {
      ++_seen;
      const auto bound = _slots.FreqBound();
      _freqs[0] = bound;
      _fetcher.Fetch(doc);
      auto score = _score.Score();
      if (score <= collector.ScoreThreshold()) {
        continue;
      }
      ++_passed;
      if (!_slots.MatchAfterBound()) {
        continue;
      }
      const auto freq = _slots.Freq();
      SDB_ASSERT(freq <= bound);
      if constexpr (kHasScale) {
        _freqs[0] = freq;
        _scales[0] = _slots.Scale();
        score = _score.Score();
        _scales[0] = kNoBoost;
      } else if (freq != bound) {
        _freqs[0] = freq;
        score = _score.Score();
      }
      _admit.Add(collector, score, doc);
    }
    if (_seen == kWindow) {
      if (_passed * kPassShare > _seen) {
        _interval = std::min(2 * _interval, kMaxInterval);
        _skip = _interval;
      } else {
        _interval = 1;
      }
      _seen = 0;
      _passed = 0;
    }
    return doc;
  }

  void FlushBatch(LoserScoreCollector& collector) {
    if (_batch != 0) {
      _fetcher.Fetch(std::span<const doc_id_t>{_docs, _batch});
      _score.Score(_scores, _batch);
      _admit.AddDocs(collector, _docs, _batch, _scores);
      _batch = 0;
    }
  }

  ABSL_CACHELINE_ALIGNED doc_id_t _docs[kScoreBlock];
  ABSL_CACHELINE_ALIGNED score_t _scores[kScoreBlock];
  ABSL_CACHELINE_ALIGNED uint32_t _freqs[kScoreBlock];
  [[no_unique_address]] utils::Need<kHasScale, score_t[kScoreBlock]> _scales{};
  Slots _slots;
  Provider _provider;
  ScoreFunction _score;
  ScoreFunction _bound_func;
  ScoreBoundSource::ptr _bound_source;
  ColumnArgsFetcher& _fetcher;
  doc_id_t _block_last = doc_limits::invalid();
  uint32_t _run = std::numeric_limits<uint32_t>::max();
  score_t _run_score = 0;
  scores_size_t _batch = 0;
  uint32_t _seen = 0;
  uint32_t _passed = 0;
  uint32_t _interval = 1;
  uint32_t _skip = 0;
  [[no_unique_address]] Admit<Table> _admit;
};

}  // namespace irs::top
