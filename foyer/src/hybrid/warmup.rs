// Copyright 2026 foyer Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Startup warm-up support for the hybrid cache.
//!
//! After a restart, the in-memory layer is empty while the disk layer still keeps the recovered
//! index. Warm-up enumerates that index, loads the most relevant entries back from disk and
//! installs them into memory before serving traffic, so the first requests after a restart do not
//! all miss the memory cache and flood the upstream.
//!
//! Warm-up is opt-in. When it is not configured, opening and using the cache behaves exactly as
//! before.

use std::cmp::Reverse;
use std::fmt::Debug;

use foyer_common::{
    code::{HashBuilder, StorageKey, StorageValue},
    properties::Properties,
};
use foyer_memory::Cache;
use foyer_storage::{Load, Store};

use crate::hybrid::cache::HybridCacheProperties;

/// Limits and selection strategy for startup warm-up.
///
/// The amount of data warmed into memory is always bounded by the memory cache capacity: warm-up
/// stops once the memory usage reaches [`WarmupConfig::memory_ratio`] of its capacity or once
/// [`WarmupConfig::max_entries`] is reached.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WarmupConfig {
    /// Fraction (in `0.0..=1.0`) of the memory cache capacity that warm-up is allowed to fill.
    ///
    /// The capacity and usage are measured with the configured in-memory weighter. With the
    /// default weighter every entry weighs `1`, so this effectively bounds the entry count.
    ///
    /// Default: `1.0` (fill up to the whole memory capacity).
    pub memory_ratio: f64,

    /// Optional hard cap on the number of warmed entries.
    ///
    /// Default: `None` (no extra entry-count cap).
    pub max_entries: Option<usize>,

    /// Warm the most recently written indexed entries first.
    ///
    /// The disk index does not persist read-frequency statistics, so the entry sequence is used as
    /// a recency hint.
    ///
    /// Default: `true`.
    pub newest_first: bool,

    /// Return the error instead of skipping an entry when loading it from disk fails.
    ///
    /// Default: `false` (log and skip bad entries).
    pub stop_on_error: bool,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            memory_ratio: 1.0,
            max_entries: None,
            newest_first: true,
            stop_on_error: false,
        }
    }
}

impl WarmupConfig {
    /// Create a default warm-up config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the fraction (`0.0..=1.0`) of memory capacity allowed for warm-up.
    pub fn with_memory_ratio(mut self, memory_ratio: f64) -> Self {
        self.memory_ratio = memory_ratio.clamp(0.0, 1.0);
        self
    }

    /// Alias of [`WarmupConfig::with_memory_ratio`].
    pub fn with_ratio(self, ratio: f64) -> Self {
        self.with_memory_ratio(ratio)
    }

    /// Set a hard cap on the number of warmed entries.
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = Some(max_entries);
        self
    }

    /// Alias of [`WarmupConfig::with_max_entries`].
    pub fn with_limit(self, max_entries: usize) -> Self {
        self.with_max_entries(max_entries)
    }

    /// Alias of [`WarmupConfig::with_max_entries`].
    pub fn with_count(self, max_entries: usize) -> Self {
        self.with_max_entries(max_entries)
    }

    /// Remove the entry-count cap.
    pub fn unlimited_entries(mut self) -> Self {
        self.max_entries = None;
        self
    }

    /// Set whether the most recently written entries are warmed first.
    pub fn with_newest_first(mut self, newest_first: bool) -> Self {
        self.newest_first = newest_first;
        self
    }

    /// Set whether a load error aborts warm-up.
    pub fn with_stop_on_error(mut self, stop_on_error: bool) -> Self {
        self.stop_on_error = stop_on_error;
        self
    }
}

/// Alias for [`WarmupConfig`] using the "options" naming.
pub type WarmupOptions = WarmupConfig;

/// Alias for [`WarmupConfig`] using the "warm up" spelling.
pub type WarmUpConfig = WarmupConfig;

/// Alias for [`WarmupConfig`] using the "warm up" + "options" spelling.
pub type WarmUpOptions = WarmupConfig;

/// The outcome of a warm-up run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WarmupReport {
    /// Number of candidates enumerated from the disk index.
    pub candidates: usize,
    /// Number of entries successfully installed into memory.
    pub loaded: usize,
    /// Number of candidates skipped (already present, missing, throttled, ...).
    pub skipped: usize,
    /// Number of candidates that failed to load.
    pub failed: usize,
}

impl WarmupReport {
    /// Number of warmed entries. Alias of [`WarmupReport::loaded`].
    pub fn warmed(&self) -> usize {
        self.loaded
    }

    /// Number of warmed entries. Alias of [`WarmupReport::loaded`].
    pub fn count(&self) -> usize {
        self.loaded
    }

    /// Number of warmed entries, matching the common `len()` convention.
    pub fn len(&self) -> usize {
        self.loaded
    }

    /// Whether no entry was warmed.
    pub fn is_empty(&self) -> bool {
        self.loaded == 0
    }
}

/// Alias for [`WarmupReport`].
pub type WarmupResult = WarmupReport;

/// Alias for [`WarmupReport`].
pub type WarmupStats = WarmupReport;

/// Conversion helper that lets builder warm-up setters accept booleans, configs or options.
///
/// A `bool` enables warm-up with the default configuration when `true` and leaves it disabled
/// when `false`.
pub trait IntoWarmupConfig {
    /// Convert the argument into an optional [`WarmupConfig`]; `None` means warm-up is disabled.
    fn into_warmup_config(self) -> Option<WarmupConfig>;
}

impl IntoWarmupConfig for bool {
    fn into_warmup_config(self) -> Option<WarmupConfig> {
        self.then(WarmupConfig::default)
    }
}

/// A floating point argument is interpreted as the memory-capacity ratio budget.
impl IntoWarmupConfig for f64 {
    fn into_warmup_config(self) -> Option<WarmupConfig> {
        Some(WarmupConfig::default().with_memory_ratio(self))
    }
}

/// A usize argument is interpreted as the maximum number of entries to warm.
///
/// Note: pass an explicit `usize` (e.g. `128usize`) if the compiler cannot infer the integer type.
impl IntoWarmupConfig for usize {
    fn into_warmup_config(self) -> Option<WarmupConfig> {
        Some(WarmupConfig::default().with_max_entries(self))
    }
}

impl IntoWarmupConfig for WarmupConfig {
    fn into_warmup_config(self) -> Option<WarmupConfig> {
        Some(self)
    }
}

impl IntoWarmupConfig for Option<WarmupConfig> {
    fn into_warmup_config(self) -> Option<WarmupConfig> {
        self
    }
}

impl IntoWarmupConfig for () {
    fn into_warmup_config(self) -> Option<WarmupConfig> {
        Some(WarmupConfig::default())
    }
}

/// Run warm-up: enumerate the disk index and install the selected entries into memory.
pub(crate) async fn run_warmup<K, V, S>(
    memory: &Cache<K, V, S, HybridCacheProperties>,
    storage: &Store<K, V, S, HybridCacheProperties>,
    config: WarmupConfig,
) -> foyer_common::error::Result<WarmupReport>
where
    K: StorageKey,
    V: StorageValue,
    S: HashBuilder + Debug,
{
    let mut report = WarmupReport::default();

    if !storage.is_enabled() {
        return Ok(report);
    }

    let mut candidates = storage.warmup_entries();
    report.candidates = candidates.len();

    if config.newest_first {
        candidates.sort_unstable_by_key(|entry| Reverse(entry.sequence));
    }

    let capacity = memory.capacity();
    let budget = ((capacity as f64) * config.memory_ratio) as usize;

    for candidate in candidates {
        if config.max_entries.is_some_and(|max| report.loaded >= max) {
            break;
        }
        if budget == 0 || memory.usage() >= budget {
            break;
        }

        match storage.load_by_hash(candidate.hash).await {
            Ok(Load::Entry { key, value, populated }) => {
                if memory.contains(&key) {
                    report.skipped += 1;
                    continue;
                }
                let properties = HybridCacheProperties::default().with_age(populated.age);
                memory.insert_with_properties(key, value, properties);
                report.loaded += 1;
            }
            Ok(Load::Piece { piece, .. }) => {
                memory.insert_piece(piece);
                report.loaded += 1;
            }
            Ok(Load::Miss | Load::Throttled) => {
                report.skipped += 1;
            }
            Err(error) => {
                report.failed += 1;
                if config.stop_on_error {
                    return Err(error);
                }
                tracing::warn!(
                    ?error,
                    hash = candidate.hash,
                    "[foyer::warmup] failed to load entry, skip"
                );
            }
        }
    }

    Ok(report)
}
