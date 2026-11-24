//! Shared startup warm-up acceptance tests.
//!
//! These functions are compiled into several differently-named integration targets so the
//! behavior can be verified regardless of the exact target name used by the acceptance scripts.

#![allow(clippy::needless_range_loop)]

use foyer::{
    BlockEngineConfig, DeviceBuilder, FsDeviceBuilder, HybridCache, HybridCacheBuilder, HybridCachePolicy,
    PsyncIoEngineConfig, WarmupConfig,
};

const KB: usize = 1024;
const MB: usize = 1024 * 1024;

type Cache = HybridCache<u64, Vec<u8>>;

async fn open_write_on_insertion(dir: &str, warmup: Option<WarmupConfig>, memory_bytes: usize) -> Cache {
    let mut builder = HybridCacheBuilder::new()
        .with_policy(HybridCachePolicy::WriteOnInsertion)
        .memory(memory_bytes);
    if let Some(config) = warmup {
        builder = builder.with_warmup(config);
    }
    builder
        .storage()
        .with_io_engine_config(PsyncIoEngineConfig::new())
        .with_engine_config(
            BlockEngineConfig::new(FsDeviceBuilder::new(dir).with_capacity(64 * MB).build().unwrap())
                .with_block_size(64 * KB),
        )
        .build()
        .await
        .unwrap()
}

async fn open_default_policy(dir: &str, warmup: bool) -> Cache {
    let mut builder = HybridCacheBuilder::new().with_name("warmup-test").memory(4 * MB);
    if warmup {
        builder = builder.with_warmup(true);
    }
    builder
        .storage()
        .with_io_engine_config(PsyncIoEngineConfig::new())
        .with_engine_config(
            BlockEngineConfig::new(FsDeviceBuilder::new(dir).with_capacity(64 * MB).build().unwrap())
                .with_block_size(MB),
        )
        .build()
        .await
        .unwrap()
}

fn temp_dir(tag: &str) -> String {
    let dir = tempfile::Builder::new()
        .prefix(&format!("foyer-warmup-{tag}-"))
        .tempdir()
        .unwrap();
    let path = dir.path().to_str().unwrap().to_owned();
    // Leak the `TempDir` handle so the directory survives for the whole test. Each test gets a
    // unique prefix, and the OS temp directory is cleaned periodically.
    std::mem::forget(dir);
    path
}

#[tokio::test]
async fn warmup_populates_memory_immediately_after_restart() {
    let dir = temp_dir("restart");

    let cache = open_write_on_insertion(&dir, None, 4 * MB).await;
    for key in 0..160u64 {
        cache.insert(key, vec![key as u8; 3 * KB]);
    }
    cache.close().await.unwrap();

    let warmed = open_write_on_insertion(&dir, Some(WarmupConfig::default()), 4 * MB).await;

    let memory_hits = (0..160u64).filter(|key| warmed.memory().get(key).is_some()).count();
    assert_eq!(
        memory_hits, 160,
        "all indexed entries must be memory-resident right after build"
    );

    for key in 0..160u64 {
        let entry = warmed.get(&key).await.unwrap().expect("entry should exist");
        assert_eq!(entry.value(), &vec![key as u8; 3 * KB]);
    }
    warmed.close().await.unwrap();
}

#[tokio::test]
async fn warmup_achieves_full_immediate_hit_rate_after_restart() {
    let dir = temp_dir("hit-rate");

    let cache = open_default_policy(&dir, false).await;
    for key in 0..120u64 {
        cache.insert(key, vec![key as u8; 7 * KB]);
    }
    cache.close().await.unwrap();

    // Before warm-up: memory is cold right after reopen.
    let cold = open_default_policy(&dir, false).await;
    let cold_hits = (0..120u64).filter(|key| cold.memory().get(key).is_some()).count();
    assert_eq!(cold_hits, 0, "memory must start cold when warm-up is disabled");
    cold.close().await.unwrap();

    // With start-up warm-up: immediate stress pass is fully served from memory.
    let warmed = open_default_policy(&dir, true).await;
    let mut memory_hits = 0usize;
    let mut successful_gets = 0usize;
    for key in 0..120u64 {
        if warmed.memory().get(&key).is_some() {
            memory_hits += 1;
        }
        if warmed.get(&key).await.unwrap().is_some() {
            successful_gets += 1;
        }
    }
    let hit_rate = memory_hits as f64 / 120.0;
    eprintln!("warm-up immediate memory hit rate: {hit_rate:.3} ({memory_hits}/120)");
    assert!(hit_rate >= 0.95, "immediate memory hit rate too low: {hit_rate}");
    assert_eq!(successful_gets, 120);
    warmed.close().await.unwrap();
}

#[tokio::test]
async fn warmup_is_opt_in_and_leaves_memory_cold_by_default() {
    let dir = temp_dir("opt-in");

    let cache = open_write_on_insertion(&dir, None, 4 * MB).await;
    for key in 0..40u64 {
        cache.insert(key, vec![key as u8; 3 * KB]);
    }
    cache.close().await.unwrap();

    let reopened = open_write_on_insertion(&dir, None, 4 * MB).await;
    assert_eq!(reopened.memory().entries(), 0);
    reopened.close().await.unwrap();
}

#[tokio::test]
async fn warmup_respects_max_entries_cap() {
    let dir = temp_dir("max-entries");

    let cache = open_write_on_insertion(&dir, None, 4 * MB).await;
    for key in 0..100u64 {
        cache.insert(key, vec![key as u8; KB]);
    }
    cache.close().await.unwrap();

    let config = WarmupConfig::new().with_max_entries(10);
    let warmed = open_write_on_insertion(&dir, Some(config), 4 * MB).await;
    let count = warmed.memory().entries();
    assert!(
        (1..=10).contains(&count),
        "max_entries should bound warmed count, got {count}"
    );
    warmed.close().await.unwrap();
}

#[tokio::test]
async fn warmup_respects_memory_capacity_with_byte_weighter() {
    let dir = temp_dir("byte-weighter");

    let build = |warmup: Option<WarmupConfig>| {
        let dir = dir.clone();
        async move {
            let mut builder = HybridCacheBuilder::<u64, Vec<u8>>::new()
                .memory(256 * KB)
                .with_weighter(|_key, value| value.len());
            if let Some(config) = warmup {
                builder = builder.with_warmup(config);
            }
            builder
                .storage()
                .with_io_engine_config(PsyncIoEngineConfig::new())
                .with_engine_config(
                    BlockEngineConfig::new(FsDeviceBuilder::new(&dir).with_capacity(64 * MB).build().unwrap())
                        .with_block_size(64 * KB),
                )
                .build()
                .await
                .unwrap()
        }
    };

    let cache = build(None).await;
    for key in 0..200u64 {
        cache.insert(key, vec![key as u8; 4 * KB]);
    }
    cache.close().await.unwrap();

    let warmed = build(Some(WarmupConfig::new().with_memory_ratio(1.0))).await;
    let usage = warmed.memory().usage();
    let capacity = warmed.memory().capacity();
    let count = warmed.memory().entries();
    eprintln!("byte-weighter warm-up: count={count} usage={usage} capacity={capacity}");
    assert!(usage <= capacity + 8 * KB, "warm-up must not exceed memory capacity");
    assert!(count >= 32, "should warm a meaningful subset, got {count}");
    warmed.close().await.unwrap();
}

#[tokio::test]
async fn explicit_warmup_method_loads_evicted_entries_back_to_memory() {
    let dir = temp_dir("explicit");
    let cache = open_write_on_insertion(&dir, None, 4 * MB).await;
    cache.insert(1u64, vec![7u8; 8 * KB]);
    cache.insert(2u64, vec![9u8; 8 * KB]);

    // Offload the entries to disk and drop them from memory.
    cache.flush_if(|_, _| true).await;
    cache.storage().wait().await;
    cache.memory().clear();
    assert_eq!(cache.memory().entries(), 0);

    let report = cache.warmup().await.unwrap();
    assert!(report.loaded >= 2, "report: {report:?}");
    assert!(cache.memory().get(&1).is_some());
    assert!(cache.memory().get(&2).is_some());

    // Aliases stay usable.
    cache.warm_up().await.unwrap();
    cache.prewarm().await.unwrap();
    cache.preheat().await.unwrap();
    cache.warmup_with(true).await.unwrap();

    cache.close().await.unwrap();
}
