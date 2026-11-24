use rayon::prelude::*;
use std::iter::once_with;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct ActiveGuard<'a> {
    active: &'a AtomicUsize,
    peak: &'a AtomicUsize,
    limit: usize,
}

impl<'a> ActiveGuard<'a> {
    fn enter(active: &'a AtomicUsize, peak: &'a AtomicUsize, limit: usize) -> Self {
        let now = active.fetch_add(1, Ordering::AcqRel) + 1;
        peak.fetch_max(now, Ordering::AcqRel);
        assert!(
            now <= limit,
            "active consumers ({now}) exceeded limit ({limit})"
        );
        Self {
            active,
            peak,
            limit,
        }
    }
}

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        let prev = self.active.fetch_sub(1, Ordering::AcqRel);
        assert!(prev >= 1 && prev <= self.limit);
        let _ = self.peak;
    }
}

fn burn_cpu(seed: usize, rounds: usize) -> usize {
    let mut x = seed ^ 0x9e37_79b9;
    for _ in 0..rounds {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        x ^= x >> 13;
    }
    x
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn bounded_concurrency_pool_and_limit_matrix() {
    for pool_threads in [1, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .build()
            .unwrap();

        pool.install(|| {
            // Unrestricted par_bridge() preserves full completeness across pool sizes.
            let mut unrestricted: Vec<_> = (0..500usize).par_bridge().collect();
            unrestricted.sort_unstable();
            assert_eq!(unrestricted, (0..500usize).collect::<Vec<_>>());

            for limit in [1, 2, 3, 4, 8, pool_threads + 3, usize::MAX] {
                let effective = limit.min(pool_threads);
                let max_leaves = if limit < pool_threads || effective == 1 {
                    effective
                } else {
                    pool_threads + 1
                };
                let leaves = AtomicUsize::new(0);
                let active_leaves = AtomicUsize::new(0);
                let active = AtomicUsize::new(0);
                let peak = AtomicUsize::new(0);

                let mut items: Vec<_> = (0..1000usize)
                    .par_bridge()
                    .with_max_concurrency(limit)
                    .fold(
                        || {
                            leaves.fetch_add(1, Ordering::Relaxed);
                            (Vec::new(), false)
                        },
                        |(mut items, mut seen), item| {
                            if !seen {
                                active_leaves.fetch_add(1, Ordering::Relaxed);
                                seen = true;
                            }
                            let _guard = ActiveGuard::enter(&active, &peak, effective);
                            if item % 17 == 0 {
                                std::thread::yield_now();
                            }
                            items.push(item);
                            (items, seen)
                        },
                    )
                    .map(|(items, _)| items)
                    .flatten_iter()
                    .collect();

                items.sort_unstable();
                assert_eq!(items, (0..1000usize).collect::<Vec<_>>());
                assert_eq!(active.load(Ordering::Acquire), 0);

                let leaf_count = leaves.load(Ordering::Relaxed);
                assert!(
                    (1..=max_leaves).contains(&leaf_count),
                    "pool={pool_threads}, limit={limit}: leaves={leaf_count}, max_leaves={max_leaves}"
                );
                let active_leaf_count = active_leaves.load(Ordering::Relaxed);
                assert!(
                    (1..=effective).contains(&active_leaf_count),
                    "pool={pool_threads}, limit={limit}: active_leaves={active_leaf_count}, effective={effective}"
                );
                let peak_active = peak.load(Ordering::Acquire);
                assert!(
                    (1..=effective).contains(&peak_active),
                    "pool={pool_threads}, limit={limit}: peak={peak_active}, effective={effective}"
                );
            }
        });
    }
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn bounded_bridge_heavy_contention_and_variable_duration() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();

    pool.install(|| {
        for limit in [1, 2, 3, 4, 8, 16, usize::MAX] {
            let effective = limit.min(8);
            let max_leaves = if limit < 8 || effective == 1 {
                effective
            } else {
                9
            };
            let leaves = AtomicUsize::new(0);
            let active_leaves = AtomicUsize::new(0);
            let active = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);

            let count: usize = (0..2000usize)
                .par_bridge()
                .with_max_concurrency(limit)
                .fold(
                    || {
                        leaves.fetch_add(1, Ordering::Relaxed);
                        (0usize, false)
                    },
                    |(acc, mut seen), i| {
                        if !seen {
                            active_leaves.fetch_add(1, Ordering::Relaxed);
                            seen = true;
                        }
                        let _guard = ActiveGuard::enter(&active, &peak, effective);
                        // Mix very short, medium, and heavy CPU bursts plus thread yields.
                        let rounds = match i % 23 {
                            0 => 15_000,
                            1 | 2 => 3_000,
                            _ => 50,
                        };
                        let mixed = burn_cpu(i, rounds);
                        if i % 11 == 0 {
                            std::thread::yield_now();
                        }
                        (acc + usize::from(mixed != usize::MAX || i == 0), seen)
                    },
                )
                .map(|(acc, _)| acc)
                .sum();

            assert_eq!(count, 2000);
            assert_eq!(active.load(Ordering::Acquire), 0);
            assert!((1..=max_leaves).contains(&leaves.load(Ordering::Relaxed)));
            assert!((1..=effective).contains(&active_leaves.load(Ordering::Relaxed)));
            assert!((1..=effective).contains(&peak.load(Ordering::Acquire)));
        }
    });
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn bounded_bridge_clone_debug_and_reconfigure() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    pool.install(|| {
        let base = (0..100).par_bridge();
        assert!(format!("{base:?}").contains("IterBridge"));

        let bridge = base.with_max_concurrency(1);
        assert!(format!("{bridge:?}").contains("IterBridge"));
        assert_eq!(bridge.clone().count(), 100);
        assert_eq!(bridge.clone().fold(|| (), |(), _| ()).count(), 1);

        let bridge = bridge.with_max_concurrency(3);
        let consumers = bridge.clone().fold(|| (), |(), _| ()).count();
        assert!((1..=3).contains(&consumers));

        assert_eq!(
            bridge
                .with_max_concurrency(1)
                .fold(|| (), |(), _| ())
                .count(),
            1
        );
    });
}

#[test]
fn bounded_bridge_empty_short_circuit_and_unbounded_iter() {
    for limit in [1, 2, 4, usize::MAX] {
        assert_eq!(
            std::iter::empty::<usize>()
                .par_bridge()
                .with_max_concurrency(limit)
                .count(),
            0
        );
        assert_eq!(
            (0..100usize)
                .par_bridge()
                .with_max_concurrency(limit)
                .find_any(|&i| i == 42),
            Some(42)
        );
        // Unbounded iterator terminates cleanly via short-circuiting without buffering.
        assert_eq!(
            (0usize..)
                .par_bridge()
                .with_max_concurrency(limit)
                .find_any(|&i| i == 500),
            Some(500)
        );
        let err = (0usize..)
            .par_bridge()
            .with_max_concurrency(limit)
            .try_for_each(|i| if i >= 200 { Err(i) } else { Ok(()) });
        assert!(matches!(err, Err(x) if x >= 200));
        assert!(
            (0usize..)
                .par_bridge()
                .with_max_concurrency(limit)
                .any(|i| i == 150)
        );
        assert_eq!(
            (0usize..)
                .par_bridge()
                .with_max_concurrency(limit)
                .take_any(25)
                .count(),
            25
        );
    }

    // With limit 1, take_any(0) must not pull any item and take_any(k) pulls exactly k items.
    let pulled = AtomicUsize::new(0);
    let count = (0..100usize)
        .inspect(|_| {
            pulled.fetch_add(1, Ordering::Relaxed);
        })
        .par_bridge()
        .with_max_concurrency(1)
        .take_any(0)
        .count();
    assert_eq!(count, 0);
    assert_eq!(pulled.load(Ordering::Relaxed), 0);

    let pulled = AtomicUsize::new(0);
    let count = (0..100usize)
        .inspect(|_| {
            pulled.fetch_add(1, Ordering::Relaxed);
        })
        .par_bridge()
        .with_max_concurrency(1)
        .take_any(7)
        .count();
    assert_eq!(count, 7);
    assert_eq!(pulled.load(Ordering::Relaxed), 7);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn bounded_bridge_short_circuit_respects_bound_in_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();

    pool.install(|| {
        for limit in [1, 2, 3, 4, 8] {
            let active = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            let found = (0usize..)
                .par_bridge()
                .with_max_concurrency(limit)
                .find_any(|&i| {
                    let _guard = ActiveGuard::enter(&active, &peak, limit);
                    if i % 19 == 0 {
                        std::thread::yield_now();
                    }
                    i == 1_000
                });
            assert_eq!(found, Some(1_000));
            assert_eq!(active.load(Ordering::Acquire), 0);
            assert!((1..=limit).contains(&peak.load(Ordering::Acquire)));
        }
    });
}

#[test]
#[should_panic(expected = "max_concurrency must not be zero")]
fn zero_concurrency() {
    let _ = (0..10).par_bridge().with_max_concurrency(0);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn bounded_bridge_nested_rayon_and_recursion_guard() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();

    pool.install(|| {
        for limit in [1, 2, 4, 8] {
            let mut items: Vec<_> = (0..1000usize)
                .map(|i| rayon::join(|| i, || i + 1))
                .par_bridge()
                .with_max_concurrency(limit)
                .map(|(a, b)| rayon::join(|| a, || b))
                .collect();
            items.sort_unstable();
            assert_eq!(
                items,
                (0..1000usize).map(|i| (i, i + 1)).collect::<Vec<_>>()
            );
        }

        // Nested par_bridge inside par_bridge: each enforces its own bound independently.
        let outer_active = AtomicUsize::new(0);
        let outer_peak = AtomicUsize::new(0);
        let total: usize = (0..40usize)
            .par_bridge()
            .with_max_concurrency(2)
            .map(|outer| {
                let _outer_guard = ActiveGuard::enter(&outer_active, &outer_peak, 2);
                let inner_active = AtomicUsize::new(0);
                let inner_peak = AtomicUsize::new(0);
                let sum: usize = (0..50usize)
                    .par_bridge()
                    .with_max_concurrency(3)
                    .map(|inner| {
                        let _inner_guard = ActiveGuard::enter(&inner_active, &inner_peak, 3);
                        outer * 50 + inner
                    })
                    .sum();
                assert_eq!(inner_active.load(Ordering::Acquire), 0);
                assert!((1..=3).contains(&inner_peak.load(Ordering::Acquire)));
                sum
            })
            .sum();
        assert_eq!(total, (0..2000usize).sum::<usize>());
        assert_eq!(outer_active.load(Ordering::Acquire), 0);
        assert!((1..=2).contains(&outer_peak.load(Ordering::Acquire)));

        // Stress work-stealing recursion guard (mirrors tests/par_bridge_recursion.rs).
        for limit in [1, 2, 4] {
            let mut par: Vec<_> = (0..5_000usize)
                .into_par_iter()
                .flat_map(|i| {
                    once_with(move || rayon::join(move || i, move || i * 2))
                        .par_bridge()
                        .with_max_concurrency(limit)
                })
                .collect();
            par.par_sort_unstable();
            assert_eq!(par, (0..5_000usize).map(|i| (i, i * 2)).collect::<Vec<_>>());
        }
    });
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn bounded_bridge_multiple_concurrent_bridges() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();

    pool.install(|| {
        let limits = [1usize, 2, 3, 4];
        let actives: Vec<_> = limits.iter().map(|_| AtomicUsize::new(0)).collect();
        let peaks: Vec<_> = limits.iter().map(|_| AtomicUsize::new(0)).collect();
        let consumer_counts: Vec<_> = limits.iter().map(|_| AtomicUsize::new(0)).collect();
        let sums: Vec<_> = limits.iter().map(|_| AtomicUsize::new(0)).collect();

        rayon::scope(|s| {
            for (idx, &limit) in limits.iter().enumerate() {
                let active = &actives[idx];
                let peak = &peaks[idx];
                let consumers = &consumer_counts[idx];
                let sum_out = &sums[idx];
                s.spawn(move |_| {
                    let sum: usize = (0..1000usize)
                        .par_bridge()
                        .with_max_concurrency(limit)
                        .fold(
                            || {
                                consumers.fetch_add(1, Ordering::Relaxed);
                                0usize
                            },
                            |acc, i| {
                                let _guard = ActiveGuard::enter(active, peak, limit);
                                let _ = burn_cpu(i, 400);
                                if i % 13 == 0 {
                                    std::thread::yield_now();
                                }
                                acc + i
                            },
                        )
                        .sum();
                    sum_out.store(sum, Ordering::Release);
                });
            }
        });

        let expected: usize = (0..1000usize).sum();
        for (idx, &limit) in limits.iter().enumerate() {
            assert_eq!(sums[idx].load(Ordering::Acquire), expected);
            assert_eq!(actives[idx].load(Ordering::Acquire), 0);
            assert!((1..=limit).contains(&consumer_counts[idx].load(Ordering::Relaxed)));
            assert!((1..=limit).contains(&peaks[idx].load(Ordering::Acquire)));
        }
    });
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn bounded_bridge_single_thread_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

    pool.install(|| {
        for limit in [1, 2, 3, 4, 8, usize::MAX] {
            let consumers = (0..100)
                .par_bridge()
                .with_max_concurrency(limit)
                .fold(|| (), |(), _| ())
                .count();
            assert_eq!(consumers, 1);
        }
    });
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn bounded_bridge_uses_execution_pool() {
    let bridge = (0..100).par_bridge().with_max_concurrency(4);
    for threads in [1, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let effective = threads.min(4);
            let max_leaves = if 4 < threads || effective == 1 {
                effective
            } else {
                threads + 1
            };
            let active_leaves = AtomicUsize::new(0);
            let leaves = bridge
                .clone()
                .fold(
                    || false,
                    |mut seen, _| {
                        if !seen {
                            active_leaves.fetch_add(1, Ordering::Relaxed);
                            seen = true;
                        }
                        seen
                    },
                )
                .count();
            assert!((1..=max_leaves).contains(&leaves));
            assert!((1..=effective).contains(&active_leaves.load(Ordering::Relaxed)));
        });
    }
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn bounded_bridge_nested_work_is_not_limited() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    pool.install(|| {
        let mut indices: Vec<_> = std::iter::once(())
            .par_bridge()
            .with_max_concurrency(1)
            .map(|()| rayon::broadcast(|ctx| ctx.index()))
            .flatten_iter()
            .collect();
        indices.sort_unstable();
        assert_eq!(indices, vec![0, 1, 2, 3]);
    });
}

#[test]
#[cfg_attr(
    any(
        not(panic = "unwind"),
        target_os = "emscripten",
        target_family = "wasm"
    ),
    ignore
)]
fn bounded_bridge_panic_propagation_and_poisoning() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    pool.install(|| {
        // Panic inside Iterator::next() poisons the internal mutex; sibling consumers exit
        // cleanly and the original panic propagates to the caller.
        for limit in [1, 2, 4, 8] {
            let iter = (0..500usize).inspect(|&i| {
                if i == 100 {
                    panic!("next boom {limit}");
                }
            });
            let res = panic::catch_unwind(AssertUnwindSafe(|| {
                iter.par_bridge().with_max_concurrency(limit).for_each(|i| {
                    let _ = burn_cpu(i, 200);
                });
            }));
            let payload = res.expect_err("expected panic from Iterator::next");
            let msg = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(
                msg.contains(&format!("next boom {limit}")),
                "unexpected panic payload: {msg:?}"
            );
        }

        // Panic inside consumer closure propagates cleanly, and panic_fuse stops early.
        for limit in [1, 2, 4] {
            let processed = AtomicUsize::new(0);
            let res = panic::catch_unwind(AssertUnwindSafe(|| {
                (0..10_000usize)
                    .par_bridge()
                    .with_max_concurrency(limit)
                    .panic_fuse()
                    .for_each(|i| {
                        if i == 50 {
                            panic!("consumer boom {limit}");
                        }
                        processed.fetch_add(1, Ordering::Relaxed);
                        if i % 16 == 0 {
                            std::thread::yield_now();
                        }
                    });
            }));
            let payload = res.expect_err("expected panic from consumer");
            let msg = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(
                msg.contains(&format!("consumer boom {limit}")),
                "unexpected panic payload: {msg:?}"
            );
            assert!(processed.load(Ordering::Relaxed) < 10_000);
        }
    });
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn bounded_bridge_non_fused_iterator() {
    struct NonFused {
        next: usize,
        end: usize,
        yielded_none: AtomicBool,
    }

    impl Iterator for NonFused {
        type Item = usize;

        fn next(&mut self) -> Option<usize> {
            if self.next < self.end {
                let v = self.next;
                self.next += 1;
                Some(v)
            } else if !self.yielded_none.swap(true, Ordering::Relaxed) {
                None
            } else {
                // Would yield a spurious value if polled again without fusing.
                Some(999_999)
            }
        }
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    pool.install(|| {
        for limit in [1, 2, 4] {
            let iter = NonFused {
                next: 0,
                end: 200,
                yielded_none: AtomicBool::new(false),
            };
            let mut items: Vec<_> = iter.par_bridge().with_max_concurrency(limit).collect();
            items.sort_unstable();
            assert_eq!(items, (0..200usize).collect::<Vec<_>>());
        }
    });
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn bounded_bridge_single_consumer_execution_context_and_tls() {
    use std::cell::Cell;

    thread_local! {
        static TAG: Cell<u32> = const { Cell::new(0) };
    }

    // Outside any Rayon worker: limit = 1 runs on the calling thread, preserving caller TLS,
    // while nested Rayon calls still execute on the global pool.
    let handle = std::thread::spawn(|| {
        TAG.set(0xCAFE);
        let caller_tid = std::thread::current().id();
        let nested_ran = AtomicBool::new(false);

        let sum: usize = (0..32usize)
            .par_bridge()
            .with_max_concurrency(1)
            .map(|i| {
                assert_eq!(std::thread::current().id(), caller_tid);
                assert_eq!(rayon::current_thread_index(), None);
                assert_eq!(TAG.get(), 0xCAFE);
                if i == 0 {
                    let (a, b) = rayon::join(
                        || {
                            assert!(rayon::current_thread_index().is_some());
                            1usize
                        },
                        || {
                            assert!(rayon::current_thread_index().is_some());
                            2usize
                        },
                    );
                    assert_eq!(a + b, 3);
                    nested_ran.store(true, Ordering::Relaxed);
                }
                i
            })
            .sum();

        assert_eq!(sum, (0..32usize).sum::<usize>());
        assert!(nested_ran.load(Ordering::Relaxed));
    });
    handle.join().unwrap();

    // Inside a custom scoped pool (both multi-thread with limit = 1 and single-thread with
    // limit = 4): execution stays on the custom pool worker, preserves pool TLS, and nested
    // Rayon operations stay within that custom pool.
    for (pool_threads, limit) in [(4usize, 1usize), (1usize, 4usize)] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .build_scoped(
                |thread| {
                    TAG.set(0xBEEF);
                    thread.run();
                },
                |pool| {
                    pool.install(|| {
                        let caller_tid = std::thread::current().id();
                        let caller_idx = rayon::current_thread_index();
                        assert!(caller_idx.is_some());

                        let mut broadcast_indices: Vec<usize> = (0..16usize)
                            .par_bridge()
                            .with_max_concurrency(limit)
                            .map(|i| {
                                assert_eq!(std::thread::current().id(), caller_tid);
                                assert_eq!(rayon::current_thread_index(), caller_idx);
                                assert_eq!(TAG.get(), 0xBEEF);
                                if i == 0 {
                                    rayon::broadcast(|ctx| {
                                        assert_eq!(TAG.get(), 0xBEEF);
                                        ctx.index()
                                    })
                                } else {
                                    Vec::new()
                                }
                            })
                            .flatten_iter()
                            .collect();

                        broadcast_indices.sort_unstable();
                        assert_eq!(broadcast_indices, (0..pool_threads).collect::<Vec<_>>());
                    });
                },
            )
            .unwrap();
    }
}

#[test]
fn bounded_bridge_single_consumer_pre_full_and_drop_order() {
    use rayon::iter::plumbing::{Consumer, Folder, Reducer, UnindexedConsumer};
    use std::sync::Mutex;

    struct AlreadyFullConsumer;
    struct AlreadyFullFolder;
    struct NoopReducer;

    impl Reducer<()> for NoopReducer {
        fn reduce(self, (): (), (): ()) {}
    }

    impl Folder<usize> for AlreadyFullFolder {
        type Result = ();
        fn consume(self, _: usize) -> Self {
            panic!("consume should not be called when consumer starts full");
        }
        fn complete(self) {}
        fn full(&self) -> bool {
            false
        }
    }

    impl Consumer<usize> for AlreadyFullConsumer {
        type Folder = AlreadyFullFolder;
        type Reducer = NoopReducer;
        type Result = ();

        fn split_at(self, _: usize) -> (Self, Self, Self::Reducer) {
            (Self, Self, NoopReducer)
        }
        fn into_folder(self) -> Self::Folder {
            AlreadyFullFolder
        }
        fn full(&self) -> bool {
            true
        }
    }

    impl UnindexedConsumer<usize> for AlreadyFullConsumer {
        fn split_off_left(&self) -> Self {
            Self
        }
        fn to_reducer(&self) -> Self::Reducer {
            NoopReducer
        }
    }

    // Even if Folder::full() returns false, Consumer::full() returning true must short-circuit
    // before polling the iterator (matching bridge_unindexed).
    let pulled = AtomicUsize::new(0);
    let iter = (0..10usize).inspect(|_| {
        pulled.fetch_add(1, Ordering::Relaxed);
    });
    iter.par_bridge()
        .with_max_concurrency(1)
        .drive_unindexed(AlreadyFullConsumer);
    assert_eq!(pulled.load(Ordering::Relaxed), 0);

    // Drop order on limit = 1 matches bridge_unindexed: folder drops before the iterator.
    let events = Mutex::new(Vec::<&'static str>::new());
    struct TrackedIter<'a> {
        remaining: usize,
        events: &'a Mutex<Vec<&'static str>>,
    }
    impl Iterator for TrackedIter<'_> {
        type Item = usize;
        fn next(&mut self) -> Option<usize> {
            if self.remaining == 0 {
                None
            } else {
                self.remaining -= 1;
                Some(self.remaining)
            }
        }
    }
    impl Drop for TrackedIter<'_> {
        fn drop(&mut self) {
            self.events.lock().unwrap().push("iter");
        }
    }

    struct TrackedFolder<'a>(&'a Mutex<Vec<&'static str>>);
    impl Drop for TrackedFolder<'_> {
        fn drop(&mut self) {
            self.0.lock().unwrap().push("folder");
        }
    }

    let _count: usize = (TrackedIter {
        remaining: 5,
        events: &events,
    })
    .par_bridge()
    .with_max_concurrency(1)
    .fold(|| TrackedFolder(&events), |f, _| f)
    .count();

    assert_eq!(*events.lock().unwrap(), vec!["folder", "iter"]);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn bounded_bridge_nested_par_iter_bounds() {
    for pool_threads in [2usize, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .build()
            .unwrap();

        pool.install(|| {
            for limit in [1usize, 2, 3, 4, 8, usize::MAX] {
                let effective = limit.min(pool_threads);
                let max_leaves = if limit < pool_threads || effective == 1 {
                    effective
                } else {
                    pool_threads + 1
                };
                let leaves = AtomicUsize::new(0);
                let active_leaves = AtomicUsize::new(0);
                let active = AtomicUsize::new(0);
                let peak = AtomicUsize::new(0);

                let sum: usize = (0..32usize)
                    .par_bridge()
                    .with_max_concurrency(limit)
                    .fold(
                        || {
                            leaves.fetch_add(1, Ordering::Relaxed);
                            (0usize, false)
                        },
                        |(acc, mut seen), outer| {
                            if !seen {
                                active_leaves.fetch_add(1, Ordering::Relaxed);
                                seen = true;
                            }
                            let _guard = ActiveGuard::enter(&active, &peak, effective);
                            let inner: usize =
                                (0..128usize).into_par_iter().map(|j| outer * 128 + j).sum();
                            (acc + inner, seen)
                        },
                    )
                    .map(|(acc, _)| acc)
                    .sum();

                assert_eq!(sum, (0..(32 * 128)).sum::<usize>());
                assert_eq!(active.load(Ordering::Acquire), 0);
                assert!((1..=max_leaves).contains(&leaves.load(Ordering::Relaxed)));
                assert!((1..=effective).contains(&active_leaves.load(Ordering::Relaxed)));
                assert!((1..=effective).contains(&peak.load(Ordering::Acquire)));
            }
        });
    }
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn bounded_bridge_stateful_consumers_and_split_leaves() {
    struct CountedClone<'a>(&'a AtomicUsize);

    impl Clone for CountedClone<'_> {
        fn clone(&self) -> Self {
            self.0.fetch_add(1, Ordering::Relaxed);
            Self(self.0)
        }
    }

    for pool_threads in [1usize, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .build()
            .unwrap();

        pool.install(|| {
            for limit in [1usize, 2, 4, usize::MAX] {
                let effective = limit.min(pool_threads);
                let max_leaves = if limit < pool_threads || effective == 1 {
                    effective
                } else {
                    pool_threads + 1
                };

                // for_each_init: init() runs once per split leaf; active workers processing items
                // stay strictly <= effective even with nested Rayon work.
                let inits = AtomicUsize::new(0);
                let active_leaves = AtomicUsize::new(0);
                let active = AtomicUsize::new(0);
                let peak = AtomicUsize::new(0);
                let total = AtomicUsize::new(0);
                (0..64usize)
                    .par_bridge()
                    .with_max_concurrency(limit)
                    .for_each_init(
                        || {
                            inits.fetch_add(1, Ordering::Relaxed);
                            false
                        },
                        |seen, i| {
                            if !*seen {
                                active_leaves.fetch_add(1, Ordering::Relaxed);
                                *seen = true;
                            }
                            let _guard = ActiveGuard::enter(&active, &peak, effective);
                            let sub: usize = (0..32usize).into_par_iter().map(|j| i + j).sum();
                            total.fetch_add(sub, Ordering::Relaxed);
                        },
                    );
                let expected: usize = (0..64usize)
                    .map(|i| (0..32usize).map(|j| i + j).sum::<usize>())
                    .sum();
                assert_eq!(total.load(Ordering::Acquire), expected);
                assert_eq!(active.load(Ordering::Acquire), 0);
                assert!((1..=max_leaves).contains(&inits.load(Ordering::Relaxed)));
                assert!((1..=effective).contains(&active_leaves.load(Ordering::Relaxed)));
                assert!((1..=effective).contains(&peak.load(Ordering::Acquire)));

                // for_each_with: clones happen in split_off_left() (at most max_leaves - 1 clones,
                // i.e. at most max_leaves total instances), while active concurrency <= effective.
                let clones = AtomicUsize::new(0);
                let active = AtomicUsize::new(0);
                let peak = AtomicUsize::new(0);
                let sum = AtomicUsize::new(0);
                (0..64usize)
                    .par_bridge()
                    .with_max_concurrency(limit)
                    .for_each_with(CountedClone(&clones), |_state, i| {
                        let _guard = ActiveGuard::enter(&active, &peak, effective);
                        if i % 7 == 0 {
                            std::thread::yield_now();
                        }
                        sum.fetch_add(i, Ordering::Relaxed);
                    });
                assert_eq!(sum.load(Ordering::Acquire), (0..64usize).sum::<usize>());
                assert_eq!(active.load(Ordering::Acquire), 0);
                assert!(
                    clones.load(Ordering::Relaxed) < max_leaves,
                    "pool={pool_threads}, limit={limit}: clones={}, max_leaves={max_leaves}",
                    clones.load(Ordering::Relaxed)
                );
                assert!((1..=effective).contains(&peak.load(Ordering::Acquire)));

                // map_init: verifies active concurrency and split-leaf init bound on custom pools.
                let map_inits = AtomicUsize::new(0);
                let active = AtomicUsize::new(0);
                let peak = AtomicUsize::new(0);
                let mapped_sum: usize = (0..64usize)
                    .par_bridge()
                    .with_max_concurrency(limit)
                    .map_init(
                        || {
                            map_inits.fetch_add(1, Ordering::Relaxed);
                            1usize
                        },
                        |bias, i| {
                            let _guard = ActiveGuard::enter(&active, &peak, effective);
                            i + *bias
                        },
                    )
                    .sum();
                assert_eq!(mapped_sum, (0..64usize).map(|i| i + 1).sum::<usize>());
                assert_eq!(active.load(Ordering::Acquire), 0);
                assert!((1..=max_leaves).contains(&map_inits.load(Ordering::Relaxed)));
                assert!((1..=effective).contains(&peak.load(Ordering::Acquire)));
            }
        });
    }
}
