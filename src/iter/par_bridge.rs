#[cfg(not(feature = "web_spin_lock"))]
use std::sync::Mutex;

#[cfg(feature = "web_spin_lock")]
use wasm_sync::Mutex;

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::iter::ParallelIterator;
use crate::iter::plumbing::{Folder, UnindexedConsumer, UnindexedProducer, bridge_unindexed};
use crate::{current_num_threads, current_thread_index};

/// Conversion trait to convert an `Iterator` to a `ParallelIterator`.
///
/// This creates a "bridge" from a sequential iterator to a parallel one, by distributing its items
/// across the Rayon thread pool. This has the advantage of being able to parallelize just about
/// anything, but the resulting `ParallelIterator` can be less efficient than if you started with
/// `par_iter` instead. However, it can still be useful for iterators that are difficult to
/// parallelize by other means, like channels or file or network I/O.
///
/// Iterator items are pulled by `next()` one at a time, synchronized from each thread that is
/// ready for work, so this may become a bottleneck if the serial iterator can't keep up with the
/// parallel demand. The items are not buffered by `IterBridge`, so it's fine to use this with
/// large or even unbounded iterators.
///
/// The resulting iterator is not guaranteed to keep the order of the original iterator.
///
/// # Examples
///
/// To use this trait, take an existing `Iterator` and call `par_bridge` on it. After that, you can
/// use any of the `ParallelIterator` methods:
///
/// ```
/// use rayon::iter::ParallelBridge;
/// use rayon::prelude::ParallelIterator;
/// use std::sync::mpsc::channel;
///
/// let rx = {
///     let (tx, rx) = channel();
///
///     tx.send("one!");
///     tx.send("two!");
///     tx.send("three!");
///
///     rx
/// };
///
/// let mut output: Vec<&'static str> = rx.into_iter().par_bridge().collect();
/// output.sort_unstable();
///
/// assert_eq!(&*output, &["one!", "three!", "two!"]);
/// ```
pub trait ParallelBridge: Sized {
    /// Creates a bridge from this type to a `ParallelIterator`.
    ///
    /// Use [`IterBridge::with_max_concurrency`] to limit the number of consumers.
    fn par_bridge(self) -> IterBridge<Self>;
}

impl<T> ParallelBridge for T
where
    T: Iterator<Item: Send> + Send,
{
    fn par_bridge(self) -> IterBridge<Self> {
        IterBridge {
            iter: self,
            max_concurrency: None,
        }
    }
}

/// `IterBridge` is a parallel iterator that wraps a sequential iterator.
///
/// This type is created when using the [`par_bridge`](ParallelBridge::par_bridge) method on
/// [`ParallelBridge`]. Use [`IterBridge::with_max_concurrency`] to limit how many consumers pull
/// and process items concurrently. See the [`ParallelBridge`] documentation for details.
#[derive(Debug, Clone)]
pub struct IterBridge<Iter> {
    iter: Iter,
    max_concurrency: Option<NonZeroUsize>,
}

impl<Iter> IterBridge<Iter> {
    /// Limits the maximum number of concurrent active consumers of this bridge.
    ///
    /// At most `min(max_concurrency, current_num_threads())` consumers will pull and process items
    /// concurrently in the executing Rayon thread pool ([`current_num_threads`]). Fewer consumers
    /// may be used depending on workload size and scheduling, and thread affinity is not
    /// guaranteed. Items are still pulled dynamically from the underlying iterator across active
    /// consumers.
    ///
    /// Note the distinction between *active consumers* (which pull and process items) and
    /// *internal split leaves* (which determine how many times adaptors such as [`fold`],
    /// [`for_each_init`], [`map_init`], or [`for_each_with`] may initialize or clone per-leaf
    /// state):
    ///
    /// - When `max_concurrency == 1` or the executing pool has a single thread, splitting is
    ///   disabled: exactly `1` consumer runs sequentially on the calling thread without internal
    ///   synchronization (`max_concurrency == 1` does not initialize the global thread pool).
    /// - When `1 < max_concurrency < current_num_threads()`, at most `max_concurrency` split
    ///   leaves are created (`max_concurrency - 1` splits), avoiding unnecessary worker wakeups in
    ///   larger pools.
    /// - When `max_concurrency >= current_num_threads()` in a multi-thread pool, splitting matches
    ///   unrestricted `par_bridge()` (`current_num_threads()` splits): Rayon may create up to
    ///   `current_num_threads() + 1` internal split leaves (initializing/cloning consumer state
    ///   for each leaf), while at most `current_num_threads()` consumers can actively pull and
    ///   process items (any extra leaf exits without pulling an item).
    ///
    /// This bounds only the consumers created by this bridge, not the total number of worker
    /// threads used by downstream or nested parallel operations. It does not reserve workers for
    /// other operations or guarantee scheduling fairness. By default, no explicit limit is applied.
    /// Calling this method again replaces the previous limit.
    ///
    /// [`fold`]: ParallelIterator::fold
    /// [`for_each_init`]: ParallelIterator::for_each_init
    /// [`map_init`]: ParallelIterator::map_init
    /// [`for_each_with`]: ParallelIterator::for_each_with
    ///
    /// # Panics
    ///
    /// Panics if `max_concurrency` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rayon::prelude::*;
    ///
    /// let sum: u32 = (0..100).par_bridge().with_max_concurrency(4).sum();
    /// assert_eq!(sum, 4950);
    ///
    /// // Calling `with_max_concurrency` again replaces the previous limit.
    /// let single_leaf_count = (0..100)
    ///     .par_bridge()
    ///     .with_max_concurrency(4)
    ///     .with_max_concurrency(1)
    ///     .fold(|| (), |(), _| ())
    ///     .count();
    /// assert_eq!(single_leaf_count, 1);
    /// ```
    #[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
    #[track_caller]
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency =
            Some(NonZeroUsize::new(max_concurrency).expect("max_concurrency must not be zero"));
        self
    }
}

impl<Iter> ParallelIterator for IterBridge<Iter>
where
    Iter: Iterator<Item: Send> + Send,
{
    type Item = Iter::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let num_threads = if self.max_concurrency == Some(NonZeroUsize::MIN) {
            1
        } else {
            current_num_threads()
        };
        let split_count = split_budget(self.max_concurrency, num_threads);
        if split_count == 0 {
            return drive_sequential(self.iter, consumer);
        }

        let threads_started: Vec<_> = (0..num_threads).map(|_| AtomicBool::new(false)).collect();

        bridge_unindexed(
            &IterParallelProducer {
                split_count: AtomicUsize::new(split_count),
                iter: Mutex::new(self.iter.fuse()),
                threads_started: &threads_started,
            },
            consumer,
        )
    }
}

#[inline]
fn drive_sequential<Iter, C>(mut iter: Iter, consumer: C) -> C::Result
where
    Iter: Iterator,
    C: UnindexedConsumer<Iter::Item>,
{
    if consumer.full() {
        return consumer.into_folder().complete();
    }
    let mut folder = consumer.into_folder();
    if !folder.full() {
        for item in &mut iter {
            folder = folder.consume(item);
            if folder.full() {
                break;
            }
        }
    }
    folder.complete()
}

// Each successful split adds one leaf consumer to the initial consumer.
//
// When `max_concurrency < num_threads`, we allow `max_concurrency - 1` splits so at most
// `max_concurrency` leaves are created and extra pool workers are not woken unnecessarily.
// When `max_concurrency >= num_threads` in a multi-thread pool, we keep upstream's `num_threads`
// split budget: `threads_started` still bounds active consumers to at most `num_threads`, while
// the extra split absorbs a worker self-popping its own leaf during nested Rayon work.
fn split_budget(max_concurrency: Option<NonZeroUsize>, num_threads: usize) -> usize {
    match max_concurrency {
        Some(max) if max.get() < num_threads => max.get() - 1,
        Some(_) if num_threads <= 1 => 0,
        _ => num_threads,
    }
}

struct IterParallelProducer<'a, Iter> {
    split_count: AtomicUsize,
    iter: Mutex<std::iter::Fuse<Iter>>,
    threads_started: &'a [AtomicBool],
}

impl<Iter: Iterator + Send> UnindexedProducer for &IterParallelProducer<'_, Iter> {
    type Item = Iter::Item;

    fn split(self) -> (Self, Option<Self>) {
        // Check if the iterator is exhausted
        #[allow(deprecated, reason = "TODO (MSRV 1.95): use try_update")]
        let update = self
            .split_count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| c.checked_sub(1));
        (self, update.is_ok().then_some(self))
    }

    fn fold_with<F>(self, mut folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        // Guard against work-stealing-induced recursion, in case `Iter::next()`
        // calls rayon internally, so we don't deadlock our mutex. We might also
        // be recursing via `folder` methods, which doesn't present a mutex hazard,
        // but it's lower overhead for us to just check this once, rather than
        // updating additional shared state on every mutex lock/unlock.
        // (If this isn't a rayon thread, then there's no work-stealing anyway...)
        if let Some(i) = current_thread_index() {
            // Note: If the number of threads in the pool ever grows dynamically, then
            // we'll end up sharing flags and may falsely detect recursion -- that's
            // still fine for overall correctness, just not optimal for parallelism.
            let thread_started = &self.threads_started[i % self.threads_started.len()];
            if thread_started.swap(true, Ordering::Relaxed) {
                // We can't make progress with a nested mutex, so just return and let
                // the outermost loop continue with the rest of the iterator items.
                return folder;
            }
        }

        loop {
            if let Ok(mut iter) = self.iter.lock() {
                if let Some(it) = iter.next() {
                    drop(iter);
                    folder = folder.consume(it);
                    if folder.full() {
                        return folder;
                    }
                } else {
                    return folder;
                }
            } else {
                // any panics from other threads will have been caught by the pool,
                // and will be re-thrown when joined - just exit
                return folder;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_split_budget() {
        assert_eq!(split_budget(None, 0), 0);
        for limit in [1, 2, usize::MAX] {
            assert_eq!(split_budget(NonZeroUsize::new(limit), 0), 0);
        }

        for num_threads in [1, 2, 3, 4, 8] {
            assert_eq!(split_budget(None, num_threads), num_threads);
            for limit in [1, 2, 3, 4, 8, 16, usize::MAX] {
                let expected_splits = if limit < num_threads {
                    limit - 1
                } else if num_threads <= 1 {
                    0
                } else {
                    num_threads
                };
                let budget = split_budget(NonZeroUsize::new(limit), num_threads);
                assert_eq!(budget, expected_splits);

                let producer = IterParallelProducer {
                    split_count: AtomicUsize::new(budget),
                    iter: Mutex::new(std::iter::empty::<()>().fuse()),
                    threads_started: &[],
                };
                for _ in 0..expected_splits {
                    assert!((&producer).split().1.is_some());
                }
                assert!((&producer).split().1.is_none());
                assert!((&producer).split().1.is_none());
            }
        }
    }
}
