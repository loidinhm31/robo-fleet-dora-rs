//! Process-level cumulative counters for web_bridge audio delivery.
//!
//! These counters are intentionally decoupled from the per-client
//! [`ClientState`] counters in [`crate::main`] so that the lifetime totals
//! reported at shutdown (and via future metric endpoints) are **not** lost
//! when a client disconnects.
//!
//! The previous behaviour summed the per-client counters over the still-
//! connected clients only, because [`ClientState`] entries are dropped from
//! `SharedState::video_clients` on every disconnect path (graceful
//! disconnect, idle sweep, session-expiry sweep). That made shutdown totals
//! undercount the work the bridge actually performed.
//!
//! # Concurrency
//!
//! Every counter is an [`AtomicU64`] with [`Ordering::Relaxed`] semantics.
//! The hot path performs one atomic increment per frame (sent or dropped),
//! so this must stay lock-free to avoid adding contention to the audio
//! dispatch loop. The shutdown handler reads the counters with
//! [`Ordering::Relaxed`] as well; the totals are eventually consistent and
//! only used for at-shutdown logging.
//!
//! # Per-client vs cumulative
//!
//! [`ClientState::audio_frames_sent`] and [`ClientState::audio_frames_dropped`]
//! are still maintained for live per-client debugging (e.g. an operator
//! investigating a single bad client). They are no longer the source of
//! truth for shutdown totals.

use std::sync::atomic::{AtomicU64, Ordering};

/// Process-level cumulative audio delivery counters.
///
/// All methods are `&self` and use [`Ordering::Relaxed`], so the type is
/// safe to share via `Arc<AudioDeliveryCounters>` across the audio dispatch
/// loop, the disconnect handlers, the shutdown handler, and any future
/// metric endpoints.
#[derive(Debug, Default)]
pub struct AudioDeliveryCounters {
    /// Total audio frames successfully emitted to a Socket.IO client since
    /// the bridge started. Increments exactly once per successful emit.
    pub frames_sent: AtomicU64,
    /// Total audio frames that were dropped before reaching a client since
    /// the bridge started. Covers every drop path: missing socket, emit
    /// error, sequence-drops recorded at the bridge boundary.
    pub frames_dropped: AtomicU64,
    /// Total client disconnects observed since the bridge started. Counts
    /// every disconnect regardless of cause (graceful, idle sweep, session
    /// expiry sweep, transport error). This makes the cumulative emit
    /// totals interpretable — operators can reason about "per-client
    /// average" without re-deriving disconnect counts.
    pub client_disconnects: AtomicU64,
}

impl AudioDeliveryCounters {
    /// Construct a fresh counter set. All values start at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one successful audio emit (sent to a single client).
    ///
    /// Hot path: called once per successful `socket.emit("audio_frame", …)`.
    pub fn record_emit_success(&self) {
        self.frames_sent.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one audio drop (missing socket, emit error, or sequence drop
    /// recorded at the bridge boundary).
    ///
    /// Hot path: called once per drop event. Distinct from
    /// [`ClientState::mark_audio_dropped`], which is the per-client
    /// counter; both must be incremented together so the per-client and
    /// cumulative numbers stay in lockstep.
    pub fn record_emit_drop(&self) {
        self.frames_dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Add `n` sequence-drop events observed at the bridge boundary in one
    /// observation call. Sequence drops come in batches (one missing
    /// `frame_id` per gap), so the hot path takes a count rather than
    /// firing the atomic once per missing frame.
    pub fn record_sequence_drops(&self, n: u64) {
        self.frames_dropped.fetch_add(n, Ordering::Relaxed);
    }

    /// Record one client disconnect. Call from every disconnect path:
    /// graceful `socket.on_disconnect`, idle sweep, and session-expiry
    /// sweep. The counter exists so cumulative emit totals can be
    /// divided by clients-served when reasoning about per-client load.
    pub fn record_client_disconnect(&self) {
        self.client_disconnects.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot of the lifetime cumulative totals. Used at shutdown and by
    /// future metric endpoints. Read with [`Ordering::Relaxed`]; eventually
    /// consistent.
    pub fn cumulative_totals(&self) -> CumulativeTotals {
        CumulativeTotals {
            frames_sent: self.frames_sent.load(Ordering::Relaxed),
            frames_dropped: self.frames_dropped.load(Ordering::Relaxed),
            client_disconnects: self.client_disconnects.load(Ordering::Relaxed),
        }
    }
}

/// Plain-data snapshot returned by [`AudioDeliveryCounters::cumulative_totals`].
///
/// Implements `Add` so multiple snapshots can be summed (e.g. aggregating
/// across processes or test fixtures).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CumulativeTotals {
    pub frames_sent: u64,
    pub frames_dropped: u64,
    pub client_disconnects: u64,
}

impl std::ops::Add for CumulativeTotals {
    type Output = CumulativeTotals;
    fn add(self, rhs: CumulativeTotals) -> CumulativeTotals {
        CumulativeTotals {
            frames_sent: self.frames_sent.saturating_add(rhs.frames_sent),
            frames_dropped: self.frames_dropped.saturating_add(rhs.frames_dropped),
            client_disconnects: self.client_disconnects.saturating_add(rhs.client_disconnects),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn new_counters_start_at_zero() {
        let counters = AudioDeliveryCounters::new();
        let snapshot = counters.cumulative_totals();
        assert_eq!(snapshot.frames_sent, 0);
        assert_eq!(snapshot.frames_dropped, 0);
        assert_eq!(snapshot.client_disconnects, 0);
    }

    #[test]
    fn record_emit_success_increments_frames_sent() {
        let counters = AudioDeliveryCounters::new();
        counters.record_emit_success();
        counters.record_emit_success();
        counters.record_emit_success();
        assert_eq!(counters.cumulative_totals().frames_sent, 3);
        assert_eq!(counters.cumulative_totals().frames_dropped, 0);
    }

    #[test]
    fn record_emit_drop_increments_frames_dropped() {
        let counters = AudioDeliveryCounters::new();
        counters.record_emit_drop();
        counters.record_emit_drop();
        assert_eq!(counters.cumulative_totals().frames_dropped, 2);
        assert_eq!(counters.cumulative_totals().frames_sent, 0);
    }

    #[test]
    fn record_sequence_drops_adds_in_batch() {
        let counters = AudioDeliveryCounters::new();
        counters.record_sequence_drops(5);
        counters.record_sequence_drops(2);
        assert_eq!(counters.cumulative_totals().frames_dropped, 7);
    }

    #[test]
    fn record_client_disconnect_increments_disconnect_counter() {
        let counters = AudioDeliveryCounters::new();
        counters.record_client_disconnect();
        counters.record_client_disconnect();
        assert_eq!(counters.cumulative_totals().client_disconnects, 2);
    }

    /// The whole point of the backlog fix: cumulative totals must retain
    /// their values even when the *caller* (the `ClientState` that owned
    /// the per-client counter) has been dropped. We simulate that here by
    /// recording events, "dropping" the caller (just letting it go out of
    /// scope), and asserting the cumulative counters still report the
    /// events.
    ///
    /// In production this matches the real situation: a client disconnects,
    /// the `ClientState` is removed from `video_clients` (dropping its
    /// per-client counters), but the `Arc<AudioDeliveryCounters>` lives on
    /// in `SharedState` and must still report the work the bridge did for
    /// that client during its lifetime.
    #[test]
    fn cumulative_totals_retain_counts_after_caller_drops() {
        let shared = Arc::new(AudioDeliveryCounters::new());

        // Simulate one connected client emitting 10 frames and dropping 2.
        {
            let client_view = shared.clone();
            for _ in 0..10 {
                client_view.record_emit_success();
            }
            client_view.record_emit_drop();
            client_view.record_emit_drop();
            // Client disconnects — its per-client `ClientState` would be
            // removed from `video_clients` here in production, dropping the
            // per-client counters. We model that with scope exit.
        }

        // Now the "client" is gone. The cumulative counters must still
        // reflect the 10 sends and 2 drops.
        let snapshot = shared.cumulative_totals();
        assert_eq!(
            snapshot.frames_sent, 10,
            "cumulative frames_sent must survive client disconnect"
        );
        assert_eq!(
            snapshot.frames_dropped, 2,
            "cumulative frames_dropped must survive client disconnect"
        );
        assert_eq!(snapshot.client_disconnects, 1);

        // A second client joins, emits, and disconnects.
        {
            let client_view = shared.clone();
            for _ in 0..5 {
                client_view.record_emit_success();
            }
            client_view.record_client_disconnect();
        }

        // Totals now reflect BOTH clients' lifetimes.
        let snapshot = shared.cumulative_totals();
        assert_eq!(snapshot.frames_sent, 15, "second client's sends must accumulate");
        assert_eq!(snapshot.frames_dropped, 2, "drops from first client must persist");
        assert_eq!(snapshot.client_disconnects, 2, "disconnect counter must increment");
    }

    /// Multiple clients operating concurrently must not lose increments.
    /// Uses real OS threads and `Arc<AudioDeliveryCounters>` exactly as
    /// the production code does.
    #[test]
    fn concurrent_clients_preserve_all_increments() {
        let shared = Arc::new(AudioDeliveryCounters::new());
        let clients = 8;
        let frames_per_client = 10_000;
        let drops_per_client = 100;

        let mut handles = Vec::new();
        for _ in 0..clients {
            let counters = shared.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..frames_per_client {
                    counters.record_emit_success();
                }
                for _ in 0..drops_per_client {
                    counters.record_emit_drop();
                }
                counters.record_client_disconnect();
            }));
        }
        for handle in handles {
            handle.join().expect("worker thread must not panic");
        }

        let snapshot = shared.cumulative_totals();
        assert_eq!(snapshot.frames_sent, clients * frames_per_client);
        assert_eq!(snapshot.frames_dropped, clients * drops_per_client);
        assert_eq!(snapshot.client_disconnects, clients as u64);
    }

    #[test]
    fn cumulative_totals_add_combines_snapshots() {
        let a = CumulativeTotals {
            frames_sent: 10,
            frames_dropped: 2,
            client_disconnects: 1,
        };
        let b = CumulativeTotals {
            frames_sent: 5,
            frames_dropped: 1,
            client_disconnects: 1,
        };
        let combined = a + b;
        assert_eq!(combined.frames_sent, 15);
        assert_eq!(combined.frames_dropped, 3);
        assert_eq!(combined.client_disconnects, 2);
    }

    /// Saturated add must not wrap on absurd inputs. A misbehaving caller
    /// passing `u64::MAX` would otherwise corrupt the snapshot.
    #[test]
    fn cumulative_totals_add_saturates() {
        let a = CumulativeTotals {
            frames_sent: u64::MAX,
            frames_dropped: u64::MAX,
            client_disconnects: u64::MAX,
        };
        let b = CumulativeTotals {
            frames_sent: 1,
            frames_dropped: 1,
            client_disconnects: 1,
        };
        let combined = a + b;
        assert_eq!(combined.frames_sent, u64::MAX);
        assert_eq!(combined.frames_dropped, u64::MAX);
        assert_eq!(combined.client_disconnects, u64::MAX);
    }
}

