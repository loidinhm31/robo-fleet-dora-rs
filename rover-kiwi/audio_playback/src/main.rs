mod arbiter;
mod buffers;
mod device;
#[path = "playback-event.rs"]
mod playback_event;
#[path = "playback-result.rs"]
mod playback_result;
mod protocol;
mod resampler;
mod runtime;
mod state;
#[path = "tts-arbiter.rs"]
mod tts_arbiter;
mod tts_result;

#[cfg(test)]
#[path = "arbiter-tests.rs"]
mod arbiter_tests;
#[cfg(test)]
#[path = "buffers-tests.rs"]
mod buffers_tests;

use eyre::Result;
use robo_rover_lib::init_tracing;

fn main() -> Result<()> {
    let _guard = init_tracing();
    runtime::run()
}
