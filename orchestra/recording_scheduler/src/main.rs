use eyre::Result;
use robo_rover_lib::init_tracing;

fn main() -> Result<()> {
    let _guard = init_tracing();
    recording_scheduler::node_loop::run()
}
