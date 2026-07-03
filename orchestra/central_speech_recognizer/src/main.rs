use eyre::Result;
use robo_rover_lib::init_tracing;

fn main() -> Result<()> {
    let _guard = init_tracing();
    central_speech_recognizer::runtime::run()
}
