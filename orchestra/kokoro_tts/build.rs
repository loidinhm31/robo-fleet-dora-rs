fn main() {
    // espeak-rs-sys builds espeak-ng from source but doesn't emit link
    // directives for its runtime dependencies (sonic and pcaudio).
    // When the system libraries are used (detected via CMake), we must
    // link them explicitly here.
    println!("cargo:rustc-link-lib=sonic");
    println!("cargo:rustc-link-lib=pcaudio");
}
