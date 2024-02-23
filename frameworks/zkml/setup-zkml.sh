git clone https://github.com/ddkang/zkml.git
cd zkml
rustup override set nightly
cargo build --release

mv ./target/release/test_circuit ./target/release/time_circuit ./target/release/verify_circuit ./target/release/verify_wav ../bin/