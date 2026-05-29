# Single-turn
cargo run --release -p bitnet_rs --bin bitnet-cli -- --prompt "Hello"

# Interactive
cargo run -p bitnet_rs --bin bitnet-cli --no-default-features

# Greedy decoding (most deterministic, best for debugging)
cargo run -p bitnet_rs --bin bitnet-cli --no-default-features -- --prompt "Hello" --temperature 0
