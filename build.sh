#!/bin/sh
# Build the Rust game core to WebAssembly and drop it next to the web files.
set -e
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/obby.wasm web/obby.wasm
echo "built web/obby.wasm ($(wc -c < web/obby.wasm) bytes) — serve with: python3 -m http.server -d web 8080"
