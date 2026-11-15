# Neon Obby

A Roblox-style obstacle course with a hybrid **raster + ray traced** renderer, running in the browser.

- **Rust → WebAssembly** game core (no wasm-bindgen): level, physics, R6-style character rig, and a BVH rebuilt every frame.
- **WebGPU** renderer:
  1. Raster pass → G-buffer (albedo with procedural studs, normal + box id, reversed-Z depth)
  2. Trace pass (compute) → 1 spp soft sun shadow + AO rays through the BVH
  3. Shade pass (compute) → temporal accumulation with variance clamping, traced mirror/metal reflections and glass transmission
  4. Post → ACES tonemap

Needs a WebGPU browser (Chrome/Edge, Safari 26+).

## Controls
`WASD` move · `Space` jump · mouse / right-drag look · wheel zoom · `R` toggle ray tracing · `G` render scale · `K` respawn · `T` restart · `H` hide help

## Build
```sh
rustup target add wasm32-unknown-unknown
./build.sh
python3 -m http.server -d web 8080
```
Pushing to `main` builds and deploys to GitHub Pages via `.github/workflows/pages.yml`.
