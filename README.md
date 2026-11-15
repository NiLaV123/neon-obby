# Neon Obby

A Roblox-style obstacle course with a hybrid **raster + ray traced** renderer, running in the browser.

- **Rust → WebAssembly** game core (no wasm-bindgen): level, physics, R6-style character rig, and a BVH rebuilt every frame.
- **WebGPU** renderer:
  1. Raster pass → G-buffer (albedo with procedural studs, normal + box id, reversed-Z depth)
  2. Trace pass (compute) → 1 spp soft sun shadow + AO rays through the BVH
  3. **ReSTIR PT Enhanced** ([Lin, Kettunen, Wyman 2026](https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/)) at full render resolution, 4 compute passes:
     - *initial + temporal*: 1-spp path tree (up to 3 bounces, sun NEE at every vertex incl. x1 = unified direct + indirect light, Russian roulette in the initial source pdf only), RIS to one path, then temporal reuse with pairwise MIS against last frame's G-buffer and a duplication-map-driven confidence cap;
     - *paired spatial pre-pass*: 3 Gaussian-shuffled, self-inverting pairing textures (randomly flipped / transposed / offset every frame); each pixel shifts its path to each partner once;
     - *paired spatial resampling*: confidence-weighted pairwise MIS reusing those shifts, vector-valued weights for shading (colour-noise reduction);
     - *duplication map*: 17×17 seed-match count for the next frame's adaptive cCap.
     Shifts are the hybrid shift: random replay with stored seed + forced lobes up to the reconnection vertex chosen by the dual ray-footprint criterion (c = 0.02) + roughness test, with invertibility checks; NEE light vertices are forced reconnections.
     Its output goes through an **SVGF-style denoiser** (2 + 5 compute passes): albedo demodulation, reprojected temporal accumulation of illumination and luminance moments (variance), a neighbourhood clamp against ghosting from moving objects, then a 5-level edge-aware à-trous wavelet filter (normal, surface-plane and variance-scaled luminance edge-stops), re-modulated by albedo in the shade pass.
     In PT mode the internal resolution is capped to ~960×540 (software BVH rays are expensive without RT cores).
     Not implemented from the paper: dual motion vectors, light tiles (one sun, so unnecessary), stream compaction of replays.  4. Shade pass (compute) → temporal accumulation with variance clamping, GI upsample, traced mirror/metal reflections and glass transmission
  5. Post → ACES tonemap

Needs a WebGPU browser (Chrome/Edge, Safari 26+).

## Controls
`WASD` move · `Space` jump · mouse / right-drag look · wheel zoom · `R` toggle ray tracing · `P` toggle ReSTIR PT Enhanced · `G` render scale · `K` respawn · `T` restart · `M` mute · `H` hide help

## Sound
Eleven original Web Audio effects cover starting, jumping, landing, footsteps, bounce pads, checkpoints, death, respawning, restarting, UI changes, and winning. Layered tones and filtered noise are synthesized locally; no sound files or external services are needed. Footsteps vary in pitch and alternate subtly between left and right, while landing volume follows time spent airborne.

Use the sound button or `M` to mute, and the volume slider to adjust the mix. Both settings are saved in your browser. Audio starts after interaction and pauses when the tab is hidden.

Effect definitions live in `web/audio.js`: add a named array of tone/noise layers to `SOUND_LIBRARY`, then call `sounds.play(name)`. Each layer supports a delay (`at`), duration, gain, and either oscillator frequency/type/pitch sweep or a noise filter cutoff. Melodies use the audio clock to stay in time during slow frames.

## Build
```sh
rustup target add wasm32-unknown-unknown
./build.sh
python3 -m http.server -d web 8080
```
Pushing to `main` builds and deploys to GitHub Pages via `.github/workflows/pages.yml`.

## Checks
```sh
cargo test --locked
node --test tools/audio.test.mjs
./build.sh
node tools/sim.mjs web/obby.wasm
```
The run timer starts when PLAY is clicked. The checks cover this lifecycle in Rust and through the WebAssembly API.
