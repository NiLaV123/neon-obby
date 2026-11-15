# Neon Obby

A Roblox-style obstacle course running in the browser, with three renderers: a production-style **raster lighting** pipeline (default), **hybrid ray tracing**, and **ReSTIR PT Enhanced** path tracing.

**[Playtest the live game](https://nilav123.github.io/neon-obby/)** in a WebGPU-capable browser.

- **Rust → WebAssembly** game core (no wasm-bindgen): level, physics, R6-style character rig, and a BVH rebuilt every frame.
- **Raster lighting** (default; how a shipped game lights a scene — no per-pixel ray tracing):
  - **Hand-placed lights**: 40 point lights authored in the Rust level (neon letters, hazard glows that ride on spinners and sliders, checkpoint pads, uplights, under-glass fills, the win arch), with **tiled light culling** (8×8 tiles, world-space tile bounds reduced in shared memory).
  - **Cascaded shadow maps**: 3 × 2048² sun cascades fitted to frustum slices and texel-snapped, with **PCSS** contact-hardening soft shadows (blocker search + rotated 16-tap Poisson PCF).
  - **DDGI irradiance probes**: a 24×12×32 grid (3-stud spacing) that scrolls with the player using toroidal addressing; half of the probes trace 48 rays per frame through the BVH (sun, local lights, emission and last frame's probes → infinite bounces), blended into 8×8 octahedral irradiance and 16×16 depth-moment maps with hysteresis; Chebyshev visibility and backface weights stop leaks, probes stuck inside geometry are switched off.
  - **AO**: half-resolution horizon-based ambient occlusion with a depth-aware upsampling blur, applied with multi-bounce AO and specular occlusion.
  - **Shading**: GGX specular per material (plastic, smooth, polished, metal), split-sum environment BRDF, **screen-space reflections** from last frame's image over a probe/sky fallback (sky visibility from the probe depth maps).
  - **Forward transparency** for glass and blinking ghost blocks (Fresnel, sun glint, probe-lit body).
  - **Procedural animated lava** (domain-warped crust with glowing, pulsing cracks), emissive treatment of neon, exponential **height fog** with sun in-scattering, 6-level **dual-filter bloom**, hand-tuned exposure and grade.
- **WebGPU** ray-traced modes (`R`):
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
     Not implemented from the paper: dual motion vectors, light tiles (one sun, so unnecessary), stream compaction of replays.
  4. Shade pass (compute) → temporal accumulation with variance clamping, GI upsample, traced mirror/metal reflections and glass transmission
  5. Post → ACES tonemap

## Asynchronous reprojection (timewarp)
Press `F` to separate the display rate from the render rate. A second WebGPU device does only one thing: every display refresh it warps the newest finished frame to the current camera. It unprojects each pixel with that frame's depth and inverse view-projection, reprojects it with the latest camera, and solves for the inverse with a 6-step fixed-point iteration. The scene device renders at its own pace, with a 1.2× field-of-view overscan so fast turns don't expose edges. Each finished colour and depth frame is handed to the presenting device through a mapped staging buffer.

Because the presenting device's queue never waits behind a 60–100 ms path-traced frame, camera motion stays at display rate (60 fps while ReSTIR PT renders at about 15). `V` caps the scene rate so you can see the effect. **Freeze frame (`X`)** stops scene rendering and keeps warping the last frame, so you can walk the camera around a single still image and see how far reprojection holds up.

**Infill.** Each display pixel first tries a cheap fixed-point solve. Where that fails (steep parallax or a large camera move), it marches the pixel's epipolar line through the source frame from near to far. The first point where the view ray passes behind the stored depth is the visible surface, found with a bisection. If the stored depth jumps at that crossing, the ray went behind a foreground object: that's a disocclusion, and it's filled with a depth-weighted blur of the background next to it instead of stretched foreground. Areas beyond the overscan border get a blur that widens with distance. `J` tints filled holes magenta and filled edges cyan; `I` compares against the plain warp.

Known limitations: there are no motion vectors, so moving objects such as the player judder at the scene rate, and disocclusions stretch. If a second device isn't available, the warp runs on the scene device; that path is correct but gives no async benefit.

Needs a WebGPU browser (Chrome/Edge, Safari 26+).

## Controls
`WASD` move · `Space` jump · mouse / right-drag look · wheel zoom · `R` raster lighting ↔ ray tracing · `P` (in ray tracing) ReSTIR PT Enhanced ↔ hybrid · `G` render scale · `F` asynchronous reprojection · `V` cycle scene fps cap (∞/30/15/8) · `X` freeze frame · `I` infill on/off · `J` tint filled holes · `K` respawn · `T` restart · `M` mute · `H` hide help

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
