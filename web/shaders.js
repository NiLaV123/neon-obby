// WGSL for the hybrid renderer:
//   1. RASTER  – instanced oriented boxes -> G-buffer (albedo, normal+id, depth)
//   2a. TRACE  – compute: jittered sun-shadow + AO ray per pixel (noisy, 1 spp)
//   2b. PT_*   – compute: ReSTIR PT Enhanced (path tree + temporal reuse, paired spatial
//                pre-pass, paired spatial resampling, duplication map)
//   2c. SHADE  – compute: clamped temporal accumulation and compositing; in the non-PT
//                modes also the direct-light + traced mirror/glass shading
//   3. POST    – ACES tonemap, vignette, damage flash

const COMMON = /* wgsl */ `
struct U {
  viewProj: mat4x4f,
  invViewProj: mat4x4f,
  prevViewProj: mat4x4f,
  prevInvViewProj: mat4x4f,
  camPos: vec4f,   // xyz camera, w time
  sunDir: vec4f,   // xyz to-sun, w frame
  res: vec4f,      // w, h, rt on, box count
  misc: vec4f,     // x flash, y exposure, zw canvas size
  opts: vec4f,     // x ReSTIR PT Enhanced on, y probe stride, z probe phase
  prevCam: vec4f,  // previous frame camera position
  pair: array<vec4f, 3>,  // per-frame transforms of the 3 pairing textures (flags, offset)
  // ---- raster lighting
  sunVP: array<mat4x4f, 3>, // cascaded shadow map view-projections
  cascade: vec4f,           // xyz far split (view distance) per cascade, w sun penumbra spread
  cascadeTexel: vec4f,      // world size of one shadow texel per cascade
  cascadeRange: vec4f,      // world depth range per cascade (light space z)
  probeGrid: vec4f,         // xyz first probe cell of the scrolling DDGI volume, w spacing
  probeInfo: vec4f,         // x hysteresis, y frame (ray rotation), z light count, w bloom strength
  rasterInfo: vec4f,        // x raster lighting mode on, y SSR on, z AO strength, w exposure
};
struct Box { c: vec3f, mat: f32, h: vec3f, emis: f32, q: vec4f, col: vec3f, refl: f32 };

const M_PLASTIC = 0u; const M_SMOOTH = 1u; const M_NEON = 2u; const M_GLASS = 3u;
const M_METAL = 4u; const M_POLISHED = 5u; const M_GHOST = 6u; const M_HEAD = 7u; const M_CHECK = 8u;

fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
fn qinv(q: vec4f, v: vec3f) -> vec3f { return qrot(vec4f(-q.xyz, q.w), v); }
fn matOf(b: Box) -> u32 { return u32(b.mat + 0.5); }

// Surface colour incl. Roblox studs, bevelled edges, the face decal and spawn pads.
fn albedoAt(b: Box, lp: vec3f, ln: vec3f) -> vec3f {
  let m = matOf(b);
  var col = b.col;
  if (m == M_NEON || m == M_GLASS || m == M_GHOST) { return col; }
  let d = b.h - abs(lp) + abs(ln) * 1e4;
  let e = min(d.x, min(d.y, d.z));
  col *= mix(0.7, 1.0, smoothstep(0.0, 0.09, e));
  if (m == M_PLASTIC && ln.y > 0.5) {
    let f = fract(lp.xz + b.h.xz) - 0.5;
    let r = length(f);
    col *= 1.0 - 0.25 * smoothstep(0.26, 0.30, r) * (1.0 - smoothstep(0.33, 0.38, r));
    col *= 1.0 + 0.08 * (1.0 - smoothstep(0.0, 0.28, r));
  }
  if (m == M_HEAD && ln.z > 0.5) {
    let p = lp.xy / b.h.xy;
    let eye = min(length(p - vec2f(-0.32, 0.22)), length(p - vec2f(0.32, 0.22)));
    var ink = 1.0 - smoothstep(0.10, 0.13, eye);
    let sp = p - vec2f(0.0, 0.12);
    if (sp.y < -0.2) { ink = max(ink, 1.0 - smoothstep(0.04, 0.07, abs(length(sp) - 0.5))); }
    col = mix(col, vec3f(0.04), ink);
  }
  if (m == M_CHECK && ln.y > 0.5) {
    let r = length(lp.xz / b.h.xz);
    let ring = 1.0 - smoothstep(0.04, 0.07, abs(r - 0.62));
    let dotc = 1.0 - smoothstep(0.2, 0.25, r);
    col = mix(col, vec3f(1.0), max(ring, dotc));
  }
  return col;
}
`;

export const RASTER_WGSL = COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> boxes: array<Box>;

struct VO {
  @builtin(position) pos: vec4f,
  @location(0) lp: vec3f,
  @location(1) @interpolate(flat) ln: vec3f,
  @location(2) @interpolate(flat) id: u32,
};

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let b = boxes[ii];
  let f = vi / 6u;
  let k = vi % 6u;
  let axis = f / 2u;
  let s = select(-1.0, 1.0, (f & 1u) == 0u);
  var n = vec3f(0.0);
  n[axis] = s;
  var tu = vec3f(0.0);
  tu[(axis + 1u) % 3u] = 1.0;
  var tv = vec3f(0.0);
  tv[(axis + 2u) % 3u] = 1.0;
  var quad = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                             vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c2 = quad[k];
  let lp = (n + tu * c2.x + tv * c2.y) * b.h;
  let wp = qrot(b.q, lp) + b.c;
  var o: VO;
  o.pos = u.viewProj * vec4f(wp, 1.0);
  // raster lighting draws see-through parts in a forward pass after the deferred lighting
  let m = matOf(b);
  if (u.rasterInfo.x > 0.5 && (m == M_GLASS || m == M_GHOST)) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); }
  o.lp = lp;
  o.ln = n;
  o.id = ii;
  return o;
}

struct FO { @location(0) albedo: vec4f, @location(1) normal: vec4f };

@fragment fn fs(i: VO) -> FO {
  let b = boxes[i.id];
  var o: FO;
  o.albedo = vec4f(albedoAt(b, i.lp, i.ln), 1.0);
  o.normal = vec4f(normalize(qrot(b.q, i.ln)), f32(i.id));
  return o;
}
`;

// Shared by the TRACE and SHADE compute passes.
// Sun, sky and flat ambient, shared by every lighting path.
const SKY_FN = /* wgsl */ `
const SUN_COL = vec3f(1.0, 0.94, 0.84) * 2.7;
fn sky(rd: vec3f) -> vec3f {
  let zen = vec3f(0.18, 0.42, 0.92);
  let hor = vec3f(0.70, 0.83, 0.97);
  var c = mix(hor, zen, pow(clamp(rd.y, 0.0, 1.0), 0.55));
  c = mix(c, vec3f(0.62, 0.5, 0.45), 1.0 - smoothstep(-0.25, 0.0, rd.y));
  let sd = max(dot(rd, u.sunDir.xyz), 0.0);
  c += vec3f(1.0, 0.9, 0.7) * (pow(sd, 1500.0) * 60.0 + pow(sd, 10.0) * 0.3);
  return c;
}

// sky without the sun disc (for probes and rough reflections)
fn skyDiffuse(rd: vec3f) -> vec3f {
  let zen = vec3f(0.18, 0.42, 0.92);
  let hor = vec3f(0.70, 0.83, 0.97);
  var c = mix(hor, zen, pow(clamp(rd.y, 0.0, 1.0), 0.55));
  c = mix(c, vec3f(0.62, 0.5, 0.45), 1.0 - smoothstep(-0.25, 0.0, rd.y));
  return c + vec3f(1.0, 0.9, 0.7) * pow(max(dot(rd, u.sunDir.xyz), 0.0), 10.0) * 0.3;
}
fn ambient(n: vec3f) -> vec3f {
  return mix(vec3f(0.45, 0.36, 0.30), vec3f(0.50, 0.64, 0.92), n.y * 0.5 + 0.5) * 0.85;
}

`;

const RT_COMMON = COMMON + /* wgsl */ `
struct Node { mn: vec3f, lf: u32, mx: vec3f, cnt: u32 };

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> boxes: array<Box>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> prims: array<u32>;
@group(0) @binding(4) var normalTex: texture_2d<f32>;
@group(0) @binding(5) var depthTex: texture_depth_2d;

const NONE = 0xffffffffu;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd(s: ptr<function, u32>) -> f32 { *s = pcg(*s); return f32(*s >> 8u) / 16777216.0; }

fn safeInv(d: vec3f) -> vec3f { return 1.0 / select(d, vec3f(1e-9), abs(d) < vec3f(1e-9)); }

${SKY_FN}
fn boxHit(b: Box, ro: vec3f, rd: vec3f, tmax: f32) -> f32 {
  let lo = qinv(b.q, ro - b.c);
  let inv = safeInv(qinv(b.q, rd));
  let t0 = (-b.h - lo) * inv;
  let t1 = (b.h - lo) * inv;
  let a = min(t0, t1);
  let c = max(t0, t1);
  let tn = max(max(a.x, a.y), a.z);
  let tf = min(min(c.x, c.y), c.z);
  if (tn > tf || tf < 1e-4) { return -1.0; }
  let t = select(tf, tn, tn > 1e-4);
  if (t >= tmax) { return -1.0; }
  return t;
}

fn aabbHit(mn: vec3f, mx: vec3f, ro: vec3f, inv: vec3f, tmax: f32) -> f32 {
  let t0 = (mn - ro) * inv;
  let t1 = (mx - ro) * inv;
  let a = min(t0, t1);
  let c = max(t0, t1);
  let tn = max(max(a.x, a.y), a.z);
  let tf = min(min(c.x, c.y), c.z);
  if (tf < max(tn, 0.0) || tn > tmax) { return 1e30; }
  return max(tn, 0.0);
}

struct Hit { t: f32, id: u32 };

// BVH traversal. 'shadow' = any-hit, ignores see-through boxes.
fn trace(ro: vec3f, rd: vec3f, tmax: f32, skip: u32, shadow: bool) -> Hit {
  var best = Hit(tmax, NONE);
  let inv = safeInv(rd);
  var stack: array<u32, 32>;
  var sp = 0u;
  var ni = 0u;
  loop {
    let n = nodes[ni];
    var pop = false;
    if (n.cnt > 0u) {
      for (var i = 0u; i < n.cnt; i++) {
        let id = prims[n.lf + i];
        if (id == skip) { continue; }
        let b = boxes[id];
        if (shadow) {
          let m = matOf(b);
          if (m == M_GLASS || m == M_GHOST) { continue; }
        }
        let t = boxHit(b, ro, rd, best.t);
        if (t > 0.0) {
          best = Hit(t, id);
          if (shadow) { return best; }
        }
      }
      pop = true;
    } else {
      let l = n.lf;
      let r = n.lf + 1u;
      let a = nodes[l];
      let c = nodes[r];
      let ta = aabbHit(a.mn, a.mx, ro, inv, best.t);
      let tc = aabbHit(c.mn, c.mx, ro, inv, best.t);
      if (ta > 1e29 && tc > 1e29) {
        pop = true;
      } else if (ta > 1e29) {
        ni = r;
      } else if (tc > 1e29) {
        ni = l;
      } else {
        var near = l;
        var far = r;
        if (tc < ta) { near = r; far = l; }
        if (sp < 32u) { stack[sp] = far; sp += 1u; }
        ni = near;
      }
    }
    if (pop) {
      if (sp == 0u) { break; }
      sp -= 1u;
      ni = stack[sp];
    }
  }
  return best;
}

struct Surf { p: vec3f, n: vec3f, alb: vec3f, id: u32 };

fn surfAt(id: u32, p: vec3f) -> Surf {
  let b = boxes[id];
  let lp = qinv(b.q, p - b.c);
  let r = abs(lp) / b.h;
  var ln = vec3f(0.0);
  if (r.x >= r.y && r.x >= r.z) { ln.x = sign(lp.x); }
  else if (r.y >= r.z) { ln.y = sign(lp.y); }
  else { ln.z = sign(lp.z); }
  return Surf(p, qrot(b.q, ln), albedoAt(b, lp, ln), id);
}

// amb = indirect irradiance (flat ambient*AO, or the ReSTIR PT result)
fn shade(s: Surf, shadow: f32, amb: vec3f) -> vec3f {
  let b = boxes[s.id];
  let ndl = max(dot(s.n, u.sunDir.xyz), 0.0);
  return s.alb * (SUN_COL * ndl * shadow + amb + b.emis);
}

fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

fn fresnel(c: f32, f0: f32) -> f32 { return f0 + (1.0 - f0) * pow(1.0 - clamp(c, 0.0, 1.0), 5.0); }

fn ortho(n: vec3f) -> vec3f {
  let a = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.x) > 0.9);
  return normalize(cross(n, a));
}

fn hardShadow(p: vec3f, n: vec3f) -> f32 {
  if (dot(n, u.sunDir.xyz) <= 0.0) { return 0.0; }
  return select(1.0, 0.0, trace(p + n * 0.02, u.sunDir.xyz, 1e4, NONE, true).id != NONE);
}

fn ndcOf(px: vec2i) -> vec2f {
  let uv = (vec2f(px) + 0.5) / u.res.xy;
  return vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}
fn worldPos(px: vec2i, d: f32) -> vec3f {
  let wh = u.invViewProj * vec4f(ndcOf(px), d, 1.0);
  return wh.xyz / wh.w;
}
`;

// Pass 2a: one jittered sun-shadow ray + one AO ray per pixel -> raw (noisy) visibility.
export const TRACE_WGSL = RT_COMMON + /* wgsl */ `
@group(0) @binding(6) var rawOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec2u(u.res.xy);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let px = vec2i(gid.xy);
  let d = textureLoad(depthTex, px, 0);
  if (d <= 0.0) { textureStore(rawOut, px, vec4f(1.0, 1.0, 0.0, 0.0)); return; }
  let wp = worldPos(px, d);
  let n = normalize(textureLoad(normalTex, px, 0).xyz);
  let o = wp + n * 0.02;
  var seed = pcg(gid.x + gid.y * 9187u) ^ pcg(u32(u.sunDir.w) * 7919u + 13u);

  var shadow = 0.0;
  if (dot(n, u.sunDir.xyz) > 0.0) {
    let t1 = ortho(u.sunDir.xyz);
    let t2 = cross(u.sunDir.xyz, t1);
    let r1 = sqrt(rnd(&seed)) * 0.03;
    let a1 = rnd(&seed) * 6.2831853;
    let sd = normalize(u.sunDir.xyz + t1 * (r1 * cos(a1)) + t2 * (r1 * sin(a1)));
    shadow = select(1.0, 0.0, trace(o, sd, 1e4, NONE, true).id != NONE);
  }

  let b1 = ortho(n);
  let b2 = cross(n, b1);
  let r2 = rnd(&seed);
  let a2 = rnd(&seed) * 6.2831853;
  let sr = sqrt(r2);
  let hd = normalize(b1 * (sr * cos(a2)) + b2 * (sr * sin(a2)) + n * sqrt(max(1.0 - r2, 0.0)));
  let ah = trace(o, hd, 3.5, NONE, true);
  let ao = select(1.0, smoothstep(0.0, 3.5, ah.t), ah.id != NONE);

  textureStore(rawOut, px, vec4f(shadow, ao, 0.0, 1.0));
}
`;

// =======================================================================================
// ReSTIR PT Enhanced (Lin, Kettunen, Wyman 2026) on top of ReSTIR PT (Lin et al. 2022),
// written for WebGPU compute + the software BVH above. Full render resolution.
//
//  * Lobe-indexed paths: every vertex picks one lobe (diffuse, delta reflect, delta transmit).
//  * Hybrid shift: random replay (stored seed + forced lobes) until the precomputed
//    reconnection vertex x_k, then reconnection; PSS Jacobian of paper Eq. 2.
//  * Reconnection criteria (paper Sec. 4): dual ray-footprint threshold (Eq. 5, c = 0.02)
//    + single-vertex roughness threshold at x_{k-1}. Invertibility checked during replay.
//  * Unified direct + indirect light (Sec. 6.1): sun NEE from x1 lives in the same reservoir;
//    NEE light vertices are forced reconnection vertices (Sec. 6.2.3).
//  * Russian roulette only modifies the initial PSS source pdf (Sec. 6.2.4); replay never rolls.
//  * Temporal + paired spatial reuse with pairwise MIS and confidence weights; temporal cCap
//    driven by the duplication map (Sec. 5); vector-valued weights for shading (Sec. 6.3).
// =======================================================================================
const PT_COMMON = RT_COMMON + /* wgsl */ `
const PI = 3.14159265;
const DMAX = 4u;                 // path vertices x1..x4 (3 bounces after the primary hit)
const SUN_COSMAX = 0.99955003;   // cos(0.03 rad): sun is a small disc light
const SUN_OMEGA = 0.0028272;     // its solid angle
const PN = 353.7;                // NEE pdf (uniform over the disc)
const C_FOOT = 0.0002;           // c / 100 with the paper's c = 0.02
const RC_NONE = 0u; const RC_SURF = 1u; const RC_INF = 2u; const RC_NEE = 3u;
const END_NEE = 0u; const END_EMIT = 1u; const END_INF = 2u;

@group(0) @binding(6) var albedoTex: texture_2d<f32>;

fn sunRad() -> vec3f { return SUN_COL * (PI / SUN_OMEGA); }

// ---- reservoir (64 bytes, like the paper's compressed layout) --------------------------
struct PRes { a: vec4u, b: vec4u, c: vec4u, d: vec4u };
struct Path {
  F: vec3f, W: f32, conf: f32, seed: u32,
  rcLocal: vec3f, rcBox: u32,   // reconnection vertex x_k in its box's local space
  R: vec3f, Rsun: f32,          // cached suffix contribution from x_k on (+ raw sun part)
  jac: f32,                     // cached base-path Jacobian terms p_{k-1}(w) G(x_{k-1} -> x_k)
  rcWi: vec3f,                  // direction for reconnections at infinity / NEE
  endJ: u32, endType: u32, k: u32, rcType: u32, lobes: u32,
};
fn emptyPath() -> Path {
  return Path(vec3f(0.0), 0.0, 0.0, 0u, vec3f(0.0), 0u, vec3f(0.0), 0.0, 0.0, vec3f(0.0, 1.0, 0.0), 0u, 0u, 0u, 0u, 0u);
}
fn octEnc(n: vec3f) -> u32 {
  var p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
  if (n.z < 0.0) { p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0)); }
  return pack2x16snorm(p);
}
fn octDec(e: u32) -> vec3f {
  let p = unpack2x16snorm(e);
  var n = vec3f(p, 1.0 - abs(p.x) - abs(p.y));
  let t = max(-n.z, 0.0);
  n.x += select(t, -t, n.x >= 0.0);
  n.y += select(t, -t, n.y >= 0.0);
  return normalize(n);
}
fn packPath(p: Path) -> PRes {
  let info = (p.endJ & 7u) | ((p.endType & 3u) << 3u) | ((p.k & 7u) << 5u) | ((p.rcType & 3u) << 8u) | (p.lobes << 16u);
  let R = min(p.R, vec3f(60000.0));
  return PRes(
    vec4u(bitcast<u32>(p.W), bitcast<u32>(p.conf), p.seed, bitcast<u32>(p.Rsun)),
    vec4u(bitcast<vec3u>(p.rcLocal), p.rcBox),
    vec4u(pack2x16float(R.xy), pack2x16float(vec2f(R.z, 0.0)), bitcast<u32>(p.jac), octEnc(p.rcWi)),
    vec4u(info, bitcast<vec3u>(p.F)));
}
fn unpackPath(r: PRes) -> Path {
  var p: Path;
  p.W = bitcast<f32>(r.a.x); p.conf = bitcast<f32>(r.a.y); p.seed = r.a.z; p.Rsun = bitcast<f32>(r.a.w);
  p.rcLocal = bitcast<vec3f>(r.b.xyz); p.rcBox = r.b.w;
  p.R = vec3f(unpack2x16float(r.c.x), unpack2x16float(r.c.y).x);
  p.jac = bitcast<f32>(r.c.z); p.rcWi = octDec(r.c.w);
  let info = r.d.x;
  p.endJ = info & 7u; p.endType = (info >> 3u) & 3u; p.k = (info >> 5u) & 7u; p.rcType = (info >> 8u) & 3u;
  p.lobes = info >> 16u;
  p.F = bitcast<vec3f>(r.d.yzw);
  return p;
}

// ---- domains (primary hits) -----------------------------------------------------------
struct PDom { x0: vec3f, p: vec3f, n: vec3f, alb: vec3f, id: u32, z: f32, ok: bool };
fn makeDom(nt: vec4f, alb: vec3f, d: f32, px: vec2i, inv: mat4x4f, cam: vec3f) -> PDom {
  var o = PDom(cam, vec3f(0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0), 0u, 0.0, false);
  if (d <= 0.0) { return o; }
  let wh = inv * vec4f(ndcOf(px), d, 1.0);
  o.p = wh.xyz / wh.w;
  o.n = normalize(nt.xyz);
  o.alb = alb;
  o.id = u32(nt.w + 0.5);
  o.z = length(o.p - cam);
  o.ok = true;
  return o;
}
fn domCur(px: vec2i) -> PDom {
  return makeDom(textureLoad(normalTex, px, 0), textureLoad(albedoTex, px, 0).rgb, textureLoad(depthTex, px, 0),
                 px, u.invViewProj, u.camPos.xyz);
}
// symmetric G-buffer similarity test for reuse
fn similar(a: PDom, b: PDom) -> bool {
  return a.ok && b.ok && dot(a.n, b.n) > 0.9 && abs(a.z - b.z) < 0.1 * max(a.z, b.z);
}

// ---- deterministic per-vertex random numbers (replayable) ----------------------------
fn rng(seed: u32, j: u32, dim: u32) -> f32 {
  return f32(pcg(seed ^ pcg(j * 8u + dim + 1u)) >> 8u) / 16777216.0;
}
fn cosSample(n: vec3f, u1: f32, u2: f32) -> vec3f {
  let b1 = ortho(n);
  let b2 = cross(n, b1);
  let r = sqrt(u1);
  let a = u2 * 6.2831853;
  return normalize(b1 * (r * cos(a)) + b2 * (r * sin(a)) + n * sqrt(max(1.0 - u1, 0.0)));
}
fn sunSample(u1: f32, u2: f32) -> vec3f {
  let s = u.sunDir.xyz;
  let t1 = ortho(s);
  let t2 = cross(s, t1);
  let r = 0.03 * sqrt(u1);
  let a = u2 * 6.2831853;
  return normalize(s + t1 * (r * cos(a)) + t2 * (r * sin(a)));
}

fn skyEnv(rd: vec3f) -> vec3f {
  let zen = vec3f(0.18, 0.42, 0.92);
  let hor = vec3f(0.70, 0.83, 0.97);
  var c = mix(hor, zen, pow(clamp(rd.y, 0.0, 1.0), 0.55));
  c = mix(c, vec3f(0.62, 0.5, 0.45), 1.0 - smoothstep(-0.25, 0.0, rd.y));
  return c * 0.9;
}

// ---- BSDF: one diffuse lobe + delta reflection + delta (straight) transmission ---------
// Lobe weights depend only on the vertex itself (not on the incoming direction) for every
// material that can be a reconnection vertex, so cached suffixes stay exact under shifts.
struct Lobes { wd: f32, ws: f32, wt: f32, ts: vec3f, tt: vec3f, pd: f32, ps: f32, pt: f32 };
fn lobesAt(id: u32, alb: vec3f, n: vec3f, wo: vec3f) -> Lobes {
  let b = boxes[id];
  let m = matOf(b);
  var L = Lobes(1.0, 0.0, 0.0, vec3f(1.0), vec3f(1.0), 1.0, 0.0, 0.0);
  if (m == M_METAL) { L.ws = b.refl; L.ts = alb; L.wd = 1.0 - b.refl; }
  else if (m == M_POLISHED) { L.ws = b.refl; L.wd = 1.0 - b.refl; }
  else if (m == M_GLASS) { let f = fresnel(max(dot(wo, n), 0.0), 0.04); L.ws = f; L.wt = (1.0 - f) * 0.9; L.tt = alb; L.wd = 0.0; }
  else if (m == M_GHOST) { L.wt = 0.82; L.tt = mix(alb, vec3f(1.0), 0.6); L.wd = 0.18; }
  let ed = L.wd * max(lum(alb), 0.02);
  let es = L.ws * lum(L.ts);
  let et = L.wt * lum(L.tt);
  let s = ed + es + et;
  if (s > 0.0) { L.pd = ed / s; L.ps = es / s; L.pt = et / s; }
  return L;
}
fn lobeProb(L: Lobes, lob: u32) -> f32 { return select(select(L.pt, L.ps, lob == 1u), L.pd, lob == 0u); }

fn localNormal(b: Box, lp: vec3f) -> vec3f {
  let r = abs(lp) / b.h;
  var ln = vec3f(0.0);
  if (r.x >= r.y && r.x >= r.z) { ln.x = sign(lp.x); }
  else if (r.y >= r.z) { ln.y = sign(lp.y); }
  else { ln.z = sign(lp.z); }
  return ln;
}

// Sun visibility through thin (non-refracting) glass / vanished platforms.
fn transmittance(o: vec3f, w: vec3f) -> vec3f {
  var T = vec3f(1.0);
  var org = o;
  var skip = NONE;
  for (var i = 0; i < 4; i++) {
    let h = trace(org, w, 1e4, skip, false);
    if (h.id == NONE) { return T; }
    let b = boxes[h.id];
    let m = matOf(b);
    if (m == M_GLASS) { T *= b.col * 0.864; }
    else if (m == M_GHOST) { T *= mix(b.col, vec3f(1.0), 0.6) * 0.82; }
    else { return vec3f(0.0); }
    org = org + w * h.t;
    skip = h.id;
  }
  return T;
}

// Primary-ray footprint term of Eq. 5: ||x0 - x1||^2 / (<n_x1, d_x1x0> / 4pi)
fn rpri2(x0: vec3f, x1: vec3f, n1: vec3f) -> f32 {
  let v = x0 - x1;
  let d2 = dot(v, v);
  return d2 * 4.0 * PI / max(abs(dot(n1, v * inverseSqrt(d2))), 1e-3);
}
// Dual footprint test for a diffuse x_{k-1} -> (diffuse or end) x_k pair. With diffuse x_k
// the inverse-footprint term is skipped (paper footnote 6), leaving
// 1 / (p_{k-1}(w) G(x_{k-1} -> x_k)) = pi d^2 / (cos_{k-1} cos_k).
fn footprintOK(d2: f32, c1: f32, ck: f32, rp2: f32) -> bool {
  return PI * d2 / max(c1 * ck, 1e-8) >= C_FOOT * rp2;
}

fn sdiv(a: vec3f, b: vec3f) -> vec3f { return select(vec3f(0.0), a / b, b > vec3f(1e-8)); }

// =============================== initial path tree + RIS ===============================
struct RisState { sel: Path, wSum: f32, r: u32 };
fn risAdd(st: ptr<function, RisState>, c: Path, w: f32) {
  if (!(w > 0.0)) { return; }
  (*st).wSum += w;
  (*st).r = pcg((*st).r);
  if (f32((*st).r >> 8u) / 16777216.0 * (*st).wSum < w) { (*st).sel = c; }
}

fn genPath(d: PDom, seed: u32) -> Path {
  var st: RisState;
  st.sel = emptyPath();
  st.wSum = 0.0;
  st.r = seed ^ 0x5bd1e995u;
  let rp2 = rpri2(d.x0, d.p, d.n);
  var beta = vec3f(1.0);
  var pRR = 1.0;
  var prevP = d.x0;
  var v = Surf(d.p, d.n, d.alb, d.id);
  var lobes = 0u;
  var misPb = -1.0;
  // reconnection bookkeeping
  var rcFound = false;
  var rc = emptyPath();          // holds rcType/k/rcLocal/rcBox/jac
  var rcBeta = vec3f(1.0);
  var pend = false;              // pair (x_{j-1}, x_j) passed; x_j becomes x_k if rough
  var pendP = emptyPath();
  var pendBeta = vec3f(1.0);

  for (var j = 1u; j <= DMAX; j++) {
    let wo = normalize(prevP - v.p);
    let L = lobesAt(v.id, v.alb, v.n, wo);

    // ---- NEE to the sun (also covers direct light at x1: unified DI/GI)
    if (L.wd > 0.0) {
      let w = sunSample(rng(seed, j, 3u), rng(seed, j, 4u));
      let c = dot(v.n, w);
      if (c > 0.0) {
        let T = transmittance(v.p + v.n * 0.02, w);
        if (lum(T) > 0.0) {
          let pb = L.pd * c / PI;
          let m = PN / (PN + pb);
          let F = beta * L.wd * v.alb * c * T * SUN_COL * m;
          var cand = emptyPath();
          cand.F = F; cand.seed = seed; cand.endJ = j; cand.endType = END_NEE; cand.lobes = lobes;
          if (rcFound) {
            cand.rcType = RC_SURF; cand.k = rc.k; cand.rcLocal = rc.rcLocal; cand.rcBox = rc.rcBox; cand.jac = rc.jac;
            cand.R = sdiv(F, rcBeta);
          } else if (pend) {
            cand.rcType = RC_SURF; cand.k = j; cand.rcLocal = pendP.rcLocal; cand.rcBox = pendP.rcBox; cand.jac = pendP.jac;
            cand.R = sdiv(F, pendBeta);
          } else {
            cand.rcType = RC_NEE; cand.k = j + 1u; cand.rcWi = w;   // forced NEE reconnection
          }
          risAdd(&st, cand, lum(F) / pRR);
        }
      }
    }
    if (j == DMAX) { break; }

    // ---- BSDF continuation with one sampled lobe
    let ul = rng(seed, j, 0u);
    var lob = 0u;
    if (ul >= L.pd) { lob = select(2u, 1u, ul < L.pd + L.ps); }
    let P = lobeProb(L, lob);
    if (P <= 0.0) { break; }
    var w = vec3f(0.0);
    var fac = vec3f(0.0);
    var org = v.p + v.n * 0.02;
    var skip = NONE;
    var cj = 0.0;
    if (lob == 0u) {
      w = cosSample(v.n, rng(seed, j, 1u), rng(seed, j, 2u));
      cj = dot(v.n, w);
      if (cj <= 1e-4) { break; }
      fac = L.wd * v.alb / P;
      misPb = L.pd * cj / PI;
    } else if (lob == 1u) {
      w = reflect(-wo, v.n);
      fac = L.ws * L.ts / P;
      skip = v.id;
      misPb = -1.0;
    } else {
      w = -wo;
      fac = L.wt * L.tt / P;
      org = v.p;
      skip = v.id;
    }
    lobes |= lob << (2u * j);
    if (pend && !rcFound) {
      if (lob == 0u) { rcFound = true; rc = pendP; rc.k = j; rcBeta = pendBeta; }
      pend = false;
    }
    let nb = beta * fac;
    let h = trace(org, w, 1e4, skip, false);

    if (h.id == NONE) {
      // escaped: sky (BSDF only) + sun disc (MIS with NEE unless the last scatter was a mirror)
      let sunIn = select(0.0, 1.0, dot(w, u.sunDir.xyz) > SUN_COSMAX);
      let ms = select(misPb / (misPb + PN), 1.0, misPb < 0.0);
      let Lsky = skyEnv(w);
      let F = nb * (Lsky + sunRad() * sunIn * ms);
      var cand = emptyPath();
      cand.F = F; cand.seed = seed; cand.endJ = j; cand.endType = END_INF; cand.lobes = lobes;
      if (rcFound) {
        cand.rcType = RC_SURF; cand.k = rc.k; cand.rcLocal = rc.rcLocal; cand.rcBox = rc.rcBox; cand.jac = rc.jac;
        cand.R = sdiv(F, rcBeta);
      } else if (lob == 0u) {
        cand.rcType = RC_INF; cand.k = j + 1u; cand.rcWi = w; cand.R = Lsky; cand.Rsun = sunIn; cand.jac = cj / PI;
      }
      risAdd(&st, cand, lum(F) / pRR);
      break;
    }

    let hp = org + w * h.t;
    let nv = surfAt(h.id, hp);
    // reconnection criteria for pair (x_j, x_{j+1})
    var pnd = false;
    if (!rcFound && lob == 0u) {
      let ck = dot(nv.n, -w);
      if (ck > 0.0 && footprintOK(h.t * h.t, cj, ck, rp2)) {
        pnd = true;
        pendP.rcLocal = qinv(boxes[h.id].q, hp - boxes[h.id].c);
        pendP.rcBox = h.id;
        pendP.jac = (cj / PI) * ck / (h.t * h.t);
        pendBeta = nb;
      }
    }
    pend = pnd;
    let Le = nv.alb * boxes[h.id].emis;
    if (lum(Le) > 0.0) {
      let F = nb * Le;
      var cand = emptyPath();
      cand.F = F; cand.seed = seed; cand.endJ = j; cand.endType = END_EMIT; cand.lobes = lobes;
      if (rcFound) {
        cand.rcType = RC_SURF; cand.k = rc.k; cand.rcLocal = rc.rcLocal; cand.rcBox = rc.rcBox; cand.jac = rc.jac;
        cand.R = sdiv(F, rcBeta);
      } else if (pend) {
        cand.rcType = RC_SURF; cand.k = j + 1u; cand.rcLocal = pendP.rcLocal; cand.rcBox = pendP.rcBox; cand.jac = pendP.jac;
        cand.R = Le;
      }
      risAdd(&st, cand, lum(F) / pRR);
    }
    beta = nb;
    prevP = v.p;
    v = nv;
    // Russian roulette: only alters the initial sample's PSS pdf, never applied in replay
    if (j >= 2u) {
      let q = clamp(lum(beta), 0.05, 0.95);
      if (rng(seed, j, 5u) >= q) { break; }
      pRR *= q;
    }
  }
  var out = st.sel;
  let ph = lum(out.F);
  out.W = select(0.0, st.wSum / ph, ph > 0.0);
  out.conf = select(0.0, 1.0, ph > 0.0);
  out.seed = seed;
  return out;
}

// =============================== hybrid shift ===========================================
struct Shift { F: vec3f, J: f32, jacNew: f32, ok: bool };

fn roughAt(r: Path, j: u32) -> bool {
  if (j == r.endJ && r.endType == END_NEE) { return true; }
  return ((r.lobes >> (2u * j)) & 3u) == 0u;
}

fn shiftPath(r: Path, d: PDom) -> Shift {
  var o = Shift(vec3f(0.0), 0.0, 0.0, false);
  if (!d.ok || r.conf <= 0.0 || r.endJ == 0u) { return o; }
  let rp2 = rpri2(d.x0, d.p, d.n);
  var beta = vec3f(1.0);
  var prevP = d.x0;
  var v = Surf(d.p, d.n, d.alb, d.id);
  var misPb = -1.0;
  // k >= 1 when RC_SURF; guard anyway: a wrapped u32 bound here spins the GPU until it hangs
  if (r.rcType == RC_SURF && r.k < 2u) { return o; }
  let last = min(select(r.endJ, r.k - 1u, r.rcType == RC_SURF), DMAX);
  if (last < 1u) { return o; }

  // random replay of the prefix y_1 .. y_last with forced lobes
  for (var j = 1u; j < last; j++) {
    let wo = normalize(prevP - v.p);
    let L = lobesAt(v.id, v.alb, v.n, wo);
    let lob = (r.lobes >> (2u * j)) & 3u;
    let P = lobeProb(L, lob);
    if (P <= 0.0) { return o; }
    var w = vec3f(0.0);
    var fac = vec3f(0.0);
    var org = v.p + v.n * 0.02;
    var skip = NONE;
    var cj = 0.0;
    if (lob == 0u) {
      w = cosSample(v.n, rng(r.seed, j, 1u), rng(r.seed, j, 2u));
      cj = dot(v.n, w);
      if (cj <= 1e-4) { return o; }
      fac = L.wd * v.alb / P;
      misPb = L.pd * cj / PI;
    } else if (lob == 1u) {
      w = reflect(-wo, v.n); fac = L.ws * L.ts / P; skip = v.id; misPb = -1.0;
    } else {
      w = -wo; fac = L.wt * L.tt / P; org = v.p; skip = v.id;
    }
    let h = trace(org, w, 1e4, skip, false);
    if (h.id == NONE) { return o; }
    let hp = org + w * h.t;
    let nv = surfAt(h.id, hp);
    // invertibility: the shifted path must not satisfy the criteria any earlier than x_k
    if (lob == 0u && roughAt(r, j + 1u)) {
      let ck = dot(nv.n, -w);
      if (ck > 0.0 && footprintOK(h.t * h.t, cj, ck, rp2)) { return o; }
    }
    beta *= fac;
    prevP = v.p;
    v = nv;
  }

  let wo = normalize(prevP - v.p);
  let L = lobesAt(v.id, v.alb, v.n, wo);

  if (r.rcType == RC_SURF) {
    // reconnect y_{k-1} -> x_k (x_{k-1} was diffuse; y_{k-1} must be too)
    if (L.wd <= 0.0) { return o; }
    let b = boxes[r.rcBox];
    let xk = qrot(b.q, r.rcLocal) + b.c;
    let nk = qrot(b.q, localNormal(b, r.rcLocal));
    let dv = xk - v.p;
    let d2 = max(dot(dv, dv), 1e-8);
    let dist = sqrt(d2);
    let w = dv / dist;
    let cy = dot(v.n, w);
    let ck = dot(nk, -w);
    if (cy <= 1e-4 || ck <= 1e-4) { return o; }
    if (!footprintOK(d2, cy, ck, rp2)) { return o; }
    if (trace(v.p + v.n * 0.02, w, dist - 0.03, NONE, false).id != NONE) { return o; }
    o.jacNew = (cy / PI) * ck / d2;
    o.J = o.jacNew / max(r.jac, 1e-12);
    o.F = beta * (L.wd * v.alb / L.pd) * r.R;
  } else if (r.rcType == RC_INF) {
    if (L.wd <= 0.0) { return o; }
    let w = r.rcWi;
    let cy = dot(v.n, w);
    if (cy <= 1e-4) { return o; }
    if (trace(v.p + v.n * 0.02, w, 1e4, NONE, false).id != NONE) { return o; }
    let pb = L.pd * cy / PI;
    let ms = pb / (pb + PN);
    o.F = beta * (L.wd * v.alb / L.pd) * (r.R + sunRad() * r.Rsun * ms);
    o.jacNew = cy / PI;
    o.J = o.jacNew / max(r.jac, 1e-12);
  } else if (r.rcType == RC_NEE) {
    if (L.wd <= 0.0) { return o; }
    let w = r.rcWi;
    let cy = dot(v.n, w);
    if (cy <= 0.0) { return o; }
    let T = transmittance(v.p + v.n * 0.02, w);
    let pb = L.pd * cy / PI;
    o.F = beta * L.wd * v.alb * cy * T * SUN_COL * (PN / (PN + pb));
    o.J = 1.0;
  } else {
    // no reconnection: replay the terminal scatter at y_endJ too
    let j = r.endJ;
    let lob = (r.lobes >> (2u * j)) & 3u;
    let P = lobeProb(L, lob);
    if (P <= 0.0) { return o; }
    var w = vec3f(0.0);
    var fac = vec3f(0.0);
    var org = v.p + v.n * 0.02;
    var skip = NONE;
    var cj = 0.0;
    if (lob == 0u) {
      w = cosSample(v.n, rng(r.seed, j, 1u), rng(r.seed, j, 2u));
      cj = dot(v.n, w);
      if (cj <= 1e-4) { return o; }
      fac = L.wd * v.alb / P;
      misPb = L.pd * cj / PI;
    } else if (lob == 1u) {
      w = reflect(-wo, v.n); fac = L.ws * L.ts / P; skip = v.id; misPb = -1.0;
    } else {
      w = -wo; fac = L.wt * L.tt / P; org = v.p; skip = v.id;
    }
    let h = trace(org, w, 1e4, skip, false);
    if (r.endType == END_INF) {
      if (h.id != NONE || lob == 0u) { return o; }
      let sunIn = select(0.0, 1.0, dot(w, u.sunDir.xyz) > SUN_COSMAX);
      let ms = select(misPb / (misPb + PN), 1.0, misPb < 0.0);
      o.F = beta * fac * (skyEnv(w) + sunRad() * sunIn * ms);
    } else {
      if (h.id == NONE) { return o; }
      let hp = org + w * h.t;
      let nv = surfAt(h.id, hp);
      if (lob == 0u) {
        let ck = dot(nv.n, -w);
        if (ck > 0.0 && footprintOK(h.t * h.t, cj, ck, rp2)) { return o; }
      }
      o.F = beta * fac * nv.alb * boxes[h.id].emis;
    }
    o.J = 1.0;
  }
  o.ok = lum(o.F) > 0.0 && o.J > 0.0;
  if (!o.ok) { o.F = vec3f(0.0); o.J = 0.0; }
  return o;
}

fn mis2(a: f32, b: f32) -> f32 { return select(0.0, a / (a + b), a + b > 0.0); }

// ---- paired reuse textures (paper Sec. 3), randomly flipped/transposed/offset per frame
fn pairDelta(px: vec2i, t: vec4f, tex: texture_2d<i32>) -> vec2i {
  let flags = u32(t.x);
  let size = vec2i(textureDimensions(tex));
  var c = px;
  if ((flags & 4u) != 0u) { c = c.yx; }
  if ((flags & 1u) != 0u) { c.x = -c.x; }
  if ((flags & 2u) != 0u) { c.y = -c.y; }
  c = c + vec2i(i32(t.y), i32(t.z));
  c = ((c % size) + size) % size;
  var dd = textureLoad(tex, c, 0).xy;
  if ((flags & 1u) != 0u) { dd.x = -dd.x; }
  if ((flags & 2u) != 0u) { dd.y = -dd.y; }
  if ((flags & 4u) != 0u) { dd = dd.yx; }
  return dd;
}
`;

// Pass 1: initial path tree + RIS, then temporal reuse (pairwise MIS, duplication-map cCap).
export const PT_TEMPORAL_WGSL = PT_COMMON + /* wgsl */ `
@group(0) @binding(7) var prevAlbedo: texture_2d<f32>;
@group(0) @binding(8) var prevNormal: texture_2d<f32>;
@group(0) @binding(9) var prevDepth: texture_depth_2d;
@group(0) @binding(10) var<storage, read> resPrev: array<PRes>;
@group(0) @binding(11) var<storage, read_write> resOut: array<PRes>;
@group(0) @binding(12) var<storage, read> dupIn: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec2u(u.res.xy);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let px = vec2i(gid.xy);
  let idx = gid.y * dims.x + gid.x;
  let dc = domCur(px);
  if (!dc.ok) { resOut[idx] = packPath(emptyPath()); return; }
  let frame = u32(u.sunDir.w);
  let seed = pcg(idx ^ pcg(frame * 0x9e3779b9u + 0x85ebca6bu)) | 1u;
  var rc = genPath(dc, seed);

  let pc = u.prevViewProj * vec4f(dc.p, 1.0);
  if (pc.w > 0.0) {
    let puv = vec2f(pc.x / pc.w * 0.5 + 0.5, 0.5 - pc.y / pc.w * 0.5);
    if (all(puv >= vec2f(0.0)) && all(puv < vec2f(1.0))) {
      let q = vec2i(puv * u.res.xy);
      let qi = u32(q.y) * dims.x + u32(q.x);
      let dq = makeDom(textureLoad(prevNormal, q, 0), textureLoad(prevAlbedo, q, 0).rgb, textureLoad(prevDepth, q, 0),
                       q, u.prevInvViewProj, u.prevCam.xyz);
      if (dq.ok && length(dq.p - dc.p) < 0.03 * dc.z + 0.05 && dot(dq.n, dc.n) > 0.9) {
        var rp = unpackPath(resPrev[qi]);
        if (rp.conf > 0.0) {
          // duplication map -> adaptive confidence cap (paper Sec. 5: cMin = 1, alpha = 0.1)
          let cap = mix(20.0, 1.0, pow(clamp(dupIn[qi], 0.0, 1.0), 0.1));
          rp.conf = min(rp.conf, cap);
          let s1 = shiftPath(rp, dc);   // temporal sample -> current pixel
          let s2 = shiftPath(rc, dq);   // canonical sample -> previous pixel (for MIS)
          let phc = lum(rc.F);
          let phn = lum(rp.F);
          let phcn = lum(s1.F);
          let phnc = lum(s2.F);
          let mc = mis2(rc.conf * phc, rp.conf * phnc * s2.J);
          let mn = mis2(rp.conf * phn, rc.conf * phcn * s1.J);
          let wc = mc * phc * rc.W;
          let wn = mn * phcn * rp.W * s1.J;
          let ws = wc + wn;
          var out = rc;
          var phs = phc;
          var rs = pcg(seed ^ 0x27d4eb2du);
          if (wn > 0.0 && f32(rs >> 8u) / 16777216.0 * ws < wn) {
            out = rp;
            out.F = s1.F;
            out.jac = s1.jacNew;
            phs = phcn;
          }
          out.W = select(0.0, ws / phs, phs > 0.0);
          out.conf = rc.conf + rp.conf;
          rc = out;
        }
      }
    }
  }
  resOut[idx] = packPath(rc);
}
`;

// Pass 2: paired spatial reuse pre-pass. Each pixel shifts its own path to each of its 3
// paired neighbours exactly once; the partner reuses the result (halves the shift cost).
export const PT_PREPASS_WGSL = PT_COMMON + /* wgsl */ `
@group(0) @binding(7) var<storage, read> resIn: array<PRes>;
@group(0) @binding(8) var<storage, read_write> preFJ: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> preJac: array<f32>;
@group(0) @binding(10) var pairTex0: texture_2d<i32>;
@group(0) @binding(11) var pairTex1: texture_2d<i32>;
@group(0) @binding(12) var pairTex2: texture_2d<i32>;

fn partner(px: vec2i, s: u32) -> vec2i {
  if (s == 0u) { return px + pairDelta(px, u.pair[0], pairTex0); }
  if (s == 1u) { return px + pairDelta(px, u.pair[1], pairTex1); }
  return px + pairDelta(px, u.pair[2], pairTex2);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec2u(u.res.xy);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let px = vec2i(gid.xy);
  let n = dims.x * dims.y;
  let idx = gid.y * dims.x + gid.x;
  let r = unpackPath(resIn[idx]);
  let dc = domCur(px);
  for (var s = 0u; s < 3u; s++) {
    var fj = vec4f(0.0);
    var jn = 0.0;
    let q = partner(px, s);
    if (all(q >= vec2i(0)) && all(q < vec2i(dims))) {
      let dq = domCur(q);
      if (similar(dc, dq)) {
        let sh = shiftPath(r, dq);
        if (sh.ok) { fj = vec4f(sh.F, sh.J); jn = sh.jacNew; }
      }
    }
    preFJ[s * n + idx] = fj;
    preJac[s * n + idx] = jn;
  }
}
`;

// Pass 3: paired spatial resampling with pairwise MIS (confidence-weighted), vector-valued
// weights for shading (colour-noise reduction), writes final reservoirs + the PT estimate.
export const PT_RESAMPLE_WGSL = PT_COMMON + /* wgsl */ `
@group(0) @binding(7) var<storage, read> resIn: array<PRes>;
@group(0) @binding(8) var<storage, read_write> resOut: array<PRes>;
@group(0) @binding(9) var<storage, read> preFJ: array<vec4f>;
@group(0) @binding(10) var<storage, read> preJac: array<f32>;
@group(0) @binding(11) var ptOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var pairTex0: texture_2d<i32>;
@group(0) @binding(13) var pairTex1: texture_2d<i32>;
@group(0) @binding(14) var pairTex2: texture_2d<i32>;

fn partner(px: vec2i, s: u32) -> vec2i {
  if (s == 0u) { return px + pairDelta(px, u.pair[0], pairTex0); }
  if (s == 1u) { return px + pairDelta(px, u.pair[1], pairTex1); }
  return px + pairDelta(px, u.pair[2], pairTex2);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec2u(u.res.xy);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let px = vec2i(gid.xy);
  let n = dims.x * dims.y;
  let idx = gid.y * dims.x + gid.x;
  let dc = domCur(px);
  if (!dc.ok) {
    resOut[idx] = packPath(emptyPath());
    textureStore(ptOut, px, vec4f(0.0));
    return;
  }
  let rc = unpackPath(resIn[idx]);
  let phc = lum(rc.F);

  var qs: array<u32, 3>;
  var ok: array<bool, 3>;
  var k = 0.0;
  var C = rc.conf;
  for (var s = 0u; s < 3u; s++) {
    ok[s] = false;
    let q = partner(px, s);
    if (all(q >= vec2i(0)) && all(q < vec2i(dims)) && similar(dc, domCur(q))) {
      qs[s] = u32(q.y) * dims.x + u32(q.x);
      ok[s] = true;
      k += 1.0;
      C += bitcast<f32>(resIn[qs[s]].a.y);
    }
  }

  var wSum = 0.0;
  var wv = vec3f(0.0);
  var mc = select(0.0, 1.0, k == 0.0);
  var sel = rc;
  var phSel = phc;
  var rs = pcg(rc.seed ^ pcg(idx + 0x632be5abu) ^ u32(u.sunDir.w));
  for (var s = 0u; s < 3u; s++) {
    if (!ok[s]) { continue; }
    let rn = unpackPath(resIn[qs[s]]);
    let fjN = preFJ[s * n + qs[s]];  // neighbour's path shifted here
    let fjC = preFJ[s * n + idx];    // our path shifted to the neighbour
    let share = select(0.0, (rn.conf + rc.conf / k) / C, C > 0.0);
    let phn = lum(rn.F);
    let phcn = lum(fjN.rgb);
    let phnc = lum(fjC.rgb);
    let mn = share * mis2(rn.conf * phn, (rc.conf / k) * phcn * fjN.a);
    mc += share * mis2((rc.conf / k) * phc, rn.conf * phnc * fjC.a);
    let w = mn * phcn * rn.W * fjN.a;
    wv += mn * fjN.rgb * rn.W * fjN.a;
    if (w > 0.0) {
      wSum += w;
      rs = pcg(rs);
      if (f32(rs >> 8u) / 16777216.0 * wSum < w) {
        sel = rn;
        sel.F = fjN.rgb;
        sel.jac = preJac[s * n + qs[s]];
        phSel = phcn;
      }
    }
  }
  let wc = mc * phc * rc.W;
  wv += mc * rc.F * rc.W;
  if (wc > 0.0) {
    wSum += wc;
    rs = pcg(rs);
    if (f32(rs >> 8u) / 16777216.0 * wSum < wc) { sel = rc; phSel = phc; }
  }
  sel.W = select(0.0, wSum / phSel, phSel > 0.0);
  sel.conf = C;
  resOut[idx] = packPath(sel);

  var e = wv;
  if (!(lum(e) >= 0.0) || !(lum(e) < 1e6)) { e = vec3f(0.0); }
  let l = lum(e);
  if (l > 40.0) { e *= 40.0 / l; }
  textureStore(ptOut, px, vec4f(e, 1.0));
}
`;

// Pass 4: duplication map (paper Sec. 5): fraction of the 17x17 neighbourhood that holds a
// copy of the same initial sample (same seed); read by next frame's temporal reuse.
export const PT_DUP_WGSL = PT_COMMON + /* wgsl */ `
@group(0) @binding(7) var<storage, read> resIn: array<PRes>;
@group(0) @binding(8) var<storage, read_write> dupOut: array<f32>;
var<workgroup> tile: array<u32, 1024>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let dims = vec2i(u.res.xy);
  let base = vec2i(wid.xy) * 16 - 8;
  for (var i = 0u; i < 4u; i++) {
    let t = lid.y * 16u + lid.x + i * 256u;
    let c = base + vec2i(i32(t % 32u), i32(t / 32u));
    var s = 0u;
    if (all(c >= vec2i(0)) && all(c < dims)) { s = resIn[u32(c.y * dims.x + c.x)].a.z; }
    tile[t] = s;
  }
  workgroupBarrier();
  if (i32(gid.x) >= dims.x || i32(gid.y) >= dims.y) { return; }
  let me = tile[(lid.y + 8u) * 32u + lid.x + 8u];
  var cnt = 0.0;
  if (me != 0u) {
    for (var dy = 0u; dy < 17u; dy++) {
      for (var dx = 0u; dx < 17u; dx++) {
        cnt += select(0.0, 1.0, tile[(lid.y + dy) * 32u + lid.x + dx] == me);
      }
    }
    cnt -= 1.0;
  }
  dupOut[gid.y * u32(dims.x) + gid.x] = cnt / 288.0;
}
`;

// Pass 2c: temporal accumulation with neighbourhood (variance) clamping, then shading,
// reflections and glass. The clamp is what keeps moving shadows (the player, spinners,
// sliding platforms) from ghosting: stale history outside the current 3x3 range is rejected.
// ---------------------------------------------------------------------------------------------
// Denoiser for the ReSTIR PT output (SVGF-style, Schied et al. 2017):
//   DN_ACCUM  – demodulate albedo, reproject + accumulate illumination and its 1st/2nd luminance
//               moments -> per-pixel variance (spatial estimate while history is short)
//   DN_ATROUS – 5 iterations of an edge-aware 5x5 a-trous wavelet (steps 1,2,4,8,16) guided by
//               normals, distance to the surface plane and variance-scaled luminance; the first
//               iteration's output becomes the colour history (feedback, as in SVGF).
const DN_COMMON = COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var normalTex: texture_2d<f32>;
@group(0) @binding(2) var depthTex: texture_depth_2d;

fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn ndcOf(px: vec2i) -> vec2f {
  let uv = (vec2f(px) + 0.5) / u.res.xy;
  return vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}
fn worldPos(px: vec2i, d: f32) -> vec3f {
  let wh = u.invViewProj * vec4f(ndcOf(px), d, 1.0);
  return wh.xyz / wh.w;
}
`;

export const DN_ACCUM_WGSL = DN_COMMON + /* wgsl */ `
@group(0) @binding(3) var albedoTex: texture_2d<f32>;
@group(0) @binding(4) var ptTex: texture_2d<f32>;
@group(0) @binding(5) var histIn: texture_2d<f32>;   // rgb illum, a box id + 1
@group(0) @binding(6) var momIn: texture_2d<f32>;    // x m1, y m2, z viewZ, w frames
@group(0) @binding(7) var momOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var dnOut: texture_storage_2d<rgba16float, write>; // rgb illum, a variance

const MAX_FRAMES = 24.0;

fn illumAt(q: vec2i) -> vec3f {
  return textureLoad(ptTex, q, 0).rgb / max(textureLoad(albedoTex, q, 0).rgb, vec3f(0.03));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec2u(u.res.xy);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let px = vec2i(gid.xy);
  let d = textureLoad(depthTex, px, 0);
  if (d <= 0.0) {
    textureStore(momOut, px, vec4f(0.0));
    textureStore(dnOut, px, vec4f(0.0));
    return;
  }
  let nt = textureLoad(normalTex, px, 0);
  let wp = worldPos(px, d);
  let viewZ = (u.viewProj * vec4f(wp, 1.0)).w;
  let c = illumAt(px);
  let l = lum(c);

  // reproject (camera motion) and validate with view depth + object id
  var hc = c;
  var hm = vec2f(l, l * l);
  var cnt = 0.0;
  let pc = u.prevViewProj * vec4f(wp, 1.0);
  if (pc.w > 0.0) {
    let puv = vec2f(pc.x / pc.w * 0.5 + 0.5, 0.5 - pc.y / pc.w * 0.5);
    if (all(puv >= vec2f(0.0)) && all(puv < vec2f(1.0))) {
      let pp = vec2i(puv * u.res.xy);
      let m = textureLoad(momIn, pp, 0);
      let h = textureLoad(histIn, pp, 0);
      if (m.w > 0.0 && abs(m.z - pc.w) < 0.03 * pc.w + 0.05 && abs(h.a - (nt.w + 1.0)) < 0.5) {
        cnt = m.w;
        hc = h.rgb;
        hm = m.xy;
      }
    }
  }
  // anti-lag: history only tracks camera motion, so moving objects (the player, platforms) leave
  // stale lighting behind. Clamp the history to the current 5x5 neighbourhood statistics and
  // shorten the history when the clamp had to move it a lot.
  if (cnt > 0.0) {
    let maxp = vec2i(dims) - 1;
    var s1 = vec3f(0.0);
    var s2 = vec3f(0.0);
    var sn = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
      for (var dx = -2; dx <= 2; dx++) {
        let q = clamp(px + vec2i(dx, dy), vec2i(0), maxp);
        if (textureLoad(depthTex, q, 0) <= 0.0) { continue; }
        if (dot(textureLoad(normalTex, q, 0).xyz, nt.xyz) < 0.8) { continue; }
        let cq = illumAt(q);
        s1 += cq;
        s2 += cq * cq;
        sn += 1.0;
      }
    }
    let mean = s1 / max(sn, 1.0);
    let sig = sqrt(max(s2 / max(sn, 1.0) - mean * mean, vec3f(0.0)));
    let cl = clamp(hc, mean - 2.0 * sig, mean + 2.0 * sig);
    if (lum(abs(cl - hc)) > 0.25 * (lum(mean) + 0.05)) { cnt = min(cnt, 3.0); }
    hc = cl;
  }
  cnt = min(cnt + 1.0, MAX_FRAMES);
  let a = 1.0 / cnt;
  let am = max(a, 0.2);
  let acc = mix(hc, c, a);
  let mom = mix(hm, vec2f(l, l * l), am);
  var variance = max(mom.y - mom.x * mom.x, 0.0);

  if (cnt < 4.0) {
    // short history: estimate the variance spatially (7x7, same surface only)
    let maxp = vec2i(dims) - 1;
    var m1 = 0.0;
    var m2 = 0.0;
    var ws = 0.0;
    for (var dy = -3; dy <= 3; dy++) {
      for (var dx = -3; dx <= 3; dx++) {
        let q = clamp(px + vec2i(dx, dy), vec2i(0), maxp);
        let dq = textureLoad(depthTex, q, 0);
        if (dq <= 0.0) { continue; }
        let w = pow(max(dot(textureLoad(normalTex, q, 0).xyz, nt.xyz), 0.0), 32.0);
        let lq = lum(illumAt(q));
        m1 += w * lq;
        m2 += w * lq * lq;
        ws += w;
      }
    }
    m1 /= max(ws, 1e-4);
    m2 /= max(ws, 1e-4);
    variance = max(m2 - m1 * m1, 0.0) * 4.0 / cnt;
  }
  textureStore(momOut, px, vec4f(mom, viewZ, cnt));
  textureStore(dnOut, px, vec4f(acc, variance));
}
`;

export const DN_ATROUS_WGSL = DN_COMMON + /* wgsl */ `
@group(0) @binding(4) var dnIn: texture_2d<f32>;
@group(0) @binding(5) var dnOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var histOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var<uniform> it: vec4u;   // x step, y write history

const SIGMA_L = 4.0;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec2u(u.res.xy);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let px = vec2i(gid.xy);
  let maxp = vec2i(dims) - 1;
  let d = textureLoad(depthTex, px, 0);
  if (d <= 0.0) {
    textureStore(dnOut, px, vec4f(0.0));
    if (it.y == 1u) { textureStore(histOut, px, vec4f(0.0)); }
    return;
  }
  let nt = textureLoad(normalTex, px, 0);
  let n = nt.xyz;
  let wp = worldPos(px, d);
  let dist = length(wp - u.camPos.xyz);
  let center = textureLoad(dnIn, px, 0);
  let lc = lum(center.rgb);

  // 3x3 gaussian of the variance for a stabler luminance edge-stop
  var vg = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let k = select(0.25, 0.5, dx == 0) * select(0.25, 0.5, dy == 0);
      vg += k * textureLoad(dnIn, clamp(px + vec2i(dx, dy), vec2i(0), maxp), 0).a;
    }
  }
  let lden = SIGMA_L * sqrt(max(vg, 0.0)) + 1e-3;

  let step = i32(it.x);
  // plane tolerance grows with the footprint of the kernel in world space
  let ptol = 0.004 * dist * f32(step) + 0.02;
  var h = array<f32, 3>(0.375, 0.25, 0.0625);
  var sum = center.rgb * 0.140625;
  var vsum = center.a * 0.140625 * 0.140625;
  var wsum = 0.140625;
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let q = px + vec2i(dx, dy) * step;
      if (any(q < vec2i(0)) || any(q > maxp)) { continue; }
      let dq = textureLoad(depthTex, q, 0);
      if (dq <= 0.0) { continue; }
      let nq = textureLoad(normalTex, q, 0).xyz;
      let wn = pow(max(dot(n, nq), 0.0), 64.0);
      let wz = exp(-abs(dot(n, worldPos(q, dq) - wp)) / ptol);
      let s = textureLoad(dnIn, q, 0);
      let wl = exp(-abs(lum(s.rgb) - lc) / lden);
      let w = h[abs(dx)] * h[abs(dy)] * wn * wz * wl;
      sum += w * s.rgb;
      vsum += w * w * s.a;
      wsum += w;
    }
  }
  let o = vec4f(sum / wsum, vsum / (wsum * wsum));
  textureStore(dnOut, px, o);
  if (it.y == 1u) { textureStore(histOut, px, vec4f(o.rgb, nt.w + 1.0)); }
}
`;

export const POST_WGSL = COMMON + /* wgsl */ `
@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var<uniform> u: U;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var bloom: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

@fragment fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let uv = fc.xy / u.misc.zw;
  var c = textureSample(hdr, samp, uv).rgb * u.misc.y;
  // bloom: the up-chain sums 6 levels, so normalise before mixing
  let bl = textureSample(bloom, samp, uv).rgb / 6.0;
  c = mix(c, bl, u.probeInfo.w);
  c = aces(c);
  // gentle grade: a touch more saturation and contrast in the mids
  let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c = mix(vec3f(l), c, 1.08);
  c = clamp(c, vec3f(0.0), vec3f(1.0));
  c = c * c * (3.0 - 2.0 * c) * 0.18 + c * 0.82;
  let v = uv * (1.0 - uv);
  c *= pow(clamp(v.x * v.y * 16.0, 0.0, 1.0), 0.18);
  c = pow(c, vec3f(1.0 / 2.2));
  c = mix(c, vec3f(1.0, 0.15, 0.1), u.misc.x * 0.45);
  return vec4f(c, 1.0);
}
`;

// =============================================================================================
// RASTER LIGHTING ("how a shipped game would light it"): no per-pixel ray tracing.
//   SHADOW        – 3 cascaded sun shadow maps (depth only, stabilised, slope-scaled bias)
//   PROBE_TRACE   – DDGI: every probe of a camera-following 24x12x32 grid shoots 48 rays per
//                   frame (spherical Fibonacci, randomly rotated); hit radiance = sun + local
//                   lights + emission + last frame's probes (infinite bounces)
//   PROBE_UPDATE  – blend rays into 8x8 octahedral irradiance + 16x16 depth-moment maps with
//                   hysteresis; classify probes stuck inside geometry
//   AO / AO_BLUR  – horizon-based ambient occlusion (4 dirs x 4 steps, 4x4 Bayer + 4x4 blur)
//   LIGHT         – deferred: GGX sun with PCSS soft shadows, hand-placed point lights,
//                   probe GI with Chebyshev visibility, multi-bounce AO, split-sum specular
//                   with screen-space reflections (from last frame) over a probe/sky fallback,
//                   height fog with sun in-scattering
//   TRANSPARENT   – forward pass for glass and ghost blocks over the lit image
//   BLOOM_DOWN/UP – 6-level dual-filter bloom (13-tap down with Karis average, tent up)
// =============================================================================================

const BRDF_FN = /* wgsl */ `
const PI = 3.14159265;
struct Mat { rough: f32, metal: f32 };
fn matParams(m: u32) -> Mat {
  if (m == M_METAL) { return Mat(0.05, 1.0); }
  if (m == M_POLISHED) { return Mat(0.07, 0.0); }
  if (m == M_SMOOTH) { return Mat(0.32, 0.0); }
  if (m == M_NEON) { return Mat(0.45, 0.0); }
  if (m == M_CHECK) { return Mat(0.25, 0.0); }
  if (m == M_GLASS || m == M_GHOST) { return Mat(0.02, 0.0); }
  return Mat(0.5, 0.0);                      // Roblox plastic
}
fn fSchlick(f0: vec3f, c: f32) -> vec3f { return f0 + (1.0 - f0) * pow(1.0 - clamp(c, 0.0, 1.0), 5.0); }
// Diffuse + GGX specular for one light, in this renderer's units (diffuse = albedo * E).
fn brdf(n: vec3f, v: vec3f, l: vec3f, alb: vec3f, mt: Mat) -> vec3f {
  let h = normalize(v + l);
  let nl = max(dot(n, l), 1e-4);
  let nv = max(dot(n, v), 1e-4);
  let nh = max(dot(n, h), 0.0);
  let a = max(mt.rough * mt.rough, 0.002);
  let a2 = a * a;
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * dd * dd);
  let k = a * 0.5;
  let Vis = 0.25 / ((nv * (1.0 - k) + k) * (nl * (1.0 - k) + k));
  let f0 = mix(vec3f(0.04), alb, mt.metal);
  let F = fSchlick(f0, dot(v, h));
  return alb * (1.0 - F) * (1.0 - mt.metal) + PI * D * Vis * F;
}
// split-sum environment BRDF (Karis' analytic fit)
fn envBRDF(f0: vec3f, nv: f32, rough: f32) -> vec3f {
  let r = rough * vec4f(-1.0, -0.0275, -0.572, 0.022) + vec4f(1.0, 0.0425, 1.04, -0.04);
  let a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
  let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
  return f0 * ab.x + ab.y;
}
`;

// Hand-placed point lights (authored in the Rust level, attached to moving parts).
const LIGHTS_FN = /* wgsl */ `
struct Light { p: vec3f, radius: f32, col: vec3f, pad: f32 };
fn shadeLight(L: Light, p: vec3f, n: vec3f, v: vec3f, alb: vec3f, mt: Mat) -> vec3f {
  let dv = L.p - p;
  let d2 = dot(dv, dv);
  let r2 = L.radius * L.radius;
  if (d2 >= r2) { return vec3f(0.0); }
  let l = dv * inverseSqrt(max(d2, 1e-6));
  let nl = dot(n, l);
  if (nl <= 0.0) { return vec3f(0.0); }
  let x = d2 / r2;
  let win = (1.0 - x * x) * (1.0 - x * x);
  return brdf(n, v, l, alb, mt) * L.col * (win / (d2 + 1.0)) * nl;
}
fn localLights(p: vec3f, n: vec3f, v: vec3f, alb: vec3f, mt: Mat) -> vec3f {
  var o = vec3f(0.0);
  let cnt = u32(u.probeInfo.z);
  for (var i = 0u; i < cnt; i++) {
    let L = lights[i];
    let dv = L.p - p;
    let d2 = dot(dv, dv);
    let r2 = L.radius * L.radius;
    if (d2 >= r2) { continue; }
    let l = dv * inverseSqrt(max(d2, 1e-6));
    let nl = dot(n, l);
    if (nl <= 0.0) { continue; }
    // inverse-square with a smooth window to zero at the light's radius (UE4 falloff)
    let x = d2 / r2;
    let win = (1.0 - x * x) * (1.0 - x * x);
    o += brdf(n, v, l, alb, mt) * L.col * (win / (d2 + 1.0)) * nl;
  }
  return o;
}
`;

// DDGI probe volume: toroidal addressing so the grid can scroll with the player and only the
// newly entered slab of probes has to start over.
const PROBE_FN = /* wgsl */ `
const PDIM = vec3i(24, 12, 32);
const PCOUNT = 9216u;
const PRAYS = 48u;

fn octEncode(n0: vec3f) -> vec2f {
  let n = n0 / (abs(n0.x) + abs(n0.y) + abs(n0.z));
  if (n.z < 0.0) {
    return (1.0 - abs(n.yx)) * select(vec2f(-1.0), vec2f(1.0), n.xy >= vec2f(0.0));
  }
  return n.xy;
}
fn octDecode(e: vec2f) -> vec3f {
  var v = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  if (v.z < 0.0) {
    let t = (1.0 - abs(v.yx)) * select(vec2f(-1.0), vec2f(1.0), v.xy >= vec2f(0.0));
    v = vec3f(t, v.z);
  }
  return normalize(v);
}
fn gridBase() -> vec3i { return vec3i(u.probeGrid.xyz); }
fn probeSlot(cell: vec3i) -> u32 {
  let m = ((cell % PDIM) + PDIM) % PDIM;
  return u32(m.x + PDIM.x * (m.y + PDIM.y * m.z));
}
fn cellOfSlot(slot: u32) -> vec3i {
  let sl = vec3i(i32(slot % 24u), i32((slot / 24u) % 12u), i32(slot / 288u));
  let b = gridBase();
  return b + (((sl - b) % PDIM) + PDIM) % PDIM;
}
fn probePos(cell: vec3i) -> vec3f { return (vec3f(cell) + 0.5) * u.probeGrid.w; }

fn irrTexel(slot: u32, t: vec2i) -> vec3f { return probeIrr[slot * 64u + u32(t.y * 8 + t.x)].rgb; }
fn sampleIrr(slot: u32, dir: vec3f) -> vec3f {
  let uv = (octEncode(dir) * 0.5 + 0.5) * 8.0 - 0.5;
  let f = fract(uv);
  let i0 = vec2i(floor(uv));
  let a = clamp(i0, vec2i(0), vec2i(7));
  let b = clamp(i0 + 1, vec2i(0), vec2i(7));
  return mix(mix(irrTexel(slot, a), irrTexel(slot, vec2i(b.x, a.y)), f.x),
             mix(irrTexel(slot, vec2i(a.x, b.y)), irrTexel(slot, b), f.x), f.y);
}
fn depTexel(slot: u32, t: vec2i) -> vec2f { return probeDepth[slot * 256u + u32(t.y * 16 + t.x)]; }
fn sampleDep(slot: u32, dir: vec3f) -> vec2f {
  let uv = (octEncode(dir) * 0.5 + 0.5) * 16.0 - 0.5;
  let f = fract(uv);
  let i0 = vec2i(floor(uv));
  let a = clamp(i0, vec2i(0), vec2i(15));
  let b = clamp(i0 + 1, vec2i(0), vec2i(15));
  return mix(mix(depTexel(slot, a), depTexel(slot, vec2i(b.x, a.y)), f.x),
             mix(depTexel(slot, vec2i(a.x, b.y)), depTexel(slot, b), f.x), f.y);
}

// Irradiance at p (normal n, v = towards the viewer / ray origin) looked up in direction sdir,
// from the 8 surrounding probes with trilinear, backface and Chebyshev visibility weights.
struct ProbeHit { irr: vec3f, refl: vec3f, w: f32 };
fn probeLookup2(p: vec3f, n: vec3f, v: vec3f, sdir: vec3f, both: bool) -> ProbeHit {
  let sp = u.probeGrid.w;
  let pb = p + (n * 0.2 + v * 0.8) * (0.3 * sp);
  let g = pb / sp - 0.5;
  let c0 = vec3i(floor(g));
  let al = g - vec3f(c0);
  let base = gridBase();
  var sum = vec3f(0.0);
  var sumR = vec3f(0.0);
  var ws = 0.0;
  for (var i = 0u; i < 8u; i++) {
    let off = vec3i(i32(i & 1u), i32((i >> 1u) & 1u), i32((i >> 2u) & 1u));
    let cell = c0 + off;
    if (any(cell < base) || any(cell >= base + PDIM)) { continue; }
    let slot = probeSlot(cell);
    let st = probeState[slot];
    if (st.w == 0 || any(st.xyz != cell)) { continue; }
    let pp = probePos(cell);
    let tri = select(1.0 - al, al, off == vec3i(1));
    var w = 1.0;
    let dirTo = normalize(pp - p);
    let bf = (dot(dirTo, n) + 1.0) * 0.5;
    w *= bf * bf + 0.2;
    let pv = pb - pp;
    let d = length(pv);
    let m = sampleDep(slot, pv / max(d, 1e-4));
    if (d > m.x) {
      let vr = abs(m.y - m.x * m.x);
      let dd = d - m.x;
      let ch = vr / (vr + dd * dd);
      w *= max(ch * ch * ch, 0.0);
    }
    w = max(w, 1e-6);
    if (w < 0.2) { w *= w * w / 0.04; }
    w *= tri.x * tri.y * tri.z;
    sum += w * sampleIrr(slot, n);
    if (both) { sumR += w * sampleIrr(slot, sdir); }
    ws += w;
  }
  if (ws <= 1e-5) {
    // outside the volume / no valid probes yet: a flat sky-ish ambient
    let fb = mix(vec3f(0.45, 0.36, 0.30), vec3f(0.50, 0.64, 0.92), n.y * 0.5 + 0.5) * 0.6;
    return ProbeHit(fb, skyDiffuse(sdir), 0.0);
  }
  return ProbeHit(sum / ws, sumR / ws, ws);
}
fn probeIrradiance(p: vec3f, n: vec3f, v: vec3f) -> vec3f { return probeLookup2(p, n, v, n, false).irr; }
// fraction of the nearest probe's rays in direction d that escaped to the sky
fn skyVisibility(p: vec3f, d: vec3f) -> f32 {
  let sp = u.probeGrid.w;
  let cell = vec3i(floor(p / sp));
  let base = gridBase();
  if (any(cell < base) || any(cell >= base + PDIM)) { return 1.0; }
  let slot = probeSlot(cell);
  let st = probeState[slot];
  if (st.w == 0 || any(st.xyz != cell)) { return 1.0; }
  let m = sampleDep(slot, d);
  return smoothstep(sp * 1.0, sp * 1.45, m.x);
}
`;

const SHADOW_FN = /* wgsl */ `
const SM_SIZE = 2048.0;
const POISSON = array<vec2f, 16>(
  vec2f(-0.94201624, -0.39906216), vec2f(0.94558609, -0.76890725), vec2f(-0.094184101, -0.92938870),
  vec2f(0.34495938, 0.29387760), vec2f(-0.91588581, 0.45771432), vec2f(-0.81544232, -0.87912464),
  vec2f(-0.38277543, 0.27676845), vec2f(0.97484398, 0.75648379), vec2f(0.44323325, -0.97511554),
  vec2f(0.53742981, -0.47373420), vec2f(-0.26496911, -0.41893023), vec2f(0.79197514, 0.19090188),
  vec2f(-0.24188840, 0.99706507), vec2f(-0.81409955, 0.91437590), vec2f(0.19984126, 0.78641367),
  vec2f(0.14383161, -0.14100790));

// Contact-hardening soft sun shadow (PCSS): blocker search, penumbra from the blocker distance,
// then a rotated 16-tap Poisson PCF with hardware 2x2 comparison filtering per tap.
fn sunShadow(wp: vec3f, n: vec3f, viewZ: f32, noise: f32) -> f32 {
  let L = u.sunDir.xyz;
  if (dot(n, L) <= 0.0) { return 0.0; }
  if (viewZ > u.cascade.z) { return 1.0; }
  var c = 2;
  if (viewZ < u.cascade.x) { c = 0; } else if (viewZ < u.cascade.y) { c = 1; }
  let texel = u.cascadeTexel[c];
  let nl = clamp(dot(n, L), 0.0, 1.0);
  let pw = wp + n * texel * (1.0 + 2.0 * (1.0 - nl)) + L * texel;
  let lc = u.sunVP[c] * vec4f(pw, 1.0);
  let uv = vec2f(lc.x * 0.5 + 0.5, 0.5 - lc.y * 0.5);
  let z = lc.z;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || z > 1.0) { return 1.0; }
  let ang = noise * 6.2831853;
  let rot = mat2x2f(cos(ang), sin(ang), -sin(ang), cos(ang));

  // blocker search (radius covers the widest penumbra we allow)
  let search = clamp(1.2 / texel, 3.0, 14.0);
  var bsum = 0.0;
  var bn = 0.0;
  for (var i = 0; i < 12; i++) {
    let o = rot * POISSON[i] * search;
    let t = clamp(vec2i(uv * SM_SIZE + o), vec2i(0), vec2i(i32(SM_SIZE) - 1));
    let d = textureLoad(shadowMap, t, c, 0);
    if (d < z) { bsum += d; bn += 1.0; }
  }
  if (bn == 0.0) { return 1.0; }
  let zb = bsum / bn;
  let penumbra = (z - zb) * u.cascadeRange[c] * u.cascade.w;
  let rad = clamp(penumbra / texel, 1.0, search);
  var vis = 0.0;
  for (var i = 0; i < 16; i++) {
    let o = rot * POISSON[i] * rad / SM_SIZE;
    vis += textureSampleCompareLevel(shadowMap, shadowSamp, uv + o, c, z);
  }
  return vis / 16.0;
}
`;

const FOG_FN = /* wgsl */ `
// Exponential height fog integrated along the view ray, lit by the sun and the lava below.
fn heightFog(col: vec3f, wp: vec3f, cam: vec3f) -> vec3f {
  let rv = wp - cam;
  let dist = length(rv);
  let rd = rv / max(dist, 1e-4);
  let dens = 0.0028;
  let fall = 0.05;
  let h0 = 2.0;
  // integrate down to at most a little below the course, so looking at the lava doesn't fog it out
  let dy = max(wp.y, -3.0) - cam.y;
  var k = 1.0;
  if (abs(dy) > 0.01) { k = (1.0 - exp(-fall * dy)) / (fall * dy); }
  let amount = dens * exp(-fall * (cam.y - h0)) * max(dist - 8.0, 0.0) * k;
  let f = 1.0 - exp(-amount);
  let sunAmt = pow(max(dot(rd, u.sunDir.xyz), 0.0), 8.0);
  var fc = mix(vec3f(0.52, 0.64, 0.84), vec3f(1.0, 0.84, 0.6) * 1.4, sunAmt);
  fc = mix(fc, vec3f(0.9, 0.36, 0.14), clamp((-wp.y - 2.0) / 14.0, 0.0, 1.0) * 0.35);
  return mix(col, fc, f);
}
`;

// Shared box expansion (36 vertices per instance) for the shadow and transparent passes.
const BOX_VS_FN = /* wgsl */ `
fn boxCorner(b: Box, vi: u32) -> array<vec3f, 2> {
  let f = vi / 6u;
  let k = vi % 6u;
  let axis = f / 2u;
  let s = select(-1.0, 1.0, (f & 1u) == 0u);
  var n = vec3f(0.0);
  n[axis] = s;
  var tu = vec3f(0.0);
  tu[(axis + 1u) % 3u] = 1.0;
  var tv = vec3f(0.0);
  tv[(axis + 2u) % 3u] = 1.0;
  var quad = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                             vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c2 = quad[k];
  let lp = (n + tu * c2.x + tv * c2.y) * b.h;
  return array<vec3f, 2>(qrot(b.q, lp) + b.c, qrot(b.q, n));
}
`;

export const SHADOW_WGSL = COMMON + BOX_VS_FN + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> boxes: array<Box>;
@group(0) @binding(2) var<uniform> cas: vec4u;

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> @builtin(position) vec4f {
  let b = boxes[ii];
  let m = matOf(b);
  // see-through parts and the huge lava slab don't cast sun shadows
  if (m == M_GLASS || m == M_GHOST || b.h.x > 300.0) { return vec4f(2.0, 2.0, 2.0, 1.0); }
  let w = boxCorner(b, vi);
  return u.sunVP[cas.x] * vec4f(w[0], 1.0);
}
`;

export const PROBE_TRACE_WGSL = RT_COMMON + BRDF_FN + /* wgsl */ `
@group(0) @binding(6) var<storage, read> probeIrr: array<vec4f>;
@group(0) @binding(7) var<storage, read> probeDepth: array<vec2f>;
@group(0) @binding(8) var<storage, read> probeState: array<vec4i>;
@group(0) @binding(9) var<storage, read> lights: array<Light>;
@group(0) @binding(10) var<storage, read_write> rays: array<vec4f>;
` + PROBE_FN + LIGHTS_FN + /* wgsl */ `
// random rotation per frame so the 48 fixed directions cover the sphere over time
fn rayRot() -> mat3x3f {
  var s = pcg(u32(u.probeInfo.y) * 9781u + 1u);
  let u1 = rnd(&s); let u2 = rnd(&s); let u3 = rnd(&s);
  let q = vec4f(sqrt(1.0 - u1) * sin(6.2831853 * u2), sqrt(1.0 - u1) * cos(6.2831853 * u2),
                sqrt(u1) * sin(6.2831853 * u3), sqrt(u1) * cos(6.2831853 * u3));
  return mat3x3f(qrot(q, vec3f(1.0, 0.0, 0.0)), qrot(q, vec3f(0.0, 1.0, 0.0)), qrot(q, vec3f(0.0, 0.0, 1.0)));
}
fn sphFib(i: u32) -> vec3f {
  let k = f32(i) + 0.5;
  let z = 1.0 - 2.0 * k / f32(PRAYS);
  let r = sqrt(max(1.0 - z * z, 0.0));
  let ph = 6.2831853 * fract(k * 0.61803399);
  return vec3f(r * cos(ph), r * sin(ph), z);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  let stride = u32(u.opts.y);
  if (t >= PCOUNT / stride * PRAYS) { return; }
  let slot = (t / PRAYS) * stride + u32(u.opts.z);
  let ri = t % PRAYS;
  let ro = slot * PRAYS + ri;
  let o = probePos(cellOfSlot(slot));
  let d = rayRot() * sphFib(ri);
  let h = trace(o, d, 1e4, NONE, false);
  if (h.id == NONE) { rays[ro] = vec4f(skyDiffuse(d), 1e4); return; }
  let sf = surfAt(h.id, o + d * h.t);
  if (dot(sf.n, d) > 0.0) { rays[ro] = vec4f(0.0, 0.0, 0.0, -h.t * 0.2); return; }  // inside a box
  let b = boxes[h.id];
  let mt = matParams(matOf(b));
  let v = -d;
  var L = vec3f(0.0);
  let ndl = dot(sf.n, u.sunDir.xyz);
  if (ndl > 0.0) { L += sf.alb * SUN_COL * ndl * hardShadow(sf.p, sf.n); }
  L += localLights(sf.p, sf.n, v, sf.alb, Mat(1.0, 0.0));
  L += sf.alb * probeIrradiance(sf.p, sf.n, v) * (1.0 - mt.metal * 0.5);
  L += sf.alb * b.emis;
  rays[ro] = vec4f(L, h.t);
}
`;

export const PROBE_UPDATE_WGSL = COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> rays: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> probeIrr: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> probeDepth: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> probeState: array<vec4i>;
` + PROBE_FN.replace(/fn irrTexel[\s\S]*$/, '') + /* wgsl */ `
fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd(s: ptr<function, u32>) -> f32 { *s = pcg(*s); return f32(*s >> 8u) / 16777216.0; }
fn rayRot() -> mat3x3f {
  var s = pcg(u32(u.probeInfo.y) * 9781u + 1u);
  let u1 = rnd(&s); let u2 = rnd(&s); let u3 = rnd(&s);
  let q = vec4f(sqrt(1.0 - u1) * sin(6.2831853 * u2), sqrt(1.0 - u1) * cos(6.2831853 * u2),
                sqrt(u1) * sin(6.2831853 * u3), sqrt(u1) * cos(6.2831853 * u3));
  return mat3x3f(qrot(q, vec3f(1.0, 0.0, 0.0)), qrot(q, vec3f(0.0, 1.0, 0.0)), qrot(q, vec3f(0.0, 0.0, 1.0)));
}
fn sphFib(i: u32) -> vec3f {
  let k = f32(i) + 0.5;
  let z = 1.0 - 2.0 * k / f32(PRAYS);
  let r = sqrt(max(1.0 - z * z, 0.0));
  let ph = 6.2831853 * fract(k * 0.61803399);
  return vec3f(r * cos(ph), r * sin(ph), z);
}

var<workgroup> rr: array<vec4f, 48>;
var<workgroup> rdir: array<vec3f, 48>;
var<workgroup> backs: atomic<u32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let stride = u32(u.opts.y);
  let slot = wg.x * stride + u32(u.opts.z);
  if (li == 0u) { atomicStore(&backs, 0u); }
  workgroupBarrier();
  if (li < PRAYS) {
    let r = rays[slot * PRAYS + li];
    rr[li] = r;
    rdir[li] = rayRot() * sphFib(li);
    if (r.w < 0.0) { atomicAdd(&backs, 1u); }
  }
  workgroupBarrier();
  let cell = cellOfSlot(slot);
  let st = probeState[slot];
  let reset = st.w == 0 || any(st.xyz != cell);
  let hyst = select(u.probeInfo.x, 0.0, reset);
  let sp = u.probeGrid.w;

  if (li < 64u) {
    let e = (vec2f(f32(li % 8u), f32(li / 8u)) + 0.5) / 8.0 * 2.0 - 1.0;
    let dir = octDecode(e);
    var sum = vec3f(0.0);
    var ws = 0.0;
    for (var i = 0u; i < PRAYS; i++) {
      let w = max(dot(dir, rdir[i]), 0.0);
      sum += w * rr[i].rgb;
      ws += w;
    }
    let nw = sum / max(ws, 1e-4);
    let old = probeIrr[slot * 64u + li].rgb;
    // react faster to big lighting changes (blinking platforms, lights switching off)
    var hy = hyst;
    let dl = abs(dot(nw - old, vec3f(0.2126, 0.7152, 0.0722)));
    if (dl > 0.6 * (dot(old, vec3f(0.2126, 0.7152, 0.0722)) + 0.1)) { hy = min(hy, 0.85); }
    probeIrr[slot * 64u + li] = vec4f(mix(nw, old, hy), 1.0);
  }
  {
    let e = (vec2f(f32(li % 16u), f32(li / 16u)) + 0.5) / 16.0 * 2.0 - 1.0;
    let dir = octDecode(e);
    var sum = vec2f(0.0);
    var ws = 0.0;
    let maxD = sp * 1.5;
    for (var i = 0u; i < PRAYS; i++) {
      let w = pow(max(dot(dir, rdir[i]), 0.0), 50.0);
      if (w <= 0.0) { continue; }
      let dd = min(abs(rr[i].w), maxD);
      sum += w * vec2f(dd, dd * dd);
      ws += w;
    }
    if (ws > 1e-5) {
      let nw = sum / ws;
      let old = probeDepth[slot * 256u + li];
      probeDepth[slot * 256u + li] = mix(nw, old, select(hyst, 0.0, reset));
    } else if (reset) {
      probeDepth[slot * 256u + li] = vec2f(maxD, maxD * maxD);
    }
  }
  workgroupBarrier();
  if (li == 0u) {
    // more than 25% of rays hitting back faces: the probe sits inside geometry -> disable it
    let alive = select(1, 0, atomicLoad(&backs) * 4u > PRAYS);
    probeState[slot] = vec4i(cell, alive);
  }
}
`;

export const AO_WGSL = COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var normalTex: texture_2d<f32>;
@group(0) @binding(2) var depthTex: texture_depth_2d;
@group(0) @binding(3) var aoOut: texture_storage_2d<r32float, write>;

fn ndcOf(px: vec2f) -> vec2f {
  let uv = (px + 0.5) / u.res.xy;
  return vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}
fn worldPos(px: vec2f, d: f32) -> vec3f {
  let wh = u.invViewProj * vec4f(ndcOf(px), d, 1.0);
  return wh.xyz / wh.w;
}
fn bayer4(p: vec2u) -> f32 {
  let m = array<f32, 16>(0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
  return (m[(p.y & 3u) * 4u + (p.x & 3u)] + 0.5) / 16.0;
}

// runs at half resolution: one AO sample per 2x2 block, upsampled by the depth-aware blur
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let hd = (vec2u(u.res.xy) + 1u) / 2u;
  if (gid.x >= hd.x || gid.y >= hd.y) { return; }
  let px = min(vec2i(gid.xy) * 2, vec2i(u.res.xy) - 1);
  let d = textureLoad(depthTex, px, 0);
  if (d <= 0.0) { textureStore(aoOut, gid.xy, vec4f(1.0)); return; }
  let n = textureLoad(normalTex, px, 0).xyz;
  let p = worldPos(vec2f(px), d);
  let viewZ = 0.1 / d;
  let R = 1.8;
  // projected radius in pixels for a 70 deg vertical fov
  let rpx = clamp(R * u.res.y * 0.714 / viewZ, 3.0, 90.0);
  let jit = bayer4(gid.xy);
  var occ = 0.0;
  for (var di = 0; di < 4; di++) {
    let a = (f32(di) + jit) * 0.78539816;   // pi/4 steps over a half circle, both sides
    let dir = vec2f(cos(a), sin(a));
    for (var side = -1.0; side <= 1.0; side += 2.0) {
      for (var si = 0; si < 4; si++) {
        let t = (f32(si) + fract(jit * 7.0 + f32(si) * 0.37)) / 4.0;
        let q = vec2f(px) + dir * side * rpx * (t * t * 0.85 + 0.15);
        if (any(q < vec2f(0.0)) || any(q >= u.res.xy)) { continue; }
        let dq = textureLoad(depthTex, vec2i(q), 0);
        if (dq <= 0.0) { continue; }
        let v = worldPos(floor(q), dq) - p;
        let l2 = dot(v, v);
        let h = dot(n, v) * inverseSqrt(max(l2, 1e-6));
        occ += max(h - 0.12, 0.0) * clamp(1.0 - l2 / (R * R), 0.0, 1.0);
      }
    }
  }
  let ao = clamp(1.0 - occ / 32.0 * 2.6 * u.rasterInfo.z, 0.0, 1.0);
  textureStore(aoOut, gid.xy, vec4f(ao));
}
`;

export const AO_BLUR_WGSL = COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(2) var depthTex: texture_depth_2d;
@group(0) @binding(3) var aoIn: texture_2d<f32>;
@group(0) @binding(4) var aoOut: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec2u(u.res.xy);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let px = vec2i(gid.xy);
  let d = textureLoad(depthTex, px, 0);
  if (d <= 0.0) { textureStore(aoOut, px, vec4f(1.0)); return; }
  let z = 0.1 / d;
  var s = 0.0;
  var ws = 0.0;
  // 4x4 half-res taps around this pixel (= 8x8 full-res footprint): blur + bilateral upsample
  let hp = px / 2;
  let hmax = vec2i(textureDimensions(aoIn)) - 1;
  for (var y = -2; y <= 1; y++) {
    for (var x = -2; x <= 1; x++) {
      let hq = clamp(hp + vec2i(x, y), vec2i(0), hmax);
      let dq = textureLoad(depthTex, min(hq * 2, vec2i(dims) - 1), 0);
      if (dq <= 0.0) { continue; }
      let w = exp(-abs(0.1 / dq - z) / (0.03 * z + 0.02));
      s += w * textureLoad(aoIn, hq, 0).r;
      ws += w;
    }
  }
  if (ws <= 1e-4) { textureStore(aoOut, px, vec4f(1.0)); return; }
  textureStore(aoOut, px, vec4f(s / max(ws, 1e-4)));
}
`;

const LAVA_FN = /* wgsl */ `
fn hash2(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
fn vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let w = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash2(i), hash2(i + vec2f(1.0, 0.0)), w.x),
             mix(hash2(i + vec2f(0.0, 1.0)), hash2(i + vec2f(1.0, 1.0)), w.x), w.y);
}
// Animated lava: slow-drifting cooled crust plates separated by glowing, pulsing cracks.
struct Lava { alb: vec3f, emis: vec3f };
fn lavaSurface(p: vec2f, t: f32) -> Lava {
  var q = p * 0.07;
  let warp = vec2f(vnoise(q * 1.7 + t * 0.05), vnoise(q * 1.7 - t * 0.04 + 7.3));
  q += (warp - 0.5) * 1.4;
  var f = 0.0;
  var a = 0.5;
  var qq = q + vec2f(t * 0.018, t * 0.011);
  for (var i = 0; i < 4; i++) { f += a * vnoise(qq); qq = qq * 2.03 + 1.7; a *= 0.5; }
  let crack = 1.0 - smoothstep(0.02, 0.11, abs(f - 0.5));
  let pulse = 0.75 + 0.25 * sin(t * 1.3 + f * 12.0);
  let heat = clamp(crack * pulse + smoothstep(0.62, 0.9, f) * 0.35, 0.0, 1.0);
  let hot = mix(vec3f(1.0, 0.07, 0.0), vec3f(1.0, 0.42, 0.04), heat * heat) * (0.12 + 4.0 * heat * heat);
  return Lava(mix(vec3f(0.035, 0.022, 0.02), vec3f(0.2, 0.05, 0.02), heat), hot);
}

`;

export const LIGHT_WGSL = COMMON + SKY_FN + BRDF_FN + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> boxes: array<Box>;
@group(0) @binding(2) var normalTex: texture_2d<f32>;
@group(0) @binding(3) var depthTex: texture_depth_2d;
@group(0) @binding(4) var albedoTex: texture_2d<f32>;
@group(0) @binding(5) var shadowMap: texture_depth_2d_array;
@group(0) @binding(6) var shadowSamp: sampler_comparison;
@group(0) @binding(7) var<storage, read> probeIrr: array<vec4f>;
@group(0) @binding(8) var<storage, read> probeDepth: array<vec2f>;
@group(0) @binding(9) var<storage, read> probeState: array<vec4i>;
@group(0) @binding(10) var<storage, read> lights: array<Light>;
@group(0) @binding(11) var aoTex: texture_2d<f32>;
@group(0) @binding(12) var hdrPrev: texture_2d<f32>;
@group(0) @binding(13) var linSamp: sampler;
@group(0) @binding(14) var hdrOut: texture_storage_2d<rgba16float, write>;
` + PROBE_FN + LIGHTS_FN + SHADOW_FN + FOG_FN + LAVA_FN + /* wgsl */ `
fn ndcOf(px: vec2f) -> vec2f {
  let uv = (px + 0.5) / u.res.xy;
  return vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}
fn worldPos(px: vec2f, d: f32) -> vec3f {
  let wh = u.invViewProj * vec4f(ndcOf(px), d, 1.0);
  return wh.xyz / wh.w;
}
fn ign(p: vec2f) -> f32 { return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715)))); }

// Screen-space reflection: march the reflected ray in clip space against the depth buffer,
// refine with a binary search, then fetch last frame's final colour at the reprojected hit.
fn ssr(p: vec3f, r: vec3f, jit: f32) -> vec4f {
  let maxD = 60.0;
  var e = p + r * maxD;
  let h0 = u.viewProj * vec4f(p, 1.0);
  var h1 = u.viewProj * vec4f(e, 1.0);
  if (h1.w < 0.2) {                        // clip the ray in front of the camera
    let t = (h0.w - 0.2) / max(h0.w - h1.w, 1e-4);
    e = p + r * maxD * t;
    h1 = u.viewProj * vec4f(e, 1.0);
  }
  let STEPS = 40;
  var prevT = 0.0;
  var hitT = -1.0;
  for (var i = 1; i <= STEPS; i++) {
    let t = (f32(i) - 1.0 + jit) / f32(STEPS);
    let tt = t * t;                       // more samples near the surface
    let hc = mix(h0, h1, tt);
    let ndc = hc.xy / hc.w;
    if (any(abs(ndc) > vec2f(1.0))) { break; }
    let px = vec2i((vec2f(ndc.x, -ndc.y) * 0.5 + 0.5) * u.res.xy);
    let ds = textureLoad(depthTex, px, 0);
    if (ds <= 0.0) { prevT = tt; continue; }
    let sz = 0.1 / ds;
    let rz = hc.w;
    if (rz > sz + 0.03 && rz - sz < 0.6 + 0.03 * rz) { hitT = tt; break; }
    prevT = tt;
  }
  if (hitT < 0.0) { return vec4f(0.0); }
  var lo = prevT;
  var hi = hitT;
  for (var k = 0; k < 5; k++) {
    let m = 0.5 * (lo + hi);
    let hc = mix(h0, h1, m);
    let ndc = hc.xy / hc.w;
    let px = vec2i((vec2f(ndc.x, -ndc.y) * 0.5 + 0.5) * u.res.xy);
    let ds = textureLoad(depthTex, px, 0);
    if (ds > 0.0 && hc.w > 0.1 / ds) { hi = m; } else { lo = m; }
  }
  let hc = mix(h0, h1, hi);
  let ndc = hc.xy / hc.w;
  let hp = vec2i((vec2f(ndc.x, -ndc.y) * 0.5 + 0.5) * u.res.xy);
  let hn = textureLoad(normalTex, hp, 0).xyz;
  if (dot(hn, r) > 0.1) { return vec4f(0.0); }     // hit a back face
  let wh = worldPos(vec2f(hp), textureLoad(depthTex, hp, 0));
  let pc = u.prevViewProj * vec4f(wh, 1.0);
  if (pc.w <= 0.0) { return vec4f(0.0); }
  let puv = vec2f(pc.x / pc.w * 0.5 + 0.5, 0.5 - pc.y / pc.w * 0.5);
  if (any(puv < vec2f(0.0)) || any(puv > vec2f(1.0))) { return vec4f(0.0); }
  let edge = smoothstep(0.0, 0.08, min(min(puv.x, 1.0 - puv.x), min(puv.y, 1.0 - puv.y)));
  let fade = edge * (1.0 - smoothstep(0.7, 1.0, hi));
  return vec4f(textureSampleLevel(hdrPrev, linSamp, puv, 0.0).rgb, fade);
}

// Tiled light culling (8x8 pixels): the tile's world-space bounds are reduced in shared memory,
// every light is tested once per tile, and pixels only loop over the survivors.
var<workgroup> tMin: array<vec3f, 64>;
var<workgroup> tMax: array<vec3f, 64>;
var<workgroup> tileLights: array<u32, 64>;
var<workgroup> tileCount: atomic<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let dims = vec2u(u.res.xy);
  let inside = gid.x < dims.x && gid.y < dims.y;
  let px = vec2i(min(gid.xy, dims - 1u));
  let cam = u.camPos.xyz;
  let d = textureLoad(depthTex, px, 0);
  let wp = worldPos(vec2f(px), max(d, 1e-6));
  let geo = inside && d > 0.0;
  tMin[li] = select(vec3f(1e9), wp, geo);
  tMax[li] = select(vec3f(-1e9), wp, geo);
  if (li == 0u) { atomicStore(&tileCount, 0u); }
  workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) {
    if (li < st) {
      tMin[li] = min(tMin[li], tMin[li + st]);
      tMax[li] = max(tMax[li], tMax[li + st]);
    }
    workgroupBarrier();
  }
  let bmin = tMin[0];
  let bmax = tMax[0];
  let cnt = u32(u.probeInfo.z);
  for (var i = li; i < cnt; i += 64u) {
    let L = lights[i];
    let q = clamp(L.p, bmin, bmax);
    if (dot(q - L.p, q - L.p) < L.radius * L.radius) {
      let k = atomicAdd(&tileCount, 1u);
      if (k < 64u) { tileLights[k] = i; }
    }
  }
  workgroupBarrier();
  let nTile = min(atomicLoad(&tileCount), 64u);
  if (!inside) { return; }
  if (d <= 0.0) {
    let fp = worldPos(vec2f(px), 1e-6);
    textureStore(hdrOut, px, vec4f(sky(normalize(fp - cam)) * u.rasterInfo.w, 1.0));
    return;
  }
  let nt = textureLoad(normalTex, px, 0);
  let n = normalize(nt.xyz);
  let id = u32(nt.w + 0.5);
  var alb = textureLoad(albedoTex, px, 0).rgb;
  let b = boxes[id];
  let mtl = matOf(b);
  let mt = matParams(mtl);
  let isLava = mtl == M_NEON && b.h.x > 300.0;
  var lava = Lava(vec3f(0.0), vec3f(0.0));
  if (isLava) { lava = lavaSurface(wp.xz, u.camPos.w); alb = lava.alb; }
  let v = normalize(cam - wp);
  let viewZ = 0.1 / d;
  let nv = max(dot(n, v), 1e-4);
  let noise = ign(vec2f(px) + f32(u32(u.sunDir.w) % 64u) * 5.588238);
  let ao = textureLoad(aoTex, px, 0).r;

  // direct: sun with soft shadows + hand-placed lights
  let L = u.sunDir.xyz;
  var col = vec3f(0.0);
  let ndl = dot(n, L);
  if (ndl > 0.0) {
    col += brdf(n, v, L, alb, mt) * SUN_COL * ndl * sunShadow(wp, n, viewZ, noise);
  }
  for (var k = 0u; k < nTile; k++) { col += shadeLight(lights[tileLights[k]], wp, n, v, alb, mt); }

  // indirect diffuse from the DDGI probes, with multi-bounce AO (Jimenez et al. 2016)
  let f0 = mix(vec3f(0.04), alb, mt.metal);
  let Fe = envBRDF(f0, nv, mt.rough);
  let r = reflect(-v, n);
  let ph = probeLookup2(wp, n, v, r, true);
  let irr = ph.irr;
  let ca = 2.0404 * alb - 0.3324;
  let cb = -4.7951 * alb + 0.6417;
  let cc = 2.7552 * alb + 0.6903;
  let aoMB = max(vec3f(ao), ((ao * ca + cb) * ao + cc) * ao);
  col += alb * irr * aoMB * (1.0 - mt.metal) * (1.0 - Fe);

  // indirect specular: SSR over a probe / sky fallback, with specular occlusion
  let skyVis = skyVisibility(wp + n * 0.3, r);
  var env = skyDiffuse(r);
  if (ph.w > 1e-5) { env = mix(ph.refl, env, skyVis); }
  if (mt.rough < 0.12 && skyVis > 0.5) { env = mix(env, sky(r), (skyVis - 0.5) * 2.0); }
  if (u.rasterInfo.y > 0.5 && mt.rough < 0.3) {
    let s = ssr(wp + n * 0.03, r, noise);
    env = mix(env, s.rgb / max(u.rasterInfo.w, 1e-3), s.a * (1.0 - mt.rough / 0.3));
  }
  let specOcc = clamp(pow(nv + ao, exp2(-16.0 * mt.rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
  col += env * Fe * specOcc;

  if (isLava) {
    col += lava.emis;
  } else if (mtl == M_NEON) {
    // emissive parts are light sources: mostly their own glow, only a hint of surface lighting
    col = col * 0.2 + alb * b.emis * 0.95;
  } else {
    col += alb * b.emis;
  }
  col = heightFog(col, wp, cam);
  textureStore(hdrOut, px, vec4f(col * u.rasterInfo.w, 1.0));
}
`;

export const TRANSPARENT_WGSL = COMMON + SKY_FN + BRDF_FN + BOX_VS_FN + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> boxes: array<Box>;
@group(0) @binding(2) var shadowMap: texture_depth_2d_array;
@group(0) @binding(3) var shadowSamp: sampler_comparison;
@group(0) @binding(4) var<storage, read> probeIrr: array<vec4f>;
@group(0) @binding(5) var<storage, read> probeDepth: array<vec2f>;
@group(0) @binding(6) var<storage, read> probeState: array<vec4i>;
@group(0) @binding(7) var<storage, read> lights: array<Light>;
` + PROBE_FN + LIGHTS_FN + SHADOW_FN + FOG_FN + /* wgsl */ `
struct VO {
  @builtin(position) pos: vec4f,
  @location(0) wp: vec3f,
  @location(1) @interpolate(flat) n: vec3f,
  @location(2) @interpolate(flat) id: u32,
};

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let b = boxes[ii];
  let m = matOf(b);
  var o: VO;
  let w = boxCorner(b, vi);
  o.pos = u.viewProj * vec4f(w[0], 1.0);
  if (m != M_GLASS && m != M_GHOST) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); }
  o.wp = w[0];
  o.n = w[1];
  o.id = ii;
  return o;
}

@fragment fn fs(i: VO) -> @location(0) vec4f {
  let cam = u.camPos.xyz;
  let v = normalize(cam - i.wp);
  let n = i.n;
  if (dot(n, v) <= 0.0) { discard; }         // front faces only
  let b = boxes[i.id];
  let ghost = matOf(b) == M_GHOST;
  let tint = select(b.col, mix(b.col, vec3f(1.0), 0.4), ghost);
  let mt = Mat(0.03, 0.0);
  let nv = max(dot(n, v), 1e-4);
  let F = fSchlick(vec3f(0.04), nv).x;
  let viewZ = (u.viewProj * vec4f(i.wp, 1.0)).w;
  let L = u.sunDir.xyz;
  // thin tinted body: lit by sun, lights and probes
  let body = select(0.22, 0.14, ghost);
  let r = reflect(-v, n);
  let ph = probeLookup2(i.wp, n, v, r, true);
  var lit = ph.irr;
  let ndl = dot(n, L);
  if (ndl > 0.0) { lit += SUN_COL * ndl * sunShadow(i.wp, n, viewZ, 0.5); }
  lit += localLights(i.wp, n, v, vec3f(1.0), Mat(0.6, 0.0));
  var c = tint * lit * body;
  // reflection of the environment + the sun glint
  var env = skyDiffuse(r);
  if (ph.w > 1e-5) { env = mix(ph.refl, sky(r), skyVisibility(i.wp + n * 0.3, r)); }
  c += env * F;
  if (ndl > 0.0) {
    let h = normalize(v + L);
    c += SUN_COL * pow(max(dot(n, h), 0.0), 600.0) * 4.0 * F;
  }
  if (ghost) { c += tint * 0.12 * (0.5 + 0.5 * sin(u.camPos.w * 6.0)); }
  let a = clamp(body + F, 0.0, 1.0);
  let fogged = heightFog(c / max(a, 1e-3), i.wp, cam) * a;
  return vec4f(fogged * u.rasterInfo.w, a);
}
`;

// Dual-filter bloom (Jimenez, "Next Generation Post Processing in Call of Duty: AW").
export const BLOOM_DOWN_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> prm: vec4f;   // x = first level (Karis average)

fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn S(uv: vec2f) -> vec3f { return min(textureSampleLevel(src, samp, uv, 0.0).rgb, vec3f(64.0)); }
fn kw(c: vec3f) -> f32 { return 1.0 / (1.0 + lum(c)); }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dd = textureDimensions(dst);
  if (gid.x >= dd.x || gid.y >= dd.y) { return; }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(dd);
  let ts = 1.0 / vec2f(textureDimensions(src));
  let a = S(uv + ts * vec2f(-2.0, -2.0)); let b = S(uv + ts * vec2f(0.0, -2.0)); let c = S(uv + ts * vec2f(2.0, -2.0));
  let d = S(uv + ts * vec2f(-1.0, -1.0)); let e = S(uv + ts * vec2f(1.0, -1.0));
  let f = S(uv + ts * vec2f(-2.0, 0.0)); let g = S(uv); let h = S(uv + ts * vec2f(2.0, 0.0));
  let i = S(uv + ts * vec2f(-1.0, 1.0)); let j = S(uv + ts * vec2f(1.0, 1.0));
  let k = S(uv + ts * vec2f(-2.0, 2.0)); let l = S(uv + ts * vec2f(0.0, 2.0)); let m = S(uv + ts * vec2f(2.0, 2.0));
  var o: vec3f;
  if (prm.x > 0.5) {
    // Karis average per 2x2 group kills fireflies on the first downsample
    let g0 = (d + e + i + j) * 0.25; let g1 = (a + b + f + g) * 0.25; let g2 = (b + c + g + h) * 0.25;
    let g3 = (f + g + k + l) * 0.25; let g4 = (g + h + l + m) * 0.25;
    let w0 = kw(g0) * 0.5; let w1 = kw(g1) * 0.125; let w2 = kw(g2) * 0.125; let w3 = kw(g3) * 0.125; let w4 = kw(g4) * 0.125;
    o = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  } else {
    o = (d + e + i + j) * 0.125 + (a + b + f + g) * 0.03125 + (b + c + g + h) * 0.03125
      + (f + g + k + l) * 0.03125 + (g + h + l + m) * 0.03125;
  }
  textureStore(dst, gid.xy, vec4f(o, 1.0));
}
`;

export const BLOOM_UP_WGSL = /* wgsl */ `
@group(0) @binding(0) var low: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var cur: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dd = textureDimensions(dst);
  if (gid.x >= dd.x || gid.y >= dd.y) { return; }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(dd);
  let ts = 1.0 / vec2f(textureDimensions(low));
  var t = textureSampleLevel(low, samp, uv, 0.0).rgb * 4.0;
  t += (textureSampleLevel(low, samp, uv + vec2f(-ts.x, 0.0), 0.0).rgb + textureSampleLevel(low, samp, uv + vec2f(ts.x, 0.0), 0.0).rgb
      + textureSampleLevel(low, samp, uv + vec2f(0.0, -ts.y), 0.0).rgb + textureSampleLevel(low, samp, uv + vec2f(0.0, ts.y), 0.0).rgb) * 2.0;
  t += textureSampleLevel(low, samp, uv - ts, 0.0).rgb + textureSampleLevel(low, samp, uv + ts, 0.0).rgb
     + textureSampleLevel(low, samp, uv + vec2f(ts.x, -ts.y), 0.0).rgb + textureSampleLevel(low, samp, uv + vec2f(-ts.x, ts.y), 0.0).rgb;
  let c = textureLoad(cur, gid.xy, 0).rgb;
  textureStore(dst, gid.xy, vec4f(c + t / 16.0, 1.0));
}
`;

export const SHADE_WGSL = RT_COMMON + BRDF_FN + /* wgsl */ `
@group(0) @binding(6) var albedoTex: texture_2d<f32>;
@group(0) @binding(7) var rawIn: texture_2d<f32>;
@group(0) @binding(8) var histIn: texture_2d<f32>;
@group(0) @binding(9) var histOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var hdrOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var ptTex: texture_2d<f32>;   // denoised PT illumination (albedo divided out)
@group(0) @binding(12) var<storage, read> probeIrr: array<vec4f>;
@group(0) @binding(13) var<storage, read> probeDepth: array<vec2f>;
@group(0) @binding(14) var<storage, read> probeState: array<vec4i>;
@group(0) @binding(15) var<storage, read> lights: array<Light>;
` + PROBE_FN + LIGHTS_FN + FOG_FN + LAVA_FN + /* wgsl */ `

var<workgroup> tMin: array<vec3f, 64>;
var<workgroup> tMax: array<vec3f, 64>;
var<workgroup> tileLights: array<u32, 64>;
var<workgroup> tileCount: atomic<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let dims = vec2u(u.res.xy);
  let cam = u.camPos.xyz;
  // tiled light culling for the primary hits (same scheme as the raster LIGHT pass)
  {
    let inside = gid.x < dims.x && gid.y < dims.y;
    let tp = vec2i(min(gid.xy, dims - 1u));
    let td = textureLoad(depthTex, tp, 0);
    let twp = worldPos(tp, max(td, 1e-6));
    let geo = inside && td > 0.0;
    tMin[li] = select(vec3f(1e9), twp, geo);
    tMax[li] = select(vec3f(-1e9), twp, geo);
    if (li == 0u) { atomicStore(&tileCount, 0u); }
    workgroupBarrier();
    for (var st = 32u; st > 0u; st >>= 1u) {
      if (li < st) {
        tMin[li] = min(tMin[li], tMin[li + st]);
        tMax[li] = max(tMax[li], tMax[li + st]);
      }
      workgroupBarrier();
    }
    let bmin = tMin[0];
    let bmax = tMax[0];
    let cnt = u32(u.probeInfo.z);
    for (var i = li; i < cnt; i += 64u) {
      let L = lights[i];
      let q = clamp(L.p, bmin, bmax);
      if (dot(q - L.p, q - L.p) < L.radius * L.radius) {
        let k = atomicAdd(&tileCount, 1u);
        if (k < 64u) { tileLights[k] = i; }
      }
    }
    workgroupBarrier();
  }
  let nTile = min(atomicLoad(&tileCount), 64u);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let px = vec2i(gid.xy);
  let rt = u.res.z > 0.5;

  let d = textureLoad(depthTex, px, 0);
  if (d <= 0.0) {
    let np = u.invViewProj * vec4f(ndcOf(px), 1.0, 1.0);
    let rd = normalize(np.xyz / np.w - cam);
    let expo = select(u.rasterInfo.w, 1.0, rt && u.opts.x > 0.5);
    textureStore(hdrOut, px, vec4f(sky(rd) * expo, 1.0));
    textureStore(histOut, px, vec4f(1.0, 1.0, 0.0, 0.0));
    return;
  }

  let wp = worldPos(px, d);
  let nt = textureLoad(normalTex, px, 0);
  let n = normalize(nt.xyz);
  let id = u32(nt.w + 0.5);
  var s = Surf(wp, n, textureLoad(albedoTex, px, 0).rgb, id);
  var rd = normalize(wp - cam);
  let dist = length(wp - cam);
  let viewZ = (u.viewProj * vec4f(wp, 1.0)).w;

  if (rt && u.opts.x > 0.5) {
    // ReSTIR PT Enhanced: primary emission + the denoised illumination re-modulated by albedo
    textureStore(histOut, px, vec4f(1.0, 1.0, viewZ, 1.0));
    let illum = textureLoad(ptTex, px, 0).rgb;
    var col = s.alb * boxes[id].emis + max(s.alb, vec3f(0.03)) * illum;
    let fogPT = 1.0 - exp(-max(dist - 60.0, 0.0) * 0.0035);
    col = mix(col, sky(rd) * 0.95, fogPT);
    textureStore(hdrOut, px, vec4f(col, 1.0));
    return;
  }

  var shadow = select(0.0, 1.0, dot(n, u.sunDir.xyz) > 0.0);
  var ao = 1.0;
  if (rt) {
    let raw = textureLoad(rawIn, px, 0).xy;
    // 3x3 neighbourhood statistics over pixels on the same surface orientation
    var m1 = vec2f(0.0);
    var m2 = vec2f(0.0);
    var wsum = 0.0;
    let maxp = vec2i(dims) - 1;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let q = clamp(px + vec2i(dx, dy), vec2i(0), maxp);
        let nq = textureLoad(normalTex, q, 0).xyz;
        if (dot(nq, n) < 0.8) { continue; }
        let r = textureLoad(rawIn, q, 0).xy;
        m1 += r;
        m2 += r * r;
        wsum += 1.0;
      }
    }
    let mean = m1 / wsum;
    let sigma = sqrt(max(m2 / wsum - mean * mean, vec2f(0.0)));

    var v = raw;
    var cnt = 1.0;
    let pc = u.prevViewProj * vec4f(wp, 1.0);
    if (pc.w > 0.0) {
      let puv = vec2f(pc.x / pc.w * 0.5 + 0.5, 0.5 - pc.y / pc.w * 0.5);
      if (all(puv >= vec2f(0.0)) && all(puv < vec2f(1.0))) {
        let h = textureLoad(histIn, vec2i(puv * u.res.xy), 0);
        if (h.z > 0.0 && abs(h.z - pc.w) < 0.02 * pc.w + 0.05) {
          let hc = clamp(h.xy, mean - 1.25 * sigma, mean + 1.25 * sigma);
          // big correction = the lighting changed here (occluder moved): restart accumulation
          cnt = select(min(h.w + 1.0, 12.0), 2.0, max(abs(hc.x - h.x), abs(hc.y - h.y)) > 0.2);
          v = mix(hc, raw, 1.0 / cnt);
        }
      }
    }
    textureStore(histOut, px, vec4f(v, viewZ, cnt));
    // freshly (re)started pixels lean on the spatial mean to hide 1-sample noise
    let disp = mix(mean, v, clamp((cnt - 1.0) / 4.0, 0.0, 1.0));
    shadow = disp.x;
    ao = disp.y;
  } else {
    textureStore(histOut, px, vec4f(1.0, 1.0, 0.0, 0.0));
  }

  // Hybrid RT: ray-traced soft sun shadow + AO (denoised above), DDGI probes for the bounce
  // light, the hand-placed lights, GGX materials, and a traced specular chain for mirrors,
  // polished floors and glass (instead of screen-space reflections).
  var col = vec3f(0.0);
  var thr = vec3f(1.0);
  var aoS = ao;
  for (var bounce = 0; bounce < 4; bounce++) {
    let b = boxes[s.id];
    let m = matOf(b);
    let mt = matParams(m);
    let v = -rd;
    let nv = max(dot(s.n, v), 1e-4);
    let isLava = m == M_NEON && b.h.x > 300.0;
    var lv = Lava(vec3f(0.0), vec3f(0.0));
    if (isLava) { lv = lavaSurface(s.p.xz, u.camPos.w); s.alb = lv.alb; }
    var kr = 0.0;   // traced reflection weight
    var kt = 0.0;   // traced transmission weight
    var ke = 0.0;   // Fresnel reflection of glass (probe / sky env)
    var tint = vec3f(1.0);
    if (m == M_METAL) { kr = b.refl; tint = s.alb; }
    else if (m == M_POLISHED) { kr = fresnel(nv, b.refl); }
    else if (m == M_GLASS) { let f = fresnel(nv, 0.04); kt = (1.0 - f) * 0.9; ke = f; tint = s.alb; }
    else if (m == M_GHOST) { kt = 0.82; tint = mix(s.alb, vec3f(1.0), 0.6); }

    let refl = reflect(rd, s.n);
    let L = u.sunDir.xyz;
    var lit = vec3f(0.0);
    let ndl = dot(s.n, L);
    if (ndl > 0.0) { lit += brdf(s.n, v, L, s.alb, mt) * SUN_COL * ndl * shadow; }
    if (bounce == 0) {
      for (var k = 0u; k < nTile; k++) { lit += shadeLight(lights[tileLights[k]], s.p, s.n, v, s.alb, mt); }
    } else {
      lit += localLights(s.p, s.n, v, s.alb, mt);
    }
    let ph = probeLookup2(s.p, s.n, v, refl, kr == 0.0 || ke > 0.0);
    let f0 = mix(vec3f(0.04), s.alb, mt.metal);
    let Fe = envBRDF(f0, nv, mt.rough);
    let ca = 2.0404 * s.alb - 0.3324;
    let cb = -4.7951 * s.alb + 0.6417;
    let cc = 2.7552 * s.alb + 0.6903;
    let aoMB = max(vec3f(aoS), ((aoS * ca + cb) * aoS + cc) * aoS);
    lit += s.alb * ph.irr * aoMB * (1.0 - mt.metal) * (1.0 - Fe);
    let skyVis = skyVisibility(s.p + s.n * 0.3, refl);
    var env = skyDiffuse(refl);
    if (ph.w > 1e-5) { env = mix(ph.refl, env, skyVis); }
    if (mt.rough < 0.12 && skyVis > 0.5) { env = mix(env, sky(refl), (skyVis - 0.5) * 2.0); }
    if (kr == 0.0) {   // rough surfaces: prefiltered probe / sky reflection instead of a traced ray
      let specOcc = clamp(pow(nv + aoS, exp2(-16.0 * mt.rough - 1.0)) - 1.0 + aoS, 0.0, 1.0);
      lit += env * Fe * specOcc;
    }
    if (isLava) { lit += lv.emis; }
    else if (m == M_NEON) { lit = lit * 0.2 + s.alb * b.emis * 0.95; }
    else { lit += s.alb * b.emis; }

    col += thr * (lit * max(1.0 - kr - kt - ke, 0.0) + env * ke);
    if (kr + kt < 0.01 || bounce == 3) { break; }

    var nd = refl;
    var org = s.p + s.n * 0.02;
    var k = kr;
    if (kt > 0.0) { nd = rd; org = s.p + rd * 0.01; k = kt; }
    thr *= k * tint;
    let h = trace(org, nd, 1e4, s.id, false);
    if (h.id == NONE) { col += thr * sky(nd); break; }
    s = surfAt(h.id, org + nd * h.t);
    rd = nd;
    shadow = hardShadow(s.p, s.n);
    aoS = 1.0;
  }

  col = heightFog(col, wp, cam);
  textureStore(hdrOut, px, vec4f(col * u.rasterInfo.w, 1.0));
}
`;


// Asynchronous reprojection ("timewarp"): presents the last finished scene frame from the
// *current* camera every display refresh. Backward warp: for each output pixel, a few fixed-point
// iterations find the source pixel whose depth-reconstructed world point lands here (handles
// camera rotation and translation; the sky, at infinite depth, warps as pure rotation).
// DEPTH_DECL / DEPTH_LOAD are patched for a depth32float (same device) or r32float (2nd device) source.
export const WARP_WGSL = /* wgsl */ `
// newVP: current camera; oldInvVP: the source frame's inverse VP;
// back = oldVP * newInvVP (current NDC -> source clip, used to seed the search);
// info.xy = source depth size, info.zw = output size; opt.x = infill on, opt.y = tint holes
struct WU { newVP: mat4x4f, oldInvVP: mat4x4f, back: mat4x4f, info: vec4f, opt: vec4f };
@group(0) @binding(0) var<uniform> wu: WU;
@group(0) @binding(1) var src: texture_2d<f32>;
DEPTH_DECL
@group(0) @binding(3) var samp: sampler;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn depthTexel(q: vec2i) -> f32 {
  let dp = clamp(q, vec2i(0), vec2i(wu.info.xy) - 1);
  return DEPTH_LOAD;
}
// Bilinear depth on continuous surfaces (so the search sees a smooth function on slopes),
// nearest texel across a real depth edge (so layers never blend into a fake middle surface).
fn depthAt(uv: vec2f) -> f32 {
  let f = uv * wu.info.xy - 0.5;
  let q = vec2i(floor(f));
  let t = f - floor(f);
  let a = depthTexel(q); let b = depthTexel(q + vec2i(1, 0));
  let c = depthTexel(q + vec2i(0, 1)); let d = depthTexel(q + vec2i(1, 1));
  let lo = min(min(a, b), min(c, d)); let hi = max(max(a, b), max(c, d));
  if (hi - lo > 0.08 * hi + 1e-5) { return depthTexel(vec2i(uv * wu.info.xy)); }
  return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}
fn ndc(uv: vec2f) -> vec2f { return vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0); }
fn warpTo(s: vec2f) -> vec3f {   // where source uv s lands on screen now (z = 1 if in front)
  let d = depthAt(s);
  let c = wu.newVP * (wu.oldInvVP * vec4f(ndc(s), d, 1.0));
  if (c.w <= 1e-6) { return vec3f(-9.0, -9.0, 0.0); }
  return vec3f(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5, 1.0);
}
fn seed(uv: vec2f, d: f32) -> vec2f {   // where uv would come from if it were at reversed-Z depth d
  let c = wu.back * vec4f(ndc(uv), d, 1.0);
  if (c.w <= 1e-6) { return uv; }
  return vec2f(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
}
// Fixed-point search for the source texel that lands on uv. Returns (s, residual in output px).
fn solve(uv: vec2f, start: vec2f) -> vec3f {
  // Back-projection iteration: assume the depth stored at the current guess and map uv back into
  // the source frame through it. Exact once the guess sits on the right surface; it converges
  // from far larger camera motion than the screen-space update s += uv - warp(s).
  var s = start;
  for (var i = 0; i < 8; i++) {
    let n = clamp(seed(uv, depthAt(clamp(s, vec2f(0.0), vec2f(1.0)))), vec2f(-0.3), vec2f(1.3));
    if (all(abs(n - s) < vec2f(1e-5))) { s = n; break; }
    s = n;
  }
  // a couple of screen-space steps polish the result on sloped or edge texels
  for (var i = 0; i < 2; i++) {
    let w = warpTo(clamp(s, vec2f(0.0), vec2f(1.0)));
    if (w.z == 0.0) { break; }
    s = clamp(s + (uv - w.xy), vec2f(-0.3), vec2f(1.3));
  }
  let w = warpTo(clamp(s, vec2f(0.0), vec2f(1.0)));
  let r = select(1e9, length((w.xy - uv) * wu.info.zw), w.z > 0.0);
  return vec3f(s, r);
}
fn tap(s: vec2f) -> vec3f {
  let size = vec2f(textureDimensions(src));
  return textureSampleLevel(src, samp, clamp(s, vec2f(0.5) / size, vec2f(1.0) - vec2f(0.5) / size), 0.0).rgb;
}
const GOLDEN = 2.39996323;

fn toUv(h: vec4f) -> vec2f { return vec2f(h.x / h.w * 0.5 + 0.5, 0.5 - h.y / h.w * 0.5); }
fn nearestDepth(p: vec2f) -> f32 { return depthTexel(vec2i(clamp(p, vec2f(0.0), vec2f(1.0)) * wu.info.xy)); }

struct Hit { s: vec2f, ok: bool, bg: vec2f, hole: bool };
// Epipolar search. The current pixel's view ray, as a function of its reversed-Z depth d, traces
// a line in the source image: p(d) = toUv(A + d*B). March it from the near end towards infinity
// in equal screen steps; the first place the ray falls behind the stored depth is the visible
// surface. If the stored depth jumps there, the ray slipped behind a foreground object: the
// true surface was never rendered (disocclusion), and the sample before the jump is background.
fn epipolar(uv: vec2f) -> Hit {
  var h: Hit; h.ok = false; h.hole = false;
  let A = wu.back * vec4f(ndc(uv), 0.0, 1.0);
  let B = wu.back[2];
  let eps = 1e-4;
  var dF = 0.0; var dN = 1.0;
  if (A.w <= eps) { if (B.w <= 0.0) { return h; } dF = (eps - A.w) / B.w * 1.01; }
  if (A.w + dN * B.w <= eps) { if (B.w >= 0.0) { return h; } dN = (eps - A.w) / B.w * 0.99; }
  if (dN <= dF) { return h; }
  let H0 = A + dF * B; let H1 = A + dN * B;
  let p0 = toUv(H0); let p1 = toUv(H1);
  // clip the segment to the source image (plus a little margin)
  let dlt = p1 - p0;
  var tA = 0.0; var tB = 1.0;
  for (var k = 0; k < 2; k++) {
    if (abs(dlt[k]) < 1e-7) { if (p0[k] < -0.02 || p0[k] > 1.02) { return h; } continue; }
    let ta = (-0.02 - p0[k]) / dlt[k]; let tb = (1.02 - p0[k]) / dlt[k];
    tA = max(tA, min(ta, tb)); tB = min(tB, max(ta, tb));
  }
  if (tA >= tB) { return h; }
  // projective interpolation: depth of the ray at screen parameter t
  let w0 = H0.w; let w1 = H1.w;
  let N = 40;
  var tPrev = tB; var hit = false; var tHit = tB;
  for (var i = 0; i <= N; i++) {
    let t = mix(tB, tA, f32(i) / f32(N));
    let k0 = (1.0 - t) / w0; let k1 = t / w1;
    let dr = (k0 * dF + k1 * dN) / (k0 + k1);
    if (dr <= nearestDepth(mix(p0, p1, t))) { hit = true; tHit = t; break; }
    tPrev = t;
  }
  if (!hit) { return h; }
  // bisection between the last "in front" and the first "behind" sample
  var lo = tPrev; var hi = tHit;
  if (tHit != tB) {
    for (var i = 0; i < 6; i++) {
      let t = 0.5 * (lo + hi);
      let k0 = (1.0 - t) / w0; let k1 = t / w1;
      let dr = (k0 * dF + k1 * dN) / (k0 + k1);
      if (dr <= depthAt(clamp(mix(p0, p1, t), vec2f(0.0), vec2f(1.0)))) { hi = t; } else { lo = t; }
    }
  }
  let k0 = (1.0 - hi) / w0; let k1 = hi / w1;
  let dr = (k0 * dF + k1 * dN) / (k0 + k1);
  h.s = mix(p0, p1, hi);
  let ds = nearestDepth(h.s);
  h.ok = true;
  h.hole = abs(ds - dr) > 0.12 * max(ds, dr) + 1e-5;
  h.bg = mix(p0, p1, lo);
  return h;
}

@fragment fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let uv = fc.xy / wu.info.zw;
  let infill = wu.opt.x > 0.5;
  var s: vec2f; var hole = false; var bgS = vec2f(0.0); var noData = false;
  if (!infill) {
    s = solve(uv, uv).xy;      // plain fixed-point warp, for comparison
  } else {
    // fast path: converges in a few steps wherever the motion is small
    let f = solve(uv, seed(uv, depthAt(uv)));
    if (f.z < 1.0) { s = f.xy; }
    else {
      let e = epipolar(uv);
      if (e.ok) { s = e.s; hole = e.hole; bgS = e.bg; }
      else { s = seed(uv, 0.0); noData = true; }
    }
  }
  let inside = all(s >= vec2f(0.0)) && all(s <= vec2f(1.0)) && !noData;
  var c = tap(s);

  if (hole) {
    // Disocclusion: fill from the background side with a depth-weighted disc blur so it reads as
    // soft background rather than foreground smeared over the hole.
    let aspect = vec2f(wu.info.w / wu.info.z, 1.0);
    let bgD = nearestDepth(bgS);
    var acc = vec3f(0.0); var wsum = 0.0;
    for (var i = 0; i < 12; i++) {
      let fi = f32(i) + 0.5;
      let a = fi * GOLDEN;
      let p = bgS + vec2f(cos(a), sin(a)) * sqrt(fi / 12.0) * 0.012 * aspect;
      let d = nearestDepth(p);
      let wgt = select(0.05, 1.0, d <= bgD * 1.15 + 1e-5);   // reject foreground taps
      acc += tap(p) * wgt; wsum += wgt;
    }
    c = acc / wsum;
    if (wu.opt.y > 0.5) { c = mix(c, vec3f(1.0, 0.0, 1.0), 0.6); }
  }
  if (!inside) {
    // Beyond the overscan border: the old frame has no data. Infill with a blur that grows with
    // the distance past the edge (no hard smear lines), slightly darkened.
    let o = max(max(-s, s - 1.0), vec2f(0.0));
    let dist = length(o);
    if (infill) {
      let e = clamp(s, vec2f(0.0), vec2f(1.0));
      let r = 0.01 + dist * 0.6;
      var acc = vec3f(0.0);
      for (var i = 0; i < 12; i++) {
        let fi = f32(i) + 0.5;
        let a = fi * GOLDEN;
        acc += tap(e + vec2f(cos(a), sin(a)) * sqrt(fi / 12.0) * r);
      }
      c = acc / 12.0;
      if (wu.opt.y > 0.5) { c = mix(c, vec3f(0.0, 1.0, 1.0), 0.5); }
    }
    c *= 1.0 - clamp(dist * 2.5, 0.0, 0.35);
  }
  return vec4f(c, 1.0);
}
`;
