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
  opts: vec4f,     // x ReSTIR PT Enhanced on
  prevCam: vec4f,  // previous frame camera position
  pair: array<vec4f, 3>,  // per-frame transforms of the 3 pairing textures (flags, offset)
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
const RT_COMMON = COMMON + /* wgsl */ `
struct Node { mn: vec3f, lf: u32, mx: vec3f, cnt: u32 };

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> boxes: array<Box>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> prims: array<u32>;
@group(0) @binding(4) var normalTex: texture_2d<f32>;
@group(0) @binding(5) var depthTex: texture_depth_2d;

const NONE = 0xffffffffu;
const SUN_COL = vec3f(1.0, 0.94, 0.84) * 2.7;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd(s: ptr<function, u32>) -> f32 { *s = pcg(*s); return f32(*s >> 8u) / 16777216.0; }

fn safeInv(d: vec3f) -> vec3f { return 1.0 / select(d, vec3f(1e-9), abs(d) < vec3f(1e-9)); }

fn sky(rd: vec3f) -> vec3f {
  let zen = vec3f(0.18, 0.42, 0.92);
  let hor = vec3f(0.70, 0.83, 0.97);
  var c = mix(hor, zen, pow(clamp(rd.y, 0.0, 1.0), 0.55));
  c = mix(c, vec3f(0.62, 0.5, 0.45), 1.0 - smoothstep(-0.25, 0.0, rd.y));
  let sd = max(dot(rd, u.sunDir.xyz), 0.0);
  c += vec3f(1.0, 0.9, 0.7) * (pow(sd, 1500.0) * 60.0 + pow(sd, 10.0) * 0.3);
  return c;
}

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

fn ambient(n: vec3f) -> vec3f {
  return mix(vec3f(0.45, 0.36, 0.30), vec3f(0.50, 0.64, 0.92), n.y * 0.5 + 0.5) * 0.85;
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

export const SHADE_WGSL = RT_COMMON + /* wgsl */ `
@group(0) @binding(6) var albedoTex: texture_2d<f32>;
@group(0) @binding(7) var rawIn: texture_2d<f32>;
@group(0) @binding(8) var histIn: texture_2d<f32>;
@group(0) @binding(9) var histOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var hdrOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var ptTex: texture_2d<f32>;   // denoised PT illumination (albedo divided out)

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec2u(u.res.xy);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let px = vec2i(gid.xy);
  let rt = u.res.z > 0.5;
  let cam = u.camPos.xyz;

  let d = textureLoad(depthTex, px, 0);
  if (d <= 0.0) {
    let np = u.invViewProj * vec4f(ndcOf(px), 1.0, 1.0);
    let rd = normalize(np.xyz / np.w - cam);
    textureStore(hdrOut, px, vec4f(sky(rd), 1.0));
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

  var amb = ambient(n) * ao;

  // shading + specular chain (mirrors, polished floors, glass)
  var col = vec3f(0.0);
  var thr = vec3f(1.0);
  var ao_s = ao;
  for (var bounce = 0; bounce < 4; bounce++) {
    let b = boxes[s.id];
    let m = matOf(b);
    let cosT = max(dot(-rd, s.n), 0.0);
    var kr = 0.0;   // traced reflection weight
    var kt = 0.0;   // traced transmission weight
    var ke = 0.0;   // cheap sky-only reflection weight
    var tint = vec3f(1.0);
    if (m == M_METAL) { kr = b.refl; tint = s.alb; }
    else if (m == M_POLISHED) { kr = fresnel(cosT, b.refl); }
    else if (m == M_GLASS) { let f = fresnel(cosT, 0.04); kt = (1.0 - f) * 0.9; ke = f; tint = s.alb; }
    else if (m == M_GHOST) { kt = 0.82; tint = mix(s.alb, vec3f(1.0), 0.6); }
    else if (m != M_NEON) { ke = fresnel(cosT, 0.03) * 0.5; }

    let refl = reflect(rd, s.n);
    if (!rt) { ke += kr; kr = 0.0; kt = 0.0; }
    let diffuseW = max(1.0 - kr - kt - ke, 0.0);
    col += thr * (shade(s, shadow, amb) * diffuseW + sky(refl) * ke * (0.35 + 0.65 * ao_s) * shadow * 0.8 + sky(refl) * ke * 0.2);
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
    ao_s = 0.85;
    amb = ambient(s.n) * ao_s;
  }

  let fog = 1.0 - exp(-max(dist - 60.0, 0.0) * 0.0035);
  col = mix(col, sky(normalize(wp - cam)) * 0.95, fog);
  textureStore(hdrOut, px, vec4f(col, 1.0));
}
`;

export const POST_WGSL = COMMON + /* wgsl */ `
@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var<uniform> u: U;
@group(0) @binding(2) var samp: sampler;

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
  c = aces(c);
  let v = uv * (1.0 - uv);
  c *= pow(clamp(v.x * v.y * 16.0, 0.0, 1.0), 0.18);
  c = pow(c, vec3f(1.0 / 2.2));
  c = mix(c, vec3f(1.0, 0.15, 0.1), u.misc.x * 0.45);
  return vec4f(c, 1.0);
}
`;
