// WGSL for the hybrid renderer:
//   1. RASTER  – instanced oriented boxes -> G-buffer (albedo, normal+id, depth)
//   2a. TRACE  – compute: jittered sun-shadow + AO ray per pixel (noisy, 1 spp)
//   2b. SHADE  – compute: clamped temporal accumulation, shading, traced
//                mirror/metal reflections and glass transmission
//   3. POST    – ACES tonemap, vignette, damage flash

const COMMON = /* wgsl */ `
struct U {
  viewProj: mat4x4f,
  invViewProj: mat4x4f,
  prevViewProj: mat4x4f,
  camPos: vec4f,   // xyz camera, w time
  sunDir: vec4f,   // xyz to-sun, w frame
  res: vec4f,      // w, h, rt on, box count
  misc: vec4f,     // x flash, y exposure, zw canvas size
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

fn shade(s: Surf, shadow: f32, ao: f32) -> vec3f {
  let b = boxes[s.id];
  let ndl = max(dot(s.n, u.sunDir.xyz), 0.0);
  return s.alb * (SUN_COL * ndl * shadow + ambient(s.n) * ao + b.emis);
}

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

// Pass 2b: temporal accumulation with neighbourhood (variance) clamping, then shading,
// reflections and glass. The clamp is what keeps moving shadows (the player, spinners,
// sliding platforms) from ghosting: stale history outside the current 3x3 range is rejected.
export const SHADE_WGSL = RT_COMMON + /* wgsl */ `
@group(0) @binding(6) var albedoTex: texture_2d<f32>;
@group(0) @binding(7) var rawIn: texture_2d<f32>;
@group(0) @binding(8) var histIn: texture_2d<f32>;
@group(0) @binding(9) var histOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var hdrOut: texture_storage_2d<rgba16float, write>;

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
    col += thr * (shade(s, shadow, ao_s) * diffuseW + sky(refl) * ke * (0.35 + 0.65 * ao_s) * shadow * 0.8 + sky(refl) * ke * 0.2);
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
