//! Obby game core: level, player physics, character rig and a per-frame BVH
//! over oriented boxes. Exposed to JS through a tiny C ABI (no wasm-bindgen).
#![allow(static_mut_refs)]

use std::f32::consts::PI;
use std::ops::{Add, Mul, Neg, Sub};

// ---------------------------------------------------------------- math

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct V3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
const fn v3(x: f32, y: f32, z: f32) -> V3 {
    V3 { x, y, z }
}
impl Add for V3 {
    type Output = V3;
    fn add(self, o: V3) -> V3 {
        v3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for V3 {
    type Output = V3;
    fn sub(self, o: V3) -> V3 {
        v3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f32> for V3 {
    type Output = V3;
    fn mul(self, s: f32) -> V3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
}
impl Neg for V3 {
    type Output = V3;
    fn neg(self) -> V3 {
        v3(-self.x, -self.y, -self.z)
    }
}
impl V3 {
    const Y: V3 = v3(0.0, 1.0, 0.0);
    fn dot(self, o: V3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn cross(self, o: V3) -> V3 {
        v3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
    fn abs(self) -> V3 {
        v3(self.x.abs(), self.y.abs(), self.z.abs())
    }
    fn min(self, o: V3) -> V3 {
        v3(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }
    fn max(self, o: V3) -> V3 {
        v3(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }
    fn get(self, i: usize) -> f32 {
        match i {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
    fn set(&mut self, i: usize, v: f32) {
        match i {
            0 => self.x = v,
            1 => self.y = v,
            _ => self.z = v,
        }
    }
    fn arr(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

type Q = [f32; 4];
const QID: Q = [0.0, 0.0, 0.0, 1.0];
fn qaxis(a: V3, ang: f32) -> Q {
    let s = (ang * 0.5).sin();
    [a.x * s, a.y * s, a.z * s, (ang * 0.5).cos()]
}
fn qmul(a: Q, b: Q) -> Q {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}
fn qconj(q: Q) -> Q {
    [-q[0], -q[1], -q[2], q[3]]
}
fn qrot(q: Q, v: V3) -> V3 {
    let u = v3(q[0], q[1], q[2]);
    let t = u.cross(v) * 2.0;
    v + t * q[3] + u.cross(t)
}

// ---------------------------------------------------------------- GPU layouts

/// Must match `struct Box` in WGSL (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GpuBox {
    c: [f32; 3],
    mat: f32,
    h: [f32; 3],
    emis: f32,
    q: [f32; 4],
    col: [f32; 3],
    refl: f32,
}

/// Must match `struct Node` in WGSL (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Node {
    mn: [f32; 3],
    left_first: u32,
    mx: [f32; 3],
    count: u32,
}

// materials (shared with WGSL)
const PLASTIC: u32 = 0;
const SMOOTH: u32 = 1;
const NEON: u32 = 2;
const GLASS: u32 = 3;
const METAL: u32 = 4;
const POLISHED: u32 = 5;
const GHOST: u32 = 6;
const HEAD: u32 = 7;
const CHECKPOINT: u32 = 8;

// gameplay flags
const SOLID: u32 = 1;
const KILL: u32 = 2;
const BOUNCE: u32 = 4;
const CHECK: u32 = 8;
const WIN: u32 = 16;

// events reported to JS
const EV_CHECKPOINT: u32 = 1;
const EV_DEATH: u32 = 2;
const EV_WIN: u32 = 4;
const EV_JUMP: u32 = 8;
const EV_BOUNCE: u32 = 16;

// Roblox-ish constants (1 unit = 1 stud)
const GRAVITY: f32 = 196.2;
const WALK: f32 = 16.0;
const JUMP_V: f32 = 50.0;
const BOUNCE_V: f32 = 95.0;
const PW: f32 = 0.9; // half width of player hitbox
const PH: f32 = 5.0; // height of player hitbox

#[derive(Clone, Copy)]
enum Motion {
    Static,
    Slide { dir: V3, amp: f32, speed: f32, phase: f32 },
    Spin { axis: V3, speed: f32, phase: f32 },
    Blink { period: f32, on: f32, phase: f32 },
}

struct Part {
    base: V3,
    h: V3,
    col: V3,
    mat: u32,
    emis: f32,
    refl: f32,
    flags: u32,
    motion: Motion,
    stage: i32,
    // runtime
    c: V3,
    q: Q,
    delta: V3,
    active: bool,
    warn: f32,
}

// ---------------------------------------------------------------- level

const GREY: V3 = v3(0.64, 0.64, 0.64);
const LAVA: V3 = v3(1.0, 0.18, 0.08);
const CP_COL: V3 = v3(0.1, 0.45, 0.9);
const GOLD: V3 = v3(1.0, 0.76, 0.32);

fn pal(i: usize) -> V3 {
    const P: [V3; 8] = [
        v3(0.77, 0.16, 0.11), // bright red
        v3(0.85, 0.52, 0.25), // bright orange
        v3(0.96, 0.80, 0.19), // bright yellow
        v3(0.29, 0.75, 0.29), // lime green
        v3(0.07, 0.60, 0.60), // teal
        v3(0.05, 0.41, 0.67), // bright blue
        v3(0.42, 0.22, 0.66), // royal purple
        v3(0.95, 0.40, 0.75), // hot pink
    ];
    P[i % P.len()]
}

struct Builder {
    parts: Vec<Part>,
    stage: i32,
}

impl Builder {
    fn add(&mut self, c: V3, size: V3, col: V3, mat: u32, flags: u32) -> usize {
        let emis = if mat == NEON { 2.2 } else { 0.0 };
        let refl = match mat {
            METAL => 0.85,
            POLISHED => 0.35,
            _ => 0.0,
        };
        self.parts.push(Part {
            base: c,
            h: size * 0.5,
            col,
            mat,
            emis,
            refl,
            flags,
            motion: Motion::Static,
            stage: -1,
            c,
            q: QID,
            delta: V3::default(),
            active: true,
            warn: 0.0,
        });
        self.parts.len() - 1
    }
    /// Platform given its top-center.
    fn plat(&mut self, top: V3, sx: f32, sz: f32, col: V3) -> usize {
        self.add(top - v3(0.0, 0.5, 0.0), v3(sx, 1.0, sz), col, PLASTIC, SOLID)
    }
    /// 12x12 checkpoint platform starting at `near_z`; returns far edge z.
    fn checkpoint(&mut self, near_z: f32, y: f32) -> f32 {
        let c = v3(0.0, y, near_z + 6.0);
        self.plat(c, 12.0, 12.0, GREY);
        self.stage += 1;
        let i = self.add(c + v3(0.0, 0.1, 0.0), v3(5.0, 0.2, 5.0), CP_COL, CHECKPOINT, SOLID | CHECK);
        self.parts[i].stage = self.stage;
        near_z + 12.0
    }
    fn motion(&mut self, i: usize, m: Motion) {
        self.parts[i].motion = m;
    }
}

const FONT: [(char, [&str; 5]); 3] = [
    ('O', [".###.", "#...#", "#...#", "#...#", ".###."]),
    ('B', ["####.", "#...#", "####.", "#...#", "####."]),
    ('Y', ["#...#", ".#.#.", "..#..", "..#..", "..#.."]),
];

fn build_level() -> (Vec<Part>, i32) {
    let mut b = Builder { parts: Vec::new(), stage: 0 };

    // lava sea + spawn baseplate
    let lava = b.add(v3(0.0, -16.0, 220.0), v3(1400.0, 2.0, 1400.0), v3(1.0, 0.32, 0.06), NEON, KILL);
    b.parts[lava].emis = 1.1;
    b.add(v3(0.0, -0.5, 0.0), v3(48.0, 1.0, 48.0), GREY, PLASTIC, SOLID);
    let sp = b.add(v3(0.0, 0.1, 0.0), v3(6.0, 0.2, 6.0), CP_COL, CHECKPOINT, SOLID | CHECK);
    b.parts[sp].stage = 0;

    // spawn showcase: mirror wall, glass cube, polished floor, pillars
    b.add(v3(-20.0, 7.0, -2.0), v3(1.0, 14.0, 26.0), v3(0.92, 0.94, 0.98), METAL, SOLID);
    b.add(v3(-19.0, 0.5, -2.0), v3(1.0, 1.0, 26.0), v3(0.2, 0.2, 0.22), SMOOTH, SOLID);
    b.add(v3(13.0, 3.0, -10.0), v3(6.0, 6.0, 6.0), v3(0.55, 0.85, 1.0), GLASS, SOLID);
    b.add(v3(13.0, 0.05, 8.0), v3(14.0, 0.1, 14.0), v3(0.06, 0.06, 0.08), POLISHED, SOLID);
    for i in 0..5 {
        let h = 2.0 + i as f32 * 2.0;
        b.add(v3(-12.0 + i as f32 * 4.0, h * 0.5, -18.0), v3(2.0, h, 2.0), pal(i), PLASTIC, SOLID);
    }
    let tb = b.add(v3(13.0, 7.0, 8.0), v3(2.0, 2.0, 2.0), GOLD, METAL, 0);
    b.motion(tb, Motion::Spin { axis: v3(0.3, 1.0, 0.2) * (1.0 / 1.063), speed: 1.2, phase: 0.0 });

    // neon "OBBY" sign over the start of the course (reads correctly from spawn)
    let px = 1.4;
    let word = "OBBY";
    let width = (word.len() * 6 - 1) as f32 * px;
    let x0 = width * 0.5;
    for (li, ch) in word.chars().enumerate() {
        let rows = FONT.iter().find(|f| f.0 == ch).unwrap().1;
        for (r, row) in rows.iter().enumerate() {
            for (cc, bit) in row.chars().enumerate() {
                if bit == '#' {
                    let x = x0 - ((li * 6 + cc) as f32) * px;
                    let y = 22.0 - r as f32 * px;
                    b.add(v3(x, y, 34.0), v3(px, px, px), pal(li * 2 + 7), NEON, 0);
                }
            }
        }
    }

    let mut z = 24.0f32;
    let mut y = 0.0f32;

    // 1: hops
    let mut cz = z;
    for i in 0..6 {
        let x = if i % 2 == 0 { -3.0 } else { 3.0 };
        y += 1.0;
        cz = z + 5.0 + 7.0 * i as f32;
        b.plat(v3(x, y, cz), 4.0, 4.0, pal(i));
    }
    z = b.checkpoint(cz + 5.0, y);

    // 2: lava walkway
    let len = 44.0;
    let start = z + 3.0;
    b.plat(v3(0.0, y, start + len * 0.5), 6.0, len, pal(6));
    for k in 0..6 {
        let hgt = if k % 2 == 0 { 0.8 } else { 1.6 };
        b.add(v3(0.0, y + hgt * 0.5, start + 6.0 + 6.0 * k as f32), v3(6.4, hgt, 1.0), LAVA, NEON, KILL);
    }
    z = b.checkpoint(start + len + 3.0, y);

    // 3: sliding platforms
    for i in 0..4 {
        cz = z + 5.5 + 8.0 * i as f32;
        let p = b.plat(v3(0.0, y, cz), 5.0, 5.0, pal(i + 2));
        b.motion(p, Motion::Slide { dir: v3(1.0, 0.0, 0.0), amp: 6.0, speed: 1.4, phase: i as f32 * 1.6 });
    }
    z = b.checkpoint(cz + 5.5, y);

    // 4: spinners
    for k in 0..2 {
        let c = v3(0.0, y, z + 12.0 + 21.0 * k as f32);
        b.plat(c, 18.0, 18.0, pal(4 + k));
        for j in 0..=k {
            let i = b.add(c + v3(0.0, 0.6, 0.0), v3(17.0, 1.0, 1.0), LAVA, NEON, KILL);
            b.motion(i, Motion::Spin { axis: V3::Y, speed: 1.5 + 0.5 * k as f32, phase: j as f32 * PI * 0.5 });
        }
        cz = c.z;
    }
    z = b.checkpoint(cz + 9.0 + 3.0, y);

    // 5: pillars
    let hs = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0];
    let xs = [-2.0, 2.0, -2.0, 2.0, -2.0, 0.0];
    for i in 0..6 {
        cz = z + 4.0 + 6.0 * i as f32;
        let top = y + hs[i];
        b.add(v3(xs[i], top - 15.0, cz), v3(2.0, 30.0, 2.0), pal(i + 1), PLASTIC, SOLID);
    }
    y += 4.0;
    z = b.checkpoint(cz + 4.0, y);

    // 6: blinking platforms
    for i in 0..6 {
        cz = z + 5.5 + 7.5 * i as f32;
        let p = b.plat(v3(0.0, y, cz), 5.0, 5.0, pal(i + 3));
        // a forward-moving wave: each platform vanishes 0.6 s *after* the previous one,
        // so you can always ride the wave (2.2 s solid, 1.0 s gone)
        b.motion(p, Motion::Blink { period: 3.2, on: 0.69, phase: -(i as f32) * 0.6 });
    }
    z = b.checkpoint(cz + 5.5, y);

    // 7: glass bridge between mirror walls
    let start = z;
    for i in 0..7 {
        cz = z + 5.0 + 6.0 * i as f32;
        let x = if i % 2 == 0 { -2.5 } else { 2.5 };
        b.add(v3(x, y - 0.25, cz), v3(4.0, 0.5, 4.0), v3(0.6, 0.9, 1.0), GLASS, SOLID);
    }
    let end = cz + 5.0;
    for s in [-1.0f32, 1.0] {
        b.add(v3(s * 10.0, y + 4.0, (start + end) * 0.5), v3(1.0, 16.0, end - start), v3(0.9, 0.93, 0.97), METAL, SOLID);
    }
    z = b.checkpoint(end, y);

    // 8: jump pad up to a high ledge
    let c = v3(0.0, y, z + 7.0);
    b.plat(c, 8.0, 8.0, pal(1));
    b.add(c + v3(0.0, 0.2, 0.0), v3(4.0, 0.4, 4.0), v3(0.2, 1.0, 0.35), NEON, SOLID | BOUNCE);
    y += 18.0;
    b.plat(v3(0.0, y, c.z + 12.0), 10.0, 10.0, pal(3));
    z = b.checkpoint(c.z + 20.0, y);

    // 9: elevators
    for i in 0..3 {
        cz = z + 5.5 + 8.0 * i as f32;
        let p = b.plat(v3(0.0, y + 3.0 + 4.0 * i as f32, cz), 5.0, 5.0, pal(5 + i));
        b.motion(p, Motion::Slide { dir: V3::Y, amp: 3.5, speed: 1.2, phase: i as f32 });
    }
    y += 13.0;
    z = b.checkpoint(cz + 5.5, y);

    // 10: final gauntlet on a narrow beam
    let start = z + 3.0;
    let len = 40.0;
    b.plat(v3(0.0, y, start + len * 0.5), 3.0, len, pal(0));
    for off in [10.0, 26.0] {
        let i = b.add(v3(0.0, y + 0.6, start + off), v3(7.0, 1.0, 1.0), LAVA, NEON, KILL);
        b.motion(i, Motion::Spin { axis: V3::Y, speed: 2.4, phase: off });
    }
    for off in [18.0, 34.0] {
        let i = b.add(v3(0.0, y + 1.25, start + off), v3(2.5, 2.5, 2.5), LAVA, NEON, KILL);
        b.motion(i, Motion::Slide { dir: v3(1.0, 0.0, 0.0), amp: 3.5, speed: 2.0, phase: off });
    }

    // win platform + trophy + arch
    let c = v3(0.0, y, start + len + 3.0 + 7.0);
    b.stage += 1;
    let w = b.add(c - v3(0.0, 0.5, 0.0), v3(14.0, 1.0, 14.0), GOLD, PLASTIC, SOLID | WIN | CHECK);
    b.parts[w].stage = b.stage;
    let t = c + v3(0.0, 0.0, 4.0);
    b.add(t + v3(0.0, 0.4, 0.0), v3(3.0, 0.8, 3.0), GOLD, METAL, SOLID);
    b.add(t + v3(0.0, 2.3, 0.0), v3(1.0, 3.0, 1.0), GOLD, METAL, SOLID);
    b.add(t + v3(0.0, 5.0, 0.0), v3(3.2, 2.4, 3.2), GOLD, METAL, SOLID);
    for s in [-1.0f32, 1.0] {
        b.add(t + v3(s * 2.0, 5.2, 0.0), v3(0.8, 1.6, 0.6), GOLD, METAL, SOLID);
        b.add(c + v3(s * 6.0, 5.0, -6.0), v3(1.0, 10.0, 1.0), v3(0.95, 0.95, 0.95), PLASTIC, SOLID);
    }
    b.add(c + v3(0.0, 10.5, -6.0), v3(13.0, 1.0, 1.0), GOLD, NEON, 0);

    // clouds
    let mut seed = 1234567u32;
    let mut rnd = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16777216.0
    };
    for _ in 0..16 {
        let cx = (rnd() - 0.5) * 320.0;
        let cy = 70.0 + rnd() * 30.0;
        let czc = -60.0 + rnd() * 520.0;
        for k in 0..3 {
            let s = 8.0 + rnd() * 10.0;
            b.add(
                v3(cx + (k as f32 - 1.0) * s * 0.7, cy + rnd() * 3.0, czc + (rnd() - 0.5) * 8.0),
                v3(s * 1.4, 3.0 + rnd() * 3.0, s),
                v3(1.0, 1.0, 1.0),
                SMOOTH,
                0,
            );
        }
    }

    let stages = b.stage;
    (b.parts, stages)
}

// ---------------------------------------------------------------- game

struct Game {
    parts: Vec<Part>,
    num_stages: i32,
    time: f32,

    pos: V3,
    vel: V3,
    grounded: bool,
    ground: Option<usize>,
    coyote: f32,
    facing: f32,
    walk_phase: f32,
    walk_amt: f32,
    air_amt: f32,

    spawn: V3,
    stage: i32,
    deaths: u32,
    started: bool,
    won: bool,
    run_time: f32,
    events: u32,
    jump_held: bool,

    boxes: Vec<GpuBox>,
    aabbs: Vec<(V3, V3)>,
    nodes: Vec<Node>,
    prims: Vec<u32>,
    out: [f32; 32],
}

impl Game {
    fn new() -> Game {
        let (parts, num_stages) = build_level();
        let mut g = Game {
            parts,
            num_stages,
            time: 0.0,
            pos: V3::default(),
            vel: V3::default(),
            grounded: false,
            ground: None,
            coyote: 0.0,
            facing: 0.0,
            walk_phase: 0.0,
            walk_amt: 0.0,
            air_amt: 0.0,
            spawn: v3(0.0, 0.2, 0.0),
            stage: 0,
            deaths: 0,
            started: false,
            won: false,
            run_time: 0.0,
            events: 0,
            jump_held: false,
            boxes: Vec::with_capacity(1024),
            aabbs: Vec::with_capacity(1024),
            nodes: Vec::with_capacity(2048),
            prims: Vec::with_capacity(1024),
            out: [0.0; 32],
        };
        g.respawn();
        g.update_parts(0.0);
        for p in g.parts.iter_mut() {
            p.delta = V3::default();
        }
        g
    }

    fn respawn(&mut self) {
        self.pos = self.spawn;
        self.vel = V3::default();
        self.ground = None;
        self.grounded = false;
        self.facing = 0.0;
    }

    fn start(&mut self) {
        self.started = true;
    }

    fn restart(&mut self) {
        self.spawn = v3(0.0, 0.2, 0.0);
        self.stage = 0;
        self.deaths = 0;
        self.won = false;
        self.run_time = 0.0;
        self.respawn();
    }

    fn update_parts(&mut self, dt: f32) {
        let t = self.time;
        for p in self.parts.iter_mut() {
            let old = p.c;
            match p.motion {
                Motion::Static => p.c = p.base,
                Motion::Slide { dir, amp, speed, phase } => p.c = p.base + dir * (amp * (t * speed + phase).sin()),
                Motion::Spin { axis, speed, phase } => p.q = qaxis(axis, t * speed + phase),
                Motion::Blink { period, on, phase } => {
                    let ph = ((t + phase) / period).rem_euclid(1.0);
                    p.active = ph < on;
                    // flash during the last 0.6 s before vanishing
                    let left = (on - ph) * period;
                    p.warn = if p.active && left < 0.8 { 0.5 + 0.5 * (t * 30.0).sin() } else { 0.0 };
                }
            }
            p.delta = if dt > 0.0 { p.c - old } else { V3::default() };
        }
    }

    fn solid(p: &Part) -> bool {
        p.active && p.flags & SOLID != 0
    }

    fn overlaps(&self, p: &Part) -> bool {
        let mn = p.c - p.h;
        let mx = p.c + p.h;
        self.pos.x + PW > mn.x
            && self.pos.x - PW < mx.x
            && self.pos.y + PH > mn.y
            && self.pos.y < mx.y
            && self.pos.z + PW > mn.z
            && self.pos.z - PW < mx.z
    }

    fn move_axis(&mut self, axis: usize, amount: f32) {
        let v = self.pos.get(axis) + amount;
        self.pos.set(axis, v);
        for i in 0..self.parts.len() {
            let p = &self.parts[i];
            if !Self::solid(p) || !self.overlaps(p) {
                continue;
            }
            let mn = p.c - p.h;
            let mx = p.c + p.h;
            if axis == 1 {
                if self.pos.y + PH * 0.5 > p.c.y {
                    self.pos.y = mx.y;
                    if self.vel.y <= 0.0 {
                        self.vel.y = 0.0;
                    }
                    self.grounded = true;
                    self.ground = Some(i);
                } else {
                    self.pos.y = mn.y - PH - 1e-4;
                    if self.vel.y > 0.0 {
                        self.vel.y = 0.0;
                    }
                }
            } else {
                // step up small lips (pads, ledges) when not rising
                let step = mx.y - self.pos.y;
                if (0.0..=0.65).contains(&step) && self.vel.y <= 0.0 {
                    self.pos.y = mx.y;
                    continue;
                }
                if self.pos.get(axis) > p.c.get(axis) {
                    self.pos.set(axis, mx.get(axis) + PW + 1e-4);
                } else {
                    self.pos.set(axis, mn.get(axis) - PW - 1e-4);
                }
            }
        }
    }

    fn touching_kill(&self) -> bool {
        for p in &self.parts {
            if !p.active || p.flags & KILL == 0 {
                continue;
            }
            let qi = qconj(p.q);
            for k in 0..3 {
                let s = self.pos + v3(0.0, 0.9 + 1.6 * k as f32, 0.0);
                let l = qrot(qi, s - p.c);
                let cl = l.max(-p.h).min(p.h);
                let d = l - cl;
                if d.dot(d) < 0.85 * 0.85 {
                    return true;
                }
            }
        }
        self.pos.y < -40.0
    }

    fn frame(&mut self, dt: f32, mx: f32, mz: f32, jump: bool, yaw: f32, pitch: f32, dist: f32) {
        let dt = dt.clamp(0.0, 0.05);
        self.events = 0;
        self.time += dt;
        if self.started && !self.won {
            self.run_time += dt;
        }
        self.update_parts(dt);

        // ride whatever we were standing on
        if let Some(g) = self.ground {
            let p = &self.parts[g];
            if p.active {
                self.pos = self.pos + p.delta;
            }
        }

        // input relative to camera
        let fwd = v3(-yaw.sin(), 0.0, -yaw.cos());
        let right = v3(yaw.cos(), 0.0, -yaw.sin());
        let mut wish = fwd * mz + right * mx;
        let wl = wish.len();
        if wl > 1.0 {
            wish = wish * (1.0 / wl);
        }
        self.vel.x = wish.x * WALK;
        self.vel.z = wish.z * WALK;
        if wl > 0.05 {
            let target = wish.x.atan2(wish.z);
            let mut d = target - self.facing;
            while d > PI {
                d -= 2.0 * PI;
            }
            while d < -PI {
                d += 2.0 * PI;
            }
            self.facing += d * (dt * 14.0).min(1.0);
        }

        if self.grounded {
            self.coyote = 0.1;
        } else {
            self.coyote -= dt;
        }
        if jump && self.coyote > 0.0 && self.vel.y <= 0.0 {
            self.vel.y = JUMP_V;
            self.coyote = 0.0;
            self.events |= EV_JUMP;
        }
        self.jump_held = jump;
        self.vel.y -= GRAVITY * dt;

        // integrate with substeps
        let speed = self.vel.len() * dt;
        let n = ((speed / 0.4).ceil() as i32).clamp(1, 12);
        let sdt = dt / n as f32;
        self.grounded = false;
        self.ground = None;
        for _ in 0..n {
            self.move_axis(0, self.vel.x * sdt);
            self.move_axis(2, self.vel.z * sdt);
            self.move_axis(1, self.vel.y * sdt);
        }

        // ground interactions
        if let Some(g) = self.ground {
            let (flags, stage, top) = {
                let p = &self.parts[g];
                (p.flags, p.stage, p.c + v3(0.0, p.h.y, 0.0))
            };
            if flags & BOUNCE != 0 {
                self.vel.y = BOUNCE_V;
                self.grounded = false;
                self.ground = None;
                self.events |= EV_BOUNCE;
            }
            if flags & CHECK != 0 && stage > self.stage {
                self.stage = stage;
                self.spawn = top;
                self.events |= EV_CHECKPOINT;
            }
            if flags & WIN != 0 && !self.won {
                self.won = true;
                self.events |= EV_WIN;
            }
        }

        if self.touching_kill() {
            self.deaths += 1;
            self.events |= EV_DEATH;
            self.respawn();
        }

        // animation
        let hs = (self.vel.x * self.vel.x + self.vel.z * self.vel.z).sqrt();
        let k = (dt * 10.0).min(1.0);
        let walk_t = if self.grounded && hs > 1.0 { 1.0 } else { 0.0 };
        let air_t = if self.grounded { 0.0 } else { 1.0 };
        self.walk_amt += (walk_t - self.walk_amt) * k;
        self.air_amt += (air_t - self.air_amt) * k;
        self.walk_phase += dt * 10.0 * (hs / WALK).max(0.2);

        self.build_boxes();
        self.build_bvh();

        // camera with collision
        let target = self.pos + v3(0.0, 4.5, 0.0);
        let dir = v3(yaw.sin() * pitch.cos(), pitch.sin(), yaw.cos() * pitch.cos());
        let hit = self.raycast_solid(target, dir, dist + 0.5);
        let d = (hit - 0.5).min(dist).max(0.3);
        let eye = target + dir * d;

        let o = &mut self.out;
        o[0..3].copy_from_slice(&eye.arr());
        o[3..6].copy_from_slice(&target.arr());
        o[6..9].copy_from_slice(&self.pos.arr());
        o[9] = self.stage as f32;
        o[10] = self.num_stages as f32;
        o[11] = self.deaths as f32;
        o[12] = self.run_time;
        o[13] = self.events as f32;
        o[14] = if self.won { 1.0 } else { 0.0 };
        o[15] = if self.grounded { 1.0 } else { 0.0 };
        o[16] = self.facing;
    }

    fn raycast_solid(&self, ro: V3, rd: V3, tmax: f32) -> f32 {
        let mut best = tmax;
        for p in &self.parts {
            if !Self::solid(p) || p.mat == GLASS || p.q != QID {
                continue;
            }
            let mut tn = -1e30f32;
            let mut tf = 1e30f32;
            let mut miss = false;
            for a in 0..3 {
                let o = ro.get(a);
                let d = rd.get(a);
                let lo = p.c.get(a) - p.h.get(a);
                let hi = p.c.get(a) + p.h.get(a);
                if d.abs() < 1e-8 {
                    if o < lo || o > hi {
                        miss = true;
                        break;
                    }
                } else {
                    let t0 = (lo - o) / d;
                    let t1 = (hi - o) / d;
                    tn = tn.max(t0.min(t1));
                    tf = tf.min(t0.max(t1));
                }
            }
            if !miss && tn <= tf && tf > 0.0 && tn > 0.0 && tn < best {
                best = tn;
            }
        }
        best
    }

    fn build_boxes(&mut self) {
        self.boxes.clear();
        for p in &self.parts {
            let mut mat = p.mat;
            let mut emis = p.emis;
            if !p.active {
                mat = GHOST;
            }
            if p.warn > 0.0 {
                emis += p.warn * 0.8;
            }
            self.boxes.push(GpuBox {
                c: p.c.arr(),
                mat: mat as f32,
                h: p.h.arr(),
                emis,
                q: p.q,
                col: p.col.arr(),
                refl: p.refl,
            });
        }

        // R6-style character
        let qy = qaxis(V3::Y, self.facing);
        let swing = self.walk_phase.sin() * 0.9 * self.walk_amt;
        let air = self.air_amt;
        let yellow = v3(0.96, 0.8, 0.19);
        let blue = v3(0.05, 0.41, 0.67);
        let green = v3(0.64, 0.77, 0.29);
        let z = V3::default();
        let rig: [(V3, V3, V3, f32, V3, u32); 6] = [
            (v3(0.0, 3.0, 0.0), z, v3(1.0, 1.0, 0.5), 0.0, blue, SMOOTH),
            (v3(0.0, 4.6, 0.0), z, v3(0.6, 0.6, 0.6), 0.0, yellow, HEAD),
            (v3(1.5, 3.75, 0.0), v3(0.0, -0.75, 0.0), v3(0.5, 1.0, 0.5), -swing - air * 2.6, yellow, SMOOTH),
            (v3(-1.5, 3.75, 0.0), v3(0.0, -0.75, 0.0), v3(0.5, 1.0, 0.5), swing - air * 2.6, yellow, SMOOTH),
            (v3(0.5, 2.0, 0.0), v3(0.0, -1.0, 0.0), v3(0.5, 1.0, 0.5), swing - air * 0.35, green, SMOOTH),
            (v3(-0.5, 2.0, 0.0), v3(0.0, -1.0, 0.0), v3(0.5, 1.0, 0.5), -swing + air * 0.35, green, SMOOTH),
        ];
        for (pivot, off, h, ang, col, mat) in rig {
            let qx = qaxis(v3(1.0, 0.0, 0.0), ang);
            let c = self.pos + qrot(qy, pivot + qrot(qx, off));
            self.boxes.push(GpuBox {
                c: c.arr(),
                mat: mat as f32,
                h: h.arr(),
                emis: 0.0,
                q: qmul(qy, qx),
                col: col.arr(),
                refl: 0.0,
            });
        }
    }

    fn build_bvh(&mut self) {
        self.aabbs.clear();
        for b in &self.boxes {
            let c = v3(b.c[0], b.c[1], b.c[2]);
            let e = qrot(b.q, v3(b.h[0], 0.0, 0.0)).abs()
                + qrot(b.q, v3(0.0, b.h[1], 0.0)).abs()
                + qrot(b.q, v3(0.0, 0.0, b.h[2])).abs();
            self.aabbs.push((c - e, c + e));
        }
        let n = self.boxes.len();
        // huge boxes (the lava sea) go into their own leaf so they don't bloat the tree
        let (mut big, rest): (Vec<u32>, Vec<u32>) = (0..n as u32).partition(|&i| {
            let (a, b) = self.aabbs[i as usize];
            let s = b - a;
            s.x.max(s.z) > 150.0
        });
        self.prims.clear();
        self.nodes.clear();
        self.nodes.push(Node::default());
        if big.is_empty() {
            self.prims.extend(rest);
            self.subdivide(0, 0, n);
        } else {
            let nb = big.len();
            self.prims.append(&mut big);
            self.prims.extend(rest);
            self.nodes.push(Node::default());
            self.nodes.push(Node::default());
            self.nodes[0].left_first = 1;
            self.subdivide(1, 0, nb);
            self.nodes[1].left_first = 0;
            self.nodes[1].count = nb as u32; // force leaf
            self.subdivide(2, nb, n - nb);
            let (a, b) = (self.nodes[1], self.nodes[2]);
            let mn = v3(a.mn[0], a.mn[1], a.mn[2]).min(v3(b.mn[0], b.mn[1], b.mn[2]));
            let mx = v3(a.mx[0], a.mx[1], a.mx[2]).max(v3(b.mx[0], b.mx[1], b.mx[2]));
            self.nodes[0].mn = mn.arr();
            self.nodes[0].mx = mx.arr();
        }
    }

    fn subdivide(&mut self, ni: usize, first: usize, count: usize) {
        let mut mn = v3(1e30, 1e30, 1e30);
        let mut mx = -mn;
        let mut cmn = mn;
        let mut cmx = mx;
        for &p in &self.prims[first..first + count] {
            let (a, b) = self.aabbs[p as usize];
            mn = mn.min(a);
            mx = mx.max(b);
            let c = (a + b) * 0.5;
            cmn = cmn.min(c);
            cmx = cmx.max(c);
        }
        self.nodes[ni].mn = mn.arr();
        self.nodes[ni].mx = mx.arr();
        if count <= 2 {
            self.nodes[ni].left_first = first as u32;
            self.nodes[ni].count = count as u32;
            return;
        }
        let ext = cmx - cmn;
        let axis = if ext.x > ext.y && ext.x > ext.z { 0 } else if ext.y > ext.z { 1 } else { 2 };
        let aabbs = &self.aabbs;
        self.prims[first..first + count].sort_by(|&a, &b| {
            let ca = aabbs[a as usize].0.get(axis) + aabbs[a as usize].1.get(axis);
            let cb = aabbs[b as usize].0.get(axis) + aabbs[b as usize].1.get(axis);
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mid = count / 2;
        let left = self.nodes.len();
        self.nodes.push(Node::default());
        self.nodes.push(Node::default());
        self.nodes[ni].left_first = left as u32;
        self.nodes[ni].count = 0;
        self.subdivide(left, first, mid);
        self.subdivide(left + 1, first + mid, count - mid);
    }
}

// ---------------------------------------------------------------- C ABI

static mut GAME: Option<Game> = None;

fn g() -> &'static mut Game {
    unsafe { GAME.get_or_insert_with(Game::new) }
}

#[no_mangle]
pub extern "C" fn init() {
    unsafe { GAME = Some(Game::new()) };
}

/// Begin timing the run after the title screen is dismissed.
#[no_mangle]
pub extern "C" fn start() {
    g().start();
}
#[no_mangle]
pub extern "C" fn frame(dt: f32, mx: f32, mz: f32, jump: u32, yaw: f32, pitch: f32, dist: f32) {
    g().frame(dt, mx, mz, jump != 0, yaw, pitch, dist);
}
#[no_mangle]
pub extern "C" fn respawn() {
    g().respawn();
}
#[no_mangle]
pub extern "C" fn restart() {
    g().restart();
}
#[no_mangle]
pub extern "C" fn out_ptr() -> *const f32 {
    g().out.as_ptr()
}
#[no_mangle]
pub extern "C" fn boxes_ptr() -> *const GpuBox {
    g().boxes.as_ptr()
}
#[no_mangle]
pub extern "C" fn box_count() -> u32 {
    g().boxes.len() as u32
}
#[no_mangle]
pub extern "C" fn nodes_ptr() -> *const Node {
    g().nodes.as_ptr()
}
#[no_mangle]
pub extern "C" fn node_count() -> u32 {
    g().nodes.len() as u32
}
#[no_mangle]
pub extern "C" fn prims_ptr() -> *const u32 {
    g().prims.as_ptr()
}
#[no_mangle]
pub extern "C" fn prim_count() -> u32 {
    g().prims.len() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idle_frame(game: &mut Game) {
        game.frame(1.0 / 60.0, 0.0, 0.0, false, PI, 0.3, 18.0);
    }

    #[test]
    fn title_screen_animates_without_timing_the_run() {
        let mut game = Game::new();
        for _ in 0..60 {
            idle_frame(&mut game);
        }
        assert!(game.time > 0.9);
        assert!(!game.boxes.is_empty());
        assert_eq!(game.out[12], 0.0);

        game.start();
        idle_frame(&mut game);
        assert!((game.out[12] - 1.0 / 60.0).abs() < 1e-6);
    }

    #[test]
    fn starting_again_and_respawning_preserve_elapsed_time() {
        let mut game = Game::new();
        game.start();
        idle_frame(&mut game);
        let elapsed = game.out[12];
        game.start();
        game.respawn();
        idle_frame(&mut game);
        assert!((game.out[12] - elapsed * 2.0).abs() < 1e-6);
    }

    #[test]
    fn restart_before_play_keeps_the_timer_stopped() {
        let mut game = Game::new();
        game.restart();
        idle_frame(&mut game);
        assert_eq!(game.out[12], 0.0);
    }

    #[test]
    fn winning_stops_the_timer_and_restart_begins_a_new_run() {
        let mut game = Game::new();
        game.start();
        idle_frame(&mut game);
        let elapsed = game.out[12];
        game.won = true;
        idle_frame(&mut game);
        assert_eq!(game.out[12], elapsed);

        game.restart();
        assert_eq!(game.run_time, 0.0);
        idle_frame(&mut game);
        assert!((game.out[12] - 1.0 / 60.0).abs() < 1e-6);
    }
}
