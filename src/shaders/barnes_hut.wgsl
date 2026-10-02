// GPU Barnes-Hut N-body solver.
//
// One step = Morton-code the particles, radix-sort them into spatial order,
// build a binary tree bottom-up over the sorted order, then traverse it per
// particle with the opening criterion s/d < theta. All data stays on the GPU.
//
// Tree layout: leaves are the sorted particles (implicit). Internal nodes are
// stored level by level: level r has ceil(n_{r-1}/2) nodes, node (r,i) covers
// leaves [i*2^r, min((i+1)*2^r, N)). Children of (r,i) are (r-1,2i) and
// (r-1,2i+1); level-1 nodes have leaf children. A node stores its center of
// mass, total mass, a conservative cell size from the Morton prefix, and its
// two children (negative child = leaf, encoded as -(leaf+1)).

struct Particle {
    pos: array<f32, 3>,
    vel: array<f32, 3>,
    acc: array<f32, 3>,
    mass: f32,
    galaxy_id: u32,
};

struct SimParams {
    dt: f32,
    g: f32,
    e: f32,
    central_mass: f32,
    num_particles: u32,
    particles_per_group: u32,
    triangle_size: f32,
    num_galaxies: u32,
    distance_between_galaxies: f32,
    galaxy_velocity: f32,
    halo_v: f32,
    halo_r: f32,
    damping: f32,
    time: f32,
    theta: f32,
};

struct BHNode {
    com: vec3<f32>,
    mass: f32,
    bb_min: vec3<f32>,
    bb_max: vec3<f32>,
    left: i32,
    right: i32,
    leaf_lo: u32,
    leaf_cnt: u32,
};

struct Center {
    pos: vec3<f32>,
    _p1: f32,
    vel: vec3<f32>,
    _p2: f32,
    mass: f32,
    galaxy_id: u32,
    _p3: vec2<f32>,
};

struct RoundParams {
    level: u32,
    src_off: u32,
    dst_off: u32,
    dst_cnt: u32,
    src_cnt: u32,
    n: u32,
};

// Per-pass radix digit, set via pipeline override constants.
override SHIFT: u32 = 0u;

// Morton codes live in [-4, 4]^3; particles outside are clamped.
const BH_HALF_EXTENT: f32 = 4.0;
const BH_MORTON_BITS: f32 = 1023.0;

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> p_src: array<Particle>;
@group(0) @binding(2) var<storage, read_write> p_tmp: array<Particle>;
@group(0) @binding(3) var<storage, read_write> p_dst: array<Particle>;
@group(0) @binding(4) var<storage, read_write> keys_a: array<u32>;
@group(0) @binding(5) var<storage, read_write> vals_a: array<u32>;
@group(0) @binding(6) var<storage, read_write> nodes: array<BHNode>;
@group(0) @binding(7) var<storage, read> centers_src: array<Center>;
@group(0) @binding(8) var<storage, read_write> centers_dst: array<Center>;
@group(0) @binding(9) var<uniform> round: RoundParams;

@group(1) @binding(0) var<storage, read_write> sk_src: array<u32>;
@group(1) @binding(1) var<storage, read_write> sk_dst: array<u32>;
@group(1) @binding(2) var<storage, read_write> sv_src: array<u32>;
@group(1) @binding(3) var<storage, read_write> sv_dst: array<u32>;
@group(1) @binding(4) var<storage, read_write> hist: array<atomic<u32>, 256>;
@group(1) @binding(5) var<storage, read_write> offsets: array<u32>;
@group(1) @binding(6) var<storage, read_write> cursors: array<atomic<u32>, 256>;

fn expand_bits(v: u32) -> u32 {
    var x = v & 0x3ffu;
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x << 8u)) & 0x0300f00fu;
    x = (x | (x << 4u)) & 0x030c30c3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}

fn morton_code(p: vec3<f32>) -> u32 {
    let ix = u32(clamp((p.x + BH_HALF_EXTENT) * 128.0, 0.0, BH_MORTON_BITS));
    let iy = u32(clamp((p.y + BH_HALF_EXTENT) * 128.0, 0.0, BH_MORTON_BITS));
    let iz = u32(clamp((p.z + BH_HALF_EXTENT) * 128.0, 0.0, BH_MORTON_BITS));
    return expand_bits(ix) | (expand_bits(iy) << 1u) | (expand_bits(iz) << 2u);
}

// 30-bit Morton codes + initial sort keys/values.
@compute @workgroup_size(256)
fn morton_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = arrayLength(&p_src);
    let i = gid.x;
    if (i >= n) { return; }
    let p = vec3<f32>(p_src[i].pos[0], p_src[i].pos[1], p_src[i].pos[2]);
    keys_a[i] = morton_code(p);
    vals_a[i] = i;
}

@compute @workgroup_size(256)
fn sort_zero_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < 256u) { atomicStore(&hist[gid.x], 0u); }
}

@compute @workgroup_size(256)
fn sort_hist_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = arrayLength(&sk_src);
    let total = nwg.x * 256u;
    for (var i = gid.x; i < n; i += total) {
        let bin = (sk_src[i] >> SHIFT) & 0xffu;
        atomicAdd(&hist[bin], 1u);
    }
}

var<workgroup> scan_mem: array<u32, 256>;

@compute @workgroup_size(256)
fn sort_scan_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    scan_mem[lid.x] = atomicLoad(&hist[lid.x]);
    workgroupBarrier();
    if (lid.x == 0u) {
        var acc = 0u;
        for (var i = 0u; i < 256u; i++) {
            let v = scan_mem[i];
            scan_mem[i] = acc;
            acc += v;
        }
    }
    workgroupBarrier();
    atomicStore(&cursors[lid.x], scan_mem[lid.x]);
}

@compute @workgroup_size(256)
fn sort_scatter_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = arrayLength(&sk_src);
    let total = nwg.x * 256u;
    for (var i = gid.x; i < n; i += total) {
        let k = sk_src[i];
        let bin = (k >> SHIFT) & 0xffu;
        let dst = atomicAdd(&cursors[bin], 1u);
        sk_dst[dst] = k;
        sv_dst[dst] = sv_src[i];
    }
}

// Reorder particles into Morton order; leaves are the sorted array.
@compute @workgroup_size(256)
fn gather_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = arrayLength(&vals_a);
    let i = gid.x;
    if (i >= n) { return; }
    p_tmp[i] = p_src[vals_a[i]];
}

// One thread per internal node: children + conservative cell size.
// Node (r,i) covers leaves [i*2^r, min((i+1)*2^r, N)); levels are laid out
// back to back, so the global index implies (r,i) via the level loop below.
@compute @workgroup_size(256)
fn build_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = arrayLength(&keys_a);
    // Internal node count = sum_{r>=1} ceil(n / 2^r); exceeds n-1 when n is
    // not a power of two because single-child nodes are allowed.
    var total = 0u;
    var c = (n + 1u) / 2u;
    loop {
        total += c;
        if (c == 1u) { break; }
        c = (c + 1u) / 2u;
    }
    let idx = gid.x;
    if (idx >= total) { return; }
    var r = 1u;
    var off = 0u;
    var cnt = (n + 1u) / 2u;
    var off_prev = 0u;
    var cnt_prev = n;
    while (idx >= off + cnt) {
        off_prev = off;
        cnt_prev = cnt;
        off += cnt;
        cnt = (cnt + 1u) / 2u;
        r += 1u;
    }
    let i = idx - off;
    let span = 1u << r;
    let L = i * span;
    var R = L + span;
    if (R > n) { R = n; }

    let c0 = 2u * i;
    let has_c1 = (2u * i + 1u) < cnt_prev;
    var left: i32;
    var right: i32;
    if (r == 1u) {
        left = -i32(L + 1u);
        if (has_c1) {
            right = -i32(L + 2u);
        } else {
            right = left;
        }
    } else {
        left = i32(off_prev + c0);
        if (has_c1) {
            right = i32(off_prev + c0 + 1u);
        } else {
            right = left;
        }
    }

    nodes[idx].left = left;
    nodes[idx].right = right;
    nodes[idx].leaf_lo = L;
    nodes[idx].leaf_cnt = R - L;
}

// One round of the bottom-up pass: level r from level r-1 (or leaves).
// Computes mass, center of mass, and the tight bounding box; the opening
// size is the box's largest extent.
@compute @workgroup_size(256)
fn accumulate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= round.dst_cnt) { return; }
    var m = 0.0;
    var com = vec3<f32>(0.0, 0.0, 0.0);
    var bb_min = vec3<f32>(1e30, 1e30, 1e30);
    var bb_max = vec3<f32>(-1e30, -1e30, -1e30);
    if (round.level == 1u) {
        for (var k = 0u; k < 2u; k++) {
            let leaf = 2u * i + k;
            if (leaf >= round.n) { break; }
            let p = p_tmp[leaf];
            let pp = vec3<f32>(p.pos[0], p.pos[1], p.pos[2]);
            m += p.mass;
            com += p.mass * pp;
            bb_min = min(bb_min, pp);
            bb_max = max(bb_max, pp);
        }
    } else {
        let c0 = round.src_off + 2u * i;
        let n0 = nodes[c0];
        m += n0.mass;
        com += n0.mass * n0.com;
        bb_min = min(bb_min, n0.bb_min);
        bb_max = max(bb_max, n0.bb_max);
        if (2u * i + 1u < round.src_cnt) {
            let n1 = nodes[c0 + 1u];
            m += n1.mass;
            com += n1.mass * n1.com;
            bb_min = min(bb_min, n1.bb_min);
            bb_max = max(bb_max, n1.bb_max);
        }
    }
    let o = round.dst_off + i;
    nodes[o].mass = m;
    if (m > 0.0) {
        nodes[o].com = com / m;
    } else {
        nodes[o].com = vec3<f32>(0.0, 0.0, 0.0);
    }
    nodes[o].bb_min = bb_min;
    nodes[o].bb_max = bb_max;
}

// Tree traversal with s/d < theta, then halo + friction + leapfrog integrate.
@compute @workgroup_size(256)
fn traverse_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = arrayLength(&vals_a);
    if (arrayLength(&nodes) == 0u) { return; }
    let p = gid.x;
    if (p >= n) { return; }
    let part = p_tmp[p];
    var pos = vec3<f32>(part.pos[0], part.pos[1], part.pos[2]);
    var vel = vec3<f32>(part.vel[0], part.vel[1], part.vel[2]);
    let acc_old = vec3<f32>(part.acc[0], part.acc[1], part.acc[2]);
    let mass = part.mass;
    let galaxy = part.galaxy_id;
    let is_center = mass == params.central_mass;

    // Leapfrog: kick with the previous acceleration, then drift.
    vel += acc_old * params.dt * 0.5;
    pos += vel * params.dt;

    var acc = vec3<f32>(0.0, 0.0, 0.0);
    let theta_sq = params.theta * params.theta;
    let root_idx = i32(arrayLength(&nodes)) - 1;

    var stack: array<i32, 64>;
    stack[0] = root_idx;
    var sp = 1u;
    while (sp > 0u) {
        sp -= 1u;
        let ni = stack[sp];
        let node = nodes[ni];
        let d = node.com - pos;
        let dist_sq = dot(d, d) + params.e;
        // Never approximate a node containing this particle: its mass
        // includes the particle itself, which would be a spurious self-force.
        let contains = p >= node.leaf_lo && p < node.leaf_lo + node.leaf_cnt;
        let ext = node.bb_max - node.bb_min;
        let size = max(ext.x, max(ext.y, ext.z));
        if (!contains && size * size < theta_sq * dist_sq) {
            let inv = params.g * node.mass / (dist_sq * sqrt(dist_sq));
            acc += inv * d;
        } else {
            let l = node.left;
            let r = node.right;
            if (r < 0) {
                let q = u32(-r - 1);
                if (q != p) {
                    let op = p_tmp[q];
                    let od = vec3<f32>(op.pos[0], op.pos[1], op.pos[2]) - pos;
                    let ods = dot(od, od);
                    if (ods > 1e-12) {
                        let odss = ods + params.e;
                        acc += params.g * op.mass * od / (odss * sqrt(odss));
                    }
                }
            } else if (r != l) {
                stack[sp] = r;
                sp += 1u;
            } else {
                stack[sp] = l;
                sp += 1u;
            }
            if (l < 0) {
                let q = u32(-l - 1);
                if (q != p) {
                    let op = p_tmp[q];
                    let od = vec3<f32>(op.pos[0], op.pos[1], op.pos[2]) - pos;
                    let ods = dot(od, od);
                    if (ods > 1e-12) {
                        let odss = ods + params.e;
                        acc += params.g * op.mass * od / (odss * sqrt(odss));
                    }
                }
            } else if (r != l) {
                stack[sp] = l;
                sp += 1u;
            }
        }
    }

    // Analytic dark-matter halo from the previous step's centers.
    let halo_v_sq = params.halo_v * params.halo_v;
    let halo_r_sq = params.halo_r * params.halo_r;
    for (var g = 0u; g < params.num_galaxies; g++) {
        let c = centers_src[g];
        if (c.mass != params.central_mass) { continue; }
        if (is_center && c.galaxy_id == galaxy) { continue; }
        let hd = pos - c.pos;
        let hds = dot(hd, hd);
        if (hds < 1e-8) { continue; }
        acc -= halo_v_sq * hd / (hds + halo_r_sq);
    }
    // Leapfrog: second half-kick with the new acceleration.
    vel += acc * params.dt * 0.5;

    // Dynamical friction on galaxy centers.
    if (is_center) {
        for (var g2 = 0u; g2 < params.num_galaxies; g2++) {
            if (g2 == galaxy) { continue; }
            let c2 = centers_src[g2];
            if (c2.mass != params.central_mass) { continue; }
            let rel = vel - c2.vel;
            vel += -params.damping * rel * params.dt;
        }
        centers_dst[galaxy] = Center(
            pos, 0.0, vel, 0.0, mass, galaxy, vec2<f32>(0.0, 0.0)
        );
    }

    p_dst[p] = Particle(
        array<f32, 3>(pos.x, pos.y, pos.z),
        array<f32, 3>(vel.x, vel.y, vel.z),
        array<f32, 3>(acc.x, acc.y, acc.z),
        mass,
        galaxy
    );
}
