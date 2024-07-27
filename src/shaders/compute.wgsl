struct Particle {
    pos: array<f32, 3>,
    vel: array<f32, 3>,
    acc: array<f32, 3>,
    mass: f32
};

struct SimParams {
    dt: f32,
    g: f32,
    e: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> particlesSrc: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesDst: array<Particle>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let totalParticles = arrayLength(&particlesSrc);
    let particleIndex = global_invocation_id.x;
    if (particleIndex >= totalParticles) {
        return;
    }

    let currentParticle = particlesSrc[particleIndex];
    var position = vec3<f32>(currentParticle.pos[0], currentParticle.pos[1], currentParticle.pos[2]);
    var velocity = vec3<f32>(currentParticle.vel[0], currentParticle.vel[1], currentParticle.vel[2]);
    var acceleration = vec3<f32>(currentParticle.acc[0], currentParticle.acc[1], currentParticle.acc[2]);

    // Leapfrog numerical integration
    velocity += acceleration * params.dt / 2.0;
    position += velocity * params.dt;

    // Calculate acceleration
    // a_i = sum_j!=i ( (G * m_j / (r_i - r_j)^3 + e) * (r_j - r_i))
    var newAcceleration = vec3<f32>(0.0, 0.0, 0.0);
    for (var i: u32 = 0u; i < totalParticles; i++) {
        if (i == particleIndex) {
            continue;
        }
        let otherParticle = particlesSrc[i];
        let otherPosition = vec3<f32>(otherParticle.pos[0], otherParticle.pos[1], otherParticle.pos[2]);
        let r = distance(position, otherPosition);
        let force = params.g * otherParticle.mass / (r * r * r + params.e) *
                  normalize(otherPosition - position);
        newAcceleration += force * params.dt;
    }

    velocity += newAcceleration * params.dt / 2.0;

    particlesDst[particleIndex] = Particle(
        array<f32, 3>(position.x, position.y, position.z),
        array<f32, 3>(velocity.x, velocity.y, velocity.z),
        array<f32, 3>(newAcceleration.x, newAcceleration.y, newAcceleration.z),
        currentParticle.mass
    );
}

