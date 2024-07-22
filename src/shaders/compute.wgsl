struct Particle {
    pos: vec3<f32>,
    vel: vec3<f32>,
    mass: f32,
};

struct SimParams {
    deltaT: f32,
    gravitationalConstant: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> particlesSrc: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesDst: array<Particle>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let total = arrayLength(&particlesSrc);
    let index = global_invocation_id.x;
    if (index >= total) {
        return;
    }

    var vPos: vec3<f32> = particlesSrc[index].pos;
    var vVel: vec3<f32> = particlesSrc[index].vel;
    var vMass: f32 = particlesSrc[index].mass;
    var netForce: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

    let G = params.gravitationalConstant;

    for (var i: u32 = 0u; i < total; i++) {
        if (i == index) {
            continue;
        }
        let pos = particlesSrc[i].pos;
        let mass = particlesSrc[i].mass;

        let direction = pos - vPos;
        let distanceSquared = dot(direction, direction);
        let distance = sqrt(distanceSquared);

        if (distance > 0.0) {
            let forceMagnitude = G * vMass * mass / distanceSquared;
            let force = normalize(direction) * forceMagnitude;

            netForce += force;
        }
    }
    var acceleration = netForce / vMass;
    vVel += acceleration * params.deltaT;
    vPos += vVel * params.deltaT;

    particlesDst[index] = Particle(vPos, vVel, vMass);
}
