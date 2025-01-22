use crate::{Particle, SimParams, SpiralGalaxyParams};
use cgmath::{InnerSpace, Vector3};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::f32::consts::PI;

pub fn create_galaxies(sim_params: &SimParams) -> Vec<Particle> {
  let mut rng = SmallRng::seed_from_u64(42);
  let mut particles = Vec::with_capacity(sim_params.num_particles as usize);
  for i in 0..sim_params.num_galaxies {
    let mut center: Vector3<f32> = Vector3::new(0.0, 0.0, 0.0);
    let mut velocity = Vector3::new(sim_params.galaxy_velocity, 0.0, 0.0);
    // based on unit circle
    if sim_params.num_galaxies > 1 {
      let theta = (2.0 * PI) / sim_params.num_galaxies as f32 * i as f32;
      center = Vector3::new(
        theta.sin() * sim_params.distance_between_galaxies,
        theta.cos() * sim_params.distance_between_galaxies,
        0.0,
      );
      velocity = Vector3::new(
        -(theta.sin() * sim_params.galaxy_velocity),
        -(theta.cos() * sim_params.galaxy_velocity),
        0.0,
      )
    }
    println!("center: {:?}", center);
    spiral(
      &mut rng,
      &mut particles,
      sim_params.num_particles,
      sim_params.gravity,
      &velocity,
      &center,
      sim_params.central_mass,
    );
  }
  particles
}

fn spiral(
  rng: &mut SmallRng,
  particles: &mut Vec<Particle>,
  num_particles: u32,
  gravity: f32,
  velocity: &Vector3<f32>,
  center: &Vector3<f32>,
  central_mass: f32,
) {
  particles.push(Particle {
    pos: [center.x, center.y, center.z],
    vel: [velocity.x, velocity.y, velocity.z],
    acc: [0.0; 3],
    mass: central_mass,
  });
  let galaxy_params = SpiralGalaxyParams::default();
  for i in 1..num_particles {
    let (pos, vel, mass) = if rng.gen::<f32>() < 0.2 {
      create_bulge_particle(rng, gravity, &galaxy_params, center, velocity)
    } else {
      create_arm_particle(
        (i - 1) as f32,
        rng,
        num_particles,
        gravity,
        &galaxy_params,
        center,
        velocity,
      )
    };

    particles.push(Particle {
      pos: [pos.x, pos.y, pos.z],
      vel: [vel.x, vel.y, vel.z],
      acc: [0.0; 3],
      mass,
    });
  }
}

fn create_bulge_particle(
  rng: &mut SmallRng,
  gravity: f32,
  galaxy_params: &SpiralGalaxyParams,
  galaxy_center: &Vector3<f32>,
  galaxy_velocity: &Vector3<f32>,
) -> (Vector3<f32>, Vector3<f32>, f32) {
  let normal_dist = Normal::new(0.0, galaxy_params.bulge_std).unwrap();
  let pos = loop {
    let x = normal_dist.sample(rng);
    let y = normal_dist.sample(rng);
    let z = normal_dist.sample(rng) * galaxy_params.width;
    let pos = Vector3::new(x, y, z) + galaxy_center;
    if (pos - galaxy_center).magnitude() <= 0.3 && (pos - galaxy_center).magnitude() >= 0.02 {
      break pos;
    }
  };
  let vel = {
    let speed = (gravity / (pos - galaxy_center).magnitude()).sqrt();
    (pos - galaxy_center).cross(Vector3::unit_z()).normalize() * speed + galaxy_velocity
  };
  let mass = 1.0;
  (pos, vel, mass)
}

fn create_arm_particle(
  i: f32,
  rng: &mut SmallRng,
  num_particles: u32,
  gravity: f32,
  galaxy_params: &SpiralGalaxyParams,
  galaxy_center: &Vector3<f32>,
  galaxy_velocity: &Vector3<f32>,
) -> (Vector3<f32>, Vector3<f32>, f32) {
  let theta = i * (galaxy_params.spiral_length * PI / (num_particles as f32 * 0.8));
  // make arms start at center
  let mut r = galaxy_params.spiral_size * theta.sqrt();
  let min_radius = 1.0 * galaxy_params.bulge_std;
  r = r.max(min_radius);
  let normal_dist = Normal::new(0.0, galaxy_params.spiral_width).unwrap();
  // choose arm 1 or 2
  let arm_theta = theta + if i % 2.0 == 0.0 { 0.0 } else { PI };
  let arm_deviation = normal_dist.sample(rng);
  let pos = Vector3::new(
    (r + arm_deviation) * arm_theta.cos(),
    (r + arm_deviation) * arm_theta.sin(),
    normal_dist.sample(rng) * galaxy_params.width,
  ) + galaxy_center;
  let vel = {
    let speed = (gravity / (pos - galaxy_center).magnitude()).sqrt();
    (pos - galaxy_center).cross(Vector3::unit_z()).normalize() * speed + galaxy_velocity
  };

  let mass = 1.0;
  (pos, vel, mass)
}
