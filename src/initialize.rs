use crate::GalaxyParams;
use crate::Particle;
use crate::SimParams;
use cgmath::{InnerSpace, Vector3};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::f32::consts::PI;

#[allow(dead_code)]
pub fn create_elliptical_galaxy(sim_params: &SimParams) -> Vec<Particle> {
  let mut rng = SmallRng::seed_from_u64(42);
  let mut initial_particles = Vec::with_capacity(sim_params.num_particles as usize);

  initial_particles.push(Particle {
    pos: [0.0; 3],
    vel: [0.0; 3],
    acc: [0.0; 3],
    mass: sim_params.central_mass,
  });
  for _ in 1..sim_params.num_particles {
    let mut pos = loop {
      let x = rng.gen::<f32>() * 2.0 - 1.0;
      let y = rng.gen::<f32>() * 2.0 - 1.0;
      let z = (rng.gen::<f32>() * 2.0 - 1.0) * 0.1;
      let pos = Vector3::new(x, y, z);
      if pos.magnitude() <= 1.0 && pos.magnitude() >= 0.25 {
        break pos;
      }
    };
    pos *= pos.magnitude();
    let vel = {
      let speed = (sim_params.gravity * 1000.0 / pos.magnitude()).sqrt();
      pos.cross(Vector3::unit_z()).normalize() * speed
    };

    initial_particles.push(Particle {
      pos: [pos.x, pos.y, pos.z],
      vel: [vel.x, vel.y, vel.z],
      acc: [0.0; 3],
      mass: 1.0,
    });
  }
  initial_particles
}

#[allow(dead_code)]
pub fn create_spiral_galaxy(sim_params: &SimParams) -> Vec<Particle> {
  let mut rng = SmallRng::seed_from_u64(42);
  let mut particle_data = Vec::with_capacity(sim_params.num_particles as usize);

  // Add central mass
  particle_data.push(Particle {
    pos: [0.0; 3],
    vel: [0.0; 3],
    acc: [0.0; 3],
    mass: sim_params.central_mass,
  });

  let galaxy_params = GalaxyParams::default();

  for i in 1..sim_params.num_particles {
    let (pos, vel, mass) = if rng.gen::<f32>() < 0.2 {
      // 20% of particles in the galactic bulge
      create_bulge_particle(&mut rng, sim_params, &galaxy_params)
    } else {
      // 80% of particles in the spiral arms
      create_arm_particle((i - 1) as f32, &mut rng, sim_params, &galaxy_params)
    };

    particle_data.push(Particle {
      pos: [pos.x, pos.y, pos.z],
      vel: [vel.x, vel.y, vel.z],
      acc: [0.0; 3],
      mass,
    });
  }

  particle_data
}

fn create_bulge_particle(
  rng: &mut SmallRng,
  sim_params: &SimParams,
  galaxy_params: &GalaxyParams,
) -> (Vector3<f32>, Vector3<f32>, f32) {
  let normal_dist = Normal::new(0.0, galaxy_params.bulge_std).unwrap();
  let pos = loop {
    let x = normal_dist.sample(rng);
    let y = normal_dist.sample(rng);
    let z = normal_dist.sample(rng);
    let pos = Vector3::new(x, y, z);
    if pos.magnitude() <= 0.6 && pos.magnitude() >= 0.25 {
      break pos;
    }
  };
  let vel = {
    let speed = (sim_params.gravity * 1000.0 / pos.magnitude()).sqrt();
    pos.cross(Vector3::unit_z()).normalize() * speed
  };
  let mass = 1.0;
  (pos, vel, mass)
}

fn create_arm_particle(
  i: f32,
  rng: &mut SmallRng,
  sim_params: &SimParams,
  galaxy_params: &GalaxyParams,
) -> (Vector3<f32>, Vector3<f32>, f32) {
  let theta = i * (galaxy_params.spiral_length * PI / (sim_params.num_particles as f32 * 0.8));

  let r = galaxy_params.spiral_size * theta.sqrt();
  let min_radius = 1.0 * galaxy_params.bulge_std;
  let r = r.max(min_radius);

  let arm_theta = theta + if i as u32 % 2 == 0 { 0.0 } else { PI };

  let arm_deviation = 0.05 * rng.gen::<f32>();
  let pos = Vector3::new(
    (r + arm_deviation) * arm_theta.cos(),
    (r + arm_deviation) * arm_theta.sin(),
    0.0,
  );
  let vel = {
    let speed = (sim_params.gravity * 1000.0 / pos.magnitude()).sqrt();
    pos.cross(Vector3::unit_z()).normalize() * speed
  };

  let mass = 1.0;
  (pos, vel, mass)
}
