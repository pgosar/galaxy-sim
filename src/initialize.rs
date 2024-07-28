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
  let mut particle_data = Vec::with_capacity((sim_params.num_particles) as usize);

  let galaxy_params = GalaxyParams::default();

  // Create first galaxy
  create_galaxy(
    &mut rng,
    sim_params,
    &galaxy_params,
    &mut particle_data,
    Vector3::new(-sim_params.distance_between_galaxies, 0.0, 0.0), // Position
    Vector3::new(sim_params.galaxy_velocity, 0.0, 0.0),            // Velocity
  );

  // Create second galaxy
  create_galaxy(
    &mut rng,
    sim_params,
    &galaxy_params,
    &mut particle_data,
    Vector3::new(sim_params.distance_between_galaxies, 0.0, 0.0), // Position
    Vector3::new(-sim_params.galaxy_velocity, 0.0, 0.0),          // Velocity
  );

  particle_data
}

fn create_galaxy(
  rng: &mut SmallRng,
  sim_params: &SimParams,
  galaxy_params: &GalaxyParams,
  particle_data: &mut Vec<Particle>,
  galaxy_center: Vector3<f32>,
  galaxy_velocity: Vector3<f32>,
) {
  // Add central mass of the galaxy
  particle_data.push(Particle {
    pos: [galaxy_center.x, galaxy_center.y, galaxy_center.z],
    vel: [galaxy_velocity.x, galaxy_velocity.y, galaxy_velocity.z],
    acc: [0.0; 3],
    mass: sim_params.central_mass,
  });

  for i in 1..sim_params.num_particles / sim_params.num_galaxies {
    let (pos, vel, mass) = if rng.gen::<f32>() < 0.2 {
      // 20% of particles in the galactic bulge
      create_bulge_particle(
        rng,
        sim_params,
        galaxy_params,
        &galaxy_center,
        &galaxy_velocity,
      )
    } else {
      // 80% of particles in the spiral arms
      create_arm_particle(
        (i - 1) as f32,
        rng,
        sim_params,
        galaxy_params,
        &galaxy_center,
        &galaxy_velocity,
      )
    };

    particle_data.push(Particle {
      pos: [pos.x, pos.y, pos.z],
      vel: [vel.x, vel.y, vel.z],
      acc: [0.0; 3],
      mass,
    });
  }
}

fn create_bulge_particle(
  rng: &mut SmallRng,
  sim_params: &SimParams,
  galaxy_params: &GalaxyParams,
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
    let speed = (sim_params.gravity * 1000.0 / (pos - galaxy_center).magnitude()).sqrt();
    (pos - galaxy_center).cross(Vector3::unit_z()).normalize() * speed + galaxy_velocity
  };
  let mass = 1.0;
  (pos, vel, mass)
}

fn create_arm_particle(
  i: f32,
  rng: &mut SmallRng,
  sim_params: &SimParams,
  galaxy_params: &GalaxyParams,
  galaxy_center: &Vector3<f32>,
  galaxy_velocity: &Vector3<f32>,
) -> (Vector3<f32>, Vector3<f32>, f32) {
  let theta = i * (galaxy_params.spiral_length * PI / (sim_params.num_particles as f32 * 0.8));

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
    let speed = (sim_params.gravity * 1000.0 / (pos - galaxy_center).magnitude()).sqrt();
    (pos - galaxy_center).cross(Vector3::unit_z()).normalize() * speed + galaxy_velocity
  };

  let mass = 1.0;
  (pos, vel, mass)
}
