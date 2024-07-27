use crate::Particle;
use crate::SimParams;
use cgmath::{InnerSpace, Point3, Vector3};
use nanorand::{Rng, WyRand};
use std::f32::consts::PI;

#[allow(dead_code)]
pub fn create_elliptical_galaxy(sim_params: &SimParams) -> Vec<Particle> {
  let mut rng = WyRand::new();
  let mut initial_particles = Vec::with_capacity(sim_params.num_particles as usize);

  initial_particles.push(Particle {
    pos: [0.0; 3],
    vel: [0.0; 3],
    acc: [0.0; 3],
    mass: 2e5,
  });
  for _ in 1..sim_params.num_particles {
    let mut pos = loop {
      let x = rng.generate::<f32>() * 2.0 - 1.0;
      let y = rng.generate::<f32>() * 2.0 - 1.0;
      let z = (rng.generate::<f32>() * 2.0 - 1.0) * 0.1;
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
  let mut rng = WyRand::new_seed(42);
  let mut particle_data = Vec::with_capacity(sim_params.num_particles as usize); // 3 for pos, 3 for vel, 1 for mass

  for _ in 0..sim_params.num_particles {
    let (pos, vel, mass) = if rng.generate::<f32>() < 0.2 {
      // 20% of particles in the galactic bulge
      create_bulge_particle(&mut rng, sim_params)
    } else {
      // 80% of particles in the spiral arms
      create_arm_particle(&mut rng, sim_params)
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
  rng: &mut WyRand,
  sim_params: &SimParams,
) -> (Point3<f32>, Vector3<f32>, f32) {
  let r = rng.generate::<f32>().powf(0.5) * sim_params.galaxy_radius * 0.1;
  let theta = rng.generate::<f32>() * 2.0 * PI;
  let phi = (rng.generate::<f32>() - 0.5) * PI;

  let pos = Point3::new(
    r * theta.cos() * phi.cos(),
    r * theta.sin() * phi.cos(),
    r * phi.sin() * 0.1, // Flatten the bulge
  );

  let speed = (r / sim_params.galaxy_radius).sqrt() * 0.1;
  let vel = Vector3::new(-speed * theta.sin(), speed * theta.cos(), 0.0);
  let mass = 1.0;

  (pos, vel, mass)
}

fn create_arm_particle(
  rng: &mut WyRand,
  sim_params: &SimParams,
) -> (Point3<f32>, Vector3<f32>, f32) {
  let r = rng.generate::<f32>().powf(0.5) * sim_params.galaxy_radius;
  let theta = rng.generate::<f32>() * 2.0 * PI;
  let arm_offset = (r * sim_params.arm_factor).exp() + rng.generate::<f32>() * 0.3;

  let pos = Point3::new(
    r * (theta + arm_offset).cos(),
    r * (theta + arm_offset).sin(),
    (rng.generate::<f32>() - 0.5) * 0.1 * r, // Add some thickness
  );

  let speed = (r / sim_params.galaxy_radius).sqrt() * 0.1;
  let vel = Vector3::new(
    -speed * (theta + arm_offset).sin(),
    speed * (theta + arm_offset).cos(),
    0.0,
  );
  let mass = 1.0;

  (pos, vel, mass)
}
