use crate::{Particle, SimParams};
use cgmath::{InnerSpace, Vector3};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::f32::consts::PI;

#[must_use]
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
      );
    }
    println!("center: {center:?}");
    elliptical(
      &mut rng,
      &mut particles,
      sim_params.num_particles,
      sim_params.gravity,
      &velocity,
      &center,
      sim_params.central_mass,
      sim_params.calibrate,
      sim_params.halo_velocity,
      sim_params.halo_radius,
    );
  }
  particles
}

#[allow(clippy::too_many_arguments)]
fn elliptical(
  rng: &mut SmallRng,
  particles: &mut Vec<Particle>,
  num_particles: u32,
  gravity: f32,
  velocity: &Vector3<f32>,
  center: &Vector3<f32>,
  central_mass: f32,
  softening: f32,
  halo_velocity: f32,
  halo_radius: f32,
) {
  particles.push(Particle {
    pos: [center.x, center.y, center.z],
    vel: [velocity.x, velocity.y, velocity.z],
    acc: [0.0; 3],
    mass: central_mass,
  });

  let bulge_fraction: f32 = 0.4;
  let bulge_scale_radius: f32 = 0.15;
  let disk_scale_radius: f32 = 0.3;
  let disk_scale_height: f32 = 0.02;

  // Generate particles
  for _ in 1..num_particles {
    let is_bulge = rng.gen::<f32>() < bulge_fraction;

    let pos = if is_bulge {
      // Generate bulge particle with spherical distribution
      loop {
        let r = -bulge_scale_radius * rng.gen::<f32>().ln();
        let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
        let phi = (2.0 * rng.gen::<f32>() - 1.0).acos();

        let x = r * phi.sin() * theta.cos();
        let y = r * phi.sin() * theta.sin();
        let z = r * phi.cos();

        let pos = Vector3::new(x, y, z);
        if pos.magnitude() <= 0.6 {
          break pos;
        }
      }
    } else {
      // Generate disk particle with exponential distribution
      loop {
        let r = -disk_scale_radius * rng.gen::<f32>().ln();
        let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
        let z =
          disk_scale_height * (2.0 * rng.gen::<f32>() - 1.0) * (-rng.gen::<f32>().ln()).sqrt();

        let x = r * theta.cos();
        let y = r * theta.sin();

        let pos = Vector3::new(x, y, z);
        if pos.magnitude() <= 0.6 && pos.magnitude() >= 0.02 {
          break pos;
        }
      }
    };

    let relative_pos = pos;
    let final_pos = pos + *center;

    let vel = {
      let rotation_dir = Vector3::new(-relative_pos.y, relative_pos.x, 0.0).normalize();
      let distance = relative_pos.magnitude();
      let dist_sq = distance * distance + softening;
      let central_speed_sq = gravity * central_mass * distance * distance / (dist_sq * dist_sq.sqrt());
      
      // Halo velocity contribution: v^2 = V_halo^2 * r^2 / (r^2 + R_c^2)
      let halo_dist_sq = distance * distance + halo_radius * halo_radius;
      let halo_speed_sq = (halo_velocity * halo_velocity * distance * distance) / halo_dist_sq;
      
      let rotation_speed = (central_speed_sq + halo_speed_sq).sqrt();
      // Add more random motion for bulge particles
      let variation = if is_bulge {
        Vector3::new(
          rng.gen::<f32>() * 0.15 - 0.075,
          rng.gen::<f32>() * 0.15 - 0.075,
          rng.gen::<f32>() * 0.15 - 0.075,
        )
      } else {
        Vector3::new(
          rng.gen::<f32>() * 0.05 - 0.025,
          rng.gen::<f32>() * 0.05 - 0.025,
          rng.gen::<f32>() * 0.01 - 0.005,
        )
      };

      rotation_dir * rotation_speed + variation + velocity
    };

    let mass = 1.0;

    particles.push(Particle {
      pos: [final_pos.x, final_pos.y, final_pos.z],
      vel: [vel.x, vel.y, vel.z],
      acc: [0.0; 3],
      mass,
    });
  }
}


