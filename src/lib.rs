pub mod camera;
pub mod initialize;
pub mod render;
pub mod state;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
  delta_t: f32,
  gravity: f32,
  calibrate: f32,
  central_mass: f32,
  num_particles: u32,
  particles_per_group: u32,
  triangle_size: f32,
  num_galaxies: u32,
  distance_between_galaxies: f32,
  galaxy_velocity: f32,
  halo_velocity: f32,
  halo_radius: f32,
  time: f32,
}

impl Default for SimParams {
  fn default() -> Self {
    Self {
      delta_t: 0.0005,
      gravity: 1e-6,
      calibrate: 0.01,
      central_mass: 1_000_000.0,
      num_particles: 10_000,
      particles_per_group: 64,
      triangle_size: 0.002f32,
      num_galaxies: 1,
      distance_between_galaxies: 0.5,
      galaxy_velocity: 0.0, // really only useful when num_galaxies > 1
      halo_velocity: 2.0,
      halo_radius: 2.0,
      time: 0.0,
    }
  }
}

pub struct CameraParams {
  pub speed: f32,
  pub rotational_speed: f32,
}

impl Default for CameraParams {
  fn default() -> Self {
    Self {
      speed: 0.02,
      rotational_speed: 0.02,
    }
  }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
  pub pos: [f32; 3],
  pub vel: [f32; 3],
  pub acc: [f32; 3],
  pub mass: f32,
}
