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
}

impl Default for SimParams {
  fn default() -> Self {
    Self {
      delta_t: 0.0016,
      gravity: 1e-6,
      calibrate: 1e-4,
      central_mass: 1500000.0,
      num_particles: 1e4 as u32,
      particles_per_group: 64,
      triangle_size: 0.002f32,
      num_galaxies: 4,
      distance_between_galaxies: 0.5,
      galaxy_velocity: 0.3, // really only useful when num_galaxies > 1
    }
  }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpiralGalaxyParams {
  pub arms: u32,
  pub spiral_size: f32,
  pub spiral_width: f32,
  pub spiral_length: f32,
  pub bulge_std: f32,
  pub width: f32,
}

impl Default for SpiralGalaxyParams {
  fn default() -> Self {
    Self {
      arms: 2,
      spiral_size: 0.2,
      spiral_width: 0.025,
      spiral_length: 4.0,
      bulge_std: 0.1,
      width: 0.4,
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
