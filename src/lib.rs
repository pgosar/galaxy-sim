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
  num_particles: u32,
  particles_per_group: u32,
  galaxy_radius: f32,
  arm_factor: f32,
  triangle_size: f32,
}

impl Default for SimParams {
  fn default() -> Self {
    Self {
      delta_t: 0.0048,
      gravity: 4e-7,
      calibrate: 1e-4,
      num_particles: 1e4 as u32,
      particles_per_group: 64,
      galaxy_radius: 1.0,
      arm_factor: 0.3,
      triangle_size: 0.002f32,
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
