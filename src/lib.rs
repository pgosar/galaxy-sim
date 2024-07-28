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
}

impl Default for SimParams {
  fn default() -> Self {
    Self {
      delta_t: 0.0016,
      gravity: 4e-7,
      calibrate: 1e-4,
      central_mass: 2e5,
      num_particles: 1e4 as u32,
      particles_per_group: 64,
      triangle_size: 0.002f32,
    }
  }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GalaxyParams {
  pub arms: u32,
  pub spiral_size: f32,
  pub spiral_width: f32,
  pub spiral_length: f32,
  pub bulge_std: f32,
  pub width: f32,
}

impl Default for GalaxyParams {
  fn default() -> Self {
    Self {
      arms: 2,
      spiral_size: 0.2,
      spiral_width: 0.025,
      spiral_length: 2.0,
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
