use cgmath::{InnerSpace, Rad, Rotation, Rotation3, SquareMatrix};
use winit::{
  event::{ElementState, KeyEvent, WindowEvent},
  keyboard::{KeyCode, PhysicalKey},
};

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

pub struct Camera {
  pub eye: cgmath::Point3<f32>,
  pub target: cgmath::Point3<f32>,
  pub up: cgmath::Vector3<f32>,
  pub aspect: f32,
  pub fovy: f32,
  pub znear: f32,
  pub zfar: f32,
}

impl Camera {
  fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
    let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
    let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
    OPENGL_TO_WGPU_MATRIX * proj * view
  }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
  view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
  pub fn new() -> Self {
    Self {
      view_proj: cgmath::Matrix4::identity().into(),
    }
  }

  pub fn update_view_proj(&mut self, camera: &Camera) {
    self.view_proj = camera.build_view_projection_matrix().into();
  }
}

pub struct CameraController {
  speed: f32,
  rotation_speed: f32,
  is_forward_pressed: bool,
  is_backward_pressed: bool,
  is_left_pressed: bool,
  is_right_pressed: bool,
  is_rotate_up_pressed: bool,
  is_rotate_down_pressed: bool,
}

impl CameraController {
  pub fn init(speed: f32, rotation_speed: f32) -> Self {
    Self {
      speed,
      rotation_speed,
      is_forward_pressed: false,
      is_backward_pressed: false,
      is_left_pressed: false,
      is_right_pressed: false,
      is_rotate_up_pressed: false,
      is_rotate_down_pressed: false,
    }
  }

  pub fn process_events(&mut self, event: &WindowEvent) -> bool {
    match event {
      WindowEvent::KeyboardInput {
        event:
          KeyEvent {
            state,
            physical_key: PhysicalKey::Code(keycode),
            ..
          },
        ..
      } => {
        let is_pressed = *state == ElementState::Pressed;
        match keycode {
          KeyCode::KeyW | KeyCode::ArrowUp => {
            self.is_forward_pressed = is_pressed;
            true
          }
          KeyCode::KeyA | KeyCode::ArrowLeft => {
            self.is_left_pressed = is_pressed;
            true
          }
          KeyCode::KeyS | KeyCode::ArrowDown => {
            self.is_backward_pressed = is_pressed;
            true
          }
          KeyCode::KeyD | KeyCode::ArrowRight => {
            self.is_right_pressed = is_pressed;
            true
          }
          KeyCode::KeyQ => {
            self.is_rotate_up_pressed = is_pressed;
            true
          }
          KeyCode::KeyE => {
            self.is_rotate_down_pressed = is_pressed;
            true
          }
          _ => false,
        }
      }
      _ => false,
    }
  }

  pub fn update_camera(&self, camera: &mut Camera) {
    let forward = camera.target - camera.eye;
    let forward_norm = forward.normalize();
    let forward_mag = forward.magnitude();

    if self.is_forward_pressed && forward_mag > self.speed {
      camera.eye += forward_norm * self.speed;
    }
    if self.is_backward_pressed {
      camera.eye -= forward_norm * self.speed;
    }

    let right = forward_norm.cross(camera.up);

    if self.is_right_pressed {
      camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
    }
    if self.is_left_pressed {
      camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
    }

    if self.is_rotate_up_pressed {
      let rot_axis = right.normalize();
      let rotation = cgmath::Quaternion::from_axis_angle(rot_axis, Rad(self.rotation_speed));
      let rotated_forward = rotation.rotate_vector(forward);
      camera.eye = camera.target - rotated_forward;
      camera.up = rotation.rotate_vector(camera.up);
    }
    if self.is_rotate_down_pressed {
      let rot_axis = right.normalize();
      let rotation = cgmath::Quaternion::from_axis_angle(rot_axis, Rad(-self.rotation_speed));
      let rotated_forward = rotation.rotate_vector(forward);
      camera.eye = camera.target - rotated_forward;
      camera.up = rotation.rotate_vector(camera.up);
    }
  }
}
