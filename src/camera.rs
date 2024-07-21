use cgmath::{InnerSpace, Rad, Rotation, Rotation3, SquareMatrix, Vector3};
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
  #[must_use]
  pub fn init() -> Self {
    Self {
      view_proj: cgmath::Matrix4::identity().into(),
    }
  }

  pub fn update_view_proj(&mut self, camera: &Camera) {
    self.view_proj = camera.build_view_projection_matrix().into();
  }
}

#[derive(Default)]
enum Movement {
  #[default]
  None,
  Forward,
  Backward,
  Left,
  Right,
  RotateUp,
  RotateDown,
}

pub struct CameraController {
  speed: f32,
  rotation_speed: f32,
  movement: Movement,
}

impl CameraController {
  #[must_use]
  pub fn init(speed: f32, rotation_speed: f32) -> Self {
    Self {
      speed,
      rotation_speed,
      movement: Movement::None,
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
        self.movement = match keycode {
          KeyCode::KeyW | KeyCode::ArrowUp if is_pressed => Movement::Forward,
          KeyCode::KeyA | KeyCode::ArrowLeft if is_pressed => Movement::Left,
          KeyCode::KeyS | KeyCode::ArrowDown if is_pressed => Movement::Backward,
          KeyCode::KeyD | KeyCode::ArrowRight if is_pressed => Movement::Right,
          KeyCode::KeyQ if is_pressed => Movement::RotateUp,
          KeyCode::KeyE if is_pressed => Movement::RotateDown,
          _ => Movement::None,
        };
        true
      }
      _ => false,
    }
  }

  pub fn update_camera(&self, camera: &mut Camera) {
    let forward = camera.target - camera.eye;
    let forward_norm = forward.normalize();
    let forward_mag = forward.magnitude();
    let right = forward_norm.cross(camera.up);

    match self.movement {
      Movement::Forward => {
        if forward_mag > self.speed {
          camera.eye += forward_norm * self.speed;
        }
      }
      Movement::Backward => {
        camera.eye -= forward_norm * self.speed;
      }
      Movement::Right => {
        camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
      }
      Movement::Left => {
        camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
      }
      Movement::RotateUp => {
        Self::rotate_camera(camera, right, self.rotation_speed);
      }
      Movement::RotateDown => {
        Self::rotate_camera(camera, right, -self.rotation_speed);
      }
      Movement::None => {}
    }
  }

  fn rotate_camera(camera: &mut Camera, axis: Vector3<f32>, angle: f32) {
    let rot_axis = axis.normalize();
    let rotation = cgmath::Quaternion::from_axis_angle(rot_axis, Rad(angle));
    let forward = camera.target - camera.eye;
    let rotated_forward = rotation.rotate_vector(forward);
    camera.eye = camera.target - rotated_forward;
    camera.up = rotation.rotate_vector(camera.up);
  }
}
