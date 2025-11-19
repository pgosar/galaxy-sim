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
  RotateLeft,
  RotateRight,
  RotateUp,
  RotateDown,
  MoveLeft,
  MoveRight,
  MoveUp,
  MoveDown,
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
          KeyCode::KeyW if is_pressed => Movement::Forward,
          KeyCode::KeyA if is_pressed => Movement::RotateLeft,
          KeyCode::KeyS if is_pressed => Movement::Backward,
          KeyCode::KeyD if is_pressed => Movement::RotateRight,
          KeyCode::KeyQ if is_pressed => Movement::RotateUp,
          KeyCode::KeyE if is_pressed => Movement::RotateDown,
          KeyCode::KeyH if is_pressed => Movement::MoveLeft,
          KeyCode::KeyJ if is_pressed => Movement::MoveDown,
          KeyCode::KeyK if is_pressed => Movement::MoveUp,
          KeyCode::KeyL if is_pressed => Movement::MoveRight,
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
      Movement::RotateRight => {
        camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
      }
      Movement::RotateLeft => {
        camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
      }
      Movement::RotateUp => {
        Self::rotate_camera(camera, right, self.rotation_speed);
      }
      Movement::RotateDown => {
        Self::rotate_camera(camera, right, -self.rotation_speed);
      }
      Movement::MoveLeft => {
        camera.eye -= right * self.speed;
        camera.target -= right * self.speed;
      }
      Movement::MoveRight => {
        camera.eye += right * self.speed;
        camera.target += right * self.speed;
      }
      Movement::MoveUp => {
        camera.eye += camera.up * self.speed;
        camera.target += camera.up * self.speed;
      }
      Movement::MoveDown => {
        camera.eye -= camera.up * self.speed;
        camera.target -= camera.up * self.speed;
      }
      Movement::None => {} // Movement::None => {}
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
