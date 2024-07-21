use crate::render::*;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::event::ElementState;
use winit::keyboard::*;
use winit::{
  dpi::PhysicalSize,
  event::{Event, KeyEvent, StartCause, WindowEvent},
  event_loop::{EventLoop, EventLoopWindowTarget},
  window::Window,
};
struct EventLoopWrapper {
  event_loop: EventLoop<()>,
  window: Arc<Window>,
}

impl EventLoopWrapper {
  pub fn new(title: &str) -> Self {
    let event_loop = EventLoop::new().unwrap();
    let mut builder = winit::window::WindowBuilder::new();
    builder = builder.with_title(title).with_resizable(false);
    let window = Arc::new(builder.build(&event_loop).unwrap());

    Self { event_loop, window }
  }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

struct Camera {
  eye: cgmath::Point3<f32>,
  target: cgmath::Point3<f32>,
  up: cgmath::Vector3<f32>,
  aspect: f32,
  fovy: f32,
  znear: f32,
  zfar: f32,
}

impl Camera {
  fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
    let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
    let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
    OPENGL_TO_WGPU_MATRIX * proj * view
  }
}

struct SurfaceWrapper {
  surface: Option<wgpu::Surface<'static>>,
  config: Option<wgpu::SurfaceConfiguration>,
}

impl SurfaceWrapper {
  fn new() -> Self {
    Self {
      surface: None,
      config: None,
    }
  }

  fn resume(&mut self, context: &State, window: Arc<Window>) {
    let window_size = window.inner_size();
    let width = window_size.width.max(1);
    let height = window_size.height.max(1);
    self.surface = Some(context.instance.create_surface(window).unwrap());
    let surface = self.surface.as_ref().unwrap();
    let mut config = surface
      .get_default_config(&context.adapter, width, height)
      .unwrap();
    let view_format = config.format.add_srgb_suffix();
    config.view_formats.push(view_format);
    surface.configure(&context.device, &config);
    self.config = Some(config);
  }

  fn acquire(&mut self, context: &State) -> wgpu::SurfaceTexture {
    let surface = self.surface.as_ref().unwrap();

    match surface.get_current_texture() {
      Ok(frame) => frame,
      Err(wgpu::SurfaceError::Timeout) => surface.get_current_texture().unwrap(),
      Err(
        wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost | wgpu::SurfaceError::OutOfMemory,
      ) => {
        surface.configure(&context.device, self.config());
        surface.get_current_texture().unwrap()
      }
    }
  }

  fn suspend(&mut self) {}

  fn config(&self) -> &wgpu::SurfaceConfiguration {
    self.config.as_ref().unwrap()
  }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
  view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
  fn new() -> Self {
    use cgmath::SquareMatrix;
    Self {
      view_proj: cgmath::Matrix4::identity().into(),
    }
  }

  fn update_view_proj(&mut self, camera: &Camera) {
    self.view_proj = camera.build_view_projection_matrix().into();
  }
}

struct CameraController {
  speed: f32,
  is_forward_pressed: bool,
  is_backward_pressed: bool,
  is_left_pressed: bool,
  is_right_pressed: bool,
}

impl CameraController {
  fn new(speed: f32) -> Self {
    Self {
      speed,
      is_forward_pressed: false,
      is_backward_pressed: false,
      is_left_pressed: false,
      is_right_pressed: false,
    }
  }

  fn process_events(&mut self, event: &WindowEvent) -> bool {
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
          _ => false,
        }
      }
      _ => false,
    }
  }

  fn update_camera(&self, camera: &mut Camera) {
    use cgmath::InnerSpace;
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
    let forward = camera.target - camera.eye;
    let forward_mag = forward.magnitude();
    if self.is_right_pressed {
      camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
    }
    if self.is_left_pressed {
      camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
    }
  }
}

struct State {
  instance: wgpu::Instance,
  adapter: wgpu::Adapter,
  device: wgpu::Device,
  queue: wgpu::Queue,
  camera: Camera,
  camera_uniform: CameraUniform,
  camera_buffer: wgpu::Buffer,
  camera_bind_group: wgpu::BindGroup,
  camera_controller: CameraController,
  camera_bind_group_layout: wgpu::BindGroupLayout,
}
impl State {
  fn input(&mut self, event: &WindowEvent) -> bool {
    self.camera_controller.process_events(event)
  }
  fn update(&mut self) {
    self.camera_controller.update_camera(&mut self.camera);
    self.camera_uniform.update_view_proj(&self.camera);
    self.queue.write_buffer(
      &self.camera_buffer,
      0,
      bytemuck::cast_slice(&[self.camera_uniform]),
    );
  }

  async fn init(surface: &SurfaceWrapper, size: &PhysicalSize<u32>) -> Self {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
      #[cfg(not(target_arch = "wasm32"))]
      backends: wgpu::Backends::PRIMARY,
      ..Default::default()
    });

    let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: surface.surface.as_ref(),
        force_fallback_adapter: false,
      })
      .await
      .unwrap();

    let (device, queue) = adapter
      .request_device(
        &wgpu::DeviceDescriptor {
          label: None,
          required_features: wgpu::Features::empty(),
          required_limits: wgpu::Limits::default(),
          memory_hints: Default::default(),
        },
        None,
      )
      .await
      .unwrap();
    let camera = Camera {
      // position the camera 1 unit up and 2 units back
      // +z is out of the screen
      eye: (0.0, 1.0, 2.0).into(),
      // have it look at the origin
      target: (0.0, 0.0, 0.0).into(),
      // which way is "up"
      up: cgmath::Vector3::unit_y(),
      aspect: size.width as f32 / size.height as f32,
      fovy: 45.0,
      znear: 0.1,
      zfar: 100.0,
    };
    let mut camera_uniform = CameraUniform::new();
    camera_uniform.update_view_proj(&camera);

    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Camera Buffer"),
      contents: bytemuck::cast_slice(&[camera_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let camera_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::VERTEX,
          ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          count: None,
        }],
        label: Some("camera_bind_group_layout"),
      });
    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &camera_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: camera_buffer.as_entire_binding(),
      }],
      label: Some("camera_bind_group"),
    });
    let camera_controller = CameraController::new(0.2);

    Self {
      instance,
      adapter,
      device,
      queue,
      camera,
      camera_uniform,
      camera_buffer,
      camera_bind_group,
      camera_controller,
      camera_bind_group_layout,
    }
  }
}

async fn start() {
  env_logger::init();
  let window_loop = EventLoopWrapper::new("Galaxy Sim");
  let mut surface = SurfaceWrapper::new();
  let mut context = State::init(&surface, &window_loop.window.inner_size()).await;
  let event_loop_function = EventLoop::run;
  let mut example = None;

  let _ = (event_loop_function)(
    window_loop.event_loop,
    move |event, target: &EventLoopWindowTarget<()>| match event {
      Event::NewEvents(StartCause::Init) => {
        surface.resume(&context, window_loop.window.clone());
        if example.is_none() {
          example = Some(Render::init(
            surface.config(),
            &context.adapter,
            &context.device,
            &context.queue,
            &context.camera_bind_group_layout,
          ));
        }
      }
      Event::Suspended => {
        surface.suspend();
      }
      Event::WindowEvent { event, window_id } if window_id == window_loop.window.id() => {
        if !context.input(&event) {
          match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
              event:
                KeyEvent {
                  state: ElementState::Pressed,
                  physical_key: PhysicalKey::Code(KeyCode::Escape),
                  ..
                },
              ..
            } => target.exit(),
            WindowEvent::RedrawRequested => {
              window_loop.window.request_redraw();
              if example.is_none() {
                return;
              }
              context.update();
              if let Some(example) = &mut example {
                let frame = surface.acquire(&context);
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                  format: Some(surface.config().view_formats[0]),
                  ..wgpu::TextureViewDescriptor::default()
                });
                example.render(
                  &view,
                  &context.device,
                  &context.queue,
                  &context.camera_bind_group,
                );
                frame.present();
              }
            }
            _ => {}
          }
        }
      }
      _ => {}
    },
  );
}

pub fn run() {
  pollster::block_on(start());
}
