use crate::render::*;
use std::sync::Arc;
use winit::{
  event::{Event, KeyEvent, StartCause, WindowEvent},
  event_loop::{EventLoop, EventLoopWindowTarget},
  keyboard::{Key, NamedKey},
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

struct State {
  instance: wgpu::Instance,
  adapter: wgpu::Adapter,
  device: wgpu::Device,
  queue: wgpu::Queue,
}
impl State {
  async fn init_async(surface: &SurfaceWrapper) -> Self {
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
    Self {
      instance,
      adapter,
      device,
      queue,
    }
  }
}

async fn start() {
  env_logger::init();
  let window_loop = EventLoopWrapper::new("Galaxy Sim");
  let mut surface = SurfaceWrapper::new();
  let context = State::init_async(&surface).await;
  let mut example = None;
  let event_loop_function = EventLoop::run;

  log::info!("Entering event loop...");
  let _ = (event_loop_function)(
    window_loop.event_loop,
    move |event: Event<()>, target: &EventLoopWindowTarget<()>| match event {
      ref e if matches!(e, Event::NewEvents(StartCause::Init)) => {
        surface.resume(&context, window_loop.window.clone());
        if example.is_none() {
          example = Some(Render::init(
            surface.config(),
            &context.adapter,
            &context.device,
            &context.queue,
          ));
        }
      }
      Event::Suspended => {
        surface.suspend();
      }
      Event::WindowEvent { event, .. } => match event {
        WindowEvent::KeyboardInput {
          event:
            KeyEvent {
              logical_key: Key::Named(NamedKey::Escape),
              ..
            },
          ..
        }
        | WindowEvent::CloseRequested => {
          target.exit();
        }
        WindowEvent::RedrawRequested => {
          if example.is_none() {
            return;
          }

          let frame = surface.acquire(&context);
          let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(surface.config().view_formats[0]),
            ..wgpu::TextureViewDescriptor::default()
          });

          example
            .as_mut()
            .unwrap()
            .render(&view, &context.device, &context.queue);
          frame.present();

          window_loop.window.request_redraw();
        }
        _ => {}
      },
      _ => {}
    },
  );
}

pub fn run() {
  pollster::block_on(start());
}
