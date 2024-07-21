use std::sync::Arc;

use crate::boids::*;
use wgpu::Surface;
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
    builder = builder.with_title(title);
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

  fn start_condition(e: &Event<()>) -> bool {
    matches!(e, Event::NewEvents(StartCause::Init))
  }

  fn resume(&mut self, context: &ExampleContext, window: Arc<Window>) {
    let window_size = window.inner_size();
    let width = window_size.width.max(1);
    let height = window_size.height.max(1);
    self.surface = Some(context.instance.create_surface(window).unwrap());
    let surface = self.surface.as_ref().unwrap();
    let mut config = surface
      .get_default_config(&context.adapter, width, height)
      .expect("Surface isn't supported by the adapter.");
    let view_format = config.format.add_srgb_suffix();
    config.view_formats.push(view_format);
    surface.configure(&context.device, &config);
    self.config = Some(config);
  }

  fn acquire(&mut self, context: &ExampleContext) -> wgpu::SurfaceTexture {
    let surface = self.surface.as_ref().unwrap();

    match surface.get_current_texture() {
      Ok(frame) => frame,
      Err(wgpu::SurfaceError::Timeout) => surface
        .get_current_texture()
        .expect("Failed to acquire next surface texture!"),
      Err(
        wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost | wgpu::SurfaceError::OutOfMemory,
      ) => {
        surface.configure(&context.device, self.config());
        surface
          .get_current_texture()
          .expect("Failed to acquire next surface texture!")
      }
    }
  }

  fn suspend(&mut self) {}

  fn get(&self) -> Option<&Surface> {
    self.surface.as_ref()
  }

  fn config(&self) -> &wgpu::SurfaceConfiguration {
    self.config.as_ref().unwrap()
  }
}

struct ExampleContext {
  instance: wgpu::Instance,
  adapter: wgpu::Adapter,
  device: wgpu::Device,
  queue: wgpu::Queue,
}
impl ExampleContext {
  async fn init_async(surface: &mut SurfaceWrapper) -> Self {
    log::info!("Initializing wgpu...");

    let backends = wgpu::util::backend_bits_from_env().unwrap_or_default();
    let dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default();
    let gles_minor_version = wgpu::util::gles_minor_version_from_env().unwrap_or_default();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
      backends,
      flags: wgpu::InstanceFlags::from_build_config().with_env(),
      dx12_shader_compiler,
      gles_minor_version,
    });
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, surface.get())
      .await
      .expect("No suitable GPU adapters found on the system!");

    let adapter_info = adapter.get_info();
    log::info!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

    let optional_features = wgpu::Features::empty();
    let required_features = wgpu::Features::empty();
    let adapter_features = adapter.features();
    assert!(
      adapter_features.contains(required_features),
      "Adapter does not support required features for this example: {:?}",
      required_features - adapter_features
    );

    let trace_dir = std::env::var("WGPU_TRACE");
    let (device, queue) = adapter
      .request_device(
        &wgpu::DeviceDescriptor {
          label: None,
          required_features: (optional_features & adapter_features) | required_features,
          required_limits: wgpu::Limits::downlevel_defaults(),
          memory_hints: wgpu::MemoryHints::MemoryUsage,
        },
        trace_dir.ok().as_ref().map(std::path::Path::new),
      )
      .await
      .expect("Failed to create device");

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
  let context = ExampleContext::init_async(&mut surface).await;
  let mut example = None;
  let event_loop_function = EventLoop::run;

  log::info!("Entering event loop...");
  let _ = (event_loop_function)(
    window_loop.event_loop,
    move |event: Event<()>, target: &EventLoopWindowTarget<()>| match event {
      ref e if SurfaceWrapper::start_condition(e) => {
        surface.resume(&context, window_loop.window.clone());
        if example.is_none() {
          example = Some(State::init(
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
        _ => example.as_mut().unwrap().update(event),
      },
      _ => {}
    },
  );
}

pub fn run() {
  pollster::block_on(start());
}
