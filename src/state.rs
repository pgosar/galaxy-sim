use crate::{
  camera::{Camera, CameraController, CameraUniform},
  render::Render,
  CameraParams, Particle, RunConfig, SimParams,
};
use std::{sync::Arc, time::Instant};
use wgpu::util::DeviceExt;
use wgpu::MemoryHints;
use winit::{
  dpi::PhysicalSize,
  event::{ElementState, Event, KeyEvent, StartCause, WindowEvent},
  event_loop::{EventLoop, EventLoopWindowTarget},
  keyboard::{KeyCode, PhysicalKey},
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

  fn suspend(&mut self) {
    // No-op: surface cleanup handled by drop
  }

  fn config(&self) -> &wgpu::SurfaceConfiguration {
    self.config.as_ref().unwrap()
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

  async fn init(surface: Option<&SurfaceWrapper>, size: &PhysicalSize<u32>) -> Self {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
      backends: wgpu::Backends::PRIMARY,
      ..Default::default()
    });

    let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: surface.and_then(|s| s.surface.as_ref()),
        force_fallback_adapter: false,
      })
      .await
      .unwrap();

    let (device, queue) = adapter
      .request_device(
        &wgpu::DeviceDescriptor {
          label: Some("Device Descriptor"),
          required_features: wgpu::Features::empty(),
          required_limits: wgpu::Limits::default(),
          memory_hints: MemoryHints::default(),
        },
        None,
      )
      .await
      .unwrap();
    let camera = Camera {
      // position the camera 1 unit up and 2 units back
      eye: (0.0, 1.0, 2.0).into(),
      target: (0.0, 0.0, 0.0).into(),
      up: cgmath::Vector3::unit_y(),
      aspect: size.width as f32 / size.height as f32,
      fovy: 45.0,
      znear: 0.1,
      zfar: 100.0,
    };
    let mut camera_uniform = CameraUniform::init();
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
    let camera_params = CameraParams::default();
    let camera_controller =
      CameraController::init(camera_params.speed, camera_params.rotational_speed);

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

pub async fn start(config: RunConfig) {
  env_logger::init();
  let mut sim_params = SimParams::default();
  sim_params.num_galaxies = config.galaxies;
  if let Some(particles) = config.particles {
    sim_params.num_particles = particles;
  }
  if let Some(theta) = config.theta {
    sim_params.theta = theta;
  }
  let headless = config.headless;

  if headless {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
      backends: wgpu::Backends::PRIMARY,
      ..Default::default()
    });

    let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
      })
      .await
      .unwrap();

    let (device, queue) = adapter
      .request_device(
        &wgpu::DeviceDescriptor {
          label: Some("Device Descriptor"),
          required_features: wgpu::Features::empty(),
          required_limits: wgpu::Limits::default(),
          memory_hints: MemoryHints::default(),
        },
        None,
      )
      .await
      .unwrap();

    // Offscreen render target so headless runs can save snapshots.
    let headless_render = if config.snapshot_every > 0 {
      Some(HeadlessRender::init(&device, &sim_params))
    } else {
      None
    };

    let mut renderer = Render::init(
      headless_render.as_ref().map(|hr| &hr.config),
      &adapter,
      &device,
      &queue,
      headless_render.as_ref().map(|hr| &hr.camera_layout),
      sim_params,
      config.exact,
    );
    let mut frame_count = 0;
    let mut frame_deltas = Vec::new();

    let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
      r.store(false, std::sync::atomic::Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    println!("Running in headless mode. Press Ctrl+C to exit.");

    let mut last_frame_time = Instant::now();
    let mut timer = Instant::now();

    std::fs::create_dir_all(&config.out_dir).expect("could not create output dir");
    let mut step: u64 = 0;
    while running.load(std::sync::atomic::Ordering::SeqCst)
      && (config.max_steps == 0 || step < config.max_steps)
    {
      let now = Instant::now();
      let delta = now.duration_since(last_frame_time);
      last_frame_time = now;

      frame_deltas.push(delta.as_secs_f32());

      if timer.elapsed().as_secs_f32() >= 1.0 {
        println!(
          "FPS: {:.2}, Time: {:.2}",
          frame_count as f32 / timer.elapsed().as_secs_f32(),
          sim_params.time
        );
        timer = Instant::now();
        frame_count = 0;
      }

      sim_params.time += sim_params.delta_t;
      if let Some(hr) = &headless_render {
        if step.is_multiple_of(config.snapshot_every) {
          renderer.render(
            &hr.view,
            &device,
            &queue,
            &hr.camera_bind_group,
            &sim_params,
            true,
          );
          hr.save_snapshot(
            &device,
            &queue,
            &format!("{}/snap_{step:06}.png", config.out_dir),
          );
        } else {
          renderer.compute(&device, &queue, &sim_params);
        }
      } else {
        renderer.compute(&device, &queue, &sim_params);
      }
      // Block until the GPU finishes this step: steps are sequential anyway
      // (each reads the previous step's buffer), and without this the FPS
      // counter would only measure CPU submission throughput.
      device.poll(wgpu::Maintain::Wait);
      if config.dump_every > 0 && step.is_multiple_of(config.dump_every) {
        dump_particles(
          &device,
          &queue,
          &renderer,
          &sim_params,
          &format!("{}/dump_{step:06}.csv", config.out_dir),
        );
      }
      step += 1;
      frame_count += 1;
    }

    println!("\nSimulation stopped after {step} steps.");
    if !frame_deltas.is_empty() {
      let total_time: f32 = frame_deltas.iter().sum();
      let avg_fps = frame_deltas.len() as f32 / total_time;

      // find 1% lows
      frame_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
      let one_percent_index = (frame_deltas.len() as f32 * 0.99) as usize;
      let low_1_percent_delta = frame_deltas[one_percent_index];
      let low_1_percent_fps = 1.0 / low_1_percent_delta;

      println!("Average FPS: {:.2}", avg_fps);
      println!("1% Low FPS:  {:.2}", low_1_percent_fps);
    }
    // Headless runs never open a window.
    return;
  }

  let window_loop = EventLoopWrapper::new("Galaxy Sim");
  let mut surface = SurfaceWrapper::new();
  let mut context = State::init(Some(&surface), &window_loop.window.inner_size()).await;
  let event_loop_function = EventLoop::run;
  let mut example = None;
  let mut tick = Instant::now();

  // main runner
  let _ = (event_loop_function)(
    window_loop.event_loop,
    move |event, target: &EventLoopWindowTarget<()>| match event {
      Event::NewEvents(StartCause::Init) => {
        surface.resume(&context, window_loop.window.clone());
        if example.is_none() {
          example = Some(Render::init(
            Some(surface.config()),
            &context.adapter,
            &context.device,
            &context.queue,
            Some(&context.camera_bind_group_layout),
            sim_params,
            config.exact,
          ));
        }
      }
      Event::Suspended => {
        surface.suspend();
      }
      Event::WindowEvent { event, window_id } if window_id == window_loop.window.id() => {
        // need to save whether escape key was sent before it is consumed by input()
        let mut exit_requested = false;
        if let WindowEvent::KeyboardInput {
          event:
            KeyEvent {
              state: ElementState::Pressed,
              physical_key: PhysicalKey::Code(KeyCode::Escape),
              ..
            },
          ..
        } = event
        {
          exit_requested = true;
        }
        if let WindowEvent::KeyboardInput {
          event:
            KeyEvent {
              state: ElementState::Pressed,
              physical_key: PhysicalKey::Code(KeyCode::KeyF),
              ..
            },
          ..
        } = event
        {
          let delta = tick.elapsed();
          println!("delta: {:?}, fps: {:.2}", delta, 1.0 / delta.as_secs_f32());
        }
        if exit_requested {
          target.exit();
        } else if !context.input(&event) {
          match event {
            WindowEvent::CloseRequested => target.exit(),
            WindowEvent::RedrawRequested => {
              window_loop.window.request_redraw();
              if example.is_none() {
                return;
              }
              tick = Instant::now();
              sim_params.time += sim_params.delta_t;
              context.update();
              if let Some(example) = &mut example {
                let frame = surface.acquire(&context);
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                  format: Some(surface.config().view_formats[0]),
                  ..wgpu::TextureViewDescriptor::default()
                });
                // start rendering
                example.render(
                  &view,
                  &context.device,
                  &context.queue,
                  &context.camera_bind_group,
                  &sim_params,
                  false,
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

/// Offscreen camera + render target for headless snapshots.
struct HeadlessRender {
  config: wgpu::SurfaceConfiguration,
  camera_layout: wgpu::BindGroupLayout,
  camera_bind_group: wgpu::BindGroup,
  texture: wgpu::Texture,
  view: wgpu::TextureView,
  width: u32,
  height: u32,
}

impl HeadlessRender {
  fn init(device: &wgpu::Device, sim_params: &SimParams) -> Self {
    let width = 1280;
    let height = 720;
    // Frame single galaxies closer; mergers need the wider view.
    let eye: cgmath::Point3<f32> = if sim_params.num_galaxies > 1 {
      (0.0, 2.0, 4.0).into()
    } else {
      (0.0, 1.0, 2.0).into()
    };
    let camera = Camera {
      eye,
      target: (0.0, 0.0, 0.0).into(),
      up: cgmath::Vector3::unit_y(),
      aspect: width as f32 / height as f32,
      fovy: 45.0,
      znear: 0.1,
      zfar: 100.0,
    };
    let mut camera_uniform = CameraUniform::init();
    camera_uniform.update_view_proj(&camera);
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Headless Camera Buffer"),
      contents: bytemuck::cast_slice(&[camera_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
      label: Some("headless_camera_bind_group_layout"),
    });
    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &camera_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: camera_buffer.as_entire_binding(),
      }],
      label: Some("headless_camera_bind_group"),
    });
    // Fabricated config: render.rs only needs format + view_formats from it.
    let linear = wgpu::TextureFormat::Rgba8Unorm;
    let config = wgpu::SurfaceConfiguration {
      usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
      format: linear,
      width,
      height,
      present_mode: wgpu::PresentMode::Fifo,
      desired_maximum_frame_latency: 2,
      alpha_mode: wgpu::CompositeAlphaMode::Auto,
      view_formats: vec![linear.add_srgb_suffix()],
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
      label: Some("Snapshot Texture"),
      size: wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
      },
      mip_level_count: 1,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format: linear.add_srgb_suffix(),
      usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
      view_formats: &[linear.add_srgb_suffix()],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
      format: Some(linear.add_srgb_suffix()),
      ..wgpu::TextureViewDescriptor::default()
    });
    Self {
      config,
      camera_layout,
      camera_bind_group,
      texture,
      view,
      width,
      height,
    }
  }

  fn save_snapshot(&self, device: &wgpu::Device, queue: &wgpu::Queue, path: &str) {
    let bytes_per_pixel = 4u32;
    let unpadded = self.width * bytes_per_pixel;
    let padded =
      unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("Snapshot Staging Buffer"),
      size: u64::from(padded * self.height),
      usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
      mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
      label: Some("Snapshot Encoder"),
    });
    encoder.copy_texture_to_buffer(
      wgpu::ImageCopyTexture {
        texture: &self.texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
      },
      wgpu::ImageCopyBuffer {
        buffer: &staging,
        layout: wgpu::ImageDataLayout {
          offset: 0,
          bytes_per_row: Some(padded),
          rows_per_image: Some(self.height),
        },
      },
      wgpu::Extent3d {
        width: self.width,
        height: self.height,
        depth_or_array_layers: 1,
      },
    );
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();
    // Strip row padding before writing the PNG.
    let mut pixels = Vec::with_capacity((unpadded * self.height) as usize);
    for row in 0..self.height {
      let start = (row * padded) as usize;
      pixels.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    staging.unmap();
    image::save_buffer(
      path,
      &pixels,
      self.width,
      self.height,
      image::ColorType::Rgba8,
    )
    .expect("could not write snapshot");
    println!("wrote {path}");
  }
}

/// Copy the latest particle state back to the CPU and write a CSV.
fn dump_particles(
  device: &wgpu::Device,
  queue: &wgpu::Queue,
  renderer: &Render,
  sim_params: &SimParams,
  path: &str,
) {
  let total = (sim_params.num_particles * sim_params.num_galaxies) as usize;
  let byte_size = (total * std::mem::size_of::<Particle>()) as u64;
  let staging = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("Dump Staging Buffer"),
    size: byte_size,
    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    mapped_at_creation: false,
  });
  let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
    label: Some("Dump Encoder"),
  });
  encoder.copy_buffer_to_buffer(renderer.latest_particle_buffer(), 0, &staging, 0, byte_size);
  queue.submit(Some(encoder.finish()));
  let slice = staging.slice(..);
  slice.map_async(wgpu::MapMode::Read, |_| {});
  device.poll(wgpu::Maintain::Wait);
  let data = slice.get_mapped_range();
  let particles: &[Particle] = bytemuck::cast_slice(&data);
  let mut csv = String::with_capacity(total * 64);
  csv.push_str("x,y,z,vx,vy,vz,ax,ay,az,galaxy_id\n");
  for p in particles {
    csv.push_str(&format!(
      "{},{},{},{},{},{},{},{},{},{}\n",
      p.pos[0],
      p.pos[1],
      p.pos[2],
      p.vel[0],
      p.vel[1],
      p.vel[2],
      p.acc[0],
      p.acc[1],
      p.acc[2],
      p.galaxy_id
    ));
  }
  drop(data);
  staging.unmap();
  std::fs::write(path, csv).expect("could not write dump");
  println!("wrote {path}");
}

pub fn run(config: RunConfig) {
  pollster::block_on(start(config));
}
