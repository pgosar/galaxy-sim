use crate::state::run;
use nanorand::{Rng, WyRand};
use std::{borrow::Cow, mem};
use wgpu::{util::DeviceExt, PipelineCompilationOptions};
const NUM_PARTICLES: u32 = 1500;
const PARTICLES_PER_GROUP: u32 = 64;

pub struct Render {
  particle_bind_groups: Vec<wgpu::BindGroup>,
  particle_buffers: Vec<wgpu::Buffer>,
  vertices_buffer: wgpu::Buffer,
  compute_pipeline: wgpu::ComputePipeline,
  render_pipeline: wgpu::RenderPipeline,
  work_group_count: u32,
  frame_num: usize,
}

impl Render {
  #[must_use]
  pub fn init(
    config: &wgpu::SurfaceConfiguration,
    _adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    _queue: &wgpu::Queue,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
  ) -> Self {
    let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: None,
      source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/compute.wgsl"))),
    });
    let draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: None,
      source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/draw.wgsl"))),
    });
    let sim_param_data = [
      0.04f32, // deltaT
      0.1,     // rule1Distance
      0.025,   // rule2Distance
      0.025,   // rule3Distance
      0.02,    // rule1Scale
      0.05,    // rule2Scale
      0.005,   // rule3Scale
    ]
    .to_vec();
    let sim_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Simulation Parameter Buffer"),
      contents: bytemuck::cast_slice(&sim_param_data),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // ========================================================================
    // compute pipeline stuff
    // ========================================================================

    let compute_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
          wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
              ty: wgpu::BufferBindingType::Uniform,
              has_dynamic_offset: false,
              min_binding_size: wgpu::BufferSize::new(
                (sim_param_data.len() * mem::size_of::<f32>()) as _,
              ),
            },
            count: None,
          },
          wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
              ty: wgpu::BufferBindingType::Storage { read_only: true },
              has_dynamic_offset: false,
              min_binding_size: wgpu::BufferSize::new((u64::from(NUM_PARTICLES) * 16) as _),
            },
            count: None,
          },
          wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
              ty: wgpu::BufferBindingType::Storage { read_only: false },
              has_dynamic_offset: false,
              min_binding_size: wgpu::BufferSize::new((u64::from(NUM_PARTICLES) * 16) as _),
            },
            count: None,
          },
        ],
        label: None,
      });
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("compute"),
      bind_group_layouts: &[&compute_bind_group_layout],
      push_constant_ranges: &[],
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: Some("Compute pipeline"),
      layout: Some(&compute_pipeline_layout),
      module: &compute_shader,
      entry_point: "main",
      compilation_options: PipelineCompilationOptions::default(),
      cache: None,
    });

    // ========================================================================
    // render pipeline stuff
    // ========================================================================

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("render"),
      bind_group_layouts: &[camera_bind_group_layout],
      push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: None,
      layout: Some(&render_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &draw_shader,
        entry_point: "main_vs",
        compilation_options: PipelineCompilationOptions::default(),
        buffers: &[
          wgpu::VertexBufferLayout {
            array_stride: 6 * 4,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
          },
          wgpu::VertexBufferLayout {
            array_stride: 2 * 4,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![2 => Float32x3],
          },
        ],
      },
      fragment: Some(wgpu::FragmentState {
        module: &draw_shader,
        entry_point: "main_fs",
        compilation_options: PipelineCompilationOptions::default(),
        targets: &[Some(config.view_formats[0].into())],
      }),
      primitive: wgpu::PrimitiveState::default(),
      depth_stencil: None,
      multisample: wgpu::MultisampleState::default(),
      multiview: None,
      cache: None,
    });

    // triangle sizes
    let vertex_buffer_data = [
      -0.01f32, -0.02, 0.0, // First vertex
      0.01, -0.02, 0.0, // Second vertex
      0.00, 0.02, 0.0, // Third vertex
    ];
    let vertices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Vertex Buffer"),
      contents: bytemuck::bytes_of(&vertex_buffer_data),
      usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    let mut initial_particle_data = vec![0.0f32; (6 * NUM_PARTICLES) as usize];
    let mut rng = WyRand::new_seed(42);
    let mut unif = || rng.generate::<f32>() * 2f32 - 1f32;
    // randomly generate initial positions and velocities for each particle
    for particle_instance_chunk in initial_particle_data.chunks_mut(6) {
      particle_instance_chunk[0] = unif(); // posx
      particle_instance_chunk[1] = unif(); // posy
      particle_instance_chunk[2] = unif(); // posz
      particle_instance_chunk[3] = unif() * 0.1; // velx
      particle_instance_chunk[4] = unif() * 0.1; // vely
      particle_instance_chunk[5] = unif() * 0.1; // velz
    }
    let mut particle_buffers = Vec::<wgpu::Buffer>::new();
    let mut particle_bind_groups = Vec::<wgpu::BindGroup>::new();
    for i in 0..2 {
      particle_buffers.push(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
          label: Some(&format!("Particle Buffer {i}")),
          contents: bytemuck::cast_slice(&initial_particle_data),
          usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        }),
      );
    }
    for i in 0..2 {
      particle_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &compute_bind_group_layout,
        entries: &[
          wgpu::BindGroupEntry {
            binding: 0,
            resource: sim_param_buffer.as_entire_binding(),
          },
          wgpu::BindGroupEntry {
            binding: 1,
            resource: particle_buffers[i].as_entire_binding(),
          },
          wgpu::BindGroupEntry {
            binding: 2,
            resource: particle_buffers[(i + 1) % 2].as_entire_binding(),
          },
        ],
        label: None,
      }));
    }
    let work_group_count = ((NUM_PARTICLES as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;
    Render {
      particle_bind_groups,
      particle_buffers,
      vertices_buffer,
      compute_pipeline,
      render_pipeline,
      work_group_count,
      frame_num: 0,
    }
  }

  pub fn render(
    &mut self,
    view: &wgpu::TextureView,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    camera_bind_group: &wgpu::BindGroup,
  ) {
    let color_attachments = [Some(wgpu::RenderPassColorAttachment {
      view,
      resolve_target: None,
      ops: wgpu::Operations {
        load: wgpu::LoadOp::Load,
        store: wgpu::StoreOp::Store,
      },
    })];
    let render_pass_descriptor = wgpu::RenderPassDescriptor {
      label: None,
      color_attachments: &color_attachments,
      depth_stencil_attachment: None,
      timestamp_writes: None,
      occlusion_query_set: None,
    };
    let mut command_encoder =
      device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    // compute pass
    {
      let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
      });
      cpass.set_pipeline(&self.compute_pipeline);
      cpass.set_bind_group(0, &self.particle_bind_groups[self.frame_num % 2], &[]);
      cpass.dispatch_workgroups(self.work_group_count, 1, 1);
    }
    // render pass
    {
      let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
      rpass.set_pipeline(&self.render_pipeline);
      rpass.set_bind_group(0, camera_bind_group, &[]);
      rpass.set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2].slice(..));
      rpass.set_vertex_buffer(1, self.vertices_buffer.slice(..));
      rpass.draw(0..3, 0..NUM_PARTICLES);
    }
    command_encoder.pop_debug_group();
    self.frame_num += 1;
    queue.submit(Some(command_encoder.finish()));
  }
}

pub fn main() {
  run();
}
