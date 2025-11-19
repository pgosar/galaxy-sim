use crate::{initialize, Particle, SimParams};
use std::borrow::Cow;
use wgpu::{util::DeviceExt, PipelineCompilationOptions};

pub struct Render {
  particle_bind_groups: Vec<wgpu::BindGroup>,
  particle_buffers: Vec<wgpu::Buffer>,
  vertices_buffer: wgpu::Buffer,
  compute_pipeline: wgpu::ComputePipeline,
  render_pipeline: wgpu::RenderPipeline,
  work_group_count: u32,
  frame_num: usize,
  sim_param_buffer: wgpu::Buffer,
}

impl Render {
  #[must_use]
  #[allow(clippy::too_many_lines)]
  pub fn init(
    config: &wgpu::SurfaceConfiguration,
    _adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    _queue: &wgpu::Queue,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    sim_params: SimParams,
  ) -> Self {
    let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("compute_shader"),
      source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/compute.wgsl"))),
    });
    let draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("draw_shader"),
      source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/draw.wgsl"))),
    });
    let sim_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Simulation Parameter Buffer"),
      contents: bytemuck::cast_slice(&[sim_params]),
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
              min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<SimParams>() as _),
            },
            count: None,
          },
          wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
              ty: wgpu::BufferBindingType::Storage { read_only: true },
              has_dynamic_offset: false,
              min_binding_size: wgpu::BufferSize::new(
                ((sim_params.num_particles * sim_params.num_galaxies) as usize
                  * std::mem::size_of::<Particle>()) as _,
              ),
            },
            count: None,
          },
          wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
              ty: wgpu::BufferBindingType::Storage { read_only: false },
              has_dynamic_offset: false,
              min_binding_size: wgpu::BufferSize::new(
                ((sim_params.num_particles * sim_params.num_galaxies) as usize
                  * std::mem::size_of::<Particle>()) as _,
              ),
            },
            count: None,
          },
        ],
        label: Some("compute_bind_group_layout"),
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
    let particle_buffer = wgpu::VertexBufferLayout {
      array_stride: std::mem::size_of::<Particle>() as u64,
      step_mode: wgpu::VertexStepMode::Instance,
      attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32],
    };
    let vertex_buffer = wgpu::VertexBufferLayout {
      array_stride: 3 * 4, // vertex data
      step_mode: wgpu::VertexStepMode::Vertex,
      attributes: &wgpu::vertex_attr_array![3 => Float32x3],
    };
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Render Pipeline"),
      layout: Some(&render_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &draw_shader,
        entry_point: "main_vs",
        compilation_options: PipelineCompilationOptions::default(),

        buffers: &[particle_buffer, vertex_buffer],
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
    let vertex_buffer_data = [
      // First vertex (bottom left)
      -0.866 * sim_params.triangle_size,
      -0.5 * sim_params.triangle_size,
      0.0,
      // Second vertex (bottom right)
      0.866 * sim_params.triangle_size,
      -0.5 * sim_params.triangle_size,
      0.0,
      // Third vertex (top)
      0.0,
      sim_params.triangle_size,
      0.0,
    ];
    let vertices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Vertex Buffer"),
      contents: bytemuck::bytes_of(&vertex_buffer_data),
      usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    let initial_particle_data = initialize::create_galaxies(&sim_params);
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
        label: Some(&format!("Particle Bind Group {i}")),
      }));
    }
    #[allow(
      clippy::cast_possible_truncation,
      clippy::cast_sign_loss,
      clippy::cast_precision_loss
    )]
    let work_group_count = (((sim_params.num_particles * sim_params.num_galaxies) as f32)
      / (sim_params.particles_per_group as f32))
      .ceil() as u32;
    Render {
      particle_bind_groups,
      particle_buffers,
      vertices_buffer,
      compute_pipeline,
      render_pipeline,
      work_group_count,
      frame_num: 0,
      sim_param_buffer,
    }
  }

  pub fn render(
    &mut self,
    view: &wgpu::TextureView,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    camera_bind_group: &wgpu::BindGroup,
    sim_params: &SimParams,
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
      label: Some("Render Pass Descriptor"),
      color_attachments: &color_attachments,
      depth_stencil_attachment: None,
      timestamp_writes: None,
      occlusion_query_set: None,
    };
    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
      label: Some("Command Encoder Descriptor"),
    });

    queue.write_buffer(
      &self.sim_param_buffer,
      0,
      bytemuck::cast_slice(&[*sim_params]),
    );

    // Compute pass
    {
      let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Compute Pass Descriptor"),
        timestamp_writes: None,
      });
      cpass.set_pipeline(&self.compute_pipeline);
      cpass.set_bind_group(0, &self.particle_bind_groups[self.frame_num % 2], &[]);
      cpass.dispatch_workgroups(self.work_group_count, 1, 1);
    }
    // Render pass
    {
      let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
      rpass.set_pipeline(&self.render_pipeline);
      rpass.set_bind_group(0, camera_bind_group, &[]);
      rpass.set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2].slice(..));
      rpass.set_vertex_buffer(1, self.vertices_buffer.slice(..));
      rpass.draw(0..3, 0..sim_params.num_particles * sim_params.num_galaxies);
    }
    command_encoder.pop_debug_group();
    self.frame_num += 1;
    queue.submit(Some(command_encoder.finish()));
  }
}
