use crate::{Particle, SimParams};
use std::borrow::Cow;
use std::collections::HashMap;
use wgpu::PipelineCompilationOptions;

// Must match BHNode in barnes_hut.wgsl. Field offsets: com@0, mass@12,
// bb_min@16, bb_max@32, left@44, right@48, leaf_lo@52, leaf_cnt@56;
// struct size 60 rounds up to stride 64 (16-byte alignment).
const NODE_BYTES: usize = 64;
// Must match Center in barnes_hut.wgsl.
const CENTER_BYTES: usize = 48;
const SORT_RADIX_PASSES: usize = 4;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RoundParams {
  level: u32,
  src_off: u32,
  dst_off: u32,
  dst_cnt: u32,
  src_cnt: u32,
  n: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Center {
  pos: [f32; 4],
  vel: [f32; 4],
  mass: f32,
  galaxy_id: u32,
  _pad: [f32; 2],
}

fn compute_pipeline(
  device: &wgpu::Device,
  layout: &wgpu::PipelineLayout,
  module: &wgpu::ShaderModule,
  entry_point: &str,
  shift: Option<u32>,
) -> wgpu::ComputePipeline {
  let mut constants: HashMap<String, f64> = HashMap::new();
  if let Some(s) = shift {
    constants.insert("SHIFT".to_string(), f64::from(s));
  }
  device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: Some(&format!("BH {entry_point}")),
    layout: Some(layout),
    module,
    entry_point,
    compilation_options: PipelineCompilationOptions {
      constants: &constants,
      ..PipelineCompilationOptions::default()
    },
    cache: None,
  })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
  wgpu::BindGroupLayoutEntry {
    binding,
    visibility: wgpu::ShaderStages::COMPUTE,
    ty: wgpu::BindingType::Buffer {
      ty: wgpu::BufferBindingType::Storage { read_only },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  }
}

pub struct BarnesHut {
  morton_pipe: wgpu::ComputePipeline,
  zero_pipe: wgpu::ComputePipeline,
  hist_pipes: Vec<wgpu::ComputePipeline>,
  scan_pipe: wgpu::ComputePipeline,
  scatter_pipes: Vec<wgpu::ComputePipeline>,
  gather_pipe: wgpu::ComputePipeline,
  build_pipe: wgpu::ComputePipeline,
  accumulate_pipe: wgpu::ComputePipeline,
  traverse_pipe: wgpu::ComputePipeline,
  main_bg: Vec<wgpu::BindGroup>,
  sort_bg: Vec<wgpu::BindGroup>,
  round_buffer: wgpu::Buffer,
  rounds: Vec<RoundParams>,
  workgroup_count: u32,
}

impl BarnesHut {
  #[allow(clippy::too_many_lines)]
  pub fn init(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sim_params: &SimParams,
    initial_particles: &[Particle],
    sim_param_buffer: &wgpu::Buffer,
    particle_buffers: &[wgpu::Buffer],
  ) -> Self {
    let n = sim_params.num_particles * sim_params.num_galaxies;
    let particle_bytes = std::mem::size_of::<Particle>() as u64;
    let storage_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

    // Bottom-up rounds, precomputed: level r covers 2^r leaves per node.
    let mut rounds = Vec::new();
    let mut cnt = n.div_ceil(2);
    let mut dst_off = 0u32;
    let mut src_off = 0u32;
    let mut src_cnt = n;
    let mut level = 1u32;
    loop {
      rounds.push(RoundParams {
        level,
        src_off,
        dst_off,
        dst_cnt: cnt,
        src_cnt,
        n,
      });
      if cnt == 1 {
        break;
      }
      src_off = dst_off;
      src_cnt = cnt;
      dst_off += cnt;
      cnt = cnt.div_ceil(2);
      level += 1;
    }
    // Internal node count exceeds n-1 when n is not a power of two because
    // single-child nodes are allowed.
    let node_count: u32 = rounds.iter().map(|r| r.dst_cnt).sum();

    let keys_a = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("BH keys A"),
      size: u64::from(n) * 4,
      usage: storage_usage,
      mapped_at_creation: false,
    });
    let keys_b = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("BH keys B"),
      size: u64::from(n) * 4,
      usage: storage_usage,
      mapped_at_creation: false,
    });
    let vals_a = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("BH vals A"),
      size: u64::from(n) * 4,
      usage: storage_usage,
      mapped_at_creation: false,
    });
    let vals_b = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("BH vals B"),
      size: u64::from(n) * 4,
      usage: storage_usage,
      mapped_at_creation: false,
    });
    let new_u32_buffer = |label: &str| {
      device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: 256 * 4,
        usage: storage_usage,
        mapped_at_creation: false,
      })
    };
    let hist = new_u32_buffer("BH hist");
    let offsets = new_u32_buffer("BH offsets");
    let cursors = new_u32_buffer("BH cursors");
    let nodes = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("BH nodes"),
      size: u64::from(node_count.max(1)) * NODE_BYTES as u64,
      usage: storage_usage,
      mapped_at_creation: false,
    });
    let scratch = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("BH scratch particles"),
      size: u64::from(n) * particle_bytes,
      usage: storage_usage,
      mapped_at_creation: false,
    });
    let new_centers = |label: &str| {
      device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: u64::from(sim_params.num_galaxies) * CENTER_BYTES as u64,
        usage: storage_usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
      })
    };
    let centers_a = new_centers("BH centers A");
    let centers_b = new_centers("BH centers B");
    // Seed both center buffers with the initial galaxy centers; the traverse
    // kernel ping-pongs between them, reading the previous step's values.
    let mut centers_init = Vec::with_capacity(sim_params.num_galaxies as usize);
    for g in 0..sim_params.num_galaxies {
      let p = &initial_particles[(g * sim_params.num_particles) as usize];
      centers_init.push(Center {
        pos: [p.pos[0], p.pos[1], p.pos[2], 0.0],
        vel: [p.vel[0], p.vel[1], p.vel[2], 0.0],
        mass: p.mass,
        galaxy_id: p.galaxy_id,
        _pad: [0.0, 0.0],
      });
    }
    queue.write_buffer(&centers_a, 0, bytemuck::cast_slice(&centers_init));
    queue.write_buffer(&centers_b, 0, bytemuck::cast_slice(&centers_init));

    let round_buffer = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("BH round params"),
      size: (rounds.len() * 256) as u64,
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    // One 256-byte slot per round (dynamic-offset alignment); written once.
    let mut round_bytes = vec![0u8; rounds.len() * 256];
    for (i, rp) in rounds.iter().enumerate() {
      let bytes = bytemuck::bytes_of(rp);
      round_bytes[i * 256..i * 256 + bytes.len()].copy_from_slice(bytes);
    }
    queue.write_buffer(&round_buffer, 0, &round_bytes);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("barnes_hut_shader"),
      source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/barnes_hut.wgsl"))),
    });

    let main_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label: Some("bh_main_layout"),
      entries: &[
        wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::COMPUTE,
          ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          count: None,
        },
        storage_entry(1, true),
        storage_entry(2, false),
        storage_entry(3, false),
        storage_entry(4, false),
        storage_entry(5, false),
        storage_entry(6, false),
        storage_entry(7, true),
        storage_entry(8, false),
        wgpu::BindGroupLayoutEntry {
          binding: 9,
          visibility: wgpu::ShaderStages::COMPUTE,
          ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<RoundParams>() as u64),
          },
          count: None,
        },
      ],
    });
    let sort_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label: Some("bh_sort_layout"),
      // All read_write: the key/value buffers are also bound read_write in
      // the main layout, and mixing declared usages in one pass is an error.
      entries: &[
        storage_entry(0, false),
        storage_entry(1, false),
        storage_entry(2, false),
        storage_entry(3, false),
        storage_entry(4, false),
        storage_entry(5, false),
        storage_entry(6, false),
      ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("bh_pipeline_layout"),
      bind_group_layouts: &[&main_layout, &sort_layout],
      push_constant_ranges: &[],
    });

    fn entry(binding: u32, resource: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
      wgpu::BindGroupEntry {
        binding,
        resource: resource.as_entire_binding(),
      }
    }
    // Frame parity: particle/center buffers ping-pong; scratch is fixed.
    let mut main_bg = Vec::with_capacity(2);
    for s in 0..2 {
      main_bg.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &main_layout,
        entries: &[
          entry(0, sim_param_buffer),
          entry(1, &particle_buffers[s]),
          entry(2, &scratch),
          entry(3, &particle_buffers[1 - s]),
          entry(4, &keys_a),
          entry(5, &vals_a),
          entry(6, &nodes),
          entry(7, if s == 0 { &centers_a } else { &centers_b }),
          entry(8, if s == 0 { &centers_b } else { &centers_a }),
          // Explicit small range so nonzero dynamic offsets stay in bounds.
          wgpu::BindGroupEntry {
            binding: 9,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
              buffer: &round_buffer,
              offset: 0,
              size: wgpu::BufferSize::new(std::mem::size_of::<RoundParams>() as u64),
            }),
          },
        ],
        label: Some(&format!("BH main BG {s}")),
      }));
    }
    // Sort parity: key/value buffers swap source and destination per pass.
    let mut sort_bg = Vec::with_capacity(2);
    for p in 0..2 {
      let (ka, kb) = if p == 0 {
        (&keys_a, &keys_b)
      } else {
        (&keys_b, &keys_a)
      };
      let (va, vb) = if p == 0 {
        (&vals_a, &vals_b)
      } else {
        (&vals_b, &vals_a)
      };
      sort_bg.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &sort_layout,
        entries: &[
          entry(0, ka),
          entry(1, kb),
          entry(2, va),
          entry(3, vb),
          entry(4, &hist),
          entry(5, &offsets),
          entry(6, &cursors),
        ],
        label: Some(&format!("BH sort BG {p}")),
      }));
    }

    let morton_pipe = compute_pipeline(&device, &pipeline_layout, &shader, "morton_main", None);
    let zero_pipe = compute_pipeline(&device, &pipeline_layout, &shader, "sort_zero_main", None);
    let hist_pipes = (0..SORT_RADIX_PASSES)
      .map(|p| {
        compute_pipeline(
          &device,
          &pipeline_layout,
          &shader,
          "sort_hist_main",
          Some((p * 8) as u32),
        )
      })
      .collect();
    let scan_pipe = compute_pipeline(&device, &pipeline_layout, &shader, "sort_scan_main", None);
    let scatter_pipes = (0..SORT_RADIX_PASSES)
      .map(|p| {
        compute_pipeline(
          &device,
          &pipeline_layout,
          &shader,
          "sort_scatter_main",
          Some((p * 8) as u32),
        )
      })
      .collect();
    let gather_pipe = compute_pipeline(&device, &pipeline_layout, &shader, "gather_main", None);
    let build_pipe = compute_pipeline(&device, &pipeline_layout, &shader, "build_main", None);
    let accumulate_pipe =
      compute_pipeline(&device, &pipeline_layout, &shader, "accumulate_main", None);
    let traverse_pipe = compute_pipeline(&device, &pipeline_layout, &shader, "traverse_main", None);

    #[allow(clippy::cast_possible_truncation)]
    let workgroup_count = n.div_ceil(256);

    Self {
      morton_pipe,
      zero_pipe,
      hist_pipes,
      scan_pipe,
      scatter_pipes,
      gather_pipe,
      build_pipe,
      accumulate_pipe,
      traverse_pipe,
      main_bg,
      sort_bg,
      round_buffer,
      rounds,
      workgroup_count,
    }
  }

  /// One full Barnes-Hut step. `frame` selects the ping-pong bind group.
  /// Morton-codes the particles, radix-sorts them, rebuilds the tree
  /// bottom-up, then traverses it per particle.
  pub fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: usize) {
    let s = frame % 2;
    let wgc = self.workgroup_count;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
      label: Some("BH step encoder"),
    });
    {
      let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("BH step pass"),
        timestamp_writes: None,
      });
      pass.set_bind_group(0, &self.main_bg[s], &[0]);
      // The pipeline layout always carries group 1; bind it even for
      // kernels that only read group 0.
      pass.set_bind_group(1, &self.sort_bg[0], &[]);
      pass.set_pipeline(&self.morton_pipe);
      pass.dispatch_workgroups(wgc, 1, 1);
      for p in 0..SORT_RADIX_PASSES {
        pass.set_bind_group(1, &self.sort_bg[p % 2], &[]);
        pass.set_pipeline(&self.zero_pipe);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.hist_pipes[p]);
        pass.dispatch_workgroups(wgc, 1, 1);
        pass.set_pipeline(&self.scan_pipe);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.scatter_pipes[p]);
        pass.dispatch_workgroups(wgc, 1, 1);
      }
      pass.set_bind_group(0, &self.main_bg[s], &[0]);
      pass.set_bind_group(1, &self.sort_bg[0], &[]);
      pass.set_pipeline(&self.gather_pipe);
      pass.dispatch_workgroups(wgc, 1, 1);
      pass.set_pipeline(&self.build_pipe);
      pass.dispatch_workgroups(wgc, 1, 1);
      // Bottom-up mass/COM: round params live in one uniform buffer, selected
      // per dispatch via dynamic offset (no write_buffer between dispatches).
      pass.set_pipeline(&self.accumulate_pipe);
      for (r, rp) in self.rounds.iter().enumerate() {
        pass.set_bind_group(0, &self.main_bg[s], &[(r * 256) as u32]);
        pass.dispatch_workgroups(rp.dst_cnt.div_ceil(256), 1, 1);
      }
      pass.set_bind_group(0, &self.main_bg[s], &[0]);
      pass.set_bind_group(1, &self.sort_bg[0], &[]);
      pass.set_pipeline(&self.traverse_pipe);
      pass.dispatch_workgroups(wgc, 1, 1);
    }
    queue.submit(Some(encoder.finish()));
  }
}
