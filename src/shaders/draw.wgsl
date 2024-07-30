struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) particle_pos: vec3<f32>,
    @location(1) particle_vel: vec3<f32>,
    @location(2) mass: f32,
    @location(3) position: vec3<f32>,
    @builtin(instance_index) particle_index: u32,  // Add this line to get the particle index
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn main_vs(
    model: VertexInput,
) -> VertexOutput {
    let offset_position = model.particle_pos + model.position;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(offset_position, 1.0);

    // Determine color based on the particle index
    let total_particles = 1e4;
    let half_particles = total_particles / 2.0;

    if f32(model.particle_index) < half_particles {
        out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red color for the first half
    } else {
        out.color = vec4<f32>(0.0, 0.0, 1.0, 1.0); // Blue color for the second half
    }

    return out;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
