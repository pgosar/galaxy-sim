struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) particle_pos: vec3<f32>,
    @location(1) particle_vel: vec3<f32>,
    // location(2) is mass which is not needed here
    @location(3) position: vec3<f32>,
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
    out.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return out;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
