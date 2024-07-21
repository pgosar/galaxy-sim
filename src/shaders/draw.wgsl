// Vertex shader
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) particle_pos: vec2<f32>,
    @location(1) particle_vel: vec2<f32>,
    @location(2) position: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn main_vs(
    model: VertexInput,
) -> VertexOutput {
    let angle = -atan2(model.particle_vel.x, model.particle_vel.y);
    let rotated_pos = vec2<f32>(
        model.position.x * cos(angle) - model.position.y * sin(angle),
        model.position.x * sin(angle) + model.position.y * cos(angle)
    );
    let world_pos = vec3<f32>(rotated_pos + model.particle_pos, 0.0);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return out;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
