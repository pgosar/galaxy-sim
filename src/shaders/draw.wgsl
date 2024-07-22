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
    // Calculate rotation angles
    let angle_xz = -atan2(model.particle_vel.x, model.particle_vel.z);
    let angle_y = asin(model.particle_vel.y / length(model.particle_vel));

    // Rotate around Y axis
    let rotated_xz = vec3<f32>(
        model.position.x * cos(angle_xz) - model.position.z * sin(angle_xz),
        model.position.y,
        model.position.x * sin(angle_xz) + model.position.z * cos(angle_xz)
    );

    // Rotate around X axis
    let rotated_pos = vec3<f32>(
        rotated_xz.x,
        rotated_xz.y * cos(angle_y) - rotated_xz.z * sin(angle_y),
        rotated_xz.y * sin(angle_y) + rotated_xz.z * cos(angle_y)
    );

    let world_pos = rotated_pos + model.particle_pos;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return out;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
