struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) particle_pos_x: f32,
    @location(1) particle_pos_y: f32,
    @location(2) particle_pos_z: f32,
    @location(3) particle_vel_x: f32,
    @location(4) particle_vel_y: f32,
    @location(5) particle_vel_z: f32,
    @location(6) particle_acc_x: f32,
    @location(7) particle_acc_y: f32,
    @location(8) particle_acc_z: f32,
    @location(9) mass: f32,
    @location(10) galaxy_id: u32,
    @location(11) position: vec3<f32>,
    @builtin(instance_index) particle_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn main_vs(
    model: VertexInput,
) -> VertexOutput {
    let particle_pos = vec3<f32>(model.particle_pos_x, model.particle_pos_y, model.particle_pos_z);
    let offset_position = particle_pos + model.position;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(offset_position, 1.0);

    // Generate color based on galaxy_id using golden ratio
    // This ensures maximum color separation for any number of galaxies
    // Each galaxy gets a unique, visually distinct hue
    let golden_ratio_conjugate = 0.618033988749895;
    let hue = fract(f32(model.galaxy_id) * golden_ratio_conjugate);
    
    // HSL to RGB conversion for vibrant colors
   // Convert hue (0-1) to RGB
    let h = hue * 6.0;
    let x = 1.0 - abs((h % 2.0) - 1.0);
    var rgb: vec3<f32>;
    if (h < 1.0) {
        rgb = vec3<f32>(1.0, x, 0.0);
    } else if (h < 2.0) {
        rgb = vec3<f32>(x, 1.0, 0.0);
    } else if (h < 3.0) {
        rgb = vec3<f32>(0.0, 1.0, x);
    } else if (h < 4.0) {
        rgb = vec3<f32>(0.0, x, 1.0);
    } else if (h < 5.0) {
        rgb = vec3<f32>(x, 0.0, 1.0);
    } else {
        rgb = vec3<f32>(1.0, 0.0, x);
    }
    
    out.color = vec4<f32>(rgb, 1.0);

    return out;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
