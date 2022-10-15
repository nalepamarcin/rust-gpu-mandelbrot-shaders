#![no_std]
#![feature(register_attr)]
#![register_attr(spirv)]

use spirv_std::glam;


#[repr(C)]
pub struct InputParameters {
    draw_bounds: glam::Vec4, // -x, x, -y, y
    img_size_px: u32,        // in pixels
    max_iter: u32,           // max number of iterations to run
}


fn mix(a: f32, b: f32, v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    a * (1.0 - v) + b * v
}


fn dist_sq(v: glam::Vec2) -> f32 {
    v.x * v.x + v.y * v.y
}


// c - coordinates of complex point to check
// returns 0 if point inside set, otherwise number of iterations (up to max_iter) necessary to escape the set for sure
fn mandelbrot(c: glam::Vec2, max_iter: u8) -> u8 {
    // mandelbrot
    // z_n+1 = z_n * z_n + c
    let mut z = glam::Vec2::new(0.0, 0.0);
    for i in 0..max_iter {
        if dist_sq(z) > 4.0 {
            return i;
        }
        z = glam::Vec2::new(
            z.x*z.x - z.y*z.y,
            z.x*z.y + z.y*z.x
        ) + c;
    }
    return 0;
}


fn mandelmsaax16(c: glam::Vec2, max_iter: u8, img_size: f32) -> u8 {
    // uniform distribution of 16 points across pixel
    let dpx = 1.0 / 8.0 / img_size;
    let dpx2 = 3.0 / 8.0 / img_size;

    let sum =
        mandelbrot(c + glam::Vec2::new(-dpx2, -dpx2), max_iter) as u32 +
        mandelbrot(c + glam::Vec2::new(-dpx,  -dpx2), max_iter) as u32 +
        mandelbrot(c + glam::Vec2::new( dpx,  -dpx2), max_iter) as u32 +
        mandelbrot(c + glam::Vec2::new( dpx2, -dpx2), max_iter) as u32 +

        mandelbrot(c + glam::Vec2::new(-dpx2, -dpx), max_iter) as u32 +
        mandelbrot(c + glam::Vec2::new(-dpx,  -dpx), max_iter) as u32 +
        mandelbrot(c + glam::Vec2::new( dpx,  -dpx), max_iter) as u32 +
        mandelbrot(c + glam::Vec2::new( dpx2, -dpx), max_iter) as u32 +

        mandelbrot(c + glam::Vec2::new(-dpx2, dpx), max_iter) as u32 +
        mandelbrot(c + glam::Vec2::new(-dpx,  dpx), max_iter) as u32 +
        mandelbrot(c + glam::Vec2::new( dpx,  dpx), max_iter) as u32 +
        mandelbrot(c + glam::Vec2::new( dpx2, dpx), max_iter) as u32 +

        mandelbrot(c + glam::Vec2::new(-dpx2, dpx2), max_iter) as u32 +
        mandelbrot(c + glam::Vec2::new(-dpx,  dpx2), max_iter) as u32 +
        mandelbrot(c + glam::Vec2::new( dpx,  dpx2), max_iter) as u32 +
        mandelbrot(c + glam::Vec2::new( dpx2, dpx2), max_iter) as u32;

    return (sum / 16u32) as u8;
}


fn mandelproc(c: glam::Vec2, max_iter: u8, img_size: f32) -> u8 {
    (mandelmsaax16(c, max_iter, img_size) as f32 / max_iter as f32 * 1.5 * 255.0).clamp(
        0.0,
        255.0
    ) as u8
}


#[spirv(compute(threads(16, 16)))]
pub fn main(
    #[spirv(global_invocation_id)] global_id: glam::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] input_parameters: &InputParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] v_pixels: &mut [u8],
) {
    let img_size_f = input_parameters.img_size_px as f32;

    let img_x = mix(
        input_parameters.draw_bounds.x,
        input_parameters.draw_bounds.y,
        global_id.x as f32 / img_size_f
    );
    let img_y = mix(
        input_parameters.draw_bounds.z,
        input_parameters.draw_bounds.w,
        global_id.y as f32 / img_size_f
    );

    let global_index = global_id.y * input_parameters.img_size_px + global_id.x;
    v_pixels[global_index as usize] = mandelproc(
        glam::Vec2::new(img_x, img_y),
        input_parameters.max_iter as u8,
        img_size_f
    );
}
