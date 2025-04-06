use ocl::{ProQue, Result};

fn main() -> Result<()> {
    // Kernel source code (OpenCL, C)
    let kernel_src = r#"
        __kernel void add(
            __global const float* a,
            __global const float* b,
            __global float* c
        ) {
            int i = get_global_id(0);
            c[i] = i+2 // a[i] + b[i]
        }
    "#;

    // Initialize ProQue
    let proque = ProQue::builder().src(kernel_src).dims(128).build()?;

    // Input data
    let a_data = vec![1.0f32; 128];
    let b_data = vec![2.0f32; 128];

    // Create buffers
    let a_buffer = proque.create_buffer::<f32>()?;
    let b_buffer = proque.create_buffer::<f32>()?;
    let mut c_buffer = proque.create_buffer::<f32>()?;

    // Write data to device buffers
    a_buffer.cmd().write(&a_data).enq()?;
    b_buffer.cmd().write(&b_data).enq()?;

    // Build kernel and set arguments
    let kernel = proque
        .kernel_builder("add")
        .arg(&a_buffer)
        .arg(&b_buffer)
        .arg(&c_buffer)
        .build()?;

    // Execute the kernel
    unsafe {
        kernel.cmd().local_work_size(1).enq()?;
    }

    // Read result back
    let mut c_data = vec![0.0f32; 128];
    c_buffer.cmd().read(&mut c_data).enq()?;

    // Verify the output
    let mut i = 0;
    for &c in &c_data {
        if i % 16 == 0 && i != 0 {
            println!("");
        }
        i += 1;
        print!("{:>3}", c);
    }

    Ok(())
}
