use ocl::{ProQue, Result};
// https://github.com/geohot/gpunoob

//            OpenCL        CUDA       HIP
// G cores   (get_group_id, blockIdx,  __ockl_get_group_id, threadgroup_position_in_grid)
// L threads (get_local_id, threadIdx, __ockl_get_local_id, thread_position_in_threadgroup)

// On NVIDIA, cores are streaming multiprocessors (SMs).
//   AD102 (4090) has 144 SMs with 128 threads each (18432 CUDA Cores)
//   GB203 (5080) has  84 SMs with 128 threads each (10752 CUDA Cores)
// On AMD, cores are compute units.
//   7900XTX has 96 CUs with with 64 threads each

// GPUs have warps. Warps are groups of threads, and all modern GPUs have them as 32 threads.
// GPUs are multicore processors with 32 threads

fn main() -> Result<()> {
    // Kernel source code (OpenCL, C)
    // __global const float* a,
    // __global const float* b,
    let kernel_src = r#"
        __kernel void add(
            __global float* c
        ) {
            int a = get_local_id(0);
            for (int i = 0; i < 10000000; i++) {a *= 2;}
            c[get_global_id(0)] = a + get_local_id(0);
        }
    "#;

    // Initialize ProQue
    let proque = ProQue::builder().src(kernel_src).dims(262144).build()?;

    // Input data (In CPU)
    // let a_data = vec![1.0f32; 128];
    // let b_data = vec![2.0f32; 128];

    // Create buffers (In GPU)
    // let a_buffer = proque.create_buffer::<f32>()?;
    // let b_buffer = proque.create_buffer::<f32>()?;
    let c_buffer = proque.create_buffer::<f32>()?;

    // Write data to device buffers
    // a_buffer.cmd().write(&a_data).enq()?;
    // b_buffer.cmd().write(&b_data).enq()?;

    // Build kernel and set arguments
    let kernel = proque
        .kernel_builder("add")
        // .arg(&a_buffer)
        // .arg(&b_buffer)
        .arg(&c_buffer)
        .build()?;

    use std::time::Instant;
    let now = Instant::now();

    // Execute the kernel
    unsafe {
        kernel.cmd().local_work_size(32).enq()?;
    }

    let _ = proque.finish();
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    // Read result back
    let mut c_data = vec![0.0f32; 128]; // only the first 128 working groups
    c_buffer.cmd().read(&mut c_data).enq()?;


    // Verify the output
    let mut i = 0;
    for &c in &c_data {
        if i % 16 == 0 && i != 0 {
            println!("");
        }
        i += 1;
        print!("{:>4} ", c);
    }
    println!("");
    Ok(())
}
