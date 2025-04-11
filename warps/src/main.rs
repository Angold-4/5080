use ocl::{ProQue, Result};

// GPUs have Warps. Warps are groups of threads, and all modern GPUs have them as 32 threads.
// The warp size (32 threads) defines the minimum unit of parallel execution within a single SM.
// In Nvidia GPU, the maximum number of threads per SM is 2048, which means each SM can handle up
// to 64 warps (64 × 32 = 2048 threads). For example, on GB203 (5080) with 84 SMs, 
// the total number of threads the GPU can execute is 84 SMs × 2048 threads/SM = 172,032 threads.

fn main() -> Result<()> {
    let kernel_src = r#"
        __kernel void divergent_kernel(__global float* c, int K) {
            int a = get_local_id(0);
            if (get_local_id(0) % K < K / 2) {
                for (int i = 0; i < 10000000; i++) {a *= 2;}
            } else {
                for (int i = 0; i < 10000000; i++) {a *= 3;}
            }
            c[get_global_id(0)] = a + get_local_id(0);
        }
    "#;

    let proque = ProQue::builder().src(kernel_src).dims(4194304).build()?;
    let c_buffer = proque.create_buffer::<f32>()?;

    let k_values = vec![16, 32, 48, 63, 64, 128, 256]; // Test different K
    for k in k_values {
        let kernel = proque.kernel_builder("divergent_kernel")
            .arg(&c_buffer)
            .arg(k)
            .build()?;

        use std::time::Instant;
        let now = Instant::now();
        unsafe {
            kernel.cmd().local_work_size(256).enq()?;
        }
        proque.finish()?;
        let elapsed = now.elapsed();
        println!("K = {}: Elapsed: {:.2?}", k, elapsed);
    }

    Ok(())
}

/*
    awang@vienna:~/learn/5080/warps$ cargo run
    Compiling warps v0.1.0 (/home/awang/learn/5080/warps)
        Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.23s
        Running `target/debug/warps`
    K = 16: Elapsed: 1.15ms
    K = 32: Elapsed: 1.17ms
    K = 48: Elapsed: 1.15ms
    K = 63: Elapsed: 1.20ms
    K = 64: Elapsed: 664.23µs
    K = 128: Elapsed: 681.54µs
    K = 256: Elapsed: 659.77µs
*/