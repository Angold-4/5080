use ocl::{ProQue, Result};

// On NVIDIA, cores are streaming multiprocessors (SMs).
//   AD102 (4090) has 144 SMs with 128 threads each (18432 CUDA Cores)
//   GB203 (5080) has  84 SMs with 128 threads each (10752 CUDA Cores)
// On AMD, cores are compute units.
//   7900XTX has 96 CUs with with 64 threads each


/*  OpenCLL:
    Global Size: The total number of work-items (think threads) that execute the kernel.
    For example, a global_size of 21,504 means 21,504 work-items run the kernel code.

    Local Size: The number of work-items grouped into a single work-group.
    Work-items in the same work-group run on the same SM (Nvidia), can synchronize with each other, and share fast local (shared) memory.
*/

use std::time::Instant;
fn main() -> Result<()> {
    let kernel_src = r#"
        __kernel void compute(__global float* input, __global float* output) {
            int gid = get_global_id(0);
            float a = input[gid];
            for (int i = 0; i < 1000000; i++) {
                a += sin((float)(gid + i));
            }
            output[gid] = a;
        }
    "#;

    let local_size = 128; // Matches CUDA cores per SM
    let sm_count = 84;    // RTX 5080 SMs
    let max_work_groups_per_sm = 4;
    let max_concurrent_work_groups = sm_count * max_work_groups_per_sm; // 336

    let global_sizes = vec![
        local_size,                              // 1 work-group (1 SM)
        local_size * sm_count,                   // 84 work-groups (all SMs, 1 per SM)
        local_size * sm_count * 2,               // 168 work-groups
        local_size * max_concurrent_work_groups, // 336 work-groups (full concurrency)
        local_size * max_concurrent_work_groups * 2, // 672 work-groups (2 waves)
    ];

    let proque = ProQue::builder().src(kernel_src).dims(430080).build()?;
    let input_buffer = proque.buffer_builder::<f32>().len(430080).fill_val(1.0).build()?;
    let output_buffer = proque.create_buffer::<f32>()?;

    for global_size in global_sizes {
        let kernel = proque.kernel_builder("compute")
            .arg(&input_buffer)
            .arg(&output_buffer)
            .global_work_size(global_size)
            .build()?;

        let now = Instant::now();
        unsafe {
            kernel.cmd().local_work_size(local_size).enq()?;
        }
        proque.finish()?;
        let elapsed = now.elapsed();
        let work_groups = global_size / local_size;
        println!(
            "Global Size = {}, Work-Groups = {}: Elapsed: {:.2?}",
            global_size, work_groups, elapsed
        );
    }

    Ok(())
}

/*
    awang@vienna:~/learn/5080/threads$ cargo run
        Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.06s
        Running `target/debug/threads`
    Global Size = 128, Work-Groups = 1: Elapsed: 159.57ms
    The 156.57 ms reflects the time for one SM to process 128 threads, with the rest of the GPU idle.

    Global Size = 10752, Work-Groups = 84: Elapsed: 163.39ms
    The time barely increases (163.39 ms), demonstrating excellent parallel scaling as all 84 SMs work simultaneously.

    Global Size = 21504, Work-Groups = 168: Elapsed: 200.05ms
    As the number of work groups for each SMs doubled (2 for each SM), the time don't increase that much.
    This sub-linear increase suggests that the SMs can process multiple work-groups concurrently, overlapping their execution efficiently.

    Global Size = 43008, Work-Groups = 336: Elapsed: 265.10ms
    With 336 work-groups, and assuming each SM can handle up to 4 work-groups concurrently (84 SMs × 4 = 336), the GPU is fully utilized.
    
    Global Size = 86016, Work-Groups = 672: Elapsed: 567.05ms
    With 672 work-groups, the GPU must process two waves of 336 work-groups each (672 / 336 = 2).
*/