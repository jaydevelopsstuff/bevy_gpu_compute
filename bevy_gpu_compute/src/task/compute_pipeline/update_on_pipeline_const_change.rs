use bevy::{log, render::renderer::RenderDevice};

use wgpu::{ComputePipelineDescriptor, PipelineCompilationOptions};

use crate::task::lib::BevyGpuComputeTask;

use super::pipeline_cache::PipelineKey;

pub fn update_compute_pipeline(task: &mut BevyGpuComputeTask, render_device: &RenderDevice) {
    if task.current_data().input_lengths().is_none() {
        return;
    }
    log::trace!("Updating pipeline for task {}", task.name());
    let key = PipelineKey {
        pipeline_consts_version: task.configuration().version(),
    };
    if task
        .runtime_state()
        .pipeline_cache()
        .cache
        .contains_key(&key)
    {
    } else {
        log::trace!("Creating new pipeline for task {}", task.name());
        log::trace!(
            "pipeline layout {:?}",
            task.runtime_state().pipeline_layout()
        );

        let pipeline_consts = task.get_pipeline_consts();
        let constants_vec: Vec<(&str, f64)> = pipeline_consts
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();

        let compute_pipeline = render_device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(task.name()),
            layout: Some(task.runtime_state().pipeline_layout()),
            module: task.configuration().shader().shader_module(),
            entry_point: Some(task.configuration().shader().entry_point_function_name()),
            // this is where we specify new values for pipeline constants...
            compilation_options: PipelineCompilationOptions {
                constants: &constants_vec,
                zero_initialize_workgroup_memory: Default::default(),
            },
            cache: None,
        });
        task.runtime_state_mut()
            .pipeline_cache_mut()
            .cache
            .insert(key, compute_pipeline);
    }
}
