use std::collections::HashMap;

use bevy::{ecs::component::Component, log, render::renderer::RenderDevice};
use bevy_gpu_compute_core::{InputTypesMetadataTrait, OutputTypesMetadataTrait};
use bevy_gpu_compute_core::{
    MaxOutputLengths, TypesSpec,
    wgsl::shader_module::{
        complete_shader_module::WgslShaderModule, user_defined_portion::WgslShaderModuleUserPortion,
    },
};

use super::task_components::{
    buffers::TaskBuffers,
    configuration::{
        input_spec::InputSpec, iteration_space::IterationSpace, lib::TaskConfiguration,
        output_spec::OutputSpec, wgsl_code::WgslCode,
    },
    data::TaskData,
    runtime_state::{
        gpu_workgroup_sizes::GpuWorkgroupSizes,
        gpu_workgroup_space::GpuWorkgroupSpace,
        lib::{TaskRuntimeState, TaskRuntimeStateBuilder},
        max_output_bytes::MaxOutputBytes,
    },
};

/**
A task can only run once per run of the BevyGpuComputeRunTaskSet system set
By default this means once per frame
*/

#[derive(Component)]
pub struct BevyGpuComputeTask {
    name: String,
    configuration: TaskConfiguration,
    runtime_state: TaskRuntimeState,
    buffers: TaskBuffers,
    current_data: TaskData,
}

impl BevyGpuComputeTask {
    //getters
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn configuration(&self) -> &TaskConfiguration {
        &self.configuration
    }

    pub fn runtime_state(&self) -> &TaskRuntimeState {
        &self.runtime_state
    }
    pub fn runtime_state_mut(&mut self) -> &mut TaskRuntimeState {
        &mut self.runtime_state
    }
    pub fn buffers(&self) -> &TaskBuffers {
        &self.buffers
    }
    pub fn buffers_mut(&mut self) -> &mut TaskBuffers {
        &mut self.buffers
    }
    pub fn current_data(&self) -> &TaskData {
        &self.current_data
    }
    pub fn current_data_mut(&mut self) -> &mut TaskData {
        &mut self.current_data
    }
    pub fn from_shader<ShaderModuleTypes: TypesSpec>(
        name: &str,
        render_device: &RenderDevice,
        wgsl_shader_module: WgslShaderModuleUserPortion,
        iteration_space: IterationSpace,
        max_output_vector_lengths: MaxOutputLengths,
    ) -> Self {
        let full_module = WgslShaderModule::new(wgsl_shader_module);
        log::debug!(
            "generated wgsl code : {}",
            full_module.wgsl_code(iteration_space.num_dimmensions())
        );
        Self::create_manually::<ShaderModuleTypes>(
            name,
            render_device,
            iteration_space,
            max_output_vector_lengths,
            unsafe {
                WgslCode::from_string(
                    name,
                    render_device,
                    full_module.wgsl_code(iteration_space.num_dimmensions()),
                    "main".to_string(),
                )
            },
        )
    }

    /// ensure that you send relevant update events after calling this function
    pub fn create_manually<ShaderModuleTypes: TypesSpec>(
        name: &str,
        render_device: &RenderDevice,
        iteration_space: IterationSpace,
        max_output_array_lengths: MaxOutputLengths,
        wgsl_code: WgslCode,
    ) -> Self {
        let config_input_metadata = ShaderModuleTypes::ConfigInputTypes::get_all();
        let input_metadata = ShaderModuleTypes::InputArrayTypes::get_all();
        let output_metadata = ShaderModuleTypes::OutputArrayTypes::get_all();
        let data = TaskData::default();
        let buffers = TaskBuffers::default();
        let configuration = TaskConfiguration::new(
            wgsl_code,
            iteration_space,
            InputSpec::new(input_metadata, config_input_metadata),
            OutputSpec::new(output_metadata, max_output_array_lengths),
        );
        let runtime_state =
            TaskRuntimeStateBuilder::new(render_device, name, &configuration).build();
        Self {
            name: name.to_string(),
            configuration,
            runtime_state,
            buffers,
            current_data: data,
        }
    }

    /// runtime state has to be updated if either iteration space or output array lengths is changed, so more efficient to combine updates into a single method
    /// If a parameter is None then the existing value is retained
    pub fn mutate(
        &mut self,
        new_iteration_space: Option<IterationSpace>,
        new_max_output_array_lengths: Option<MaxOutputLengths>,
    ) {
        if let Some(iter_space) = new_iteration_space {
            // ensure that the number of dimmensions has not been changed
            assert!(
                iter_space.num_dimmensions()
                    == self.configuration.iteration_space().num_dimmensions(),
                "The number of dimmensions cannot be changed after creating the task. Currently the iteration space for this task is {:?}, but you are trying to change it to be {:?}. For example: an iteration space of x = 30, y = 20 and z = 1 has 2 dimmensions, and an iteration space of x = 30, y=1, z=1 has 1 dimmension.",
                self.configuration
                    .iteration_space()
                    .num_dimmensions()
                    .to_usize(),
                iter_space.num_dimmensions().to_usize()
            );
            self.configuration._internal_set_iteration_space(iter_space);
        }
        if let Some(output_lengths) = new_max_output_array_lengths {
            self.configuration
                .outputs_mut()
                ._internal_set_max_lengths(output_lengths);
        }
        self.update_runtime_state_on_iter_space_or_max_output_lengths_change();
    }

    pub fn get_pipeline_consts(&self) -> HashMap<String, f64> {
        let mut n: HashMap<String, f64> = HashMap::new();
        if self.current_data().input_lengths().is_none() {
            panic!("input_lengths not set for task {}", self.name());
        }
        // input and output array lengths
        for (i, metadata) in self.configuration().inputs().arrays().iter().enumerate() {
            let length = self
                .current_data()
                .input_lengths()
                .as_ref()
                .unwrap()
                .get(metadata.name.name());
            let name = metadata.name.input_array_length();
            log::debug!("input_array_lengths = {:?}, for {}", length, name);
            assert!(
                length.is_some(),
                "input_array_lengths not set for input array {}, {}",
                i,
                name.clone()
            );
            n.insert(name.clone(), *length.unwrap() as f64);
        }
        for metadata in self.configuration.outputs().arrays().iter() {
            n.insert(
                metadata.name.output_array_length(),
                self.configuration()
                    .outputs()
                    .max_lengths()
                    .get_by_name(&metadata.name) as f64,
            );
        }
        log::debug!("pipeline consts  = {:?}", n);
        n
    }

    fn update_runtime_state_on_iter_space_or_max_output_lengths_change(&mut self) {
        // update task max output bytes
        self.runtime_state._internal_set_max_output_bytes(
            MaxOutputBytes::from_max_lengths_and_spec(
                self.configuration.outputs().max_lengths(),
                self.configuration.outputs().arrays(),
            ),
        );
        let mut wg_sizes = self.runtime_state.workgroup_sizes().clone();
        // update workgroup sizes
        if self
            .configuration
            .iteration_space()
            .num_dimmensions()
            .to_usize()
            != wg_sizes.num_dimmensions()
        {
            wg_sizes = GpuWorkgroupSizes::from_iter_space(self.configuration.iteration_space());
            self.runtime_state
                ._internal_set_workgroup_sizes(wg_sizes.clone());
        }
        self.runtime_state._internal_set_workgroup_space(
            GpuWorkgroupSpace::from_iter_space_and_wrkgrp_sizes(
                self.configuration.iteration_space(),
                &wg_sizes,
            ),
        );
    }
}
