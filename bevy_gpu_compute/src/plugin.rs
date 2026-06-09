use bevy::{
    app::{App, Plugin, Startup, Update},
    prelude::{AppExtStates, IntoScheduleConfigs, States, in_state},
};

use crate::{
    ram_limit::RamLimit,
    spawn_fallback_camera::{spawn_fallback_camera, spawn_fallback_camera_runif},
};

/// state for activating or deactivating the plugin
#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum BevyGpuComputeState {
    #[default]
    Running,
    #[allow(dead_code)]
    Stopped,
}

pub struct BevyGpuComputePlugin {
    with_default_schedule: bool,
}

impl Plugin for BevyGpuComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RamLimit>()
            .init_state::<BevyGpuComputeState>();
        if self.with_default_schedule {
            app.add_systems(Startup, spawn_fallback_camera).add_systems(
                Update,
                (spawn_fallback_camera.run_if(spawn_fallback_camera_runif),)
                    .chain()
                    .run_if(in_state(BevyGpuComputeState::Running)),
            );
        } else {
            app.add_systems(
                Update,
                spawn_fallback_camera
                    .run_if(spawn_fallback_camera_runif)
                    .run_if(in_state(BevyGpuComputeState::Running)),
            );
        }
    }
}

impl Default for BevyGpuComputePlugin {
    fn default() -> Self {
        BevyGpuComputePlugin {
            with_default_schedule: true,
        }
    }
}
impl BevyGpuComputePlugin {
    pub fn no_default_schedule() -> Self {
        BevyGpuComputePlugin {
            with_default_schedule: false,
        }
    }
}
