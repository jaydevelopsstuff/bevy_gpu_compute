use bevy::{
    ecs::system::SystemParam,
    prelude::{Commands, Entity, Query},
};

use crate::task::lib::BevyGpuComputeTask;

#[derive(SystemParam)]

pub struct BevyGpuComputeTaskDeleter<'w, 's> {
    commands: Commands<'w, 's>,
    tasks: Query<'w, 's, (Entity, &'static mut BevyGpuComputeTask)>,
}

impl BevyGpuComputeTaskDeleter<'_, '_> {
    /// spawns all components needed for the task to run
    pub fn delete(&mut self, name: &str) {
        let (entity, _) = self
            .tasks
            .iter_mut()
            .find(|(_, task)| task.name() == name)
            .expect("Task not found");
        self.commands.entity(entity).despawn();
    }
}
