/*
Demonstrates all features of the BevyGpuCompute library
 */

use bevy::{
    DefaultPlugins,
    app::{App, AppExit, PluginGroup, Startup, Update},
    log::{self, LogPlugin},
    prelude::{IntoScheduleConfigs, MessageWriter, Local, Query, Res, ResMut, Resource},
};
use bevy_gpu_compute::prelude::*;
mod visuals;
use visuals::{BoundingCircleComponent, ColorHandles, spawn_camera, spawn_entities};

fn main() {
    let mut binding = App::new();
    let _app = binding
        .add_plugins(DefaultPlugins.set(LogPlugin {
            level: log::Level::INFO,
            ..Default::default()
        }))
        .add_plugins(BevyGpuComputePlugin::default())
        .init_resource::<ColorHandles>()
        .init_resource::<State>()
        .add_systems(
            Startup,
            (spawn_camera, spawn_entities, create_task, modify_task).chain(),
        )
        .add_systems(Update, (modify_task_config_inputs, run_task).chain())
        .add_systems(Update, (handle_task_results, exit_and_show_results).chain())
        .run();
}

// constants used to produce predictable collision results
const SPAWN_RANGE_MIN: i32 = -2;
const SPAWN_RANGE_MAX: i32 = 2;
const ENTITY_RADIUS: f32 = 100.1;
const RADIUS_MULTIPLIER_FRAME_0: f32 = 1.0;
const RADIUS_MULTIPLIER_FRAME_1: f32 = 0.01;
const RADIUS_MULTIPLIER_FRAME_2: f32 = 0.002;
const EXIT_AFTER_FRAMES: u32 = 3;
// expected results
const EXPECTED_NUM_ENTITIES: u32 = 16;
// 16 entities 100% intersecting should produce 120 collisions
const EXPECTED_COLLISIONS_FRAME_0: usize = 120;
const EXPECTED_COLLISIONS_FRAME_1: usize = 58;
const EXPECTED_COLLISIONS_FRAME_2: usize = 0;

#[derive(Resource, Default)]
struct State {
    pub num_entities: u32,
    pub collisions_per_frame: Vec<usize>,
}

#[wgsl_shader_module]
mod collision_detection_module {
    use bevy_gpu_compute::prelude::*;

    const MY_CONST: u32 = 1;
    #[wgsl_config]
    struct Config {
        pub radius_multiplier: f32,
    }
    #[wgsl_input_array]
    struct Position {
        pub v: Vec2F32,
    }
    #[wgsl_input_array]
    type Radius = f32;
    #[wgsl_output_vec]
    struct CollisionResult {
        entity1: u32,
        entity2: u32,
    }
    #[wgsl_output_array]
    struct MyDebugInfo {
        entity1: u32,
        entity2: u32,
        counter_value: u32,
        is_collision: i32,
        dist_squared: f32,
        rad_sum_sq: f32,
        rad_mult: f32,
    }
    fn calculate_distance_squared(p1: Vec2F32, p2: Vec2F32) -> f32 {
        let dx = p1.x - p2[0];
        let dy = p1.y - p2[1];
        return dx * dx + dy * dy;
    }
    fn main(iter_pos: WgslIterationPosition) {
        let current_entity = iter_pos.x;
        let other_entity = iter_pos.y;
        // Early exit conditions
        let out_of_bounds = current_entity >= WgslVecInput::vec_len::<Position>() as u32
            || other_entity >= WgslVecInput::vec_len::<Position>() as u32;
        if out_of_bounds || current_entity == other_entity || current_entity >= other_entity {
            return;
        }
        let current_radius = WgslVecInput::vec_val::<Radius>(current_entity);
        let other_radius = WgslVecInput::vec_val::<Radius>(other_entity);
        if current_radius <= 0.0 || other_radius <= 0.0 {
            return;
        }
        let current_pos = WgslVecInput::vec_val::<Position>(current_entity);
        let other_pos = WgslVecInput::vec_val::<Position>(other_entity);
        let dist_squared = calculate_distance_squared(current_pos.v, other_pos.v);
        let radius_sum = (current_radius + other_radius)
            * WgslConfigInput::get::<Config>().radius_multiplier
            * MY_CONST as f32;
        let rad_sum_sq = radius_sum * radius_sum;
        // index = y * width + x
        let debug_index = other_entity * WgslVecInput::vec_len::<Radius>() as u32 + current_entity;
        let is_collision = dist_squared < rad_sum_sq;
        WgslOutput::set::<MyDebugInfo>(
            debug_index,
            MyDebugInfo {
                entity1: current_entity,
                entity2: other_entity,
                counter_value: WgslOutput::len::<CollisionResult>() as u32,
                is_collision: is_collision as i32,
                dist_squared: dist_squared,
                rad_sum_sq: rad_sum_sq,
                rad_mult: WgslConfigInput::get::<Config>().radius_multiplier,
            },
        );
        if is_collision {
            WgslOutput::push::<CollisionResult>(CollisionResult {
                entity1: current_entity,
                entity2: other_entity,
            });
        }
    }
}

fn create_task(mut gpu_task_creator: BevyGpuComputeTaskCreator) {
    let initial_iteration_space = IterationSpace::new(
        // set incorrectly here, just so that we can demonstrate changing it later
        100, 100, 1,
    );
    //* There are two methods of creating the MaxOutputLengths config object: */
    // Method 1:
    let initial_max_output_lengths = collision_detection_module::MaxOutputLengthsBuilder::new()
        .set_collision_result(100)
        .set_my_debug_info(100)
        .finish();
    // Method 2:
    let mut alternate_max_output_lengths = MaxOutputLengths::empty();
    alternate_max_output_lengths.set("CollisionResult", 100);
    alternate_max_output_lengths.set("MyDebugInfo", 100);
    //
    gpu_task_creator.create_task_from_rust_shader::<collision_detection_module::Types>(
        "collision_detection", // ensure name is unique
        collision_detection_module::parsed(),
        initial_iteration_space,
        initial_max_output_lengths,
    );
}
/// This is here for reference, but is not used in this example
#[allow(dead_code)]
fn delete_task(mut gpu_task_deleter: BevyGpuComputeTaskDeleter) {
    gpu_task_deleter.delete("collision_detection");
}
fn modify_task(mut gpu_tasks: GpuTaskRunner, state: Res<State>) {
    let num_entities = state.num_entities;
    let max_output_lengths = collision_detection_module::MaxOutputLengthsBuilder::new()
        .set_collision_result((num_entities * num_entities) as usize)
        .set_my_debug_info((num_entities * num_entities) as usize)
        .finish();
    let iteration_space =
        IterationSpace::new(state.num_entities as usize, state.num_entities as usize, 1);
    let pending_commands = gpu_tasks
        .task("collision_detection")
        .mutate(Some(iteration_space), Some(max_output_lengths));
    gpu_tasks.run_commands(pending_commands);
}
fn modify_task_config_inputs(mut count: Local<u32>, mut gpu_tasks: GpuTaskRunner) {
    let radius_multiplier = if *count <= 0 {
        RADIUS_MULTIPLIER_FRAME_0
    } else if *count == 1 {
        RADIUS_MULTIPLIER_FRAME_1
    } else {
        RADIUS_MULTIPLIER_FRAME_2
    };
    let configs = collision_detection_module::ConfigInputDataBuilder::new()
        .set_config(collision_detection_module::Config { radius_multiplier })
        .finish();
    let commands = gpu_tasks
        .task("collision_detection")
        .set_config_inputs(configs);
    gpu_tasks.run_commands(commands);

    *count += 1;
}

fn run_task(mut gpu_tasks: GpuTaskRunner, entities: Query<&BoundingCircleComponent>) {
    let input_data = collision_detection_module::InputDataBuilder::new()
        .set_position(
            entities
                .iter()
                .map(|e| collision_detection_module::Position {
                    v: Vec2F32::new(e.0.center.x, e.0.center.y),
                })
                .collect(),
        )
        .set_radius(entities.iter().map(|e| e.0.radius()).collect())
        .into();
    let task = gpu_tasks
        .task("collision_detection")
        .set_inputs(input_data)
        .run();
    gpu_tasks.run_commands(task);
}

fn handle_task_results(mut gpu_task_reader: GpuTaskReader, mut state: ResMut<State>) {
    let results = gpu_task_reader
        .latest_results::<collision_detection_module::OutputDataBuilder>("collision_detection");

    if let Ok(results) = results {
        #[allow(unused_variables)]
        let debug_results = results.my_debug_info.unwrap();
        log::debug!("debug results: {:?}", debug_results);
        //fully type-safe results
        let collision_results = results.collision_result.unwrap();
        // your logic here
        let count = collision_results.len();
        log::info!("collisions this frame: {}", count);
        log::trace!("collision_results: {:?}", collision_results);
        state.collisions_per_frame.push(count);
    }
}

// when the local variable "count" goes above a certain number (representing frame count), exit the app
fn exit_and_show_results(
    mut count: Local<u32>,
    state: Res<State>,
    mut exit: MessageWriter<AppExit>,
) {
    if *count > EXIT_AFTER_FRAMES {
        let total_collisions = state.collisions_per_frame.iter().sum::<usize>();
        log::trace!("total collisions count at exit: {}", total_collisions);
        // verify results are what we expect
        assert_eq!(
            state.collisions_per_frame[0], EXPECTED_COLLISIONS_FRAME_0,
            "unexpected collision count for frame 0: {}",
            state.collisions_per_frame[0]
        );
        assert_eq!(
            state.collisions_per_frame[1], EXPECTED_COLLISIONS_FRAME_1,
            "unexpected collision count for frame 1: {}",
            state.collisions_per_frame[1]
        );
        assert_eq!(
            state.collisions_per_frame[2], EXPECTED_COLLISIONS_FRAME_2,
            "unexpected collision count for frame 2: {}",
            state.collisions_per_frame[2]
        );
        log::info!("Example completed successfully");
        exit.write(AppExit::Success);
    }
    *count += 1;
}
