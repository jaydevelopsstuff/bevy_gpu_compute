use bevy::{
    camera::Projection,
    log,
    prelude::{
        Camera, Camera2d, Commands, Component, Entity, OrthographicProjection, Query, Res,
        Transform,
    },
    time::Time,
};

#[derive(Component)]
pub struct BevyGpuComputeFallbackCamera;

/**
Testing indicates GPU performance vastly reduced if bevy does not spawn a window or camera. Unsure why. If the user doesn't spawn a camera we spawn one for them.
 */
pub fn spawn_fallback_camera(
    cameras: Query<&Camera>,
    fallback_cameras: Query<(Entity, &BevyGpuComputeFallbackCamera)>,
    mut commands: Commands,
) {
    let len = cameras.iter().len();
    match len {
        0 => {
            log::debug!(
                "GPU Compute: Spawning fallback camera in order to improve gpu performance."
            );
            commands.spawn((
                Camera2d,
                Projection::Orthographic(OrthographicProjection {
                    near: -10.0,
                    far: 10.0,
                    scale: 1.,
                    ..OrthographicProjection::default_2d()
                }),
                Transform::from_xyz(
                    0., 0., 10.0, // 100.0,
                ),
                BevyGpuComputeFallbackCamera,
            ));
        }
        1 => {
            // do nothing
        }
        _ => {
            log::trace!("GPU Compute: Despawning extra fallback cameras.");
            let fallback_cam_len = fallback_cameras.iter().len();
            if fallback_cam_len > 0 {
                fallback_cameras.iter().for_each(|(e, _)| {
                    commands.entity(e).despawn();
                });
            }
        }
    }
}

pub fn spawn_fallback_camera_runif(time: Res<Time>) -> bool {
    // stop running after a certain point, assuming that if the user was going to add a camera, they would have done so by now
    let delta = time.delta_secs();
    let elapsed = time.elapsed_secs();
    let in_first_frame = elapsed <= delta;
    if !in_first_frame {
        // should stop running after 5 frames or 5 seconds, whichever takes longer
        // assume average frame time is equal to delta
        let num_frames = elapsed / delta;
        if num_frames > 5. && elapsed > 5. {
            return false;
        }
    }
    true
}
