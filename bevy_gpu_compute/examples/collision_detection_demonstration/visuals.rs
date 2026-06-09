use bevy::{
    asset::Handle,
    camera::Projection,
    log,
    platform::collections::HashMap,
    prelude::{Color, ColorMaterial, Component, FromWorld, Resource, World},
};
use bevy::{
    asset::{Assets, RenderAssetUsages},
    math::{Vec2, Vec3, bounding::BoundingCircle},
    prelude::{
        Camera2d, Commands, Mesh, Mesh2d, MeshMaterial2d, OrthographicProjection, Res, ResMut,
        Transform,
    },
    utils::default,
};

use crate::{ENTITY_RADIUS, EXPECTED_NUM_ENTITIES, SPAWN_RANGE_MAX, SPAWN_RANGE_MIN, State};

#[derive(Debug, Component)]
pub struct BoundingCircleComponent(pub BoundingCircle);
pub fn spawn_entities(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    color_handles: Res<ColorHandles>,
    mut state: ResMut<State>,
) {
    let mut count = 0;
    for x in SPAWN_RANGE_MIN..SPAWN_RANGE_MAX {
        for y in SPAWN_RANGE_MIN..SPAWN_RANGE_MAX {
            commands.spawn((
                create_circle_outline_components(
                    ENTITY_RADIUS,
                    AvailableColor::Green,
                    &color_handles,
                    &mut meshes,
                ),
                Transform {
                    translation: Vec3::new(x as f32, y as f32, 0.0),
                    ..default()
                },
                BoundingCircleComponent(BoundingCircle::new(
                    Vec2::new(x as f32, y as f32),
                    ENTITY_RADIUS,
                )),
            ));
            count += 1;
        }
    }
    log::info!("total of {} entities spawned", count);
    assert!(count == EXPECTED_NUM_ENTITIES);
    state.num_entities = count;
}

pub fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            near: -1000.0,
            far: 1000.0,
            scale: 0.1,
            ..OrthographicProjection::default_2d()
        }),
        Transform::from_xyz(
            0., 0., 10.0, // 100.0,
        ),
    ));
}

fn create_circle_outline_components(
    radius: f32,
    outline_color: AvailableColor,
    color_handles: &Res<ColorHandles>,
    meshes: &mut ResMut<Assets<Mesh>>,
) -> (Mesh2d, MeshMaterial2d<ColorMaterial>) {
    let color = color_handles.handles.get(&outline_color).unwrap().clone();

    // Create a path for the circle outline
    let mut path = Vec::new();
    let segments = 32; // Number of segments to approximate the circle
    for i in 0..=segments {
        let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let point = Vec2::new(radius * angle.cos(), radius * angle.sin());
        path.push(point);
    }

    // Create the line strip mesh
    let mut mesh = Mesh::new(
        bevy::render::render_resource::PrimitiveTopology::LineStrip,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        path.iter()
            .map(|p| [p.x, p.y, 0.0])
            .collect::<Vec<[f32; 3]>>(),
    );
    (Mesh2d(meshes.add(mesh)), MeshMaterial2d(color))
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum AvailableColor {
    Green,
    Blue,
    Red,
    Yellow,
    Black,
}
#[derive(Resource)]
pub struct ColorHandles {
    pub handles: HashMap<AvailableColor, Handle<ColorMaterial>>,
    pub _colors: HashMap<AvailableColor, Color>,
}

impl FromWorld for ColorHandles {
    fn from_world(world: &mut World) -> Self {
        let mut colors = HashMap::new();
        colors.insert(AvailableColor::Green, Color::srgb(0.0, 1.0, 0.0));
        colors.insert(AvailableColor::Blue, Color::srgb(0.0, 0.0, 1.0));
        colors.insert(AvailableColor::Red, Color::srgb(1.0, 0.0, 0.0));
        colors.insert(AvailableColor::Yellow, Color::srgb(1.0, 1.0, 0.0));
        colors.insert(AvailableColor::Black, Color::srgb(0.0, 0.0, 0.0));
        let mut materials = world.resource_mut::<Assets<ColorMaterial>>();
        let mut handles = HashMap::new();
        for (color, color_value) in colors.iter() {
            let handle = materials.add(*color_value);
            handles.insert(*color, handle);
        }
        Self {
            handles,
            _colors: colors,
        }
    }
}
