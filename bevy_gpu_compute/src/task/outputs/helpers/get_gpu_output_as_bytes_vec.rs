use bevy::render::renderer::{RenderDevice, RenderQueue};
use pollster::FutureExt;
use wgpu::Buffer;

pub fn get_gpu_output_as_bytes_vec(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    output_buffer: &Buffer,
    staging_buffer: &Buffer,
    total_byte_size: u64,
) -> Option<Vec<u8>> {
    let mut encoder = render_device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, total_byte_size);
    render_queue.submit(std::iter::once(encoder.finish()));

    let slice = staging_buffer.slice(0..total_byte_size);
    let (sender, receiver) = futures::channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    let _ = render_device.poll(wgpu::PollType::wait_indefinitely());

    let result: Option<Vec<u8>> = if receiver.block_on().unwrap().is_ok() {
        let data = slice.get_mapped_range();
        let transformed_data = &*data;
        let r = Some(transformed_data.to_vec());
        drop(data);
        staging_buffer.unmap();
        r
    } else {
        None
    };
    result
}
