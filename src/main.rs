#![feature(portable_simd)]
mod aec;
mod feedback_suppressor;
mod low_pass_filter;
mod noise_gate;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use low_pass_filter::LowPassFilter;
use noise_gate::NoiseGate;
use opus::{packet, Decoder, Encoder};
use rtrb::RingBuffer;
use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use webrtc_audio_processing::Config as ProcessorConfig;
use webrtc_audio_processing::*;

fn create_processor() -> Processor {
    let mut processor = Processor::new(&InitializationConfig {
        num_capture_channels: 1,
        num_render_channels: 1,
        ..InitializationConfig::default()
    })
    .unwrap();

    let config = ProcessorConfig {
        echo_cancellation: Some(EchoCancellation {
            suppression_level: EchoCancellationSuppressionLevel::Low,
            stream_delay_ms: Some(16),
            enable_delay_agnostic: false,
            enable_extended_filter: false
        }),
        noise_suppression: Some(NoiseSuppression {
            suppression_level: NoiseSuppressionLevel::High,
        }),
        // gain_control: Some(GainControl{
        //     mode: GainControlMode::FixedDigital,
        //     compression_gain_db: 20,
        //     target_level_dbfs: 20,
        //     enable_limiter: true
        // }),
        // enable_transient_suppressor: true,
        enable_high_pass_filter: false,
        ..ProcessorConfig::default()
    };
    processor.set_config(config);

    processor
} 

fn main() {
    const SAMPLE_RATE: usize = 48000;
    const BUFFER_SIZE: usize = 480;
    // https://arc.net/e/BDDA7654-F9B3-44BC-91A8-3FC502FDB960
    const RINGBUFFER_SIZE: usize = BUFFER_SIZE * 2;

    let echo_processor = Arc::new(Mutex::new(create_processor()));
    let echo_processor_output = Arc::clone(&echo_processor);
    let echo_processor_input = Arc::clone(&echo_processor);

    let (mut prod, mut cons) = RingBuffer::<u8>::new(RINGBUFFER_SIZE);

    let host = cpal::default_host();

    let input_device = host.default_input_device().unwrap();
    let output_device = host.default_output_device().unwrap();

    let input_config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(SAMPLE_RATE as u32),
        buffer_size: cpal::BufferSize::Fixed(BUFFER_SIZE as u32),
    };
    let output_config: cpal::StreamConfig = input_config.clone();

    println!("Input config: {:?}", input_config);
    println!("Output config: {:?}", output_config);

    println!(
        "Actual input buffer size: {:?}",
        input_device.default_input_config().unwrap().buffer_size()
    );
    println!(
        "Actual output buffer size: {:?}",
        output_device.default_output_config().unwrap().buffer_size()
    );

    let (tx_i, rx_i) = mpsc::sync_channel::<[f32; BUFFER_SIZE]>(2);

    let input_stream = input_device
        .build_input_stream(
            &input_config,
            move |data: &[f32], _: &_| {
                let mut buffer = [0.0f32; BUFFER_SIZE];
                let len = data.len().min(BUFFER_SIZE);
                buffer[..len].copy_from_slice(&data[..len]);
                let _ = tx_i.send(buffer);
                // if tx_i.try_send(buffer).is_err() {
                //     println!("Dropping frame: processing thread can't keep up");
                // }
            },
            move |err| {
                eprintln!("There was an input error: {:?}", err);
            },
            None,
        )
        .unwrap();

    let _input_processing_thread = thread::spawn(move || {
        // let mut processor_input = create_processor();

        let mut noise_gate = NoiseGate::new(-23.5, 0.01, 0.0001, 0.02, 0.3, 48000.0);
        let cutoff_freq = 3000.0;
        let mut low_pass_filter = LowPassFilter::new(cutoff_freq, 48000.0);

        let mut encoder =
            Encoder::new(48000, opus::Channels::Mono, opus::Application::Voip).unwrap();
        // encoder.set_bitrate(opus::Bitrate::Auto).unwrap();

        while let Ok(mut data) = rx_i.recv() {
            let mut ap = echo_processor_input.lock().unwrap();
            let _ = ap.process_capture_frame(&mut data);
            drop(ap);

            // let mut low_pass_buffer = [0.0f32; BUFFER_SIZE];
            // low_pass_filter.process(&data, &mut low_pass_buffer);

            // let mut final_buffer = [0.0f32; BUFFER_SIZE];
            // noise_gate.process_block(&low_pass_buffer, &mut final_buffer);

            if let Ok(chunk) = prod.write_chunk_uninit(BUFFER_SIZE) {
                let mut encoded_buffer = [0u8; BUFFER_SIZE];
                let _ = encoder.encode_float(&data, &mut encoded_buffer);
                chunk.fill_from_iter(encoded_buffer.to_owned());
            }
        }
    });

    const MAX_DECODE_BUFFER_SIZE: usize = 5760;
    let mut decode_buffer = [0.0f32; MAX_DECODE_BUFFER_SIZE];
    let mut output_queue: VecDeque<f32> = VecDeque::new();

    let mut decoder = Decoder::new(48000, opus::Channels::Mono).unwrap();

    let (tx_o, rx_o) = mpsc::sync_channel::<[f32; BUFFER_SIZE]>(2);

    let output_stream = output_device
        .build_output_stream(
            &output_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                while let Ok(new_data) = rx_o.try_recv() {
                    output_queue.extend(new_data);
                }

                // fill the output buffer
                let samples_to_play = output_queue.len().min(data.len());
                for (i, sample) in data.iter_mut().enumerate().take(samples_to_play) {
                    *sample = output_queue.pop_front().unwrap();
                }


                println!(
                    "Played {} samples, {} remaining in queue",
                    samples_to_play,
                    output_queue.len()
                );
                
            },
            move |err| {
                eprintln!("There was an output error: {:?}", err);
            },
            None,
        )
        .unwrap();

    let _output_processing_thread = thread::spawn(move || {
        // let mut processor_output = create_processor();
        let mut decoder = Decoder::new(48000, opus::Channels::Mono).unwrap();
        let mut decode_buffer = [0.0f32; MAX_DECODE_BUFFER_SIZE];

        loop {
            if let Ok(chunk) = cons.read_chunk(BUFFER_SIZE) {
                let (first, second) = chunk.as_slices();
                let mut combined = [0u8; BUFFER_SIZE];
                combined[..first.len()].copy_from_slice(first);
                if !second.is_empty() {
                    combined[first.len()..first.len() + second.len()].copy_from_slice(second);
                }

                if let Ok(decoded_samples) =
                    decoder.decode_float(&combined, &mut decode_buffer, false)
                {
                    let mut buffer_to_send = [0.0f32; BUFFER_SIZE];
                    let len = decoded_samples.min(BUFFER_SIZE);
                    buffer_to_send[..len].copy_from_slice(&decode_buffer[..len]);

                    let mut ap = echo_processor_output.lock().unwrap();
                    ap.process_render_frame(&mut buffer_to_send);
                    drop(ap);

                    let _ = tx_o.send(buffer_to_send);
                } else {
                    println!("Decode sample error");
                }
                chunk.commit_all();
            } else {
                println!("Chunk read error");
            }
        }
    });

    output_stream.play().unwrap();
    input_stream.play().unwrap();

    println!("Playing... Press Ctrl+C to stop.");

    std::thread::park();
}
