#![feature(portable_simd)]
mod aec;
mod feedback_suppressor;
mod low_pass_filter;
mod noise_gate;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use low_pass_filter::LowPassFilter;
use noise_gate::NoiseGate;
use opus::{packet, Decoder, Encoder};
use rtcp::payload_feedbacks;
use rtp::sequence;
use rtp_rs::{RtpPacketBuilder, RtpReader, Seq};
use rtrb::RingBuffer;
use socket2::{Domain, Protocol, Socket, Type};
use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::net::SocketAddr;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::{env, thread};
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
            enable_extended_filter: false,
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

const RTP_PORT: i32 = 37069;

fn record_send(addr: &str) {
    let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP)).unwrap();
    socket.set_nonblocking(true).unwrap();

    let bind_addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
    socket.bind(&bind_addr.into()).unwrap();

    let addr: SocketAddr = format!("{}:{}", addr, RTP_PORT).parse().unwrap();

    const SAMPLE_RATE: usize = 48000;
    const BUFFER_SIZE: usize = 480;

    let echo_processor = Arc::new(Mutex::new(create_processor()));

    let host = cpal::default_host();

    let input_device = host.default_input_device().unwrap();

    let input_config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(SAMPLE_RATE as u32),
        buffer_size: cpal::BufferSize::Fixed(BUFFER_SIZE as u32),
    };

    let (tx_i, rx_i) = mpsc::sync_channel::<[f32; BUFFER_SIZE]>(2);

    let input_stream = input_device
        .build_input_stream(
            &input_config,
            move |data: &[f32], _: &_| {
                let mut buffer = [0.0f32; BUFFER_SIZE];
                let len = data.len().min(BUFFER_SIZE);
                buffer[..len].copy_from_slice(&data[..len]);
                let _ = tx_i.send(buffer);
            },
            move |err| {
                eprintln!("There was an input error: {:?}", err);
            },
            None,
        )
        .unwrap();

    let _input_processing_thread = thread::spawn(move || {
        // let mut processor_input = create_processor();

        let noise_gate = NoiseGate::new(-23.5, 0.01, 0.0001, 0.02, 0.3, 48000.0);
        let cutoff_freq = 3000.0;
        let low_pass_filter = LowPassFilter::new(cutoff_freq, 48000.0);

        let mut encoder =
            Encoder::new(48000, opus::Channels::Mono, opus::Application::Voip).unwrap();
        // encoder.set_bitrate(opus::Bitrate::Auto).unwrap();

        let mut sequence_number = 0;
        let mut timestamp = 0;

        while let Ok(mut data) = rx_i.recv() {
            let mut ap = echo_processor.lock().unwrap();
            let _ = ap.process_capture_frame(&mut data);
            drop(ap);

            let mut encoded_buffer = [0u8; BUFFER_SIZE];
            let encoded_len = encoder.encode_float(&data, &mut encoded_buffer).unwrap();

            let rtp_packet = RtpPacketBuilder::new()
                .payload_type(111)
                .sequence(sequence_number.into())
                .timestamp(timestamp)
                .ssrc(0x12345678) //random
                .payload(&encoded_buffer[..encoded_len])
                .build()
                .unwrap();

            let _ = socket.send_to(&rtp_packet, &addr.into());

            sequence_number = sequence_number.wrapping_add(1);
            timestamp += BUFFER_SIZE as u32;
        }
    });

    input_stream.play().unwrap();

    println!("Streaming... Press Ctrl+C to stop.");

    std::thread::park();
}

fn recv_play() {
    let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP)).unwrap();
    socket.set_nonblocking(true).unwrap();

    let loopback: SocketAddr = "0.0.0.0:37069".parse().unwrap();
    socket.bind(&loopback.into()).unwrap();

    const SAMPLE_RATE: usize = 48000;
    const BUFFER_SIZE: usize = 480;

    let echo_processor = Arc::new(Mutex::new(create_processor()));

    let host = cpal::default_host();
    let output_device = host.default_output_device().unwrap();
    let output_config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(SAMPLE_RATE as u32),
        buffer_size: cpal::BufferSize::Fixed(BUFFER_SIZE as u32),
    };

    let mut output_queue: VecDeque<f32> = VecDeque::new();
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

    const MAX_DECODE_BUFFER_SIZE: usize = 5760;

    let _output_processing_thread = thread::spawn(move || {
        // let mut processor_output = create_processor();
        let mut decoder = Decoder::new(48000, opus::Channels::Mono).unwrap();
        let mut decode_buffer = [0.0f32; MAX_DECODE_BUFFER_SIZE];
        let mut recv_buffer = [MaybeUninit::uninit(); 1500];

        loop {
            match socket.recv_from(&mut recv_buffer) {
                Ok((size, _)) => {
                    // let rtp_packet = RtpReader::new(&recv_buffer[..size]).unwrap();
                    let data: &[u8] = unsafe {
                        std::slice::from_raw_parts(recv_buffer.as_ptr() as *const u8, size)
                    };

                    let packet = RtpReader::new(data).unwrap().payload();

                    if let Ok(decoded_samples) =
                        decoder.decode_float(packet, &mut decode_buffer, false)
                    {
                        let mut buffer_to_send = [0.0f32; BUFFER_SIZE];
                        let len = decoded_samples.min(BUFFER_SIZE);
                        buffer_to_send[..len].copy_from_slice(&decode_buffer[..len]);

                        let mut ap = echo_processor.lock().unwrap();
                        ap.process_render_frame(&mut buffer_to_send).unwrap();
                        drop(ap);

                        let _ = tx_o.send(buffer_to_send);
                    } else {
                        println!("Decode sample error");
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // no data available, continue loop
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(e) => {
                    eprintln!("Error receiving data: {:?}", e);
                }
            }
        }
    });

    output_stream.play().unwrap();

    println!("Playing... Press Ctrl+C to stop.");

    std::thread::park();
}

fn loopback() {
    let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP)).unwrap();
    socket.set_nonblocking(true).unwrap();

    let loopback: SocketAddr = "127.0.0.1".parse().unwrap();
    socket.bind(&loopback.into());

    let socket_send = socket.try_clone().unwrap();
    let socket_recv = socket;

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

        let mut sequence_number = 0;
        let mut timestamp = 0;

        while let Ok(mut data) = rx_i.recv() {
            let mut ap = echo_processor_input.lock().unwrap();
            let _ = ap.process_capture_frame(&mut data);
            drop(ap);

            let mut encoded_buffer = [0u8; BUFFER_SIZE];
            let encoded_len = encoder.encode_float(&data, &mut encoded_buffer).unwrap();

            let mut rtp_packet = RtpPacketBuilder::new()
                .payload_type(111)
                .sequence(sequence_number.into())
                .timestamp(timestamp)
                .ssrc(0x12345678) //random
                .payload(&encoded_buffer[..encoded_len])
                .build()
                .unwrap();

            let _ = socket_send.send_to(&rtp_packet, &loopback.into());

            sequence_number = sequence_number.wrapping_add(1);
            timestamp += BUFFER_SIZE as u32;
            // let mut low_pass_buffer = [0.0f32; BUFFER_SIZE];
            // low_pass_filter.process(&data, &mut low_pass_buffer);

            // let mut final_buffer = [0.0f32; BUFFER_SIZE];
            // noise_gate.process_block(&low_pass_buffer, &mut final_buffer);

            // if let Ok(chunk) = prod.write_chunk_uninit(BUFFER_SIZE) {
            //     let mut encoded_buffer = [0u8; BUFFER_SIZE];
            //     let _ = encoder.encode_float(&data, &mut encoded_buffer);
            //     chunk.fill_from_iter(encoded_buffer.to_owned());
            // }
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
        let mut recv_buffer = [MaybeUninit::uninit(); 1500];

        loop {
            match socket_recv.recv_from(&mut recv_buffer) {
                Ok((size, _)) => {
                    // let rtp_packet = RtpReader::new(&recv_buffer[..size]).unwrap();
                    let data: &[u8] = unsafe {
                        std::slice::from_raw_parts(recv_buffer.as_ptr() as *const u8, size)
                    };

                    let packet = RtpReader::new(data).unwrap().payload();

                    if let Ok(decoded_samples) =
                        decoder.decode_float(packet, &mut decode_buffer, false)
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
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // no data available, continue loop
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(e) => {
                    eprintln!("Error receiving data: {:?}", e);
                }
            }
        }
    });

    output_stream.play().unwrap();
    input_stream.play().unwrap();

    println!("Playing... Press Ctrl+C to stop.");

    std::thread::park();
}

fn main() {
    let mut args: Vec<String> = env::args().collect();
    args.remove(0);
    println!("{:?}", args);

    if args.len() == 0 {
        loopback();
    }

    if args[0] == "send" && args.len() == 2 {
        record_send(&args[1]);
    }

    if args[0] == "recv" {
        recv_play();
    }
}
