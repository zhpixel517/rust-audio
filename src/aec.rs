#![feature(portable_simd)]
use std::simd::{f32x4, Simd};
use std::collections::VecDeque;


pub struct AEC {
    filter_length: usize, // how many past samples to consider
    step_size: f32, // learning rate
    filter: Vec<f32>, // contains our adaptive filter coefficients
    buffer: VecDeque<f32>, // recent output samples
    double_talk_threshold: f32
}

impl AEC {
    pub fn new(filter_length: usize, step_size: f32, double_talk_threshold: f32) -> Self {
        AEC {
            filter_length,
            step_size,
            filter: vec![0.0; filter_length],
            buffer: VecDeque::with_capacity(filter_length),
            double_talk_threshold
        }
    }

    // fn process(&mut self, input: f32, output: f32) -> f32 {

    // }
}