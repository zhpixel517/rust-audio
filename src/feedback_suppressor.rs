use rustfft::{FftPlanner, num_complex::Complex};

pub struct FeedbackSuppressor {
    sample_rate: f32,
    fft_size: usize,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    threshold: f32,
}

impl FeedbackSuppressor {
    pub fn new(sample_rate: f32, fft_size: usize, threshold: f32) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        FeedbackSuppressor {
            sample_rate,
            fft_size,
            fft,
            threshold,
        }
    }

    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        
    }
}