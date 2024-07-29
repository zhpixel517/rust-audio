// use rustfft::{num_complex::Complex, FftPlanner};

// pub struct PitchBasedSuppressor {
//     sample_rate: f32,
//     fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
//     ifft: std::sync::Arc<dyn rustfft::Fft<f32>>,
//     buffer_size: usize,
//     max_pitch: f32,
// }

// impl PitchBasedSuppressor {
//     pub fn new(buffer_size: usize, sample_rate: f32, max_pitch: f32) -> Self {
//         let mut planner = FftPlanner::new();
//         let fft = planner.plan_fft_forward(buffer_size);
//         let ifft = planner.plan_fft_inverse(buffer_size);

//         PitchBasedSuppressor {
//             sample_rate,
//             fft,
//             ifft,
//             buffer_size,
//             max_pitch,
//         }
//     }

//     pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
//         assert_eq!(input.len(), output.len());
//         assert_eq!(input.len(), self.buffer_size);

//         let pitch = self.detect_pitch(input);

//         if pitch > self.max_pitch {
//             // If pitch is above threshold, output silence
//             output.fill(0.0);
//         } else {
//             // Otherwise, pass through the input
//             output.copy_from_slice(input);
//         }
//     }

//     fn detect_pitch(&self, input: &[f32]) -> f32 {
//         let mut buffer: Vec<Complex<f32>> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();

//         // Perform autocorrelation using FFT
//         self.fft.process(&mut buffer);
//         for x in &mut buffer {
//             *x = Complex::new(x.norm_sqr(), 0.0);
//         }
//         self.ifft.process(&mut buffer);

//         // Find the highest peak after the first few samples
//         let start = (self.sample_rate / 1000.0) as usize; // Start after 1ms
//         let end = (self.sample_rate / 50.0).min(input.len() as f32) as usize; // End at 50Hz (20ms)
//         let max_index = (start..end)
//             .max_by_key(|&i| (buffer[i].re * 1000.0) as i32)
//             .unwrap_or(0);

//         self.sample_rate / max_index as f32
//     }
// }

pub struct LowPassFilter {
    alpha: f32,
    prev_output: f32,
}

impl LowPassFilter {
    pub fn new(cutoff_freq: f32, sample_rate: f32) -> Self {
        let dt = 1.0 / sample_rate;
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_freq);
        let alpha = dt / (rc + dt);

        LowPassFilter {
            alpha,
            prev_output: 0.0,
        }
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32]){
        for (x, y) in input.iter().zip(output.iter_mut()){
            *y = self.alpha * x + (1.0 - self.alpha) * self.prev_output;
            self.prev_output = *y;
        }

        // map function? 
    }
}
