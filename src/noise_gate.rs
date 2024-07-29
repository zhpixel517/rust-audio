use std::time::Instant;
// threshold -> volume control
// attack -> how fast the gate opens, keep as fast as possible
// release -> how fast the gate closes

pub struct NoiseGate {
    /// Decibels scale
    threshold_linear: f32,
    alpha: f32,
    low_pass_coeff: f32,

    attack_time: f32,
    release_time: f32,
    hold_time: f32,

    sample_rate: f32,

    envelope: f32,

    hold_counter: usize,
    current_gain: f32,
}

impl NoiseGate {
    pub fn new(
        threshold_db: f32,
        alpha: f32,
        attack_time: f32,
        release_time: f32,
        hold_time: f32,
        sample_rate: f32,
    ) -> Self {
        let threshold_linear = (10.0 as f32).powf(threshold_db / 20.0);
        NoiseGate {
            threshold_linear,
            alpha,
            low_pass_coeff: 0.0,
            attack_time,
            release_time,
            hold_time,
            sample_rate,
            envelope: 0.0,
            hold_counter: 0,
            current_gain: 0.0,
        }
    }

    #[inline(always)]
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let start = Instant::now();

        let attack_coef = (-1.0 / (self.attack_time * self.sample_rate)).exp();
        let release_coef = (-1.0 / (self.release_time * self.sample_rate)).exp();

        // number of samples for hold time
        let hold_samples = (self.hold_time * self.sample_rate) as usize;

        let mut samples_attenuated = 0;

        for (i, &sample) in input.iter().enumerate() {
            let abs_sample = sample.abs();

            // low pass filter
            self.low_pass_coeff =
                self.alpha * self.low_pass_coeff + (1.0 - self.alpha) * abs_sample;

            // envelope follower
            if self.low_pass_coeff > self.envelope {
                self.envelope =
                    attack_coef * (self.envelope - self.low_pass_coeff) + self.low_pass_coeff;
            } else {
                self.envelope =
                    release_coef * (self.envelope - self.low_pass_coeff) + self.low_pass_coeff;
            }

            // if i % 480 == 0 {  // Print every 480th sample (about 10ms at 48kHz)
            //     println!("Sample: {}, Abs: {:.6}, LowPass: {:.6}, Envelope: {:.6}, Threshold: {:.6}",
            //              i, abs_sample, self.low_pass_coeff, self.envelope, self.threshold_linear);
            // }

            // gate logic
            if self.envelope >= self.threshold_linear {
                self.hold_counter = hold_samples;
            } else if self.hold_counter > 0 {
                self.hold_counter -= 1; // double check this
            }

            // Compute gain
            let target_gain = if self.hold_counter > 0 { 1.0 } else { 0.0 };

            // Smooth gain changes
            self.current_gain = if target_gain > self.current_gain {
                attack_coef * (self.current_gain - target_gain) + target_gain
            } else {
                release_coef * (self.current_gain - target_gain) + target_gain
            };

            // Apply gain
            output[i] = sample * self.current_gain;

            if self.current_gain < 1.0 {
                samples_attenuated += 1;
            }
        }

        // println!(
        //     "Processed {} samples, attenuated {}: Time was {} micros",
        //     input.len(),
        //     samples_attenuated,
        //     start.elapsed().as_micros()
        // );
    }
}
