/// Simple PID controller implementation
pub struct PIDController {
    kp: f64,  // Proportional gain
    ki: f64,  // Integral gain
    kd: f64,  // Derivative gain

    // Output limits
    output_min: f64,
    output_max: f64,

    // State
    integral: f64,
    previous_error: f64,
    first_update: bool,
}

impl PIDController {
    /// Create a new PID controller
    pub fn new(kp: f64, ki: f64, kd: f64, output_min: f64, output_max: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            output_min,
            output_max,
            integral: 0.0,
            previous_error: 0.0,
            first_update: true,
        }
    }

    /// Update the PID controller with a new error value
    /// Returns the control output
    pub fn update(&mut self, error: f64, dt: f64) -> f64 {
        // Proportional term
        let p_term = self.kp * error;

        // Integral term (with anti-windup)
        self.integral += error * dt;
        let i_term = self.ki * self.integral;

        // Derivative term (handle first update)
        let d_term = if self.first_update {
            self.first_update = false;
            0.0
        } else {
            self.kd * (error - self.previous_error) / dt
        };

        self.previous_error = error;

        // Compute output
        let output = p_term + i_term + d_term;

        // Apply output limits
        let output = output.clamp(self.output_min, self.output_max);

        // Anti-windup: If output is saturated, don't accumulate integral
        if output >= self.output_max || output <= self.output_min {
            self.integral -= error * dt;
        }

        output
    }

    /// Reset the PID controller state
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.previous_error = 0.0;
        self.first_update = true;
    }

    /// Set new PID gains
    #[allow(dead_code)]
    pub fn set_gains(&mut self, kp: f64, ki: f64, kd: f64) {
        self.kp = kp;
        self.ki = ki;
        self.kd = kd;
    }

    /// Get current gains
    #[allow(dead_code)]
    pub fn get_gains(&self) -> (f64, f64, f64) {
        (self.kp, self.ki, self.kd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_proportional_only() {
        let mut pid = PIDController::new(1.0, 0.0, 0.0, -10.0, 10.0);

        // Simple proportional response
        let output = pid.update(5.0, 0.1);
        assert_eq!(output, 5.0);

        let output = pid.update(-3.0, 0.1);
        assert_eq!(output, -3.0);
    }

    #[test]
    fn test_pid_output_limits() {
        let mut pid = PIDController::new(1.0, 0.0, 0.0, -5.0, 5.0);

        // Output should be clamped
        let output = pid.update(10.0, 0.1);
        assert_eq!(output, 5.0);

        let output = pid.update(-10.0, 0.1);
        assert_eq!(output, -5.0);
    }

    #[test]
    fn test_pid_integral_term() {
        let mut pid = PIDController::new(0.0, 1.0, 0.0, -10.0, 10.0);

        // Integral should accumulate
        let output = pid.update(1.0, 0.1);
        assert_eq!(output, 0.1);

        let output = pid.update(1.0, 0.1);
        assert_eq!(output, 0.2);

        let output = pid.update(1.0, 0.1);
        assert_eq!(output, 0.3);
    }

    #[test]
    fn test_pid_reset() {
        let mut pid = PIDController::new(1.0, 1.0, 1.0, -10.0, 10.0);

        pid.update(5.0, 0.1);
        pid.update(5.0, 0.1);

        pid.reset();

        // After reset, should behave as if first update
        let output = pid.update(5.0, 0.1);
        assert_eq!(output, 5.5);  // P + I terms only (no D on first update)
    }
}
