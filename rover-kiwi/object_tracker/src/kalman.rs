use nalgebra as na;

/// Kalman filter for tracking bounding box center (x, y) and velocity (vx, vy).
/// State: [x, y, vx, vy] — constant velocity model, dt = 1 frame.
pub struct KalmanFilter {
    pub state: na::Vector4<f32>,
    covariance: na::Matrix4<f32>,
    process_noise: na::Matrix4<f32>,
    measurement_noise: na::Matrix2<f32>,
    transition: na::Matrix4<f32>,
    measurement: na::Matrix2x4<f32>,
}

impl KalmanFilter {
    pub fn new(initial_x: f32, initial_y: f32) -> Self {
        let state = na::Vector4::new(initial_x, initial_y, 0.0, 0.0);
        let covariance = na::Matrix4::from_diagonal(&na::Vector4::new(1.0, 1.0, 10.0, 10.0));
        let process_noise = na::Matrix4::from_diagonal(&na::Vector4::new(0.01, 0.01, 0.1, 0.1));
        let measurement_noise = na::Matrix2::from_diagonal(&na::Vector2::new(0.1, 0.1));

        #[rustfmt::skip]
        let transition = na::Matrix4::new(
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        #[rustfmt::skip]
        let measurement = na::Matrix2x4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        );

        Self {
            state,
            covariance,
            process_noise,
            measurement_noise,
            transition,
            measurement,
        }
    }

    pub fn predict(&mut self) {
        self.state = self.transition * self.state;
        self.covariance =
            self.transition * self.covariance * self.transition.transpose() + self.process_noise;
    }

    pub fn update(&mut self, measurement_x: f32, measurement_y: f32) {
        let z = na::Vector2::new(measurement_x, measurement_y);
        let innovation = z - self.measurement * self.state;
        let innovation_cov = self.measurement * self.covariance * self.measurement.transpose()
            + self.measurement_noise;

        if let Some(s_inv) = innovation_cov.try_inverse() {
            let kalman_gain = self.covariance * self.measurement.transpose() * s_inv;
            self.state += kalman_gain * innovation;
            let identity = na::Matrix4::identity();
            self.covariance = (identity - kalman_gain * self.measurement) * self.covariance;
        }
    }

    pub fn get_position(&self) -> (f32, f32) {
        (self.state[0], self.state[1])
    }

    pub fn get_velocity(&self) -> (f32, f32) {
        (self.state[2], self.state[3])
    }
}
