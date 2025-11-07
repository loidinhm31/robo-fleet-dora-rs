use crate::types::{ArmConfig, DHParameter};
use eyre::Result;
use nalgebra::{DMatrix, Matrix4, Vector3};

pub struct ForwardKinematics {
    dh_params: Vec<DHParameter>,
    link_lengths: Vec<f64>,
    base_offset: Vector3<f64>,
}

impl ForwardKinematics {
    pub fn new(config: &ArmConfig) -> Result<Self> {
        Ok(Self {
            dh_params: config.kinematics.dh_parameters.clone(),
            link_lengths: config.kinematics.link_lengths.clone(),
            base_offset: Vector3::new(
                config.kinematics.base_offset[0],
                config.kinematics.base_offset[1],
                config.kinematics.base_offset[2],
            ),
        })
    }

    pub fn compute_end_effector_pose(&self, joint_angles: &[f64]) -> Result<[f64; 6]> {
        if joint_angles.len() != self.dh_params.len() {
            return Err(eyre::eyre!("Joint angles count doesn't match DH parameters"));
        }

        let mut transform = Matrix4::identity();

        transform[(0, 3)] = self.base_offset.x;
        transform[(1, 3)] = self.base_offset.y;
        transform[(2, 3)] = self.base_offset.z;

        for (i, dh) in self.dh_params.iter().enumerate() {
            let theta = joint_angles[i] + dh.theta;
            let dh_transform = self.dh_transformation(dh.a, dh.alpha, dh.d, theta);
            transform = transform * dh_transform;
        }

        let position = Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]);
        // Fixed the deprecated API call
        let rotation_matrix = transform.fixed_view::<3, 3>(0, 0);
        let euler_angles = self.rotation_matrix_to_euler(&rotation_matrix.into_owned());

        Ok([
            position.x,
            position.y,
            position.z,
            euler_angles.x,
            euler_angles.y,
            euler_angles.z,
        ])
    }

    fn dh_transformation(&self, a: f64, alpha: f64, d: f64, theta: f64) -> Matrix4<f64> {
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let cos_alpha = alpha.cos();
        let sin_alpha = alpha.sin();

        Matrix4::new(
            cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, a * cos_theta,
            sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta,
            0.0,        sin_alpha,              cos_alpha,             d,
            0.0,        0.0,                    0.0,                   1.0,
        )
    }

    fn rotation_matrix_to_euler(&self, rotation: &nalgebra::Matrix3<f64>) -> Vector3<f64> {
        let sy = (rotation[(0, 0)].powi(2) + rotation[(1, 0)].powi(2)).sqrt();

        let singular = sy < 1e-6;

        let (x, y, z) = if !singular {
            let x = rotation[(2, 1)].atan2(rotation[(2, 2)]);
            let y = (-rotation[(2, 0)]).atan2(sy);
            let z = rotation[(1, 0)].atan2(rotation[(0, 0)]);
            (x, y, z)
        } else {
            let x = (-rotation[(1, 2)]).atan2(rotation[(1, 1)]);
            let y = (-rotation[(2, 0)]).atan2(sy);
            let z = 0.0;
            (x, y, z)
        };

        Vector3::new(x, y, z)
    }

    pub fn compute_jacobian(&self, joint_angles: &[f64]) -> Result<DMatrix<f64>> {
        let n_joints = joint_angles.len();
        let mut jacobian = DMatrix::zeros(6, n_joints);

        let current_pose = self.compute_end_effector_pose(joint_angles)?;
        let ee_position = Vector3::new(current_pose[0], current_pose[1], current_pose[2]);

        let mut transforms = Vec::new();
        let mut current_transform = Matrix4::identity();

        current_transform[(0, 3)] = self.base_offset.x;
        current_transform[(1, 3)] = self.base_offset.y;
        current_transform[(2, 3)] = self.base_offset.z;
        transforms.push(current_transform);

        for (i, dh) in self.dh_params.iter().enumerate() {
            let theta = joint_angles[i] + dh.theta;
            let dh_transform = self.dh_transformation(dh.a, dh.alpha, dh.d, theta);
            current_transform = current_transform * dh_transform;
            transforms.push(current_transform);
        }

        for i in 0..n_joints {
            let transform = &transforms[i];
            let joint_position = Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]);
            let joint_axis = Vector3::new(transform[(0, 2)], transform[(1, 2)], transform[(2, 2)]);

            let linear_contrib = joint_axis.cross(&(ee_position - joint_position));
            jacobian[(0, i)] = linear_contrib.x;
            jacobian[(1, i)] = linear_contrib.y;
            jacobian[(2, i)] = linear_contrib.z;

            jacobian[(3, i)] = joint_axis.x;
            jacobian[(4, i)] = joint_axis.y;
            jacobian[(5, i)] = joint_axis.z;
        }

        Ok(jacobian)
    }
}