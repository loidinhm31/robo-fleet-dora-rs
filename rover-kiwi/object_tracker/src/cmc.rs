use image::{GrayImage, Luma};
use imageproc::corners::{corners_fast9, Corner};
use nalgebra as na;
use tracing::debug;

/// Camera Motion Compensator using sparse optical flow
pub struct CameraMotionCompensator {
    prev_frame: Option<GrayImage>,
    prev_corners: Vec<Corner>,
    max_corners: usize,
    corner_threshold: u8,
    match_threshold: f32,
    min_matches: usize,
}

impl CameraMotionCompensator {
    pub fn new() -> Self {
        Self {
            prev_frame: None,
            prev_corners: Vec::new(),
            max_corners: 200,
            corner_threshold: 40,
            match_threshold: 20.0,
            min_matches: 8,
        }
    }

    /// Estimate camera motion between previous and current frame
    /// Returns homography matrix if enough motion is detected
    pub fn estimate_motion(&mut self, current_frame: &GrayImage) -> Option<na::Matrix3<f32>> {
        // Need previous frame to estimate motion
        let prev_frame = match &self.prev_frame {
            Some(frame) => frame,
            None => {
                // First frame - store and return no motion
                self.prev_frame = Some(current_frame.clone());
                self.prev_corners = self.detect_corners(current_frame);
                return None;
            }
        };

        // Detect corners in current frame
        let curr_corners = self.detect_corners(current_frame);

        // Match corners between frames using simple patch matching
        let matches = self.match_corners(prev_frame, current_frame, &self.prev_corners, &curr_corners);

        debug!("CMC: {} corners prev, {} corners curr, {} matches",
               self.prev_corners.len(), curr_corners.len(), matches.len());

        // Need minimum matches to estimate motion
        if matches.len() < self.min_matches {
            debug!("CMC: Not enough matches ({}), returning identity", matches.len());
            self.prev_frame = Some(current_frame.clone());
            self.prev_corners = curr_corners;
            return None;
        }

        // Estimate affine transformation using RANSAC
        let transform = match self.estimate_affine_ransac(&matches) {
            Some(t) => t,
            None => {
                debug!("CMC: RANSAC failed, returning identity");
                self.prev_frame = Some(current_frame.clone());
                self.prev_corners = curr_corners;
                return None;
            }
        };

        // Update state
        self.prev_frame = Some(current_frame.clone());
        self.prev_corners = curr_corners;

        Some(transform)
    }

    /// Detect corners in grayscale image
    fn detect_corners(&self, image: &GrayImage) -> Vec<Corner> {
        let mut corners = corners_fast9(image, self.corner_threshold);

        // Limit number of corners
        if corners.len() > self.max_corners {
            // Sort by score (not directly available in Corner, so we limit arbitrarily)
            corners.truncate(self.max_corners);
        }

        corners
    }

    /// Match corners between frames using simple patch matching
    fn match_corners(
        &self,
        prev_frame: &GrayImage,
        curr_frame: &GrayImage,
        prev_corners: &[Corner],
        curr_corners: &[Corner],
    ) -> Vec<(na::Point2<f32>, na::Point2<f32>)> {
        let mut matches = Vec::new();
        let patch_size = 5;
        let half_patch = patch_size / 2;

        for prev_corner in prev_corners {
            let px = prev_corner.x as i32;
            let py = prev_corner.y as i32;

            // Skip corners too close to border
            if px < half_patch as i32 || py < half_patch as i32
                || px >= (prev_frame.width() - half_patch) as i32
                || py >= (prev_frame.height() - half_patch) as i32 {
                continue;
            }

            // Find best match in current frame
            let mut best_match = None;
            let mut best_distance = f32::MAX;

            for curr_corner in curr_corners {
                let cx = curr_corner.x as i32;
                let cy = curr_corner.y as i32;

                // Skip corners too close to border
                if cx < half_patch as i32 || cy < half_patch as i32
                    || cx >= (curr_frame.width() - half_patch) as i32
                    || cy >= (curr_frame.height() - half_patch) as i32 {
                    continue;
                }

                // Compute patch distance (SSD)
                let distance = self.patch_distance(
                    prev_frame, px as u32, py as u32,
                    curr_frame, cx as u32, cy as u32,
                    patch_size,
                );

                if distance < best_distance {
                    best_distance = distance;
                    best_match = Some((cx as f32, cy as f32));
                }
            }

            // Add match if below threshold
            if let Some((cx, cy)) = best_match {
                if best_distance < self.match_threshold {
                    matches.push((
                        na::Point2::new(px as f32, py as f32),
                        na::Point2::new(cx, cy),
                    ));
                }
            }
        }

        matches
    }

    /// Compute patch distance (sum of squared differences)
    fn patch_distance(
        &self,
        img1: &GrayImage,
        x1: u32,
        y1: u32,
        img2: &GrayImage,
        x2: u32,
        y2: u32,
        patch_size: u32,
    ) -> f32 {
        let half = patch_size / 2;
        let mut sum = 0.0;

        for dy in 0..patch_size {
            for dx in 0..patch_size {
                let px1 = (x1 + dx - half) as u32;
                let py1 = (y1 + dy - half) as u32;
                let px2 = (x2 + dx - half) as u32;
                let py2 = (y2 + dy - half) as u32;

                let p1 = img1.get_pixel(px1, py1)[0] as f32;
                let p2 = img2.get_pixel(px2, py2)[0] as f32;

                let diff = p1 - p2;
                sum += diff * diff;
            }
        }

        (sum / (patch_size * patch_size) as f32).sqrt()
    }

    /// Estimate affine transformation using RANSAC
    fn estimate_affine_ransac(
        &self,
        matches: &[(na::Point2<f32>, na::Point2<f32>)],
    ) -> Option<na::Matrix3<f32>> {
        let max_iterations = 100;
        let inlier_threshold = 3.0; // pixels
        let min_inliers = (matches.len() as f32 * 0.5).max(self.min_matches as f32) as usize;

        let mut best_transform = None;
        let mut best_inlier_count = 0;

        for _ in 0..max_iterations {
            // Randomly sample 3 matches
            if matches.len() < 3 {
                return None;
            }

            let sample: Vec<_> = (0..3)
                .map(|i| matches[i % matches.len()])
                .collect();

            // Estimate affine transform from 3 points
            if let Some(transform) = self.estimate_affine_from_points(&sample) {
                // Count inliers
                let mut inlier_count = 0;
                for (p1, p2) in matches {
                    let transformed = self.transform_point(&transform, p1);
                    let error = (transformed - p2).norm();
                    if error < inlier_threshold {
                        inlier_count += 1;
                    }
                }

                if inlier_count > best_inlier_count {
                    best_inlier_count = inlier_count;
                    best_transform = Some(transform);
                }
            }
        }

        if best_inlier_count >= min_inliers {
            debug!("CMC: RANSAC found {} inliers ({}% of matches)",
                   best_inlier_count,
                   (best_inlier_count as f32 / matches.len() as f32 * 100.0) as u32);
            best_transform
        } else {
            None
        }
    }

    /// Estimate affine transformation from point correspondences
    fn estimate_affine_from_points(
        &self,
        matches: &[(na::Point2<f32>, na::Point2<f32>)],
    ) -> Option<na::Matrix3<f32>> {
        if matches.len() < 3 {
            return None;
        }

        // Solve for affine parameters using least squares
        // [x'] = [a b tx] [x]
        // [y']   [c d ty] [y]
        // [1 ]   [0 0 1 ] [1]

        let n = matches.len();
        let mut matrix_a = na::DMatrix::zeros(2 * n, 6);
        let mut b = na::DVector::zeros(2 * n);

        for (i, (p1, p2)) in matches.iter().enumerate() {
            // Row for x coordinate
            matrix_a[(2 * i, 0)] = p1.x;
            matrix_a[(2 * i, 1)] = p1.y;
            matrix_a[(2 * i, 2)] = 1.0;
            b[2 * i] = p2.x;

            // Row for y coordinate
            matrix_a[(2 * i + 1, 3)] = p1.x;
            matrix_a[(2 * i + 1, 4)] = p1.y;
            matrix_a[(2 * i + 1, 5)] = 1.0;
            b[2 * i + 1] = p2.y;
        }

        // Solve least squares: x = (A^T A)^-1 A^T b
        let ata = matrix_a.transpose() * &matrix_a;
        let atb = matrix_a.transpose() * b;

        if let Some(ata_inv) = ata.try_inverse() {
            let params = ata_inv * atb;

            // Build transformation matrix
            let transform = na::Matrix3::new(
                params[0], params[1], params[2],
                params[3], params[4], params[5],
                0.0, 0.0, 1.0,
            );

            Some(transform)
        } else {
            None
        }
    }

    /// Transform a point using affine transformation
    fn transform_point(&self, transform: &na::Matrix3<f32>, point: &na::Point2<f32>) -> na::Point2<f32> {
        let p = na::Vector3::new(point.x, point.y, 1.0);
        let transformed = transform * p;
        na::Point2::new(transformed.x, transformed.y)
    }

    /// Convert RGB8 frame to grayscale
    pub fn rgb_to_gray(rgb_data: &[u8], width: u32, height: u32) -> GrayImage {
        let mut gray = GrayImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < rgb_data.len() {
                    let r = rgb_data[idx] as f32;
                    let g = rgb_data[idx + 1] as f32;
                    let b = rgb_data[idx + 2] as f32;

                    // Standard RGB to grayscale conversion
                    let gray_value = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                    gray.put_pixel(x, y, Luma([gray_value]));
                }
            }
        }

        gray
    }
}

impl Default for CameraMotionCompensator {
    fn default() -> Self {
        Self::new()
    }
}
