use eyre::Result;
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use ndarray::{Array, CowArray, IxDyn};
use ort::{GraphOptimizationLevel, SessionBuilder, Value};
use robo_rover_lib::types::DetectionFrame;
use std::sync::Arc;
use tracing::{debug, info, warn};

const REID_INPUT_HEIGHT: u32 = 256;
const REID_INPUT_WIDTH: u32 = 128;

#[derive(Clone)]
pub struct ReIdConfig {
    pub model_path: String,
    pub min_bbox_size: u32,
}

pub struct ReIdExtractor {
    session: ort::Session,
    min_bbox_size: u32,
}

impl ReIdExtractor {
    pub fn new(env: Arc<ort::Environment>, config: ReIdConfig) -> Result<Self> {
        info!("Loading ReID model from: {}", config.model_path);

        let session = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(2)?
            .with_model_from_file(&config.model_path)?;

        info!("ReID model loaded");

        Ok(Self {
            session,
            min_bbox_size: config.min_bbox_size,
        })
    }

    /// Crop each bbox from the frame (zero-copy borrow) and extract ReID features.
    pub fn process_detections(
        &mut self,
        frame_data: &[u8],
        width: u32,
        height: u32,
        mut detection_frame: DetectionFrame,
    ) -> Result<DetectionFrame> {
        debug!(
            "Extracting ReID features for {} detections",
            detection_frame.detections.len()
        );

        for detection in &mut detection_frame.detections {
            match self.crop_detection(
                frame_data,
                width,
                height,
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.x2,
                detection.bbox.y2,
            ) {
                Ok(crop) => match self.extract_features(&crop) {
                    Ok(features) => {
                        debug!(
                            "ReID extracted for {} ({}x{})",
                            detection.class_name,
                            crop.width(),
                            crop.height()
                        );
                        detection.reid_features = Some(features);
                    }
                    Err(e) => warn!("ReID inference failed for {}: {}", detection.class_name, e),
                },
                Err(e) => debug!("Skipping ReID for {}: {}", detection.class_name, e),
            }
        }

        Ok(detection_frame)
    }

    /// Crop a detection from the frame using a borrowed buffer — no frame copy per detection.
    fn crop_detection(
        &self,
        frame_data: &[u8],
        width: u32,
        height: u32,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
    ) -> Result<DynamicImage> {
        let px1 = (x1 * width as f32).max(0.0) as u32;
        let py1 = (y1 * height as f32).max(0.0) as u32;
        let px2 = (x2 * width as f32).min(width as f32) as u32;
        let py2 = (y2 * height as f32).min(height as f32) as u32;

        let bbox_width = px2.saturating_sub(px1);
        let bbox_height = py2.saturating_sub(py1);

        if bbox_width < self.min_bbox_size || bbox_height < self.min_bbox_size {
            return Err(eyre::eyre!(
                "Bbox too small: {}x{} (min: {})",
                bbox_width,
                bbox_height,
                self.min_bbox_size
            ));
        }

        // Borrow the frame slice directly — avoids a full frame copy per detection
        let img_buffer = ImageBuffer::<Rgb<u8>, &[u8]>::from_raw(width, height, frame_data)
            .ok_or_else(|| eyre::eyre!("Invalid frame dimensions: {}x{}", width, height))?;

        let mut crop_buffer = RgbImage::new(bbox_width, bbox_height);
        for y in 0..bbox_height {
            for x in 0..bbox_width {
                let src_x = px1 + x;
                let src_y = py1 + y;
                if src_x < width && src_y < height {
                    crop_buffer.put_pixel(x, y, *img_buffer.get_pixel(src_x, src_y));
                }
            }
        }

        Ok(DynamicImage::ImageRgb8(crop_buffer))
    }

    fn extract_features(&mut self, crop: &DynamicImage) -> Result<Vec<f32>> {
        let input = self.preprocess_crop(crop)?;

        let input_cow: CowArray<f32, _> = CowArray::from(&input);
        let input_tensor = Value::from_array(self.session.allocator(), &input_cow)?;
        let outputs = self.session.run(vec![input_tensor])?;
        let output_tensor = outputs[0].try_extract::<f32>()?;

        Ok(output_tensor.view().iter().copied().collect())
    }

    fn preprocess_crop(&self, crop: &DynamicImage) -> Result<Array<f32, IxDyn>> {
        let resized = crop.resize_exact(
            REID_INPUT_WIDTH,
            REID_INPUT_HEIGHT,
            image::imageops::FilterType::Triangle,
        );
        let rgb_image = resized.to_rgb8();

        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];

        let mut array = Array::zeros(IxDyn(&[
            1,
            3,
            REID_INPUT_HEIGHT as usize,
            REID_INPUT_WIDTH as usize,
        ]));

        for (x, y, pixel) in rgb_image.enumerate_pixels() {
            for c in 0..3 {
                array[[0, c, y as usize, x as usize]] =
                    ((pixel[c] as f32 / 255.0) - mean[c]) / std[c];
            }
        }

        Ok(array)
    }
}
