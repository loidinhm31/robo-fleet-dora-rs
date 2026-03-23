use eyre::Result;
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::{Array, CowArray, IxDyn};
use ort::{GraphOptimizationLevel, SessionBuilder, Value};
use robo_rover_lib::types::{BoundingBox, DetectionFrame, DetectionResult};
use std::sync::Arc;
use tracing::{debug, info};

const YOLO_CLASSES: &[&str] = &[
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
];

#[derive(Clone)]
pub struct DetectorConfig {
    pub model_path: String,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub target_classes: Vec<String>,
}

pub struct YoloDetector {
    session: ort::Session,
    confidence_threshold: f32,
    nms_threshold: f32,
    target_classes: Vec<String>,
    input_size: (u32, u32),
    frame_counter: u64,
}

impl YoloDetector {
    pub fn new(env: Arc<ort::Environment>, config: DetectorConfig) -> Result<Self> {
        info!("Loading YOLO model from: {}", config.model_path);

        let session = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(&config.model_path)?;

        info!("YOLO model loaded, running warmup...");

        let input_size = (640u32, 640u32);
        let dummy: Array<f32, IxDyn> =
            Array::zeros(IxDyn(&[1, 3, input_size.1 as usize, input_size.0 as usize]));
        let dummy_cow: CowArray<f32, _> = CowArray::from(&dummy);
        let dummy_tensor = Value::from_array(session.allocator(), &dummy_cow)?;
        let _ = session.run(vec![dummy_tensor])?;

        info!("YOLO warmup complete");

        Ok(Self {
            session,
            confidence_threshold: config.confidence_threshold,
            nms_threshold: config.nms_threshold,
            target_classes: config.target_classes,
            input_size,
            frame_counter: 0,
        })
    }

    pub fn detect(&mut self, frame_data: &[u8], width: u32, height: u32) -> Result<DetectionFrame> {
        let img_buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, frame_data.to_vec())
            .ok_or_else(|| eyre::eyre!("Failed to create image buffer from frame data"))?;

        let img = DynamicImage::ImageRgb8(img_buffer);
        let input = self.preprocess_image(&img)?;

        let output_array = {
            let input_cow: CowArray<f32, _> = CowArray::from(&input);
            let input_tensor = Value::from_array(self.session.allocator(), &input_cow)?;
            let outputs = self.session.run(vec![input_tensor])?;
            let output_tensor = outputs[0].try_extract::<f32>()?;
            let output_view = output_tensor.view();
            debug!("YOLO output shape: {:?}", output_view.shape());
            output_view.to_owned().into_dimensionality::<IxDyn>()?
        };

        let detections = self.postprocess_output(&output_array, width, height)?;
        let frame_id = self.frame_counter;
        self.frame_counter += 1;

        Ok(DetectionFrame::new(frame_id, width, height, detections))
    }

    fn preprocess_image(&self, img: &DynamicImage) -> Result<Array<f32, IxDyn>> {
        let (target_width, target_height) = self.input_size;

        let resized = img.resize_exact(target_width, target_height, image::imageops::FilterType::Triangle);
        let rgb_image = resized.to_rgb8();

        let mut array = Array::zeros(IxDyn(&[1, 3, target_height as usize, target_width as usize]));

        for (x, y, pixel) in rgb_image.enumerate_pixels() {
            array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }

        Ok(array)
    }

    fn postprocess_output(
        &self,
        output: &Array<f32, IxDyn>,
        _original_width: u32,
        _original_height: u32,
    ) -> Result<Vec<DetectionResult>> {
        let shape = output.shape();
        debug!("YOLO postprocess output shape: {:?}", shape);

        if shape.len() != 3 {
            return Err(eyre::eyre!("Unexpected YOLO output shape: {:?}", shape));
        }

        let num_detections = shape[2];
        let num_classes = shape[1] - 4;

        let mut raw_detections = Vec::new();

        for i in 0..num_detections {
            let cx = output[[0, 0, i]];
            let cy = output[[0, 1, i]];
            let w = output[[0, 2, i]];
            let h = output[[0, 3, i]];

            let mut max_score = 0.0f32;
            let mut max_class_id = 0usize;

            for class_id in 0..num_classes {
                let score = output[[0, 4 + class_id, i]];
                if score > max_score {
                    max_score = score;
                    max_class_id = class_id;
                }
            }

            if max_score >= self.confidence_threshold {
                let class_name = YOLO_CLASSES
                    .get(max_class_id)
                    .unwrap_or(&"unknown")
                    .to_string();

                if !self.target_classes.is_empty() && !self.target_classes.contains(&class_name) {
                    continue;
                }

                let x1 = (cx - w / 2.0) / self.input_size.0 as f32;
                let y1 = (cy - h / 2.0) / self.input_size.1 as f32;
                let x2 = (cx + w / 2.0) / self.input_size.0 as f32;
                let y2 = (cy + h / 2.0) / self.input_size.1 as f32;

                let bbox = BoundingBox::new(
                    x1.clamp(0.0, 1.0),
                    y1.clamp(0.0, 1.0),
                    x2.clamp(0.0, 1.0),
                    y2.clamp(0.0, 1.0),
                );

                raw_detections.push(DetectionResult::new(bbox, max_class_id, class_name, max_score));
            }
        }

        Ok(self.apply_nms(raw_detections))
    }

    fn apply_nms(&self, mut detections: Vec<DetectionResult>) -> Vec<DetectionResult> {
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = vec![true; detections.len()];

        for i in 0..detections.len() {
            if !keep[i] {
                continue;
            }
            for j in (i + 1)..detections.len() {
                if !keep[j] {
                    continue;
                }
                if detections[i].class_id == detections[j].class_id {
                    let iou = detections[i].bbox.iou(&detections[j].bbox);
                    if iou > self.nms_threshold {
                        keep[j] = false;
                    }
                }
            }
        }

        detections
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, det)| det)
            .collect()
    }
}
