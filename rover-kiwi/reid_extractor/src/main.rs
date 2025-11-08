use dora_node_api::{
    dora_core::config::DataId,
    DoraNode,
    Event,
    arrow::array::{BinaryArray, UInt8Array},
};
use eyre::{Context, Result};
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use ndarray::{Array, IxDyn};
use ort::{
    Environment, ExecutionProvider, GraphOptimizationLevel, SessionBuilder, Value,
};
use robo_rover_lib::{init_tracing, types::DetectionFrame};
use std::env;
use tracing::{debug, error, info, warn};

const REID_INPUT_HEIGHT: u32 = 256;
const REID_INPUT_WIDTH: u32 = 128;

struct ReidExtractor {
    session: ort::Session,
    feature_dim: usize,
    min_bbox_size: u32,
    current_frame: Option<(Vec<u8>, u32, u32)>,  // (data, width, height)
}

impl ReidExtractor {
    fn new(model_path: &str, min_bbox_size: u32) -> Result<Self> {
        info!("Loading ReID model from: {}", model_path);

        // Create ONNX Runtime environment
        let environment = Environment::builder()
            .with_name("reid")
            .with_execution_providers([ExecutionProvider::CPU(Default::default())])
            .build()?
            .into_arc();

        // Load session from file
        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(2)?
            .with_model_from_file(model_path)?;

        info!("ReID model loaded successfully");

        // OSNet x0.25 outputs 512-dimensional features
        let feature_dim = 512;

        Ok(Self {
            session,
            feature_dim,
            min_bbox_size,
            current_frame: None,
        })
    }

    fn preprocess_crop(&self, crop: &DynamicImage) -> Result<Array<f32, IxDyn>> {
        // Resize to ReID model input size (256x128)
        let resized = crop.resize_exact(
            REID_INPUT_WIDTH,
            REID_INPUT_HEIGHT,
            image::imageops::FilterType::Triangle,
        );

        // Convert to RGB if needed
        let rgb_image = resized.to_rgb8();

        // ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        // Create ndarray in CHW format (Channels, Height, Width)
        let mut array = Array::zeros(IxDyn(&[1, 3, REID_INPUT_HEIGHT as usize, REID_INPUT_WIDTH as usize]));

        for (x, y, pixel) in rgb_image.enumerate_pixels() {
            for c in 0..3 {
                let normalized = ((pixel[c] as f32 / 255.0) - mean[c]) / std[c];
                array[[0, c, y as usize, x as usize]] = normalized;
            }
        }

        Ok(array)
    }

    fn extract_features(&mut self, crop: &DynamicImage) -> Result<Vec<f32>> {
        // Preprocess
        let input = self.preprocess_crop(crop)?;

        // Run inference and extract features
        let features = {
            use ndarray::CowArray;
            let input_cow: CowArray<f32, _> = CowArray::from(&input);
            let input_tensor = Value::from_array(self.session.allocator(), &input_cow)?;

            // Run inference
            let outputs = self.session.run(vec![input_tensor])?;

            // Get output tensor and convert to ndarray
            let output_tensor = outputs[0].try_extract::<f32>()?;
            let output_view = output_tensor.view();

            // Convert to Vec<f32>
            let features: Vec<f32> = output_view.iter().copied().collect();

            if features.len() != self.feature_dim {
                return Err(eyre::eyre!(
                    "Unexpected feature dimension: got {}, expected {}",
                    features.len(),
                    self.feature_dim
                ));
            }

            features
        };

        Ok(features)
    }

    fn crop_detection(&self, frame_data: &[u8], width: u32, height: u32,
                      x1: f32, y1: f32, x2: f32, y2: f32) -> Result<DynamicImage> {
        // Convert normalized coordinates to pixels
        let px1 = (x1 * width as f32).max(0.0) as u32;
        let py1 = (y1 * height as f32).max(0.0) as u32;
        let px2 = (x2 * width as f32).min(width as f32) as u32;
        let py2 = (y2 * height as f32).min(height as f32) as u32;

        // Validate bbox dimensions
        let bbox_width = px2.saturating_sub(px1);
        let bbox_height = py2.saturating_sub(py1);

        if bbox_width < self.min_bbox_size || bbox_height < self.min_bbox_size {
            return Err(eyre::eyre!(
                "Bounding box too small: {}x{} (min: {})",
                bbox_width, bbox_height, self.min_bbox_size
            ));
        }

        // Create image from frame data
        let img_buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, frame_data.to_vec())
            .ok_or_else(|| eyre::eyre!("Failed to create image buffer"))?;

        // Crop the region
        let mut crop_buffer = RgbImage::new(bbox_width, bbox_height);
        for y in 0..bbox_height {
            for x in 0..bbox_width {
                let src_x = px1 + x;
                let src_y = py1 + y;
                if src_x < width && src_y < height {
                    let pixel = img_buffer.get_pixel(src_x, src_y);
                    crop_buffer.put_pixel(x, y, *pixel);
                }
            }
        }

        Ok(DynamicImage::ImageRgb8(crop_buffer))
    }

    fn process_detections(&mut self, mut detection_frame: DetectionFrame) -> Result<DetectionFrame> {
        // Check if we have a current frame and clone the data we need
        let frame_info = match &self.current_frame {
            Some((data, w, h)) => Some((data.clone(), *w, *h)),
            None => {
                warn!("No frame available for ReID extraction");
                return Ok(detection_frame);
            }
        };

        let (frame_data, width, height) = match frame_info {
            Some(info) => info,
            None => return Ok(detection_frame),
        };

        debug!("Extracting ReID features for {} detections", detection_frame.detections.len());

        // Process each detection
        for detection in &mut detection_frame.detections {
            // Crop detection region
            match self.crop_detection(
                &frame_data,
                width,
                height,
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.x2,
                detection.bbox.y2,
            ) {
                Ok(crop) => {
                    // Extract features
                    match self.extract_features(&crop) {
                        Ok(features) => {
                            detection.reid_features = Some(features);
                            debug!(
                                "Extracted ReID features for {} ({}x{} crop)",
                                detection.class_name,
                                crop.width(),
                                crop.height()
                            );
                        }
                        Err(e) => {
                            warn!("Failed to extract ReID features for {}: {}", detection.class_name, e);
                        }
                    }
                }
                Err(e) => {
                    debug!("Skipping ReID for {}: {}", detection.class_name, e);
                }
            }
        }

        Ok(detection_frame)
    }
}

fn main() -> Result<()> {
    let _guard = init_tracing();

    info!("Starting reid_extractor node");

    // Read configuration from environment
    let model_path = env::var("REID_MODEL_PATH")
        .unwrap_or_else(|_| format!("{}/.cache/reid/osnet_x0_25.onnx", env::var("HOME").unwrap()));

    let min_bbox_size = env::var("MIN_BBOX_SIZE")
        .unwrap_or_else(|_| "32".to_string())
        .parse::<u32>()
        .context("Invalid MIN_BBOX_SIZE")?;

    info!("Configuration:");
    info!("Model path: {}", model_path);
    info!("Min bbox size: {}x{}", min_bbox_size, min_bbox_size);

    // Initialize ReID extractor
    let mut extractor = ReidExtractor::new(&model_path, min_bbox_size)?;

    // Initialize Dora node
    let (mut node, mut events) = DoraNode::init_from_env()?;
    info!("Dora node initialized");

    // Main event loop
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, metadata, .. } => {
                match id.as_str() {
                    "frame" => {
                        // Store the current frame for later use
                        let width = metadata.parameters.get("width")
                            .and_then(|v| match v {
                                dora_node_api::Parameter::Integer(i) => Some(*i as u32),
                                _ => None,
                            })
                            .unwrap_or(640);

                        let height = metadata.parameters.get("height")
                            .and_then(|v| match v {
                                dora_node_api::Parameter::Integer(i) => Some(*i as u32),
                                _ => None,
                            })
                            .unwrap_or(480);

                        let encoding = metadata.parameters.get("encoding")
                            .and_then(|v| match v {
                                dora_node_api::Parameter::String(s) => Some(s.as_str()),
                                _ => None,
                            })
                            .unwrap_or("rgb8");

                        if encoding.to_lowercase() != "rgb8" {
                            warn!("Unsupported encoding: {}. Expected RGB8", encoding);
                            continue;
                        }

                        let frame_data = if let Some(array) = data.as_any().downcast_ref::<UInt8Array>() {
                            array.values().as_ref().to_vec()
                        } else {
                            error!("Failed to cast frame data to UInt8Array");
                            continue;
                        };

                        extractor.current_frame = Some((frame_data, width, height));
                        debug!("Received frame: {}x{}", width, height);
                    }
                    "detections" => {
                        // Parse detection frame
                        let detection_data = if let Some(array) = data.as_any().downcast_ref::<BinaryArray>() {
                            array.value(0)
                        } else {
                            error!("Failed to cast detection data to BinaryArray");
                            continue;
                        };

                        let detection_frame: DetectionFrame = match serde_json::from_slice(detection_data) {
                            Ok(frame) => frame,
                            Err(e) => {
                                error!("Failed to deserialize detection frame: {}", e);
                                continue;
                            }
                        };

                        // Process detections and extract ReID features
                        match extractor.process_detections(detection_frame) {
                            Ok(enriched_frame) => {
                                let features_count = enriched_frame.detections.iter()
                                    .filter(|d| d.reid_features.is_some())
                                    .count();

                                debug!("Enriched {} detections with ReID features", features_count);

                                // Serialize and send enriched detections
                                let json = serde_json::to_vec(&enriched_frame)?;
                                let arrow_data = BinaryArray::from_vec(vec![json.as_slice()]);
                                node.send_output(
                                    DataId::from("detections_with_reid".to_owned()),
                                    Default::default(),
                                    arrow_data,
                                )?;
                            }
                            Err(e) => {
                                error!("Failed to process detections: {:?}", e);
                            }
                        }
                    }
                    other => {
                        warn!("Received unexpected input: {}", other);
                    }
                }
            }
            Event::InputClosed { id } => {
                info!("Input {} closed", id);
                break;
            }
            Event::Stop(_) => {
                info!("Received stop signal");
                break;
            }
            other => {
                debug!("Received other event: {:?}", other);
            }
        }
    }

    info!("ReID extractor node shutting down");
    Ok(())
}
