use dora_node_api::{
    dora_core::config::DataId,
    DoraNode,
    Event,
    arrow::array::{BinaryArray, UInt8Array},
};
use eyre::{Context, Result};
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::{Array, CowArray, IxDyn};
use ort::{
    Environment, ExecutionProvider, GraphOptimizationLevel, SessionBuilder, Value,
};
use robo_rover_lib::{init_tracing, types::{BoundingBox, DetectionFrame, DetectionResult}};
use std::env;
use tracing::{debug, error, info, warn};

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

struct YoloDetector {
    session: ort::Session,
    confidence_threshold: f32,
    nms_threshold: f32,
    target_classes: Vec<String>,
    input_size: (u32, u32),
    frame_counter: u64,
}

impl YoloDetector {
    fn new(model_path: &str, confidence_threshold: f32, nms_threshold: f32, target_classes: Vec<String>) -> Result<Self> {
        info!("Loading YOLO model from: {}", model_path);

        // Create ONNX Runtime environment with GPU support (fallback to CPU)
        let environment = Environment::builder()
            .with_name("yolo")
            .with_execution_providers([
                ExecutionProvider::CUDA(Default::default()),
                ExecutionProvider::TensorRT(Default::default()),
                ExecutionProvider::CPU(Default::default()),
            ])
            .build()?
            .into_arc();

        // Load session from file
        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(model_path)?;

        info!("Model loaded successfully");

        // YOLOv12 typically uses 640x640 input
        let input_size = (640, 640);

        // Warmup inference to initialize GPU/runtime
        info!("Warming up model...");
        let dummy_input: Array<f32, IxDyn> = Array::zeros(IxDyn(&[1, 3, input_size.1 as usize, input_size.0 as usize]));
        let input_cow = CowArray::from(&dummy_input);
        let input_tensor = Value::from_array(session.allocator(), &input_cow)?;
        let _ = session.run(vec![input_tensor])?;
        info!("Warmup complete - model ready");

        Ok(Self {
            session,
            confidence_threshold,
            nms_threshold,
            target_classes,
            input_size,
            frame_counter: 0,
        })
    }
    
    fn preprocess_image(&self, img: &DynamicImage) -> Result<Array<f32, IxDyn>> {
        let (target_width, target_height) = self.input_size;
        
        // Resize image
        let resized = img.resize_exact(
            target_width,
            target_height,
            image::imageops::FilterType::Triangle,
        );
        
        // Convert to RGB if needed
        let rgb_image = resized.to_rgb8();
        
        // Create ndarray in CHW format (Channels, Height, Width) normalized to [0, 1]
        let mut array = Array::zeros(IxDyn(&[1, 3, target_height as usize, target_width as usize]));
        
        for (x, y, pixel) in rgb_image.enumerate_pixels() {
            array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }
        
        Ok(array)
    }
    
    fn postprocess_output(&self, output: &Array<f32, IxDyn>, _original_width: u32, _original_height: u32) -> Result<Vec<DetectionResult>> {
        // YOLOv12 output format: [batch, num_features, num_detections]
        // num_features = 4 bbox coords (xywh) + num_classes (typically 80)
        let shape = output.shape();
        debug!("Output shape: {:?}", shape);

        if shape.len() != 3 {
            return Err(eyre::eyre!("Unexpected output shape: {:?}", shape));
        }

        let num_detections = shape[2];
        let num_classes = shape[1] - 4; // Subtract bbox coordinates
        
        let mut raw_detections = Vec::new();
        
        // Extract detections
        for i in 0..num_detections {
            // Get bbox coordinates (center x, center y, width, height)
            let cx = output[[0, 0, i]];
            let cy = output[[0, 1, i]];
            let w = output[[0, 2, i]];
            let h = output[[0, 3, i]];
            
            // Get class scores
            let mut max_score = 0.0f32;
            let mut max_class_id = 0usize;
            
            for class_id in 0..num_classes {
                let score = output[[0, 4 + class_id, i]];
                if score > max_score {
                    max_score = score;
                    max_class_id = class_id;
                }
            }
            
            // Filter by confidence threshold
            if max_score >= self.confidence_threshold {
                // Check if class is in target list
                let class_name = YOLO_CLASSES.get(max_class_id)
                    .unwrap_or(&"unknown")
                    .to_string();
                
                if !self.target_classes.is_empty() && !self.target_classes.contains(&class_name) {
                    continue;
                }
                
                // Convert from center format to corner format, normalized to [0, 1]
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
                
                raw_detections.push(DetectionResult::new(
                    bbox,
                    max_class_id,
                    class_name,
                    max_score,
                ));
            }
        }
        
        // Apply NMS (Non-Maximum Suppression)
        let detections = self.apply_nms(raw_detections);
        
        Ok(detections)
    }
    
    fn apply_nms(&self, mut detections: Vec<DetectionResult>) -> Vec<DetectionResult> {
        // Sort by confidence score (descending)
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
                
                // Only apply NMS for same class
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
    
    fn detect(&mut self, frame_data: &[u8], width: u32, height: u32) -> Result<DetectionFrame> {
        // Convert raw RGB8 data to image
        let img_buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, frame_data.to_vec())
            .ok_or_else(|| eyre::eyre!("Failed to create image buffer"))?;
        
        let img = DynamicImage::ImageRgb8(img_buffer);
        
        // Preprocess
        let input = self.preprocess_image(&img)?;
        
        // Run inference and extract output
        let output_array = {
            // Create input tensor from ndarray - convert to CowArray for ort API
            use ndarray::CowArray;
            let input_cow: CowArray<f32, _> = CowArray::from(&input);
            let input_tensor = Value::from_array(self.session.allocator(), &input_cow)?;

            // Run inference
            let outputs = self.session.run(vec![input_tensor])?;

            // Get output tensor and convert to ndarray
            let output_tensor = outputs[0].try_extract::<f32>()?;
            let output_view = output_tensor.view();
            debug!("Model output shape: {:?}", output_view.shape());

            // Convert to owned array
            output_view.to_owned().into_dimensionality::<IxDyn>()?
        };

        // Postprocess
        let detections = self.postprocess_output(&output_array, width, height)?;
        
        let frame_id = self.frame_counter;
        self.frame_counter += 1;
        
        let detection_frame = DetectionFrame::new(frame_id, width, height, detections);
        
        // info!(
        //     "Frame {}: Detected {} objects",
        //     frame_id,
        //     detection_frame.detections.len()
        // );
        
        Ok(detection_frame)
    }
}

fn main() -> Result<()> {
    let _guard = init_tracing();

    info!("Starting Object Detector node");
    
    // Read configuration from environment
    let model_path = env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/yolov12n.onnx".to_string());
    
    let confidence_threshold = env::var("CONFIDENCE_THRESHOLD")
        .unwrap_or_else(|_| "0.5".to_string())
        .parse::<f32>()
        .context("Invalid CONFIDENCE_THRESHOLD")?;
    
    let nms_threshold = env::var("NMS_THRESHOLD")
        .unwrap_or_else(|_| "0.4".to_string())
        .parse::<f32>()
        .context("Invalid NMS_THRESHOLD")?;
    
    let target_classes = env::var("TARGET_CLASSES")
        .unwrap_or_else(|_| "".to_string())
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();
    
    info!("Configuration:");
    info!("Model path: {}", model_path);
    info!("Confidence threshold: {}", confidence_threshold);
    info!("NMS threshold: {}", nms_threshold);
    info!("Target classes: {:?}", target_classes);
    
    // Initialize detector
    let mut detector = YoloDetector::new(
        &model_path,
        confidence_threshold,
        nms_threshold,
        target_classes,
    )?;
    
    // Initialize Dora node
    let (mut node, mut events) = DoraNode::init_from_env()?;
    info!("Dora node initialized");
    
    // Main event loop
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, metadata, .. } => {
                match id.as_str() {
                    "frame" => {
                        // Get width and height from metadata
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
                            warn!("Unsupported encoding: {}. Expected RGB8 or rgb8", encoding);
                            continue;
                        }

                        // Get frame data from UInt8Array
                        let frame_data = if let Some(array) = data.as_any().downcast_ref::<UInt8Array>() {
                            array.values().as_ref()
                        } else {
                            error!("Failed to cast data to UInt8Array");
                            continue;
                        };
                        
                        // Run detection
                        match detector.detect(frame_data, width, height) {
                            Ok(detection_frame) => {
                                // Serialize and send detections
                                let json = serde_json::to_vec(&detection_frame)?;
                                let arrow_data = BinaryArray::from_vec(vec![json.as_slice()]);
                                node.send_output(
                                    DataId::from("detections".to_owned()),
                                    Default::default(),
                                    arrow_data,
                                )?;

                                debug!("Sent {} detections", detection_frame.detections.len());
                            }
                            Err(e) => {
                                error!("Detection failed: {:?}", e);
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
    
    info!("Object detector node shutting down");
    Ok(())
}
