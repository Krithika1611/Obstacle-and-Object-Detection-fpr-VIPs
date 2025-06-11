import cv2
import numpy as np
from gtts import gTTS
import os
import time
import torch
import pygame
import streamlit as st
import tempfile
import uuid
from ultralytics import YOLO

# Initialize pygame for audio playback
pygame.init()

# App state management
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_speak_time' not in st.session_state:
    st.session_state.last_speak_time = time.time()
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'midas' not in st.session_state:
    st.session_state.midas = None
if 'transform' not in st.session_state:
    st.session_state.transform = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'yolov11x_model' not in st.session_state:
    st.session_state.yolov11x_model = None
if 'classes' not in st.session_state:
    st.session_state.classes = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "YOLOv11x"

# Streamlit page configuration
st.set_page_config(page_title="Advanced Object Detection with Depth Estimation", layout="wide")
st.title("Advanced Object Detection with Depth Estimation")
st.subheader("Using YOLOv11x")

# Fixed parameters - no user controls
confidence_threshold = 0.5
speech_interval = 5
show_depth_map = False
yolov11x_model_path = "yolo11x.pt"  # Path to YOLOv11x weights
coco_names_path = "coco.names"

# Create circular buttons for start and stop
st.markdown("""
<style>
div.stButton > button {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    font-size: 18px;
    font-weight: bold;
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# Create two buttons in the center
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Create two buttons side by side in the center column
    start_col, stop_col = st.columns(2)
    with start_col:
        start_button = st.button("Start Detection")
    with stop_col:
        stop_button = st.button("Stop Detection")

# Status indicator
status_text = st.empty()

# Single main view for combined results
main_view = st.empty()

def load_models():
    """Load all required models"""
    status_text.text("Loading models, please wait...")
    
    # Load MiDaS model
    try:
        midas_model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()
        
        # Load appropriate transforms for the model
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform
        
        st.success(f"MiDaS model loaded successfully on {device}")
    except Exception as e:
        st.error(f"Error loading MiDaS model: {e}")
        return None, None, None, None, None
    
    # Load YOLOv11x model
    try:
        yolov11x_model = YOLO(yolov11x_model_path)  # Load using YOLO class
        st.success("YOLOv11x model loaded successfully via YOLO class")
    except Exception as e2:
        st.error(f"Error loading YOLOv11x model: {e2}")
        yolov11x_model = None
    
    
    # Load COCO class labels
    try:
        with open(coco_names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        st.success(f"Loaded {len(classes)} class labels")
    except Exception as e:
        st.error(f"Error loading class labels: {e}")
        classes = []
    
    status_text.text("Models loaded successfully!")
    return midas, transform, device, yolov11x_model, classes

def process_depth(frame, midas, transform, device):
    """Process frame with MiDaS to get depth estimation"""
    # Transform input for MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert to numpy and normalize
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    
    return depth_map

def detect_with_yolov11x(frame, yolov11x_model, classes, depth_map, confidence_threshold=0.5):
    """Detect objects using YOLOv11x"""
    try:
        # Convert frame to RGB for YOLOv11x
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = yolov11x_model(rgb_frame)
        
        # Create a copy of the frame for drawing
        frame_with_objects = frame.copy()
        h, w, _ = frame.shape
        
        detected_objects = []  # To collect objects for speech
        
        # Process results based on format (handle different output formats)
        if hasattr(results, 'pandas'):  # Ultralytics YOLO format
            # Convert to pandas and get the detections
            result_pandas = results.pandas().xyxy[0]
            
            for _, detection in result_pandas.iterrows():
                x_min, y_min, x_max, y_max = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
                score = detection['confidence']
                class_id = detection['class']
                
                # Filter by confidence
                if score < confidence_threshold:
                    continue
                
                # Convert to integers
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                
                # Process detection (same as below)
                process_detection(frame_with_objects, depth_map, x_min, y_min, x_max, y_max, 
                                 score, class_id, classes, detected_objects)
                
        elif isinstance(results, list):  # List format
            # Process each detection in results list
            for result in results:
                # Check if result has boxes attribute (newer YOLO versions)
                if hasattr(result, 'boxes'):
                    boxes = result.boxes
                    for box in boxes:
                        # Get box coordinates
                        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                        score = box.conf.item()
                        class_id = int(box.cls.item())
                        
                        # Filter by confidence
                        if score < confidence_threshold:
                            continue
                            
                        # Convert to integers
                        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                        
                        # Process detection (same as below)
                        process_detection(frame_with_objects, depth_map, x_min, y_min, x_max, y_max, 
                                         score, class_id, classes, detected_objects)
                
                # Alternative format for direct box detections
                elif hasattr(result, 'xywh') or hasattr(result, 'xyxy'):
                    detections = result.xyxy[0].cpu().numpy() if hasattr(result, 'xyxy') else result.xywh[0].cpu().numpy()
                    
                    for detection in detections:
                        if len(detection) >= 6:  # Ensure we have enough values
                            x_min, y_min, x_max, y_max, score, class_id = detection[:6]
                            
                            # Filter by confidence
                            if score < confidence_threshold:
                                continue
                                
                            # Convert to integers (and handle xywh format if needed)
                            if hasattr(result, 'xywh'):
                                # Convert xywh to xyxy
                                x_center, y_center, width, height = x_min, y_min, x_max, y_max
                                x_min = int(x_center - width/2)
                                y_min = int(y_center - height/2)
                                x_max = int(x_center + width/2)
                                y_max = int(y_center + height/2)
                            else:
                                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                            
                            # Process detection (same as below)
                            process_detection(frame_with_objects, depth_map, x_min, y_min, x_max, y_max, 
                                             score, class_id, classes, detected_objects)
        
        # Fallback processing for raw numpy array output
        elif hasattr(results, 'pred') or hasattr(results, 'prediction'):
            detections = results.pred[0].cpu().numpy() if hasattr(results, 'pred') else results.prediction[0].cpu().numpy()
            
            for detection in detections:
                if len(detection) >= 6:  # Standard format: x1, y1, x2, y2, conf, cls
                    x_min, y_min, x_max, y_max, score, class_id = detection[:6]
                    
                    # Filter by confidence
                    if score < confidence_threshold:
                        continue
                        
                    # Convert to integers
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    
                    # Process detection (extracted for reuse)
                    process_detection(frame_with_objects, depth_map, x_min, y_min, x_max, y_max, 
                                     score, class_id, classes, detected_objects)
        
        return frame_with_objects, detected_objects
    except Exception as e:
        st.error(f"Error in YOLOv11x detection: {e}")
        return frame.copy(), []

def process_detection(frame, depth_map, x_min, y_min, x_max, y_max, score, class_id, classes, detected_objects):
    """Helper function to process a single detection"""
    h, w, _ = frame.shape
    
    # Ensure coordinates are within frame bounds
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    
    # Get average depth in detection region
    if x_min < x_max and y_min < y_max:
        object_region = depth_map[y_min:y_max, x_min:x_max]
        if object_region.size > 0:
            # Apply depth equation
            raw_depth = np.mean(object_region)
            avg_dist = -4.8163 * raw_depth + 2.7610 
            if avg_dist<0:
                avg_dist=avg_dist+1.2

            # Get class name
            class_id = int(class_id)
            if 0 <= class_id < len(classes):
                label = classes[class_id]
            else:
                label = f"Unknown ({class_id})"
            
            # Draw bounding box with BLUE color for YOLOv11x
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, f"{label}: {avg_dist:.2f}m", 
                      (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Add to detected objects
            detected_objects.append((label, avg_dist, (x_min, y_min, x_max, y_max), score))

def speak(text):
    """Generate and play speech using pygame with robust error handling"""
    try:
        # Initialize pygame mixer if not already initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        # Create a temporary directory with a unique name to avoid conflicts
        temp_dir = os.path.join(tempfile.gettempdir(), f"speech_app_{uuid.uuid4().hex[:8]}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Use a unique filename each time to avoid file locks
        output_file = os.path.join(temp_dir, f"speech_{uuid.uuid4().hex[:8]}.mp3")
        
        # Generate speech
        tts = gTTS(text=text, lang='en')
        tts.save(output_file)
        
        # Play the audio file
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        
        # Print as fallback
        st.info(f"SPEAKING: {text}")
        
        # Optional: Clean up old files
        try:
            for file in os.listdir(temp_dir):
                if file != os.path.basename(output_file):  # Don't delete the file we're using
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except:
                        pass
        except:
            pass
            
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        st.info(f"SPEECH MESSAGE: {text}")

def run_detection():
    """Main function to run the detection loop"""
    # Use the pre-loaded models from session state
    midas = st.session_state.midas
    transform = st.session_state.transform
    device = st.session_state.device
    yolov11x_model = st.session_state.yolov11x_model
    classes = st.session_state.classes
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_text.error("Error: Could not open camera")
        st.session_state.running = False
        return
    
    status_text.text("Detection is running... Press 'Stop Detection' to end.")
    
    # Main detection loop - runs in the main thread
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            status_text.error("Error: Failed to grab frame")
            break
        
        # Get depth estimation using MiDaS
        try:
            depth_map = process_depth(frame, midas, transform, device)
            depth_map_display = (depth_map * 255).astype(np.uint8)  # Convert for display
            depth_colored = cv2.applyColorMap(depth_map_display, cv2.COLORMAP_JET)
        except Exception as e:
            status_text.error(f"Error processing depth: {e}")
            continue
        
        # Perform object detection with YOLOv11x
        if yolov11x_model is not None:
            try:
                frame_with_objects, detected_objects = detect_with_yolov11x(
                    frame, yolov11x_model, classes, depth_map, confidence_threshold)
                main_view.image(cv2.cvtColor(frame_with_objects, cv2.COLOR_BGR2RGB), channels="RGB")
            except Exception as e:
                st.error(f"Error in YOLOv11x detection: {e}")
                detected_objects = []
                main_view.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        else:
            # No YOLOv11x model
            main_view.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            detected_objects = []
        
        # Voice announcement (only for the closest object, every X seconds)
        current_time = time.time()
        if detected_objects and current_time - st.session_state.last_speak_time > speech_interval:
            # Sort by distance and announce closest object
            closest_object = min(detected_objects, key=lambda x: x[1])
            speak(f"{closest_object[0]} detected at {closest_object[1]:.2f} meters")
            st.session_state.last_speak_time = current_time
        
        # Short sleep to reduce CPU usage and allow Streamlit to update
        time.sleep(0.05)
    
    # Cleanup
    cap.release()
    status_text.text("Detection stopped.")

# Handle button clicks
if start_button:
    if not st.session_state.models_loaded:
        # Load models in the main thread
        midas, transform, device, yolov11x_model, classes = load_models()
        if midas is not None:
            st.session_state.midas = midas
            st.session_state.transform = transform
            st.session_state.device = device
            st.session_state.yolov11x_model = yolov11x_model
            st.session_state.classes = classes
            st.session_state.models_loaded = True
        else:
            st.error("Failed to load models. Please check paths and try again.")
            st.stop()
    
    # Start detection
    st.session_state.running = True
    # Manually trigger the run_detection function
    if st.session_state.models_loaded:
        run_detection()

if stop_button:
    st.session_state.running = False
    status_text.text("Detection stopped")

# Run detection if it's active (this is needed when page refreshes)
if st.session_state.running and st.session_state.models_loaded:
    run_detection()

# Footer
st.markdown("---")
st.markdown("© 2025 Advanced Object Detection with Depth Estimation - YOLOv11x")