import os
import cv2
import json
import time
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
from tqdm import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from huggingface_hub import login
from dotenv import load_dotenv

# Authenticate with Hugging Face
login(token=os.getenv("HUGGING_FACE_TOKEN"))

load_dotenv()

class WeaponDetector:
    def __init__(self):
        # Initialize OWLv2
        self.owlv2_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
        self.owlv2_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")
        
        # Move models to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.owlv2_model.to(self.device)
        
        # Cache for tracking objects between frames
        self.previous_detections = []
        self.tracking_threshold = 0.5  # IOU threshold for tracking

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        # Calculate intersection
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def extract_frames(self, video_path: str, fps: int = 1, max_size: int = 640) -> List[Dict]:
        """Extract frames from video at specified FPS with size limit for optimization."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)
        
        # Calculate resize ratio if needed
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resize_ratio = min(max_size / width, max_size / height, 1.0)
        new_width = int(width * resize_ratio)
        new_height = int(height * resize_ratio)
        
        frame_count = 0
        batch_frames = []
        batch_timestamps = []
        batch_size = 8  # Process frames in batches for better GPU utilization
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Resize frame if needed
                if resize_ratio < 1.0:
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = frame_count / video_fps
                
                batch_frames.append(Image.fromarray(frame_rgb))
                batch_timestamps.append(timestamp)
                
                # Process batch when full
                if len(batch_frames) >= batch_size:
                    frames.extend([
                        {"frame": frame, "timestamp": ts}
                        for frame, ts in zip(batch_frames, batch_timestamps)
                    ])
                    batch_frames = []
                    batch_timestamps = []
            
            frame_count += 1
        
        # Process remaining frames
        if batch_frames:
            frames.extend([
                {"frame": frame, "timestamp": ts}
                for frame, ts in zip(batch_frames, batch_timestamps)
            ])
        
        cap.release()
        return frames

    def detect_objects(self, image: Image.Image, text_queries: List[str]) -> List[Dict]:
        """Detect objects in image using OWLv2 with temporal consistency."""
        # Prepare inputs with more specific prompts
        detailed_queries = []
        for query in text_queries:
            detailed_queries.extend([
                f"dangerous {query} being used as a weapon",
                f"threatening {query} in attack position",
                f"visible {query} that poses immediate danger",
                f"{query} being brandished as a weapon"
            ])
        
        # Process inputs properly
        inputs = self.owlv2_processor(
            images=image,
            text=detailed_queries,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions with higher threshold for better precision
        with torch.no_grad():
            outputs = self.owlv2_model(**inputs)
        
        target_sizes = torch.Tensor([[image.size[1], image.size[0]]]).to(self.device)
        results = self.owlv2_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]
        
        # Process results with temporal consistency
        current_detections = []
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            # Get base label (remove the detailed prompt parts)
            base_label = text_queries[label_id // 4]
            
            # Check if this detection matches any previous detection
            matched = False
            if self.previous_detections:
                for prev_detection in self.previous_detections:
                    if prev_detection["label"] == base_label:
                        iou = self.calculate_iou(box.tolist(), prev_detection["box"])
                        if iou > self.tracking_threshold:
                            # Average the scores for temporal smoothing
                            score = (float(score) + prev_detection["score"]) / 2
                            matched = True
                            break
            
            # Add detection with confidence boost if matched
            confidence_boost = 1.2 if matched else 1.0
            detection = {
                "label": base_label,
                "score": float(score) * confidence_boost,
                "box": box.tolist()
            }
            current_detections.append(detection)
        
        # Update previous detections for next frame
        self.previous_detections = current_detections
        
        # Filter out lower confidence detections for same object
        filtered_detections = []
        seen_labels = set()
        for det in sorted(current_detections, key=lambda x: x["score"], reverse=True):
            if det["label"] not in seen_labels and det["score"] > 0.35:  # Higher threshold for final output
                filtered_detections.append(det)
                seen_labels.add(det["label"])
        
        return filtered_detections

    def analyze_video(self, video_path: str) -> Tuple[List[Dict], Dict, Dict]:
        """Analyze video for dangerous objects."""
        # Define dangerous objects to detect - focused on bladed weapons and sharp objects
        dangerous_objects = [
            # Armas brancas principais
            "knife", "machete", "sword", "dagger", "bayonet", "blade",
            "combat knife", "hunting knife", "military knife", "tactical knife",
            "kitchen knife", "butcher knife", "pocket knife", "utility knife",
            
            # Objetos cortantes
            "razor", "box cutter", "glass shard", "broken glass", "broken bottle",
            "scissors", "sharp metal", "sharp object", "blade weapon",
            "scalpel", "exacto knife", "craft knife", "paper cutter",
            
            # Objetos perfurantes
            "ice pick", "awl", "needle", "screwdriver", "metal spike",
            "sharp stick", "sharp pole", "pointed metal", "metal rod",
            
            # Ferramentas perigosas
            "saw blade", "circular saw", "chainsaw", "axe", "hatchet",
            "cleaver", "metal file", "chisel", "wire cutter",
            
            # Armas improvisadas
            "sharpened object", "improvised blade", "makeshift weapon",
            "concealed blade", "hidden blade", "modified tool"
        ]
        
        # Performance metrics
        metrics = {
            "start_time": time.time(),
            "frame_extraction_time": 0,
            "analysis_time": 0,
            "total_time": 0,
            "frames_analyzed": 0,
            "video_duration": 0
        }
        
        # Extract frames with higher FPS
        print("Extracting frames from video...")
        frame_extraction_start = time.time()
        frames = self.extract_frames(video_path, fps=2)  # Aumentado para 2 FPS
        metrics["frame_extraction_time"] = time.time() - frame_extraction_start
        metrics["frames_analyzed"] = len(frames)
        if frames:
            metrics["video_duration"] = frames[-1]["timestamp"]
        
        # Analyze frames
        print(f"\nAnalyzing {len(frames)} frames for dangerous objects...")
        analysis_start = time.time()
        
        detections = []
        timestamps = set()
        
        for frame_data in tqdm(frames, desc="Processing frames"):
            frame_detections = self.detect_objects(frame_data["frame"], dangerous_objects)
            
            for detection in frame_detections:
                timestamps.add(frame_data["timestamp"])
                detection_info = {
                    "type": detection["label"],
                    "timestamp": frame_data["timestamp"],
                    "confidence": f"{detection['score']*100:.1f}%",
                    "description": f"A {detection['label']} was detected in the scene",
                    "detection_score": detection["score"],
                    "box": detection["box"]
                }
                detections.append(detection_info)
        
        metrics["analysis_time"] = time.time() - analysis_start
        
        # Create time ranges
        timestamps_list = sorted(list(timestamps))
        ranges = []
        if timestamps_list:
            range_start = timestamps_list[0]
            prev_time = timestamps_list[0]
            
            for t in timestamps_list[1:]:
                if t - prev_time > 1.0:
                    ranges.append((range_start, prev_time))
                    range_start = t
                prev_time = t
            ranges.append((range_start, prev_time))
        
        # Create summary
        json_summary = {
            "risk_status": "Risk found!!" if detections else "No risk detected",
            "timestamps": list(timestamps),
            "time_ranges": [
                {
                    "start": start,
                    "end": end,
                    "duration": end - start
                }
                for start, end in ranges
            ]
        }
        
        metrics["total_time"] = time.time() - metrics["start_time"]
        
        return detections, json_summary, metrics

def main():
    try:
        # Initialize detector
        detector = WeaponDetector()
        
        # Video path
        video_path = "video.mp4"
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print("Starting video analysis...")
        results, json_summary, performance_metrics = detector.analyze_video(video_path)
        
        # Print results
        if not results:
            print("\nNo dangerous objects were detected in the video.")
        else:
            print(f"\nDetected {len(results)} dangerous objects:")
            current_timestamp = None
            
            for obj in results:
                if current_timestamp != obj['timestamp']:
                    current_timestamp = obj['timestamp']
                    print(f"\nAt {obj['timestamp']} seconds:")
                
                print(f"- Type: {obj['type']}")
                print(f"  Confidence: {obj['confidence']}")
                print(f"  Description: {obj['description']}")
        
        print("\nJSON Summary:")
        print(json.dumps(json_summary, indent=2))
        
        if json_summary["time_ranges"]:
            print("\nTime Ranges Analysis:")
            for range_info in json_summary["time_ranges"]:
                print(f"- Risk detected from {range_info['start']} to {range_info['end']} seconds (duration: {range_info['duration']} seconds)")
            
            total_duration = sum(r['duration'] for r in json_summary["time_ranges"])
            print(f"\nTotal time with risks detected: {total_duration} seconds")
        
        # Print performance metrics
        print("\nPerformance Metrics:")
        print(f"Video Duration: {performance_metrics['video_duration']:.2f} seconds")
        print(f"Frames Analyzed: {performance_metrics['frames_analyzed']}")
        print(f"Frame Extraction Time: {performance_metrics['frame_extraction_time']:.2f} seconds")
        print(f"Analysis Time: {performance_metrics['analysis_time']:.2f} seconds")
        print(f"Total Processing Time: {performance_metrics['total_time']:.2f} seconds")
        if results:
            print(f"Average Time per Detection: {performance_metrics['analysis_time']/len(results):.3f} seconds")
            print(f"Processing Speed: {performance_metrics['video_duration']/performance_metrics['total_time']:.2f}x realtime")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main() 