import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch
import threading
import time
from datetime import datetime
import logging

from .centroid_tracker import CentroidTracker
from .line_counter import LineCounter
from .posture_classifier import PostureClassifier
from .face_processor import FaceProcessor
from .zone_analyzer import ZoneAnalyzer
from .alert_system import AlertSystem

logger = logging.getLogger(__name__)

class Enhanced2A2SDetector:
    """Main enhanced surveillance system with accurate detection"""
    
    def __init__(self, cap):
        self.cap = cap
        
        # Core components
        self.yolo_model = None
        self.tracker = CentroidTracker(max_disappeared=50, max_distance=120)
        self.posture_classifier = PostureClassifier()
        self.face_processor = FaceProcessor()
        self.alert_system = AlertSystem()
        
        # Threading
        self.running = False
        self.detection_thread = None
        self._frame_lock = threading.Lock()
        
        # Frame management
        self.export_frame = None
        self.frame_width = 1280
        self.frame_height = 720
        
        # Zone analytics
        self.zone_analyzer = ZoneAnalyzer(self.frame_width, self.frame_height)
        
        # Line counter for entry/exit
        line_y = self.frame_height // 2
        self.line_counter = LineCounter((100, line_y), (self.frame_width - 100, line_y))
        
        # Parameters
        self.crowd_threshold = 5
        self.inactivity_threshold = 120  # 2 minutes
        self.blur_faces = True
        self.show_poses = True
        self.show_zones = False
        self.show_line_counter = True
        
        # Statistics (JSON serializable)
        self.stats = {
            'total_detections': 0,
            'current_people_count': 0,
            'entry_count': 0,
            'exit_count': 0,
            'crowd_alerts': 0,
            'inactivity_alerts': 0
        }
        
        # Initialize
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize YOLO model"""
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            if torch.cuda.is_available():
                self.yolo_model.to('cuda')
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
    
    def start_detection(self):
        """Start detection in separate thread"""
        if not self.running:
            self.running = True
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            logger.info("Enhanced 2A2S Detection started")
    
    def stop_detection(self):
        """Stop detection"""
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)
        logger.info("Detection stopped")
    
    def detection_loop(self):
        """Main detection loop with error handling"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("End of video reached, rewinding...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue                    
                    # consecutive_errors += 1
                    # if consecutive_errors > max_consecutive_errors:
                    #     logger.error("Too many consecutive frame read errors")
                    #     break
                    # time.sleep(0.1)
                    # continue
                
                consecutive_errors = 0  # Reset error counter
                
                # Resize frame
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Update export frame
                with self._frame_lock:
                    self.export_frame = processed_frame.copy()
                
                # Control frame rate
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors:
                    break
                # Create error frame
                error_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Detection Error: {str(e)[:50]}", 
                          (50, self.frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                with self._frame_lock:
                    self.export_frame = error_frame
                time.sleep(1)
    
    def process_frame(self, frame):
        """Process single frame with all detection and analysis"""
        timestamp = datetime.now()
        
        # 1. Person Detection
        person_boxes = []
        if self.yolo_model:
            try:
                results = self.yolo_model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                
                # Filter for persons (class 0) with confidence threshold
                person_mask = (detections.class_id == 0) & (detections.confidence > 0.5)
                person_detections = detections[person_mask]
                person_boxes = person_detections.xyxy.astype(int).tolist()
                
                self.stats['total_detections'] = len(person_boxes)
                
            except Exception as e:
                logger.error(f"Error in person detection: {e}")
        
        # 2. Update tracker
        objects_info = self.tracker.update(person_boxes)
        self.stats['current_people_count'] = len(objects_info)
        
        # 3. Process faces (blur for privacy)
        if self.blur_faces:
            frame, face_info = self.face_processor.process_faces(frame, blur_faces=True)
        
        # 4. Draw tracking information with bounding boxes and IDs
        for obj_id_str, info in objects_info.items():
            try:
                obj_id = int(obj_id_str)
                centroid = info['centroid']
                bbox = info['bbox']
                
                # Draw bounding box
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw centroid
                cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, (0, 255, 0), -1)
                
                # Draw ID label with background
                label = f"ID: {obj_id}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_width + 10, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Draw trail
                positions = info['positions_history']
                if len(positions) > 1:
                    for i in range(1, len(positions)):
                        pt1 = tuple(map(int, positions[i-1]))
                        pt2 = tuple(map(int, positions[i]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
                
                # 5. Pose estimation and fall detection
                if self.show_poses:
                    posture, pose_landmarks = self.posture_classifier.classify_posture(frame, bbox)
                    
                    # Draw posture label
                    posture_label = f"{posture}"
                    cv2.putText(frame, posture_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                   
                
                # Check for inactivity
                if info['is_stationary']:
                    last_seen = datetime.fromisoformat(info['last_seen'])
                    time_stationary = (timestamp - last_seen).total_seconds()
                    if time_stationary > self.inactivity_threshold:
                        cv2.putText(frame, "INACTIVE!", (int(centroid[0]) - 30, int(centroid[1]) + 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        if self.alert_system.trigger_inactivity_alert(obj_id, time_stationary/60, timestamp):
                            self.stats['inactivity_alerts'] += 1
                            
            except Exception as e:
                logger.error(f"Error processing object {obj_id_str}: {e}")
                continue
        
        # 6. Update line counter
        self.line_counter.update(objects_info)
        self.stats['entry_count'] = self.line_counter.entry_count
        self.stats['exit_count'] = self.line_counter.exit_count
        
        # Draw counting line
        if self.show_line_counter:
            cv2.line(frame, self.line_counter.line_start, self.line_counter.line_end, (255, 255, 0), 3)
            cv2.putText(frame, f"Entry: {self.line_counter.entry_count} | Exit: {self.line_counter.exit_count}", 
                      (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 7. Zone analysis and heatmap
        current_zones = self.zone_analyzer.update_zones(objects_info)
        if self.show_zones:
            frame = self.zone_analyzer.get_heatmap_overlay(frame)
        
        # 8. Crowd detection
        if len(objects_info) >= self.crowd_threshold:
            cv2.putText(frame, f"CROWD DETECTED ({len(objects_info)} people)!", 
                      (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if self.alert_system.trigger_crowd_alert(len(objects_info), timestamp):
                self.stats['crowd_alerts'] += 1
        
        # 9. Add comprehensive information overlay
        self.add_info_overlay(frame, timestamp)
        
        return frame
    
    def add_info_overlay(self, frame, timestamp):
        """Add comprehensive information overlay"""
        try:
            # Background for stats
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, frame.shape[0]-220), (450, frame.shape[0]-10), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Stats text
            y_offset = frame.shape[0] - 200
            cv2.putText(frame, f"Current People: {self.stats['current_people_count']}", 
                      (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Entries: {self.stats['entry_count']} | Exits: {self.stats['exit_count']}", 
                      (15, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Crowd Alerts: {self.stats['crowd_alerts']}", 
                      (15, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Inactivity Alerts: {self.stats['inactivity_alerts']}", 
                      (15, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Total Detections: {self.stats['total_detections']}", 
                      (15, y_offset + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Timestamp
            cv2.putText(frame, timestamp.strftime("%Y-%m-%d %H:%M:%S"), 
                      (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # System status
            status_color = (0, 255, 0) if self.running else (0, 0, 255)
            cv2.putText(frame, "SYSTEM: ACTIVE" if self.running else "SYSTEM: INACTIVE", 
                      (frame.shape[1] - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Recent alerts
            recent_alerts = self.alert_system.get_recent_alerts(3)
            for i, alert in enumerate(recent_alerts):
                alert_text = f"{alert['type'].upper()}: {alert['message'][:30]}..."
                cv2.putText(frame, alert_text, (15, y_offset + 150 + i*20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                          
        except Exception as e:
            logger.error(f"Error adding info overlay: {e}")
    
    def get_export_frame(self):
        """Get current frame for web interface"""
        with self._frame_lock:
            return self.export_frame.copy() if self.export_frame is not None else None
    
    def update_parameters(self, params):
        """Update system parameters"""
        try:
            if 'crowd_threshold' in params:
                self.crowd_threshold = max(1, min(50, int(params['crowd_threshold'])))
            if 'inactivity_threshold' in params:
                self.inactivity_threshold = max(30, min(3600, int(params['inactivity_threshold'])))
            if 'blur_faces' in params:
                self.blur_faces = bool(params['blur_faces'])
            if 'show_poses' in params:
                self.show_poses = bool(params['show_poses'])
            if 'show_zones' in params:
                self.show_zones = bool(params['show_zones'])
            if 'show_line_counter' in params:
                self.show_line_counter = bool(params['show_line_counter'])
            logger.info(f"Parameters updated: {params}")
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
    
    def get_analytics_data(self):
        """Get current analytics data for API (JSON serializable)"""
        try:
            return {
                'stats': self.stats.copy(),
                'current_objects': len(self.tracker.objects),
                'zones': dict(self.zone_analyzer.zone_counts),
                'recent_alerts': self.alert_system.get_recent_alerts(5),
                'timestamp': datetime.now().isoformat(),
                'system_status': 'active' if self.running else 'inactive'
            }
        except Exception as e:
            logger.error(f"Error getting analytics data: {e}")
            return {'error': str(e)}
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.stats = {
            'total_detections': 0,
            'current_people_count': 0,
            'entry_count': 0,
            'exit_count': 0,
            'crowd_alerts': 0,
            'inactivity_alerts': 0
        }
        self.line_counter.entry_count = 0
        self.line_counter.exit_count = 0
        self.line_counter.crossed_objects.clear()
        self.zone_analyzer.heatmap.fill(0)
        self.zone_analyzer.zone_counts.clear()
        self.alert_system.alert_logs.clear()
        logger.info("Statistics reset successfully")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_detection()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()