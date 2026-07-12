import cv2
import os
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class PPEDetector:
    """
    Real YOLOv8 PPE detection using pretrained weights.
    """
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.model_path = os.path.join('weights', 'ppe_yolov8.pt')
        try:
            self.model = YOLO(self.model_path)
            self.model_available = True
        except Exception as e:
            logger.error(f"Failed to load PPE model from {self.model_path}: {e}")
            self.model_available = False
            self.model = None
            
    def process_frame(self, frame):
        """
        Implementation of PPE detection using YOLO
        """
        alerts = []
        if not self.model_available:
            return alerts, frame
            
        try:
            results = self.model(frame, verbose=False)[0]
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                    
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id].lower()
                
                # Check for negative classes (no hard hat, no vest, etc.)
                if 'no' in cls_name and ('hat' in cls_name or 'vest' in cls_name):
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    
                    alerts.append({
                        'type': 'ppe_violation',
                        'class': cls_name,
                        'message': f'Worker detected: {cls_name}!',
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf
                    })
                    
                    # Draw alert bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{cls_name.upper()} ({conf:.2f})", (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif 'hat' in cls_name or 'vest' in cls_name:
                    # Draw positive (compliant) bounding box
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls_name.upper()}", (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                      
        except Exception as e:
            logger.error(f"Error in PPE detection: {e}")
            
        return alerts, frame
