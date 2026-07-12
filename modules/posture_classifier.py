import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
import os

logger = logging.getLogger(__name__)

class PostureClassifier:
    """Accurate posture classification using MediaPipe Tasks API"""
    
    def __init__(self):
        try:
            # Ensure model path is correct relative to the project root
            model_path = os.path.join('weights', 'pose_landmarker_lite.task')
            if not os.path.exists(model_path):
                logger.error(f"MediaPipe pose landmarker model not found at {model_path}")
                self.mp_available = False
                return
                
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False)
            self.detector = vision.PoseLandmarker.create_from_options(options)
            self.mp_available = True
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Tasks API: {e}")
            self.mp_available = False
        
    def classify_posture(self, image, bbox=None):
        """Classify posture from image region with improved accuracy"""
        if not getattr(self, 'mp_available', False):
            return "Standing", None
            
        try:
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                # Add padding for better pose detection
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                roi = image[int(y1):int(y2), int(x1):int(x2)]
            else:
                roi = image
                
            if roi.size == 0:
                return "Unknown", None
                
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_roi)
            detection_result = self.detector.detect(mp_image)
            
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                landmarks = detection_result.pose_landmarks[0]
                posture = self.analyze_pose_landmarks(landmarks)
                return posture, landmarks
            else:
                return "Unknown", None
                
        except Exception as e:
            logger.error(f"Error in posture classification: {e}")
            return "Unknown", None
    
    def analyze_pose_landmarks(self, landmarks):
        """Analyze pose landmarks with improved accuracy"""
        try:
            # Key landmarks indexing in MediaPipe Pose
            # 11: left_shoulder, 12: right_shoulder, 23: left_hip, 24: right_hip
            # 25: left_knee, 26: right_knee, 27: left_ankle, 28: right_ankle
            
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            # Calculate key points
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            knee_center_y = (left_knee.y + right_knee.y) / 2
            ankle_center_y = (left_ankle.y + right_ankle.y) / 2
            
            # Calculate torso angle
            torso_vertical = abs(shoulder_center_y - hip_center_y)
            
            # More accurate posture classification
            if torso_vertical < 0.15:  # Very horizontal torso
                return "Lying"
            elif knee_center_y > hip_center_y + 0.15 and ankle_center_y > knee_center_y:  # Knees bent, ankles below knees
                return "Sitting"
            elif ankle_center_y > knee_center_y and knee_center_y > hip_center_y:  # Standing position
                return "Standing"
            else:
                return "Unknown"
            
        except Exception as e:
            logger.error(f"Error analyzing pose landmarks: {e}")
            return "Unknown"