
import cv2
import supervision as sv
from ultralytics import YOLO
from datetime import datetime, date, timedelta, time
import survilleance.Email_Alert as Email_Alert
import torch
import threading
import numpy as np
import os
import json
from collections import defaultdict, deque
import math
import pywhatkit as pwk
from modules.ppe_detector import PPEDetector
import time
from threading import Lock, Thread
import datetime as dt
class Detector_2A2S:
    
    def __init__(self, cap, output_video_path=None):
        # Define the VideoCapture object
        self.cap = cap
        self.output_video_path = output_video_path
        self.video_writer = None
        
        if self.output_video_path and self.cap.isOpened():
            width = 640
            height = 480
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if fps == 0: fps = 30
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        
        # Thread control
        self.running = False
        self.detection_thread = None
        
        # PARAMETERS: bg subtraction
        self.bg_sub_threshold = 30
        self.hist = 150
        self.bg_subtract = cv2.createBackgroundSubtractorMOG2(
            history=self.hist, 
            varThreshold=self.bg_sub_threshold, 
            detectShadows=True
        )

        # PARAMETERS: frame diff
        self.frame_diff_threshold = 100

        # PARAMETERS: contours
        self.min_contour_size = 500

        # PARAMETERS: log files
        self.motion_logs_path = "./logs/motion_log.txt"
        self.motion_frames_path = "./logs/motion_frames/"
        self.object_logs_path = "./logs/object_log.txt"
        self.object_frames_path = "./logs/object_frames/"
        self.analytics_logs_path = "./logs/analytics_log.json"
        self.alerts_logs_path = "./logs/alerts_log.json"
        self.last_log_time = datetime.now() - timedelta(minutes=2)

        # PARAMETERS: object detection
        self.objectDetectionIsON = False
        self.obj_scan_duration = 70
        self.obj_scan_time_start = datetime.now()
        self.model_path = "yolov8n.pt"

        # PARAMETERS: Alert system
        self.alert_time_start = "20:00"
        self.alert_time_end = "19:00"
        self.isSendingAlerts = False
        self.user_alerter = Email_Alert.email_alert_system()

        # PARAMETERS: export to GUI/Web
        self.export_frame = None
        self._frame_lock = threading.Lock()
        
        # Frame storage for comparison
        self.prev_frame = None
        self.reference_frame = None
        
        # Advanced Analytics Parameters - FIXED: All default to False
        self.shadow_detection_enabled = False
        self.abandoned_object_detection_enabled = False
        self.danger_zone_detection_enabled = False
        self.suspicious_object_detection_enabled = False
        self.fall_detection_enabled = False
        self.machinery_proximity_enabled = True
        self.proximity_threshold_pixels = 150
        try:
            self.ppe_detector = PPEDetector()
        except:
            self.ppe_detector = None
        
        # Shadow detection parameters
        self.shadow_threshold = 50
        self.light_intensity_threshold = 20
        
        # FIXED: Abandoned object detection - Reduced to 20 seconds
        self.abandoned_objects = {}
        self.abandoned_time_threshold = 20  # Reduced to 20 seconds
        self.static_object_threshold = 50  # Increased for better stability
        
        # FIXED: Danger Zone Intrusion detection with better parameters
        self.person_tracks = {}
        self.danger_zone_time_threshold = 1.0
        self.danger_zone_area_threshold = 80
        
        # FIXED: Suspicious objects with proper tracking
        self.suspicious_classes = [24, 26, 27, 28]  # backpack, handbag, tie, suitcase
        self.suspicious_objects = {}
        self.suspicious_time_threshold = 15  # 15 seconds for suspicious objects
        
        # FIXED: Fall detection with improved parameters
        self.person_positions = {}
        self.fall_aspect_ratio_threshold = 1.5  # Better threshold
        self.fall_height_change_threshold = 30  # Height change threshold
        
        # FIXED: Analytics data storage with proper initialization
        self.analytics_data = {
            'motion_events': deque(maxlen=1000),
            'object_detections': deque(maxlen=1000),
            'shadow_detections': deque(maxlen=1000),
            'abandoned_objects': deque(maxlen=100),
            'zone_intrusion_events': deque(maxlen=100),
            'suspicious_objects': deque(maxlen=100),
            'fall_detections': deque(maxlen=100),
            'alerts_sent': deque(maxlen=500)
        }
        
        # Frame counter for tracking
        self.frame_count = 0
        self.person_id_counter = 0

        # Initialize object detection attributes
        self.initialise_object_detector()

        # List of objects that can be detected by yolov8 (COCO dataset)
        self.object_classes_id = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
            49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }
        
        # Ensure log directories exist
        self.ensure_log_directories()

    def ensure_log_directories(self):
        """Ensure all log directories exist"""
        directories = [
            os.path.dirname(self.motion_logs_path),
            self.motion_frames_path,
            self.object_frames_path,
            os.path.dirname(self.analytics_logs_path),
            os.path.dirname(self.alerts_logs_path)
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
        # PARAMETERS: WhatsApp Alert System - ROBUST IMPLEMENTATION
        self.whatsapp_enabled = True
        self.whatsapp_number = "+919967533958"  # Target WhatsApp number
        self.whatsapp_lock = Lock()  # Thread safety for WhatsApp sending
        self.last_whatsapp_alert = {}  # Track last alert times to prevent spam
        self.whatsapp_cooldown = 30  # 5 minutes cooldown between same alert types

        # WhatsApp alert triggers - ONLY for specified conditions
        self.whatsapp_triggers = {
            'abandoned_object': True,
            'fall_detection': True, 
            'shadow_detection': True,
            'motion': False,  # Disabled as per requirement
            'danger_zone_intrusion': False,  # Disabled as per requirement
            'suspicious_object': False  # Disabled as per requirement
        }
    def gpu_check(self):
        """
        Checks if a GPU is available. 
        Return: Boolean
        """
        if torch.cuda.is_available():
            print("CUDA (GPU) is available. Version: ", torch.version.cuda)
            return True
        else:
            print("CUDA (GPU) is not available. Using CPU.")
            return False

    def initialise_object_detector(self):
        """
        1. Loads object detection model (.pt file)
        2. Employs GPU if available
        3. Setup box annotator for object detections
        """
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully: {self.model_path}")
        except Exception as err:
            print(f"[!] Error occurred while loading model: {err}")
            self.model = None
            return
        
        # Use GPU if available
        if self.gpu_check():
            print("Switching to GPU...")
            try:
                self.model.to('cuda')
            except Exception as e:
                print(f"Failed to move model to GPU: {e}")
        
        # Setup supervision BoxAnnotator
        self.boxAnnotator = sv.BoxAnnotator(
            thickness=2
        )


    def check_isSendingAlerts(self, now=None):
        """
        Checks if self.isSendingAlerts should be True or False depending on the current time.
        """
        if now is None:
            now = datetime.now().time()
            
        try:
            # Parse time strings
            start_hour, start_min = map(int, self.alert_time_start.split(":"))
            end_hour, end_min = map(int, self.alert_time_end.split(":"))

            start = dt.time(hour=start_hour, minute=start_min, second=0)
            end = dt.time(hour=end_hour, minute=end_min, second=0)

            if start < end:
                return start <= now <= end
            else:
                return now >= start or now <= end
        except Exception as e:
            print(f"Error checking alert times: {e}")
            return False
    
    def send_whatsapp_alert(self, alert_type, alert_data):
        """
        ROBUST WhatsApp Alert System using pywhatkit
        Only triggers for: abandoned_object, fall_detection, shadow_detection
        """
        if not self.whatsapp_enabled:
            return
            
        # Check if this alert type should trigger WhatsApp
        if not self.whatsapp_triggers.get(alert_type, False):
            return
            
        try:
            with self.whatsapp_lock:  # Thread safety
                current_time = datetime.now()
                
                # Cooldown check to prevent spam
                last_alert_key = f"{alert_type}_whatsapp"
                if last_alert_key in self.last_whatsapp_alert:
                    time_diff = (current_time - self.last_whatsapp_alert[last_alert_key]).total_seconds()
                    if time_diff < self.whatsapp_cooldown:
                        print(f"WhatsApp alert cooldown active for {alert_type}. Skipping...")
                        return
                
                # Update last alert time
                self.last_whatsapp_alert[last_alert_key] = current_time
                
                # Create robust alert message based on type
                message = self._create_whatsapp_message(alert_type, alert_data, current_time)
                
                # Send WhatsApp message with error handling
                print(f"🔔 Attempting to send WhatsApp alert for {alert_type}...")
                
                # Get current time for immediate sending
                now = datetime.now()
                hour = now.hour
                minute = now.minute + 1  # Send 1 minute from now
                
                # Handle minute overflow
                if minute >= 60:
                    minute = 0
                    hour += 1
                    if hour >= 24:
                        hour = 0
                
                # Send WhatsApp message
                pwk.sendwhatmsg(
                    phone_no=self.whatsapp_number,
                    message=message,
                    time_hour=hour,
                    time_min=minute,
                    wait_time=15,  # Wait 15 seconds for WhatsApp to load
                    tab_close=True,  # Auto close tab after sending
                    close_time=3  # Close after 3 seconds
                )
                
                print(f"✅ WhatsApp alert sent successfully for {alert_type}")
                
                # Log WhatsApp alert
                whatsapp_log = {
                    'timestamp': current_time.isoformat(),
                    'type': f'{alert_type}_whatsapp',
                    'phone_number': self.whatsapp_number,
                    'message': message,
                    'status': 'sent'
                }
                self.log_analytics_event('whatsapp_alerts', whatsapp_log)
                
        except Exception as e:
            print(f"❌ Error sending WhatsApp alert for {alert_type}: {e}")
            # Log failed WhatsApp attempt
            try:
                error_log = {
                    'timestamp': datetime.now().isoformat(),
                    'type': f'{alert_type}_whatsapp_failed',
                    'phone_number': self.whatsapp_number,
                    'error': str(e),
                    'status': 'failed'
                }
                self.log_analytics_event('whatsapp_alerts', error_log)
            except:
                pass  # Don't let logging errors break the system

    def _create_whatsapp_message(self, alert_type, alert_data, timestamp):
        """
        Create robust, formatted WhatsApp messages for different alert types
        """
        base_header = "🛡️ *Safeguard Security Alert* 🚨\n"
        base_footer = f"\n📅 Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n🏢 Location: Security Camera System"
        
        if alert_type == 'abandoned_object':
            obj_class = alert_data.get('class', 'Unknown Object')
            duration = alert_data.get('duration', 0)
            center = alert_data.get('center', [0, 0])
            
            message = f"""{base_header}
⚠️ *ABANDONED OBJECT DETECTED* ⚠️

📦 Object Type: {obj_class}
⏱️ Duration: {duration:.1f} seconds
📍 Location: X:{center[0]:.0f}, Y:{center[1]:.0f}

🔍 An object has been left unattended for an extended period. Immediate attention required.
{base_footer}"""

        elif alert_type == 'fall_detection':
            center = alert_data.get('center', [0, 0])
            reason = alert_data.get('reason', 'unknown')
            aspect_ratio = alert_data.get('aspect_ratio', 0)
            
            message = f"""{base_header}
🚑 *FALL FROM HEIGHT ALERT* 🚑

👤 Worker Fall Detected
📍 Location: X:{center[0]:.0f}, Y:{center[1]:.0f}
🔍 Detection Method: {reason.replace('_', ' ').title()}
📊 Aspect Ratio: {aspect_ratio:.2f}

🆘 URGENT: A worker may have fallen from height (possible scaffolding/ladder incident) and requires immediate medical attention!
{base_footer}"""

        elif alert_type == 'shadow_detection':
            shadow_pixels = alert_data.get('data', {}).get('shadow_pixels', 0)
            light_change = alert_data.get('data', {}).get('light_change_detected', False)
            mean_change = alert_data.get('data', {}).get('mean_light_change', 0)
            
            message = f"""{base_header}
🌒 *SHADOW ANALYSIS ALERT* 🌒

👥 Suspicious Shadow Movement Detected
📊 Shadow Pixels: {shadow_pixels}
💡 Light Change: {'Yes' if light_change else 'No'}
📈 Intensity Change: {mean_change:.1f}

🔍 Unusual shadow patterns detected. Possible unauthorized movement in monitored area.
{base_footer}"""

        else:
            # Fallback message for any unexpected alert type
            message = f"""{base_header}
🔔 *SECURITY ALERT* 🔔

Alert Type: {alert_type.replace('_', ' ').title()}
Data: {str(alert_data)[:100]}...

🔍 Security system has detected an event requiring attention.
{base_footer}"""
        
        return message

  
    def get_export_frame(self):
        """
        Retrieves latest frame to use in GUI/Web interface
        """
        with self._frame_lock:
            return self.export_frame.copy() if self.export_frame is not None else None

    def update_parameters(self, params):
        """
        Update detection parameters from web interface
        """
        try:
            if 'min_contour_size' in params:
                self.min_contour_size = max(10, min(10000, int(params['min_contour_size'])))
                print(f"Updated min_contour_size: {self.min_contour_size}")
                
            if 'obj_scan_duration' in params:
                self.obj_scan_duration = max(30, min(300, int(params['obj_scan_duration'])))
                print(f"Updated obj_scan_duration: {self.obj_scan_duration}")
                
            if 'frame_diff_threshold' in params:
                self.frame_diff_threshold = max(10, min(1000, int(params['frame_diff_threshold'])))
                print(f"Updated frame_diff_threshold: {self.frame_diff_threshold}")
                
            if 'hist' in params:
                self.hist = max(5, min(1000, int(params['hist'])))
                self.bg_subtract = cv2.createBackgroundSubtractorMOG2(
                    history=self.hist, 
                    varThreshold=self.bg_sub_threshold, 
                    detectShadows=True
                )
                print(f"Updated background subtraction history: {self.hist}")
                
            if 'objectDetectionIsON' in params:
                self.objectDetectionIsON = bool(params['objectDetectionIsON'])
                print(f"Updated objectDetectionIsON: {self.objectDetectionIsON}")
                
            # Advanced analytics toggles
            if 'shadow_detection_enabled' in params:
                self.shadow_detection_enabled = bool(params['shadow_detection_enabled'])
                print(f"Updated shadow_detection_enabled: {self.shadow_detection_enabled}")
                
            if 'abandoned_object_detection_enabled' in params:
                self.abandoned_object_detection_enabled = bool(params['abandoned_object_detection_enabled'])
                print(f"Updated abandoned_object_detection_enabled: {self.abandoned_object_detection_enabled}")
                
            if 'danger_zone_detection_enabled' in params:
                self.danger_zone_detection_enabled = bool(params['danger_zone_detection_enabled'])
                print(f"Updated danger_zone_detection_enabled: {self.danger_zone_detection_enabled}")
                
            if 'suspicious_object_detection_enabled' in params:
                self.suspicious_object_detection_enabled = bool(params['suspicious_object_detection_enabled'])
                print(f"Updated suspicious_object_detection_enabled: {self.suspicious_object_detection_enabled}")
                
            if 'fall_detection_enabled' in params:
                self.fall_detection_enabled = bool(params['fall_detection_enabled'])
                print(f"Updated fall_detection_enabled: {self.fall_detection_enabled}")
                
        except Exception as e:
            print(f"Error updating parameters: {e}")

    def detect_shadows_and_light_changes(self, frame, prev_frame):
        """
        FIXED: Detect shadow movements and light intensity changes
        """
        if prev_frame is None or not self.shadow_detection_enabled:
            return None, False
            
        try:
            # Convert to grayscale
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate difference
            diff = cv2.absdiff(gray_current, gray_prev)
            
            # Detect shadows (darker regions in current frame with significant change)
            shadow_mask = np.logical_and(
                gray_current < gray_prev - self.shadow_threshold,
                diff > 30
            )
            
            # Detect significant light changes
            mean_diff = np.mean(diff)
            light_change_detected = mean_diff > self.light_intensity_threshold
            
            # Log shadow detection
            shadow_pixels = int(np.sum(shadow_mask))
            if shadow_pixels > 500:
                shadow_event = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'shadow_detection',
                    'data': {
                        'shadow_pixels': shadow_pixels,
                        'mean_light_change': float(mean_diff),
                        'light_change_detected': light_change_detected
                    }
                }
                self.log_analytics_event('shadow_detections', shadow_event)
                # Send WhatsApp alert for significant shadow detection - NEW ADDITION
            if self.isSendingAlerts and shadow_pixels > 1000:  # Only for significant shadows
                self.send_whatsapp_alert('shadow_detection', shadow_event)
            return shadow_mask.astype(np.uint8) * 255, light_change_detected
            
        except Exception as e:
            print(f"Error in shadow detection: {e}")
            return None, False

    def detect_abandoned_objects(self, detections, frame):
        """
        FIXED: Detect objects that have been stationary for too long - 20 seconds threshold
        """
        if not self.abandoned_object_detection_enabled:
            return []
            
        if not hasattr(detections, 'xyxy') or len(detections.xyxy) == 0:
            return []
            
        current_time = datetime.now()
        abandoned_alerts = []
        
        try:
            # Update existing objects and add new ones
            current_objects = {}
            
            for i, bbox in enumerate(detections.xyxy):
                class_id = detections.class_id[i] if i < len(detections.class_id) else 0
                
                # Only track suspicious objects
                if class_id in self.suspicious_classes:
                    obj_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    obj_key = f"{class_id}_{int(obj_center[0]//30)}_{int(obj_center[1]//30)}"
                    
                    # Check if this is close to any existing object
                    matched_key = None
                    for existing_key, existing_obj in self.abandoned_objects.items():
                        existing_center = existing_obj.get('center', (0, 0))
                        distance = math.sqrt(
                            (obj_center[0] - existing_center[0])**2 + 
                            (obj_center[1] - existing_center[1])**2
                        )
                        if distance < self.static_object_threshold:
                            matched_key = existing_key
                            break
                    
                    if matched_key:
                        # Update existing object
                        self.abandoned_objects[matched_key].update({
                            'bbox': bbox,
                            'center': obj_center,
                            'last_seen': current_time
                        })
                        current_objects[matched_key] = self.abandoned_objects[matched_key]
                        
                        # Check if object has been abandoned (20 seconds)
                        time_stationary = (current_time - self.abandoned_objects[matched_key]['first_seen']).total_seconds()
                        if time_stationary > self.abandoned_time_threshold:
                            alert_data = {
                                'type': 'abandoned_object',
                                'class': self.object_classes_id.get(class_id, 'Unknown'),
                                'bbox': [float(x) for x in bbox],
                                'center': [float(obj_center[0]), float(obj_center[1])],
                                'duration': float(time_stationary),
                                'timestamp': current_time.isoformat()
                            }
                            abandoned_alerts.append(alert_data)
                            print(f"Abandoned object detected: {alert_data['class']} for {time_stationary:.1f}s")
                    else:
                        # New object
                        current_objects[obj_key] = {
                            'bbox': bbox,
                            'center': obj_center,
                            'class_id': class_id,
                            'first_seen': current_time,
                            'last_seen': current_time
                        }
            
            # Update abandoned objects (remove old ones)
            self.abandoned_objects = {k: v for k, v in current_objects.items() 
                                    if (current_time - v['last_seen']).total_seconds() < 60}
            
            # Log abandoned object alerts
            # Log abandoned object alerts
            for alert in abandoned_alerts:
                self.log_analytics_event('abandoned_objects', alert)
                if self.isSendingAlerts:
                    # Send email alert
                    self.send_alert('abandoned_object', alert)
                    # Send WhatsApp alert - NEW ADDITION
                    self.send_whatsapp_alert('abandoned_object', alert)
                    
        except Exception as e:
            print(f"Error in abandoned object detection: {e}")
            
        return abandoned_alerts

    
    def detect_machinery_proximity(self, detections, frame):
        if not self.machinery_proximity_enabled or not hasattr(detections, 'xyxy') or len(detections.xyxy) == 0:
            return []
            
        alerts = []
        persons = []
        machinery = []
        
        # 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
        for i, bbox in enumerate(detections.xyxy):
            class_id = detections.class_id[i] if i < len(detections.class_id) else 0
            obj_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            if class_id == 0:
                persons.append(obj_center)
            elif class_id in [2, 5, 7]:  # car, bus, truck (machinery proxy)
                machinery.append(obj_center)
                
        for p in persons:
            for m in machinery:
                dist = math.sqrt((p[0] - m[0])**2 + (p[1] - m[1])**2)
                if dist < self.proximity_threshold_pixels:
                    alert = {
                        'type': 'machinery_proximity',
                        'message': 'Worker too close to heavy machinery!',
                        'timestamp': datetime.now().isoformat(),
                        'distance': float(dist)
                    }
                    alerts.append(alert)
                    cv2.putText(frame, "PROXIMITY ALERT!", (int(p[0]), int(p[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.line(frame, (int(p[0]), int(p[1])), (int(m[0]), int(m[1])), (0,0,255), 2)
                    
        return alerts

    def detect_danger_zone_intrusion(self, detections, frame):
        """
        Detect people in danger zone.
        Dwell time is computed from self.frame_count / fps — never wall-clock.
        Alert fires only after MIN_BREACH_FRAMES consecutive in-zone frames (FP guard).
        """
        if not self.danger_zone_detection_enabled:
            return []
            
        if not hasattr(detections, 'xyxy') or len(detections.xyxy) == 0:
            return []

        # Initialise frame-counter-based state dicts on first call
        if not hasattr(self, '_zone_entry_frame'):
            self._zone_entry_frame = {}    # person_id -> frame_count at zone entry
            self._zone_consec      = {}    # person_id -> consecutive in-zone frames
            self._zone_fps         = 30.0  # default; overwritten when cap fps is known
            try:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps and fps > 0:
                    self._zone_fps = fps
            except Exception:
                pass

        MIN_BREACH_FRAMES = 8  # configurable: consec frames before alert fires

        current_time = datetime.now()
        zone_intrusion_alerts = []
        
        try:
            # Collect person detections
            people_detections = []
            for i, bbox in enumerate(detections.xyxy):
                class_id = detections.class_id[i] if i < len(detections.class_id) else 0
                if class_id == 0:  # Person
                    people_detections.append((bbox, i))
            
            current_people = {}
            active_zone_ids = set()
            
            for bbox, detection_idx in people_detections:
                center_x = float((bbox[0] + bbox[2]) / 2)
                center_y = float((bbox[1] + bbox[3]) / 2)
                
                # Find matching existing person by centroid distance
                matched_id = None
                min_distance = float('inf')
                
                for person_id, person_data in self.person_tracks.items():
                    if len(person_data['positions']) > 0:
                        last_pos = person_data['positions'][-1]
                        distance = math.sqrt(
                            (center_x - last_pos[0])**2 + 
                            (center_y - last_pos[1])**2
                        )
                        if distance < min_distance and distance < 100:
                            min_distance = distance
                            matched_id = person_id
                
                if matched_id:
                    self.person_tracks[matched_id]['positions'].append((center_x, center_y))
                    self.person_tracks[matched_id]['last_seen'] = current_time
                    current_people[matched_id] = self.person_tracks[matched_id]
                    
                    # ── Dwell time: video-time only (frame count / fps) ──
                    if matched_id not in self._zone_entry_frame:
                        self._zone_entry_frame[matched_id] = self.frame_count
                        self._zone_consec[matched_id] = 1
                    else:
                        self._zone_consec[matched_id] = self._zone_consec.get(matched_id, 0) + 1

                    active_zone_ids.add(matched_id)
                    entry_frame = self._zone_entry_frame[matched_id]
                    dwell_video_sec = (self.frame_count - entry_frame) / self._zone_fps
                    consec = self._zone_consec[matched_id]

                    positions = list(self.person_tracks[matched_id]['positions'])
                    
                    # Alert requires: video dwell >= threshold AND consec >= MIN_BREACH_FRAMES
                    if (dwell_video_sec > self.danger_zone_time_threshold
                            and consec >= MIN_BREACH_FRAMES
                            and len(positions) > 5):
                        xs = [pos[0] for pos in positions]
                        ys = [pos[1] for pos in positions]
                        area_covered = (max(xs) - min(xs)) * (max(ys) - min(ys))
                        
                        if area_covered < self.danger_zone_area_threshold:
                            alert_data = {
                                'type': 'danger_zone_intrusion',
                                'person_id': matched_id,
                                'duration_video_sec': float(dwell_video_sec),
                                'consec_frames': consec,
                                'area_covered': float(area_covered),
                                'center': [center_x, center_y],
                                'timestamp': current_time.isoformat()
                            }
                            zone_intrusion_alerts.append(alert_data)
                            print(f"Danger Zone Intrusion: {matched_id} "
                                  f"dwell={dwell_video_sec:.2f}s video-time "
                                  f"({consec} consec frames)")
                else:
                    # New person
                    self.person_id_counter += 1
                    person_id = f"person_{self.person_id_counter}"
                    current_people[person_id] = {
                        'positions': deque([(center_x, center_y)], maxlen=100),
                        'first_seen': current_time,
                        'last_seen': current_time
                    }
                    self._zone_entry_frame[person_id] = self.frame_count
                    self._zone_consec[person_id] = 1
                    active_zone_ids.add(person_id)
            
            # Preserve tracks seen < 1 second ago (1s * fps frames tolerance)
            frames_tolerance = int(1.0 * self._zone_fps)
            for k, v in self.person_tracks.items():
                if k not in current_people:
                    frames_since = self.frame_count - self._zone_entry_frame.get(k, self.frame_count)
                    if frames_since <= frames_tolerance:
                        current_people[k] = v
                        active_zone_ids.add(k)
            
            # Reset consec counter for people who left zone this frame
            for tid in list(self._zone_consec.keys()):
                if tid not in active_zone_ids:
                    self._zone_consec.pop(tid, None)
                    self._zone_entry_frame.pop(tid, None)

            # Prune stale tracks (missing > 4 seconds video-time)
            stale_frames = int(4.0 * self._zone_fps)
            self.person_tracks = {
                k: v for k, v in current_people.items()
                if (self.frame_count - self._zone_entry_frame.get(k, self.frame_count)) < stale_frames
            }
            
            # Log alerts
            for alert in zone_intrusion_alerts:
                self.log_analytics_event('zone_intrusion_events', alert)
                if self.isSendingAlerts:
                    self.send_alert('danger_zone_intrusion', alert)
                    
        except Exception as e:
            print(f"Error in danger_zone_intrusion detection: {e}")
            
        return zone_intrusion_alerts


    def detect_suspicious_objects(self, detections, frame):
        """
        FIXED: Detect suspicious objects like unattended bags with proper time tracking
        """
        if not self.suspicious_object_detection_enabled:
            return []
            
        if not hasattr(detections, 'xyxy') or len(detections.xyxy) == 0:
            return []
            
        current_time = datetime.now()
        suspicious_alerts = []
        
        try:
            current_suspicious = {}
            
            # Look for suspicious objects
            for i, bbox in enumerate(detections.xyxy):
                class_id = detections.class_id[i] if i < len(detections.class_id) else 0
                
                if class_id in self.suspicious_classes:
                    obj_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    obj_key = f"sus_{class_id}_{int(obj_center[0]//30)}_{int(obj_center[1]//30)}"
                    
                    # Check if there's a person nearby
                    person_nearby = False
                    for j, person_bbox in enumerate(detections.xyxy):
                        person_class_id = detections.class_id[j] if j < len(detections.class_id) else 0
                        if person_class_id == 0:  # Person
                            person_center = ((person_bbox[0] + person_bbox[2]) / 2, 
                                           (person_bbox[1] + person_bbox[3]) / 2)
                            distance = math.sqrt(
                                (obj_center[0] - person_center[0])**2 + 
                                (obj_center[1] - person_center[1])**2
                            )
                            if distance < 150:  # 150 pixel threshold
                                person_nearby = True
                                break
                    
                    # Track suspicious object
                    if obj_key in self.suspicious_objects:
                        # Update existing suspicious object
                        self.suspicious_objects[obj_key].update({
                            'bbox': bbox,
                            'center': obj_center,
                            'last_seen': current_time,
                            'person_nearby': person_nearby
                        })
                        current_suspicious[obj_key] = self.suspicious_objects[obj_key]
                        
                        # Check if object has been unattended for suspicious time
                        time_unattended = (current_time - self.suspicious_objects[obj_key]['first_seen']).total_seconds()
                        if not person_nearby and time_unattended > self.suspicious_time_threshold:
                            alert_data = {
                                'type': 'suspicious_object',
                                'class': self.object_classes_id.get(class_id, 'Unknown'),
                                'bbox': [float(x) for x in bbox],
                                'center': [float(obj_center[0]), float(obj_center[1])],
                                'duration': float(time_unattended),
                                'timestamp': current_time.isoformat()
                            }
                            suspicious_alerts.append(alert_data)
                            print(f"Suspicious object detected: {alert_data['class']} unattended for {time_unattended:.1f}s")
                    else:
                        # New suspicious object
                        current_suspicious[obj_key] = {
                            'bbox': bbox,
                            'center': obj_center,
                            'class_id': class_id,
                            'first_seen': current_time,
                            'last_seen': current_time,
                            'person_nearby': person_nearby
                        }
            
            # Update suspicious objects (remove old ones)
            self.suspicious_objects = {k: v for k, v in current_suspicious.items() 
                                     if (current_time - v['last_seen']).total_seconds() < 60}
            
            # Log suspicious object alerts
            for alert in suspicious_alerts:
                self.log_analytics_event('suspicious_objects', alert)
                if self.isSendingAlerts:
                    self.send_alert('suspicious_object', alert)
                    
        except Exception as e:
            print(f"Error in suspicious object detection: {e}")
            
        return suspicious_alerts

    def detect_falls(self, detections, frame):
        """
        FIXED: Improved fall detection with better parameters and tracking
        """
        if not self.fall_detection_enabled:
            return []
            
        if not hasattr(detections, 'xyxy') or len(detections.xyxy) == 0:
            return []
            
        current_time = datetime.now()
        fall_alerts = []
        
        try:
            current_positions = {}
            
            # Analyze person detections for fall patterns
            for i, bbox in enumerate(detections.xyxy):
                class_id = detections.class_id[i] if i < len(detections.class_id) else 0
                
                if class_id == 0:  # Person
                    width = float(bbox[2] - bbox[0])
                    height = float(bbox[3] - bbox[1])
                    aspect_ratio = width / height if height > 0 else 0
                    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    
                    person_key = f"person_{int(center[0]//50)}_{int(center[1]//50)}"
                    
                    # Check for fall indicators
                    fall_detected = False
                    
                    # Method 1: High aspect ratio (person lying down)
                    if aspect_ratio > self.fall_aspect_ratio_threshold:
                        fall_detected = True
                        fall_reason = "high_aspect_ratio"
                    
                    # Method 2: Sudden height change from previous position
                    if person_key in self.person_positions:
                        prev_pos = self.person_positions[person_key]
                        prev_height = prev_pos.get('height', height)
                        height_change = prev_height - height
                        
                        if height_change > self.fall_height_change_threshold:
                            fall_detected = True
                            fall_reason = "sudden_height_change"
                    
                    # Method 3: Low center of mass (bottom of bounding box high up in frame)
                    frame_height = frame.shape[0]
                    relative_bottom = bbox[3] / frame_height
                    if relative_bottom < 0.3 and aspect_ratio > 1.2:  # Person in upper part of frame lying down
                        fall_detected = True
                        fall_reason = "low_center_mass"
                    
                    # Update person position
                    current_positions[person_key] = {
                        'center': center,
                        'aspect_ratio': aspect_ratio,
                        'height': height,
                        'bbox': bbox,
                        'timestamp': current_time
                    }
                    
                    # Generate fall alert if detected
                    if fall_detected:
                        alert_data = {
                            'type': 'fall_detection',
                            'bbox': [float(x) for x in bbox],
                            'aspect_ratio': float(aspect_ratio),
                            'center': [float(center[0]), float(center[1])],
                            'reason': fall_reason,
                            'timestamp': current_time.isoformat()
                        }
                        fall_alerts.append(alert_data)
                        print(f"Fall detected at {center} - Reason: {fall_reason}, Aspect ratio: {aspect_ratio:.2f}")
            
            # Update person positions
            self.person_positions.update(current_positions)
            
            # Clean old person positions
            self.person_positions = {k: v for k, v in self.person_positions.items() 
                                   if (current_time - v['timestamp']).total_seconds() < 30}
            
            # Log fall alerts
            # Log fall alerts
            for alert in fall_alerts:
                self.log_analytics_event('fall_detections', alert)
                if self.isSendingAlerts:
                    # Send WhatsApp alert - NEW ADDITION
                    self.send_whatsapp_alert('fall_detection', alert)
                    
        except Exception as e:
            print(f"Error in fall detection: {e}")
            
        return fall_alerts

    def log_analytics_event(self, event_type, data):
        """
        FIXED: Log analytics events with proper error handling
        """
        try:
            # Add to in-memory storage
            if event_type in self.analytics_data:
                self.analytics_data[event_type].append(data)
            
            # Write to file with robust error handling
            try:
                # Ensure the file exists and is valid JSON
                if os.path.exists(self.analytics_logs_path):
                    try:
                        with open(self.analytics_logs_path, 'r') as f:
                            content = f.read().strip()
                            if content:
                                logs = json.loads(content)
                                if not isinstance(logs, list):
                                    logs = []
                            else:
                                logs = []
                    except (json.JSONDecodeError, IOError):
                        logs = []
                else:
                    logs = []
                
                logs.append(data)
                
                # Keep only last 1000 events
                if len(logs) > 1000:
                    logs = logs[-1000:]
                
                with open(self.analytics_logs_path, 'w') as f:
                    json.dump(logs, f, indent=2, default=str)
                    
            except Exception as file_error:
                print(f"Error writing to analytics log file: {file_error}")
                
        except Exception as e:
            print(f"Error logging analytics event: {e}")

    def send_alert(self, alert_type, alert_data):
        """
        FIXED: Send alert with proper error handling
        """
        try:
            # Send email alert
            self.user_alerter.send_alert_cli(alert_type, datetime.now(), alert_data)
            
            # Log alert
            alert_log = {
                'timestamp': datetime.now().isoformat(),
                'type': alert_type,
                'data': alert_data,
                'status': 'sent'
            }
            
            self.analytics_data['alerts_sent'].append(alert_log)
            
            # Write to alerts log file
            try:
                if os.path.exists(self.alerts_logs_path):
                    try:
                        with open(self.alerts_logs_path, 'r') as f:
                            content = f.read().strip()
                            if content:
                                alerts = json.loads(content)
                                if not isinstance(alerts, list):
                                    alerts = []
                            else:
                                alerts = []
                    except (json.JSONDecodeError, IOError):
                        alerts = []
                else:
                    alerts = []
                
                alerts.append(alert_log)
                
                # Keep only last 500 alerts
                if len(alerts) > 500:
                    alerts = alerts[-500:]
                
                with open(self.alerts_logs_path, 'w') as f:
                    json.dump(alerts, f, indent=2, default=str)
                    
            except Exception as file_error:
                print(f"Error writing to alerts log file: {file_error}")
                
        except Exception as e:
            pass
    
    def get_analytics_data(self):
        """
        FIXED: Get analytics data with robust error handling
        """
        try:
            # Convert deques to lists for JSON serialization
            analytics_dict = {}
            for key, value in self.analytics_data.items():
                if value:
                    analytics_dict[key] = [dict(item) if isinstance(item, dict) else item for item in list(value)]
                else:
                    analytics_dict[key] = []
            
            return {
                'analytics': analytics_dict,
                'current_stats': {
                    'abandoned_objects_count': len(self.abandoned_objects),
                    'person_tracks_count': len(self.person_tracks),
                    'suspicious_objects_count': len(self.suspicious_objects),
                    'frame_count': self.frame_count,
                    'detection_enabled': {
                        'shadow': self.shadow_detection_enabled,
                        'abandoned_object': self.abandoned_object_detection_enabled,
                        'danger_zone_intrusion': self.danger_zone_detection_enabled,
                        'suspicious_object': self.suspicious_object_detection_enabled,
                        'fall': self.fall_detection_enabled
                    }
                }
            }
        except Exception as e:
            print(f"Error getting analytics data: {e}")
            return {
                'analytics': {
                    'motion_events': [],
                    'object_detections': [],
                    'shadow_detections': [],
                    'abandoned_objects': [],
                    'zone_intrusion_events': [],
                    'suspicious_objects': [],
                    'fall_detections': [],
                    'alerts_sent': []
                },
                'current_stats': {
                    'abandoned_objects_count': 0,
                    'person_tracks_count': 0,
                    'suspicious_objects_count': 0,
                    'frame_count': 0,
                    'detection_enabled': {
                        'shadow': False,
                        'abandoned_object': False,
                        'danger_zone_intrusion': False,
                        'suspicious_object': False,
                        'fall': False
                    }
                }
            }

    def start_detection(self):
        """Start the detection loop in a separate thread"""
        if not self.running:
            self.running = True
            self.detection_thread = threading.Thread(target=self.motion_object_scanner)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            print("Detection started with enhanced analytics")

    def stop_detection(self):
        """Stop the detection loop"""
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)
        print("Detection stopped")

    def motion_object_scanner(self):
        """FIXED: Main detection loop with robust error handling"""
        
        frame_buffer = deque(maxlen=3)
        
        while self.running:
            try:
                self.frame_count += 1
                
                # Check current time to send alerts or not
                self.isSendingAlerts = self.check_isSendingAlerts()

                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("[!] Failed to read frame from camera or video ended")
                    if self.output_video_path:
                        self.running = False
                        break
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Camera Error", (200, 240), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    with self._frame_lock:
                        self.export_frame = frame
                    continue

                # Resize frame for consistent processing
                frame = cv2.resize(frame, (640, 480))
                frame_buffer.append(frame.copy())
                
                # Convert frame to grayscale
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Initialize reference frame for shadow detection
                if self.reference_frame is None:
                    self.reference_frame = frame.copy()

                # Shadow detection
                shadow_mask = None
                light_change_detected = False
                if len(frame_buffer) >= 2:
                    shadow_mask, light_change_detected = self.detect_shadows_and_light_changes(
                        frame, frame_buffer[-2]
                    )

                # Initialize frame difference
                if self.prev_frame is not None:
                    frame_diff = cv2.absdiff(self.prev_frame, frame_gray)
                else:
                    frame_diff = np.zeros_like(frame_gray)
                
                self.prev_frame = frame_gray.copy()

                # Background subtraction
                foreground = self.bg_subtract.apply(frame_gray)

                # Combine bg subtraction with frame diff
                combined_frame = cv2.bitwise_and(frame_diff, foreground)
                _, combined_frame = cv2.threshold(combined_frame, self.frame_diff_threshold, 255, cv2.THRESH_BINARY)

                # Find contours for motion detection
                contours_array, _ = cv2.findContours(combined_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Reset detection flags
                motion_detected = False
                objects_detected = []
                all_alerts = []

                # OBJECT DETECTION AND ADVANCED ANALYTICS
                if self.objectDetectionIsON and self.model is not None:
                    # Check scan duration
                    scan_obj_elapsed_time = datetime.now() - self.obj_scan_time_start
                    remaining_time = max(self.obj_scan_duration - scan_obj_elapsed_time.total_seconds(), 0)
                    
                    # Show enabled analytics in the frame
                    enabled_analytics = []
                    if self.shadow_detection_enabled:
                        enabled_analytics.append("Shadow")
                    if self.abandoned_object_detection_enabled:
                        enabled_analytics.append("Abandoned")
                    if self.danger_zone_detection_enabled:
                        enabled_analytics.append("Danger Zone Intrusion")
                    if self.suspicious_object_detection_enabled:
                        enabled_analytics.append("Suspicious")
                    if self.fall_detection_enabled:
                        enabled_analytics.append("Fall")
                    
                    analytics_text = f"Analytics: {', '.join(enabled_analytics) if enabled_analytics else 'None'}"
                    cv2.putText(frame, f"[Advanced Object Scanning ({remaining_time:.0f}s)]", 
                              (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, analytics_text, 
                              (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    if scan_obj_elapsed_time > timedelta(seconds=self.obj_scan_duration):
                        self.objectDetectionIsON = False
                        print("Switching to MOTION detection")
                    
                    try:
                        # Run YOLO detection
                        results = self.model(frame, verbose=False, conf=0.3)[0]
                        detections = sv.Detections.from_ultralytics(results)

                        if len(detections.class_id) > 0:
                            objects_detected = [self.object_classes_id.get(i, f"Unknown_{i}") for i in detections.class_id]
                            
                            # Log object detection event
                            object_event = {
                                'timestamp': datetime.now().isoformat(),
                                'type': 'object_detections',
                                'data': {
                                    'objects': objects_detected,
                                    'count': len(objects_detected),
                                    'frame_count': self.frame_count
                                }
                            }
                            self.log_analytics_event('object_detections', object_event)
                            
                            # ADVANCED ANALYTICS
                            if self.abandoned_object_detection_enabled:
                                abandoned_alerts = self.detect_abandoned_objects(detections, frame)
                                all_alerts.extend(abandoned_alerts)
                            
                            if self.danger_zone_detection_enabled:
                                zone_intrusion_alerts = self.detect_danger_zone_intrusion(detections, frame)
                                all_alerts.extend(zone_intrusion_alerts)
                            
                            if self.suspicious_object_detection_enabled:
                                suspicious_alerts = self.detect_suspicious_objects(detections, frame)
                                all_alerts.extend(suspicious_alerts)
                            
                            if self.fall_detection_enabled:
                                fall_alerts = self.detect_falls(detections, frame)
                                all_alerts.extend(fall_alerts)
                            if self.machinery_proximity_enabled:
                                prox_alerts = self.detect_machinery_proximity(detections, frame)
                                all_alerts.extend(prox_alerts)
                                
                            if getattr(self, 'ppe_detector', None):
                                ppe_alerts, _ = self.ppe_detector.process_frame(frame)
                                all_alerts.extend(ppe_alerts)

                            
                            # Log object detection
                            self.write_object_logs(frame, objects_detected)

                        # Draw bounding boxes with labels
                        if len(detections.class_id) > 0:
                            labels = []
                            for i, class_id in enumerate(detections.class_id):
                                confidence = detections.confidence[i] if hasattr(detections, 'confidence') else 0.0
                                class_name = self.object_classes_id.get(class_id, 'Unknown')
                                labels.append(f"{class_name} ({confidence:.2f})")
                            
                            frame = self.boxAnnotator.annotate(
                                scene=frame,
                                detections=detections,
                                labels=labels
                            )
                        
                        # Draw advanced analytics overlays
                        self.draw_advanced_analytics_overlay(frame, all_alerts, shadow_mask)
                        
                    except Exception as e:
                        print(f"Error in object detection: {e}")
                        cv2.putText(frame, "[Object Detection Error]", 
                                  (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # MOTION DETECTION
                else:
                    cv2.putText(frame, "[Enhanced Motion Scanning]", 
                              (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    for c in contours_array:
                        size = cv2.contourArea(c)
                        if size >= self.min_contour_size:
                            motion_detected = True
                            (x, y, w, h) = cv2.boundingRect(c)                    
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, f"Motion: {int(size)}", (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    if motion_detected:
                        cv2.putText(frame, "[!] Motion Detected", 
                                  (5, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Log motion event
                        motion_event = {
                            'timestamp': datetime.now().isoformat(),
                            'type': 'motion_events',
                            'data': {
                                'contours_count': len([c for c in contours_array if cv2.contourArea(c) >= self.min_contour_size]),
                                'largest_contour_size': max([cv2.contourArea(c) for c in contours_array], default=0),
                                'frame_count': self.frame_count
                            }
                        }
                        self.log_analytics_event('motion_events', motion_event)
                        
                        # Write motion logs
                        self.write_motion_logs(frame)

                        # Switch to object detection
                        if not self.objectDetectionIsON:
                            self.objectDetectionIsON = True
                            self.obj_scan_time_start = datetime.now()
                            print("Switching to Advanced OBJ detection")

                            # Send email alert for motion
                            if self.isSendingAlerts:
                                try:
                                    alert_data = {
                                        'contours_detected': len(contours_array),
                                        'timestamp': datetime.now().isoformat()
                                    }
                                    self.send_alert("motion", alert_data)
                                except Exception as e:
                                    print(f"Error sending motion alert: {e}")

                # Add enhanced information overlay
                self.add_enhanced_info_overlay(frame, motion_detected, objects_detected, all_alerts, light_change_detected)

                # Draw shadow overlay if available
                if shadow_mask is not None and self.shadow_detection_enabled and np.any(shadow_mask):
                    shadow_overlay = cv2.applyColorMap(shadow_mask, cv2.COLORMAP_COOL)
                    frame = cv2.addWeighted(frame, 0.8, shadow_overlay, 0.2, 0)

                # Export frame for web interface
                with self._frame_lock:
                    self.export_frame = frame.copy()
                    
                if self.video_writer is not None:
                    self.video_writer.write(frame)

            except Exception as e:
                print(f"Error in detection loop: {e}")
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Detection Error:", 
                          (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(error_frame, f"{str(e)[:40]}", 
                          (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                with self._frame_lock:
                    self.export_frame = error_frame

            # Control frame rate (30 FPS)
            threading.Event().wait(0.033)

        # Loop ended, release writer if exists
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def draw_advanced_analytics_overlay(self, frame, alerts, shadow_mask):
        """FIXED: Draw overlays for advanced analytics with markers"""
        height, width = frame.shape[:2]
        
        # Alert counter
        alert_y = 100
        alert_counts = defaultdict(int)
        
        for alert in alerts:
            alert_type = alert.get('type', 'unknown')
            alert_counts[alert_type] += 1
            
            # Draw specific alert information with enhanced markers
            try:
                if alert_type == 'abandoned_object':
                    bbox = alert.get('bbox', [0, 0, 100, 100])
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        # Draw pulsing red rectangle for abandoned objects
                        pulse = int(abs(math.sin(self.frame_count * 0.2) * 100))
                        color = (0, 0, 255 - pulse)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                                    (int(bbox[2]), int(bbox[3])), color, 3)
                        # Add warning text with background
                        text = f"ABANDONED: {alert.get('class', 'Object')} ({alert.get('duration', 0):.0f}s)"
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])-text_height-10), 
                                    (int(bbox[0])+text_width, int(bbox[1])), (0, 0, 0), -1)
                        cv2.putText(frame, text, (int(bbox[0]), int(bbox[1])-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                  
                elif alert_type == 'danger_zone_intrusion':
                    center = alert.get('center', [320, 240])
                    if isinstance(center, (list, tuple)) and len(center) >= 2:
                        # Draw pulsing purple circle for danger_zone_intrusion
                        radius = int(50 + 20 * abs(math.sin(self.frame_count * 0.15)))
                        cv2.circle(frame, (int(center[0]), int(center[1])), radius, (255, 0, 255), 3)
                        cv2.circle(frame, (int(center[0]), int(center[1])), 5, (255, 0, 255), -1)
                        text = f"DANGER ZONE INTRUSION: {alert.get('duration', 0):.0f}s"
                        cv2.putText(frame, text, (int(center[0])-60, int(center[1])-radius-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                                  
                elif alert_type == 'suspicious_object':
                    bbox = alert.get('bbox', [0, 0, 100, 100])
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        # Draw blinking orange rectangle for suspicious objects
                        if self.frame_count % 30 < 15:  # Blink every 15 frames
                            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                                        (int(bbox[2]), int(bbox[3])), (0, 165, 255), 3)
                            text = f"SUSPICIOUS: {alert.get('class', 'Object')} ({alert.get('duration', 0):.0f}s)"
                            cv2.putText(frame, text, (int(bbox[0]), int(bbox[1])-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                                      
                elif alert_type == 'fall_detection':
                    bbox = alert.get('bbox', [0, 0, 100, 100])
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        # Draw urgent yellow rectangle for falls with cross marker
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                                    (int(bbox[2]), int(bbox[3])), (0, 255, 255), 4)
                        # Draw cross marker
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        cv2.line(frame, (center_x-20, center_y), (center_x+20, center_y), (0, 255, 255), 3)
                        cv2.line(frame, (center_x, center_y-20), (center_x, center_y+20), (0, 255, 255), 3)
                        text = f"FALL DETECTED! Reason: {alert.get('reason', 'unknown')}"
                        cv2.putText(frame, text, (int(bbox[0]), int(bbox[1])-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            except Exception as e:
                print(f"Error drawing alert overlay: {e}")
        
        # Display alert summary with enhanced styling
        for alert_type, count in alert_counts.items():
            # Add background for better visibility
            text = f"{alert_type.upper()}: {count}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (width - 250, alert_y-text_height-5), 
                        (width - 250 + text_width, alert_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (width - 250, alert_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            alert_y += 25

    def add_enhanced_info_overlay(self, frame, motion_detected, objects_detected, alerts, light_change_detected):
        """FIXED: Add enhanced information overlay to the frame"""
        height, width = frame.shape[:2]
        
        # Enhanced parameter info at bottom
        info_y = height - 140
        cv2.putText(frame, f"Min Contour: {self.min_contour_size} | Frame Thresh: {self.frame_diff_threshold}", 
                  (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"BG History: {self.hist} | Scan Duration: {self.obj_scan_duration}s", 
                  (10, info_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Advanced analytics status
        analytics_status = []
        if self.shadow_detection_enabled:
            analytics_status.append("Shadow✓")
        if self.abandoned_object_detection_enabled:
            analytics_status.append("Abandoned✓")
        if self.danger_zone_detection_enabled:
            analytics_status.append("Danger Zone✓")
        if self.suspicious_object_detection_enabled:
            analytics_status.append("Suspicious✓")
        if self.fall_detection_enabled:
            analytics_status.append("Fall✓")
            
        analytics_text = f"Analytics: {', '.join(analytics_status) if analytics_status else 'All Disabled'}"
        cv2.putText(frame, analytics_text, 
                  (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
                  (0, 255, 0) if analytics_status else (255, 0, 0), 1)
        
        # Alert status
        alert_status = "ON" if self.isSendingAlerts else "OFF"
        alert_color = (0, 255, 0) if self.isSendingAlerts else (0, 0, 255)
        cv2.putText(frame, f"Alerts: {alert_status} ({self.alert_time_start}-{self.alert_time_end})", 
                  (width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)
        
        # Light change indicator
        if light_change_detected and self.shadow_detection_enabled:
            cv2.putText(frame, "[!] Light Change Detected", 
                      (width - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Active tracking counts
        cv2.putText(frame, f"Tracking: {len(self.person_tracks)} people, {len(self.abandoned_objects)} objects", 
                  (10, info_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Alerts count
        if alerts:
            cv2.putText(frame, f"Active Alerts: {len(alerts)}", 
                      (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        
        # Frame counter and timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Frame: {self.frame_count} | {timestamp}", 
                  (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def get_curr_date_time(self):
        """Returns the current date and current time"""
        curr_time = datetime.now().strftime("%H:%M:%S")
        curr_date = date.today()
        return curr_date, curr_time

    def write_motion_logs(self, frame):
        """Enhanced motion log writing"""
        current_time = datetime.now()
        
        # Only write once every minute to avoid spam
        if current_time - self.last_log_time >= timedelta(minutes=1):
            try:
                with open(self.motion_logs_path, "a") as motion_log:
                    self.last_log_time = current_time 
                    curr_date, curr_time = self.get_curr_date_time()
                    motion_log_entry = f"Enhanced motion detected at {curr_time} on {curr_date} (Frame: {self.frame_count})\n"
                    motion_log.write(motion_log_entry)
                
                # Save frame
                timestamp = str(datetime.now())[:19].replace(":", ";").replace(" ", "_")
                filename = os.path.join(self.motion_frames_path, f"motion_{timestamp}_f{self.frame_count}.png")
                cv2.imwrite(filename, frame)
                    
            except Exception as e:
                print(f"Error writing motion logs: {e}")

    def write_object_logs(self, frame, objects_detected):
        """Enhanced object log writing"""
        try:
            with open(self.object_logs_path, "a") as object_log:
                curr_date, curr_time = self.get_curr_date_time()
                objects_str = ", ".join(objects_detected)
                log_entry = f"Objects detected at {curr_time} on {curr_date}: {objects_str} (Frame: {self.frame_count})\n"
                object_log.write(log_entry)
            
            # Save frame
            timestamp = str(datetime.now())[:19].replace(":", ";").replace(" ", "_")
            filename = os.path.join(self.object_frames_path, f"objects_{timestamp}_f{self.frame_count}.png")
            cv2.imwrite(filename, frame)
                
        except Exception as e:
            print(f"Error writing object logs: {e}")

    def cleanup(self):
        """Clean up resources"""
        self.stop_detection()
        
        # Clear WhatsApp queue
        with self.whatsapp_lock:
            self.whatsapp_queue.clear()
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Detector cleanup completed")
    def enable_whatsapp_alerts(self, enabled=True):
        """Enable or disable WhatsApp alerts"""
        self.whatsapp_enabled = enabled
        print(f"WhatsApp alerts {'enabled' if enabled else 'disabled'}")

    def update_whatsapp_number(self, phone_number):
        """Update WhatsApp phone number"""
        # Ensure proper format
        if not phone_number.startswith('+'):
            phone_number = '+' + phone_number
        self.whatsapp_number = phone_number
        print(f"WhatsApp number updated to: {phone_number}")

    def set_whatsapp_cooldown(self, seconds):
        """Set cooldown period between WhatsApp alerts"""
        self.whatsapp_cooldown = max(60, seconds)  # Minimum 1 minute
        print(f"WhatsApp cooldown set to: {seconds} seconds")

    def get_whatsapp_status(self):
        """Get current WhatsApp alert status"""
        return {
            'enabled': self.whatsapp_enabled,
            'phone_number': self.whatsapp_number,
            'cooldown_seconds': self.whatsapp_cooldown,
            'triggers': self.whatsapp_triggers,
            'last_alerts': self.last_whatsapp_alert
        }
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()
        
    app = Detector_2A2S(cap)
    app.start_detection()
    
    try:
        print("Enhanced 2A2S Detection System Started")
        print("Press 'q' to quit")
        
        while True:
            frame = app.get_export_frame()
            if frame is not None:
                cv2.imshow("2A2S Enhanced Detection System", frame)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping detection...")
    finally:
        app.cleanup()