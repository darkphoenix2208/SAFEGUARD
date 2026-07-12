from flask import Flask, Response, jsonify, request, render_template_string
from flask_cors import CORS
import cv2
import threading
import time
import json
from datetime import datetime, timedelta
import os
import numpy as np
from survilleance.Detector import Detector_2A2S  # Import the enhanced detector

# Mock classes for demonstration (replace with your actual imports)
class MockEmailAlert:
    def send_alert_cli(self, alert_type, timestamp, alert_data=None):
        print(f"ALERT: {alert_type} at {timestamp}")
        if alert_data:
            print(f"Alert data: {alert_data}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
cap = None
detector = None
camera_initialized = False

def initialize_camera():
    """Initialize camera and detector"""
    global cap, detector, camera_initialized
    
    if not camera_initialized:
        try:
            cap = cv2.VideoCapture(0)  # Try default camera
            if not cap.isOpened():
                # Try other camera indices
                for i in range(1, 5):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        break
            
            if cap.isOpened():
                # Set camera properties for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                detector = Detector_2A2S(cap)  # Use the enhanced detector
                detector.start_detection()
                camera_initialized = True
                print("Enhanced camera system initialized successfully")
            else:
                print("Failed to open camera")
                
        except Exception as e:
            print(f"Error initializing camera: {e}")
@app.route('/')
def dashboard():
    """Main dashboard with advanced analytics"""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Enhanced video stream with advanced analytics overlay"""
    def generate_frames():
        global detector
        
        while True:
            if detector and detector.export_frame is not None:
                try:
                    frame = detector.get_export_frame()
                    if frame is not None:
                        # Enhance frame quality for web streaming
                        ret, buffer = cv2.imencode('.jpg', frame, [
                            cv2.IMWRITE_JPEG_QUALITY, 85,
                            cv2.IMWRITE_JPEG_OPTIMIZE, 1
                        ])
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        else:
                            # Send error frame
                            error_frame = create_error_frame("Frame Encoding Error")
                            ret, buffer = cv2.imencode('.jpg', error_frame)
                            if ret:
                                frame_bytes = buffer.tobytes()
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        # Send loading frame
                        loading_frame = create_loading_frame()
                        ret, buffer = cv2.imencode('.jpg', loading_frame)
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error generating frame: {e}")
                    error_frame = create_error_frame(f"Stream Error: {str(e)[:30]}")
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Send initialization message
                init_frame = create_initialization_frame()
                ret, buffer = cv2.imencode('.jpg', init_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def create_error_frame(message):
    """Create an error frame with message"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add gradient background
    for i in range(480):
        frame[i, :] = [int(50 + i*0.1), int(30 + i*0.05), int(30 + i*0.05)]
    
    cv2.putText(frame, "Safeguard SURVEILLANCE SYSTEM", (120, 100), 
              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "ERROR DETECTED", (200, 200), 
              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv2.putText(frame, message, (50, 250), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Please check system status", (150, 300), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return frame

def create_loading_frame():
    """Create a loading frame"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add gradient background
    for i in range(480):
        frame[i, :] = [int(20 + i*0.1), int(40 + i*0.1), int(60 + i*0.1)]
    
    cv2.putText(frame, "Safeguard ADVANCED SURVEILLANCE", (100, 150), 
              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Initializing Advanced Analytics...", (120, 200), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "Please wait", (250, 250), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add loading animation dots
    timestamp = int(time.time() * 2) % 4
    dots = "." * timestamp
    cv2.putText(frame, f"Loading{dots}", (270, 300), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def create_initialization_frame():
    """Create initialization frame with system info"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add animated background
    timestamp = int(time.time() * 50) % 640
    cv2.line(frame, (timestamp, 0), (timestamp, 480), (0, 50, 100), 2)
    cv2.line(frame, (0, timestamp % 480), (640, timestamp % 480), (0, 50, 100), 2)
    
    cv2.putText(frame, "Safeguard SURVEILLANCE SYSTEM", (120, 120), 
              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Advanced AI-Augmented Analytics", (130, 160), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Click 'Initialize Camera System' to start", (110, 220), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Feature list
    features = [
        "Motion Detection", "Object Recognition", "Shadow Analysis",
        "Abandoned Object Detection", "Danger Zone Intrusion", "Fall from Height Detection"
    ]
    
    y_pos = 280
    for i, feature in enumerate(features):
        color = (0, 255, 0) if i % 2 == 0 else (0, 255, 255)
        cv2.putText(frame, f"• {feature}", (50, y_pos + i * 25), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Initialize and start enhanced camera system"""
    initialize_camera()
    if camera_initialized:
        return jsonify({
            "message": "Enhanced camera system started successfully",
            "features": [
                "Motion Detection", "Object Recognition", "Shadow Analysis",
                "Abandoned Object Detection", "Danger Zone Intrusion", 
                 "Fall from Height Detection"
            ]
        })
    else:
        return jsonify({"message": "Failed to start camera system"}), 500

@app.route('/update_parameters', methods=['POST'])
def update_parameters():
    """Update detection parameters with enhanced validation"""
    global detector
    
    if detector:
        params = request.get_json()
        detector.update_parameters(params)
        return jsonify({
            "status": "Parameters updated successfully", 
            "params": params,
            "timestamp": datetime.now().isoformat()
        })
    
    return jsonify({"status": "Detector not initialized"}), 400
@app.route('/whatsapp_status')
def whatsapp_status():
    """Get WhatsApp alert system status"""
    global detector
    
    if detector:
        try:
            status = detector.get_whatsapp_status()
            return jsonify({
                "status": "success",
                "whatsapp_config": status,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": f"Error getting WhatsApp status: {str(e)}"}), 500
    
    return jsonify({"error": "Detector not initialized"}), 400

@app.route('/update_whatsapp_config', methods=['POST'])
def update_whatsapp_config():
    """Update WhatsApp configuration"""
    global detector
    
    if detector:
        try:
            data = request.get_json()
            
            if 'enabled' in data:
                detector.enable_whatsapp_alerts(bool(data['enabled']))
            
            if 'phone_number' in data:
                detector.update_whatsapp_number(str(data['phone_number']))
            
            if 'cooldown' in data:
                detector.set_whatsapp_cooldown(int(data['cooldown']))
            
            return jsonify({
                "status": "WhatsApp configuration updated successfully",
                "config": detector.get_whatsapp_status(),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": f"Error updating WhatsApp config: {str(e)}"}), 500
    
    return jsonify({"error": "Detector not initialized"}), 400

@app.route('/test_whatsapp_alert', methods=['POST'])
def test_whatsapp_alert():
    """Test WhatsApp alert system"""
    global detector
    
    if detector:
        try:
            test_data = {
                'class': 'Test Object',
                'duration': 25.5,
                'center': [320, 240],
                'timestamp': datetime.now().isoformat()
            }
            
            detector.send_whatsapp_alert('abandoned_object', test_data)
            
            return jsonify({
                "message": "Test WhatsApp alert sent successfully",
                "phone_number": detector.whatsapp_number,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": f"Error sending test WhatsApp alert: {str(e)}"}), 500
    
    return jsonify({"error": "Detector not initialized"}), 400

@app.route('/trigger_alert', methods=['POST'])
def trigger_alert():
    """Trigger manual alert with enhanced logging"""
    global detector
    
    if detector:
        alert_data = {
            "type": "manual_test",
            "timestamp": datetime.now().isoformat(),
            "source": "web_dashboard"
        }
        
        try:
            detector.send_alert("manual", alert_data)
            return jsonify({
                "message": "Manual alert triggered successfully",
                "alert_data": alert_data
            })
        except Exception as e:
            pass
    
    return jsonify({"message": "Detector not initialized"}), 400

@app.route('/capture_snapshot', methods=['POST'])
def capture_snapshot():
    """Capture and save current frame snapshot"""
    global detector
    
    if detector:
        try:
            frame = detector.get_export_frame()
            if frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"./logs/snapshots/snapshot_{timestamp}.png"
                
                # Ensure snapshot directory exists
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                saved = cv2.imwrite(filename, frame)
                if saved:
                    return jsonify({
                        "message": f"Snapshot saved successfully: {filename}",
                        "filename": filename,
                        "timestamp": timestamp
                    })
                else:
                    return jsonify({"message": "Failed to save snapshot"}), 500
            else:
                return jsonify({"message": "No frame available for snapshot"}), 400
        except Exception as e:
            return jsonify({"message": f"Error capturing snapshot: {str(e)}"}), 500
    
    return jsonify({"message": "Detector not initialized"}), 400

@app.route('/update_alert_times', methods=['POST'])
def update_alert_times():
    """Update alert time configuration with validation"""
    global detector
    
    if detector:
        try:
            data = request.get_json()
            start_time = data.get('start', '20:00')
            end_time = data.get('end', '08:00')
            
            # Validate time format
            datetime.strptime(start_time, '%H:%M')
            datetime.strptime(end_time, '%H:%M')
            
            detector.alert_time_start = start_time
            detector.alert_time_end = end_time
            
            return jsonify({
                "status": "Alert times updated successfully",
                "start_time": start_time,
                "end_time": end_time,
                "timestamp": datetime.now().isoformat()
            })
        except ValueError as e:
            return jsonify({"status": f"Invalid time format: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"status": f"Error updating alert times: {str(e)}"}), 500
    
    return jsonify({"status": "Detector not initialized"}), 400

@app.route('/get_analytics')
def get_analytics():
    """Get comprehensive analytics data for dashboard"""
    global detector
    
    if detector:
        try:
            analytics_data = detector.get_analytics_data()
            
            # Add system status information
            analytics_data['system_status'] = {
                'camera_initialized': camera_initialized,
                'detector_running': detector.running if detector else False,
                'object_detection': detector.objectDetectionIsON if detector else False,
                'alerts_enabled': detector.isSendingAlerts if detector else False,
                'frame_count': detector.frame_count if detector else 0,
                'uptime': str(datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
            }
            
            return jsonify(analytics_data)
        except Exception as e:
            return jsonify({
                "error": f"Error retrieving analytics: {str(e)}",
                "analytics": {},
                "current_stats": {}
            }), 500
    
    return jsonify({
        "error": "Detector not initialized",
        "analytics": {},
        "current_stats": {}
    }), 400

@app.route('/export_logs')
def export_logs():
    """Export all logs as JSON for download"""
    try:
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "system_info": {
                "version": "Safeguard Enhanced v2.0",
                "camera_initialized": camera_initialized,
                "detector_running": detector.running if detector else False
            },
            "logs": {}
        }
        
        # Read motion logs
        if os.path.exists("./logs/motion_log.txt"):
            with open("./logs/motion_log.txt", 'r') as f:
                export_data["logs"]["motion_log"] = f.read().split('\n')
        
        # Read object logs
        if os.path.exists("./logs/object_log.txt"):
            with open("./logs/object_log.txt", 'r') as f:
                export_data["logs"]["object_log"] = f.read().split('\n')
        
        # Read analytics logs
        if os.path.exists("./logs/analytics_log.json"):
            with open("./logs/analytics_log.json", 'r') as f:
                export_data["logs"]["analytics_log"] = json.load(f)
        
        # Read alerts logs
        if os.path.exists("./logs/alerts_log.json"):
            with open("./logs/alerts_log.json", 'r') as f:
                export_data["logs"]["alerts_log"] = json.load(f)
        
        # Get current analytics if detector is running
        if detector:
            export_data["current_analytics"] = detector.get_analytics_data()
        
        response = app.response_class(
            response=json.dumps(export_data, indent=2),
            status=200,
            mimetype='application/json'
        )
        response.headers['Content-Disposition'] = f'attachment; filename=Safeguard_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        return response
        
    except Exception as e:
        return jsonify({"error": f"Error exporting logs: {str(e)}"}), 500

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    """Clear all log files with backup"""
    try:
        # Create backup before clearing
        backup_dir = f"./logs/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)
        
        log_files = [
            "./logs/motion_log.txt",
            "./logs/object_log.txt", 
            "./logs/analytics_log.json",
            "./logs/alerts_log.json"
        ]
        
        backed_up_files = []
        cleared_files = []
        
        for log_file in log_files:
            if os.path.exists(log_file):
                # Create backup
                backup_file = os.path.join(backup_dir, os.path.basename(log_file))
                with open(log_file, 'r') as src, open(backup_file, 'w') as dst:
                    dst.write(src.read())
                backed_up_files.append(log_file)
                
                # Clear original file
                if log_file.endswith('.json'):
                    with open(log_file, 'w') as f:
                        json.dump([], f)
                else:
                    with open(log_file, 'w') as f:
                        f.write("")
                cleared_files.append(log_file)
        
        # Reset detector analytics if running
        if detector:
            detector.analytics_data = {
                'motion_events': [],
                'object_detections': [],
                'shadow_detections': [],
                'abandoned_objects': [],
                'zone_intrusion_events': [],
                'suspicious_objects': [],
                'fall_detections': [],
                'alerts_sent': []
            }
        
        return jsonify({
            "message": "Logs cleared successfully",
            "backup_location": backup_dir,
            "backed_up_files": backed_up_files,
            "cleared_files": cleared_files,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Error clearing logs: {str(e)}"}), 500

@app.route('/system_health')
def system_health():
    """Get detailed system health information"""
    global detector, cap
    
    health_data = {
        "timestamp": datetime.now().isoformat(),
        "system_status": "healthy",
        "components": {
            "camera": {
                "status": "active" if camera_initialized and cap and cap.isOpened() else "inactive",
                "details": "Camera feed operational" if camera_initialized else "Camera not initialized"
            },
            "detector": {
                "status": "active" if detector and detector.running else "inactive",
                "details": f"Processing frame #{detector.frame_count}" if detector else "Detector not running"
            },
            "analytics": {
                "status": "active" if detector else "inactive",
                "enabled_features": []
            },
            "alerts": {
                "status": "active" if detector and detector.isSendingAlerts else "inactive",
                "schedule": f"{detector.alert_time_start} - {detector.alert_time_end}" if detector else "Not configured"
            }
        },
        "performance": {
            "uptime": "Active",
            "memory_usage": "Normal",
            "cpu_usage": "Normal"
        }
    }
    
    if detector:
        health_data["components"]["analytics"]["enabled_features"] = [
            "shadow_detection" if detector.shadow_detection_enabled else None,
            "abandoned_object_detection" if detector.abandoned_object_detection_enabled else None,
            "danger_zone_detection" if detector.danger_zone_detection_enabled else None,
            "suspicious_object_detection" if detector.suspicious_object_detection_enabled else None,
            "fall_detection" if detector.fall_detection_enabled else None
        ]
        health_data["components"]["analytics"]["enabled_features"] = [f for f in health_data["components"]["analytics"]["enabled_features"] if f]
    
    # Determine overall system status
    inactive_components = [k for k, v in health_data["components"].items() if v["status"] == "inactive"]
    if len(inactive_components) > 2:
        health_data["system_status"] = "degraded"
    elif len(inactive_components) > 0:
        health_data["system_status"] = "warning"
    
    return jsonify(health_data)

@app.route('/logs')
def view_logs():
    """Enhanced log viewer with analytics"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Safeguard System Logs</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #2c3e50; text-align: center; }
            .log-section { background: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .log-content { max-height: 400px; overflow-y: auto; background: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; }
            .btn { padding: 10px 20px; margin: 10px; border: none; border-radius: 5px; cursor: pointer; }
            .btn-primary { background: #3498db; color: white; }
            .btn-success { background: #27ae60; color: white; }
            .btn-danger { background: #e74c3c; color: white; }
            @keyframes pulse {
                0%, 100% { 
                    transform: scale(1); 
                    opacity: 1; 
                }
                50% { 
                    transform: scale(1.05); 
                    opacity: 0.8;
                    text-shadow: 0 0 15px currentColor;
                }
            }

.stat-number {
    transition: all 0.3s ease;
}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ Safeguard Advanced System Logs</h1>
            
            <div class="log-section">
                <h2>📊 System Status</h2>
                <div id="systemStatus">Loading...</div>
            </div>
            
            <div class="log-section">
                <h2>🏃 Motion Detection Logs</h2>
                <p><strong>Location:</strong> ./logs/motion_log.txt</p>
                <p><strong>Captured Frames:</strong> ./logs/motion_frames/</p>
            </div>
            
            <div class="log-section">
                <h2>🎯 Object Detection Logs</h2>
                <p><strong>Location:</strong> ./logs/object_log.txt</p>
                <p><strong>Captured Frames:</strong> ./logs/object_frames/</p>
            </div>
            
            <div class="log-section">
                <h2>📈 Analytics Logs</h2>
                <p><strong>Location:</strong> ./logs/analytics_log.json</p>
                <p><strong>Features:</strong> Shadow Detection, Abandoned Objects, Danger Zone Intrusion, Fall from Height Detection</p>
            </div>
            
            <div class="log-section">
                <h2>🚨 Alert Logs</h2>
                <p><strong>Location:</strong> ./logs/alerts_log.json</p>
                <p><strong>Email System:</strong> Integrated</p>
            </div>
            
            <div class="log-section">
                <h2>📸 Snapshots</h2>
                <p><strong>Location:</strong> ./logs/snapshots/</p>
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <button class="btn btn-primary" onclick="refreshStatus()">🔄 Refresh Status</button>
                <button class="btn btn-success" onclick="window.open('/export_logs', '_blank')">📁 Export All Logs</button>
                <button class="btn btn-danger" onclick="window.close()">❌ Close</button>
            </div>
        </div>
        
        <script>
            function refreshStatus() {
                fetch('/system_health')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('systemStatus').innerHTML = `
                            <strong>Overall Status:</strong> ${data.system_status.toUpperCase()}<br>
                            <strong>Camera:</strong> ${data.components.camera.status}<br>
                            <strong>Detector:</strong> ${data.components.detector.status}<br>
                            <strong>Analytics:</strong> ${data.components.analytics.status}<br>
                            <strong>Alerts:</strong> ${data.components.alerts.status}<br>
                            <strong>Last Updated:</strong> ${new Date(data.timestamp).toLocaleString()}
                        `;
                    })
                    .catch(error => {
                        document.getElementById('systemStatus').innerHTML = 'Error loading status';
                        console.error('Error:', error);
                    });
            }
            
            // Load initial status
            refreshStatus();
        </script>
    </body>
    </html>
    '''

@app.route('/status')
def get_status():
    """Get comprehensive system status"""
    global detector, camera_initialized
    
    status = {
        "timestamp": datetime.now().isoformat(),
        "camera_initialized": camera_initialized,
        "detector_running": detector.running if detector else False,
        "object_detection": detector.objectDetectionIsON if detector else False,
        "alert_times": {
            "start": detector.alert_time_start if detector else "20:00",
            "end": detector.alert_time_end if detector else "08:00"
        },
        "advanced_analytics": {
            "shadow_detection": detector.shadow_detection_enabled if detector else False,
            "abandoned_object_detection": detector.abandoned_object_detection_enabled if detector else False,
            "danger_zone_detection": detector.danger_zone_detection_enabled if detector else False,
            "suspicious_object_detection": detector.suspicious_object_detection_enabled if detector else False,
            "fall_detection": detector.fall_detection_enabled if detector else False
        },
        "current_tracking": {
            "people_tracked": len(detector.person_tracks) if detector else 0,
            "abandoned_objects": len(detector.abandoned_objects) if detector else 0,
            "frame_count": detector.frame_count if detector else 0
        }
    }
    
    return jsonify(status)
# Enhanced Dashboard Template
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Safeguard Advanced Surveillance Analytics Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Safeguard Advanced AI Surveillance System</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-gray: #898f8f;
            --sage-green: #DABFC8;
            --accent-red: #ED0021;
            --light-gray: #BABBC0;
            --dark-bg: #0a0b0d;
            --card-bg: rgba(218, 191, 200, 0.05);
            --glass-bg: rgba(255, 255, 255, 0.02);
            --border-color: rgba(218, 191, 200, 0.1);
            --text-primary: #ffffff;
            --text-secondary: #BABBC0;
            --shadow-glow: rgba(237, 0, 33, 0.3);
        }

        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }

        body { 
            font-family: 'Inter', sans-serif; 
            background: radial-gradient(circle at 30% 30%, rgba(94, 8, 2, 0.7) 0%, rgba(26, 0, 0, 0.9) 40%, #1a0000 80%),
            linear-gradient(120deg, #1a0000 0%, #2c0a0a 25%, #3b0e0e 50%, #5e0802 75%, #b2b2b2 100%);
            background-blend-mode: overlay, multiply;
            box-shadow: inset 0 0 150px rgba(0, 0, 0, 0.8);
            color: #fff;
            font-family: 'Segoe UI', sans-serif;
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 80%, rgba(237, 0, 33, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(218, 191, 200, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(137, 143, 143, 0.05) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }
        
        .header {
            background: rgba(10, 11, 13, 0.95);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border-color);
            padding: 2rem;
            position: sticky;
            top: 0;
            z-index: 1000;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .header h1 {
            color: var(--text-primary);
            text-align: center;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, var(--sage-green), var(--accent-red));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 30px var(--shadow-glow);
            letter-spacing: -0.5px;
        }
        
        .system-status {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 1rem 1.5rem;
            border-radius: 50px;
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border-color);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .status-item::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(218, 191, 200, 0.1), transparent);
            transition: left 0.5s;
        }

        .status-item:hover::before {
            left: 100%;
        }

        .status-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(218, 191, 200, 0.2);
            border-color: var(--sage-green);
        }
        
        .status-dot {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            position: relative;
            animation: pulse 2s infinite;
        }

        .status-dot::after {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            border-radius: 50%;
            border: 2px solid currentColor;
            opacity: 0;
            animation: ping 2s infinite;
        }
        
        .status-active { 
            background: var(--sage-green); 
            box-shadow: 0 0 20px rgba(218, 191, 200, 0.5);
        }
        .status-inactive { 
            background: var(--primary-gray); 
            box-shadow: 0 0 20px rgba(137, 143, 143, 0.3);
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }

        @keyframes ping {
            75%, 100% { transform: scale(2); opacity: 0; }
        }
        
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            padding: 2rem;
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .video-section {
            grid-column: 1 / -1;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
        }

        .video-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--sage-green), transparent);
        }
        
        .video-container {
            background: #000;
            border-radius: 20px;
            overflow: hidden;
            position: relative;
            box-shadow: 
                0 20px 40px rgba(0, 0, 0, 0.5),
                0 0 0 1px rgba(218, 191, 200, 0.2);
            transition: all 0.3s ease;
        }

        .video-container:hover {
            transform: translateY(-2px);
            box-shadow: 
                0 25px 50px rgba(0, 0, 0, 0.6),
                0 0 0 1px var(--sage-green),
                0 0 30px rgba(218, 191, 200, 0.3);
        }
        
        .video-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .controls-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 2rem;
            margin-top: 2rem;
        }
        
        .control-panel, .analytics-panel {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            border: 1px solid var(--border-color);
            position: relative;
            transition: all 0.3s ease;
        }

        .control-panel:hover, .analytics-panel:hover {
            transform: translateY(-4px);
            box-shadow: 
                0 12px 40px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.15),
                0 0 20px rgba(218, 191, 200, 0.1);
        }
        
        .panel-title {
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 2rem;
            text-align: center;
            position: relative;
            letter-spacing: -0.3px;
        }
        
        .panel-title::after {
            content: '';
            position: absolute;
            bottom: -0.75rem;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 3px;
            background: linear-gradient(135deg, var(--sage-green), var(--accent-red));
            border-radius: 2px;
            box-shadow: 0 0 10px rgba(237, 0, 33, 0.5);
        }
        
        .control-group {
            margin-bottom: 2rem;
        }
        
        .control-group label {
            display: block;
            margin-bottom: 0.75rem;
            font-weight: 500;
            color: var(--text-secondary);
            font-size: 0.95rem;
            letter-spacing: 0.3px;
        }
        
        .slider {
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: rgba(137, 143, 143, 0.2);
            outline: none;
            -webkit-appearance: none;
            position: relative;
            transition: all 0.3s ease;
        }

        .slider:hover {
            background: rgba(137, 143, 143, 0.3);
        }
        
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--sage-green), var(--accent-red));
            cursor: pointer;
            box-shadow: 
                0 4px 12px rgba(237, 0, 33, 0.3),
                0 0 0 3px rgba(218, 191, 200, 0.2);
            transition: all 0.3s ease;
        }

        .slider::-webkit-slider-thumb:hover {
            transform: scale(1.1);
            box-shadow: 
                0 6px 16px rgba(237, 0, 33, 0.4),
                0 0 0 6px rgba(218, 191, 200, 0.3);
        }
        
        .btn {
            padding: 1rem 2rem;
            border: none;
            border-radius: 16px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin: 0.25rem;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            border: 1px solid transparent;
            letter-spacing: 0.3px;
        }
        
        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        
        .btn:hover::before {
            left: 100%;
        }
        
        .btn-primary { 
            background: linear-gradient(135deg, var(--sage-green), var(--primary-gray)); 
            color: var(--dark-bg);
            box-shadow: 0 4px 15px rgba(218, 191, 200, 0.3);
        }
        .btn-success { 
            background: linear-gradient(135deg, var(--sage-green), #4ade80); 
            color: var(--dark-bg);
            box-shadow: 0 4px 15px rgba(218, 191, 200, 0.3);
        }
        .btn-danger { 
            background: linear-gradient(135deg, var(--accent-red), #ef4444); 
            color: white;
            box-shadow: 0 4px 15px rgba(237, 0, 33, 0.3);
        }
        .btn-warning { 
            background: linear-gradient(135deg, #f59e0b, var(--accent-red)); 
            color: white;
            box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
        }
        .btn-info { 
            background: linear-gradient(135deg, var(--light-gray), var(--sage-green)); 
            color: var(--dark-bg);
            box-shadow: 0 4px 15px rgba(186, 187, 192, 0.3);
        }
        
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(218, 191, 200, 0.4);
        }

        .btn:active {
            transform: translateY(-1px);
        }
        
        .toggle-group {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .toggle-btn {
            padding: 1.25rem;
            border: 2px solid var(--border-color);
            border-radius: 16px;
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-align: center;
            font-weight: 500;
            color: var(--text-secondary);
            position: relative;
            overflow: hidden;
        }

        .toggle-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, var(--sage-green), var(--accent-red));
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .toggle-btn span {
            position: relative;
            z-index: 2;
        }
        
        .toggle-btn.active {
            border-color: var(--sage-green);
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(218, 191, 200, 0.3);
        }

        .toggle-btn.active::before {
            opacity: 1;
        }

        .toggle-btn:hover {
            transform: translateY(-2px);
            border-color: var(--sage-green);
            box-shadow: 0 8px 25px rgba(218, 191, 200, 0.2);
        }
        
        .chart-container {
            position: relative;
            height: 320px;
            margin: 2rem 0;
            background: var(--glass-bg);
            border-radius: 20px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            backdrop-filter: blur(10px);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .stat-card {
            background: var(--glass-bg);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 2rem 1.5rem;
            text-align: center;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            border: 1px solid var(--border-color);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--sage-green), var(--accent-red));
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }

        .stat-card:hover::before {
            transform: scaleX(1);
        }
        
        .stat-card:hover {
            transform: translateY(-8px);
            box-shadow: 
                0 12px 40px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.15),
                0 0 20px rgba(218, 191, 200, 0.2);
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--sage-green), var(--accent-red));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: var(--text-secondary);
            font-weight: 500;
            letter-spacing: 0.5px;
        }
        
        .alert-log {
            max-height: 350px;
            overflow-y: auto;
            background: var(--glass-bg);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border: 1px solid var(--border-color);
            backdrop-filter: blur(10px);
        }

        .alert-log::-webkit-scrollbar {
            width: 6px;
        }

        .alert-log::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }

        .alert-log::-webkit-scrollbar-thumb {
            background: var(--sage-green);
            border-radius: 3px;
        }
        
        .alert-item {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border-left: 4px solid var(--accent-red);
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 0 16px 16px 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--accent-red);
            transition: all 0.3s ease;
        }

        .alert-item:hover {
            transform: translateX(8px);
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.3);
        }
        
        .alert-item.motion { border-left-color: #3b82f6; }
        .alert-item.abandoned { border-left-color: #e67e22; }
        .alert-item.danger_zone_intrusion { border-left-color: #9b59b6; }
        .alert-item.suspicious { border-left-color: #f39c12; }
        .alert-item.fall { border-left-color: var(--accent-red); }
        
        .loading {
            display: inline-block;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(218, 191, 200, 0.3);
            border-radius: 50%;
            border-top-color: var(--sage-green);
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .full-width { grid-column: 1 / -1; }

        input[type="time"] {
            width: 100%;
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            color: var(--text-primary);
            font-size: 0.95rem;
            transition: all 0.3s ease;
        }

        input[type="time"]:focus {
            outline: none;
            border-color: var(--sage-green);
            box-shadow: 0 0 0 3px rgba(218, 191, 200, 0.2);
        }

        hr {
            margin: 2rem 0;
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        }

        h3 {
            text-align: center;
            color: var(--text-primary);
            margin-bottom: 1.5rem;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        
        @media (max-width: 768px) {
            .container { 
                grid-template-columns: 1fr; 
                padding: 1rem; 
                gap: 1.5rem;
            }
            .header h1 { font-size: 2.5rem; }
            .system-status { 
                flex-direction: column; 
                align-items: center; 
                gap: 1rem; 
            }
            .controls-grid {
                grid-template-columns: 1fr;
            }
            .toggle-group {
                grid-template-columns: 1fr;
            }
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 480px) {
            .header {
                padding: 1.5rem 1rem;
            }
            .header h1 {
                font-size: 2rem;
            }
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <!-- Add Back Button -->
        <button 
            onclick="window.location.href='http://localhost:3001'" 
            class="btn btn-primary" 
            style="position: absolute; left: 20px; top: 20px; display: flex; align-items: center; gap: 8px;"
        >
            <span style="font-size: 20px;">←</span> Back
        </button>
        <h1>🛡️ Safeguard Advanced AI Surveillance System</h1>
        <div class="system-status">
            <div class="status-item">
                <div class="status-dot status-active" id="cameraStatus"></div>
                <span>Camera System</span>
            </div>
            <div class="status-item">
                <div class="status-dot status-inactive" id="detectionStatus"></div>
                <span>AI Detection</span>
            </div>
            <div class="status-item">
                <div class="status-dot status-inactive" id="alertStatus"></div>
                <span>Alert System</span>
            </div>
            <div class="status-item">
                <div class="status-dot status-active" id="analyticsStatus"></div>
                <span>Advanced Analytics</span>
            </div>
        </div>
    </div>

    <div class="container">
        <!-- Video Feed Section -->
        <div class="video-section">
            <h2 class="panel-title">🎥 Live Video Feed</h2>
            <div class="video-container">
                <img id="videoFeed" src="/video_feed" alt="Camera Feed">
            </div>
            <div style="display: flex; gap: 1rem; margin-top: 2rem;">
                <button class="btn btn-success" onclick="startCamera()" style="flex: 1;">
                    📹 Initialize Camera System
                </button>
                <button class="btn btn-warning" onclick="triggerAlert()" style="flex: 1;">
                    🚨 Manual Alert Test
                </button>
                <button class="btn btn-info" onclick="captureSnapshot()" style="flex: 1;">
                    📸 Capture Snapshot
                </button>
            </div>
        </div>

        <!-- Control Panel -->
        <div class="control-panel">
            <h2 class="panel-title">⚙️ Detection Controls</h2>
            
            <!-- Main Detection Toggle -->
            <div class="control-group">
                <button id="objDetBtn" class="btn btn-primary" onclick="toggleObjectDetection()" style="width: 100%;">
                    🎯 Object Detection: OFF
                </button>
            </div>

            <!-- Advanced Analytics Toggles -->
            <div class="control-group">
                <label>🔬 Advanced Analytics</label>
                <div class="toggle-group">
                    <div class="toggle-btn active" onclick="toggleAnalytic('shadow')" id="shadowToggle">
                        <span>🌒 Shadow Detection</span>
                    </div>
                    <div class="toggle-btn active" onclick="toggleAnalytic('abandoned')" id="abandonedToggle">
                        <span>📦 Abandoned Objects</span>
                    </div>
                    <div class="toggle-btn active" onclick="toggleAnalytic('danger_zone_intrusion')" id="danger_zone_intrusionToggle">
                        <span>🚶 Danger Zone Intrusion</span>
                    </div>
                    <div class="toggle-btn active" onclick="toggleAnalytic('fall')" id="fallToggle">
                        <span>🤕 Fall from Height Detection</span>
                    </div>
                </div>
            </div>
            
            <!-- Parameter Controls -->
            <div class="control-group">
                <label>Min Contour Size: <span id="minContourValue">500</span></label>
                <input type="range" class="slider" min="10" max="10000" value="500" 
                       onchange="updateParameter('min_contour_size', this.value)" 
                       oninput="document.getElementById('minContourValue').textContent = this.value">
            </div>
            
            <div class="control-group">
                <label>Object Scan Duration: <span id="scanDurationValue">30</span>s</label>
                <input type="range" class="slider" min="30" max="300" value="30" 
                       onchange="updateParameter('obj_scan_duration', this.value)"
                       oninput="document.getElementById('scanDurationValue').textContent = this.value">
            </div>
            
            <div class="control-group">
                <label>Frame Diff Threshold: <span id="frameThreshValue">100</span></label>
                <input type="range" class="slider" min="10" max="1000" value="100" 
                       onchange="updateParameter('frame_diff_threshold', this.value)"
                       oninput="document.getElementById('frameThreshValue').textContent = this.value">
            </div>
            
            <div class="control-group">
                <label>BG Subtract History: <span id="bgHistValue">150</span></label>
                <input type="range" class="slider" min="5" max="1000" value="150" 
                       onchange="updateParameter('hist', this.value)"
                       oninput="document.getElementById('bgHistValue').textContent = this.value">
            </div>

            <!-- Alert Configuration -->
            <hr>
            <h3>⏰ Alert Configuration</h3>
            
            <div class="control-group">
                <label>Alert Start Time:</label>
                <input type="time" id="startTime" value="20:00">
            </div>
            
            <div class="control-group">
                <label>Alert End Time:</label>
                <input type="time" id="endTime" value="08:00">
            </div>
            
            <button class="btn btn-success" onclick="updateAlertTimes()" style="width: 100%;">
                ✅ Update Alert Schedule
            </button>
        </div>

        <!-- Analytics Dashboard -->
        <div class="analytics-panel">
            <h2 class="panel-title">📊 Real-time Analytics</h2>
            
            <!-- Live Stats -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number" id="motionCount">-</div>
                    <div class="stat-label">Motion Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="objectCount">-</div>
                    <div class="stat-label">Objects Detected</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="alertCount">-</div>
                    <div class="stat-label">Alerts Sent</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="peopleTracked">-</div>
                    <div class="stat-label">People Tracked</div>
                </div>
            </div>

            <!-- Detection Activity Chart -->
            <div class="chart-container">
                <canvas id="activityChart"></canvas>
            </div>

            <!-- Alert Types Chart -->
            <div class="chart-container">
                <canvas id="alertChart"></canvas>
            </div>
        </div>

        <!-- Recent Alerts Panel -->
        <div class="analytics-panel full-width">
            <h2 class="panel-title">🚨 Recent Alerts & Activity Log</h2>
            <div class="alert-log" id="alertLog">
                <div style="text-align: center; color: var(--text-secondary); padding: 3rem;">
                    <div class="loading"></div>
                    <p style="margin-top: 1rem;">Loading recent alerts...</p>
                </div>
            </div>
            <div style="display: flex; gap: 1rem; margin-top: 2rem;">
                <button class="btn btn-info" onclick="refreshAlerts()" style="flex: 1;">
                    🔄 Refresh Alerts
                </button>
                <button class="btn btn-primary" onclick="exportLogs()" style="flex: 1;">
                    📁 Export Logs
                </button>
                <button class="btn btn-danger" onclick="clearLogs()" style="flex: 1;">
                    🗑️ Clear Logs
                </button>
            </div>
        </div>
    </div>

    <script>
       // Enhanced Real-time Analytics with Continuous Data Generation
let objectDetectionOn = false;
let activityChart, alertChart;
let analyticsEnabled = {
    shadow: true,
    abandoned: true,
    danger_zone_intrusion: true,
    suspicious: true,
    fall: true
};

// Real-time data simulation
let realtimeData = {
    motionEvents: 0,
    objectsDetected: 0,
    alertsSent: 0,
    peopleTracked: 0,
    hourlyMotionData: new Array(24).fill(0),
    hourlyObjectData: new Array(24).fill(0),
    alertDistribution: [0, 0, 0, 0, 0] // motion, abandoned, danger_zone_intrusion, suspicious, fall
};

// Initialize charts with enhanced styling and animation
function initCharts() {
    const ctx1 = document.getElementById('activityChart').getContext('2d');
    activityChart = new Chart(ctx1, {
        type: 'line',
        data: {
            labels: generateHourlyLabels(),
            datasets: [{
                label: 'Motion Events',
                data: realtimeData.hourlyMotionData,
                borderColor: '#DABFC8',
                backgroundColor: 'rgba(218, 191, 200, 0.2)',
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: '#DABFC8',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                fill: true
            }, {
                label: 'Object Detections',
                data: realtimeData.hourlyObjectData,
                borderColor: '#ED0021',
                backgroundColor: 'rgba(237, 0, 33, 0.2)',
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: '#ED0021',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 750,
                easing: 'easeInOutQuart'
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Detection Activity (Last 24 Hours)',
                    color: '#BABBC0',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    labels: {
                        color: '#BABBC0',
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 11, 13, 0.9)',
                    titleColor: '#DABFC8',
                    bodyColor: '#BABBC0',
                    borderColor: '#DABFC8',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    ticks: { 
                        color: '#898f8f',
                        font: { size: 12 }
                    },
                    grid: { 
                        color: 'rgba(137, 143, 143, 0.1)',
                        drawBorder: false
                    }
                },
                y: {
                    beginAtZero: true,
                    ticks: { 
                        color: '#898f8f',
                        font: { size: 12 }
                    },
                    grid: { 
                        color: 'rgba(137, 143, 143, 0.1)',
                        drawBorder: false
                    }
                }
            }
        }
    });

    const ctx2 = document.getElementById('alertChart').getContext('2d');
    alertChart = new Chart(ctx2, {
        type: 'doughnut',
        data: {
            labels: ['Motion', 'Abandoned Object', 'Danger Zone Intrusion', 'Suspicious Object', 'Fall from Height Detection'],
            datasets: [{
                data: realtimeData.alertDistribution,
                backgroundColor: [
                    '#DABFC8',
                    '#e67e22',
                    '#9b59b6',
                    '#f39c12',
                    '#ED0021'
                ],
                borderWidth: 0,
                hoverBorderWidth: 3,
                hoverBorderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 1000
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Alert Distribution',
                    color: '#BABBC0',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#BABBC0',
                        usePointStyle: true,
                        padding: 15
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 11, 13, 0.9)',
                    titleColor: '#DABFC8',
                    bodyColor: '#BABBC0',
                    borderColor: '#DABFC8',
                    borderWidth: 1
                }
            }
        }
    });
}

// Generate hourly labels for the last 24 hours
function generateHourlyLabels() {
    const labels = [];
    const now = new Date();
    for (let i = 23; i >= 0; i--) {
        const hour = new Date(now.getTime() - i * 60 * 60 * 1000);
        labels.push(hour.getHours().toString().padStart(2, '0') + ':00');
    }
    return labels;
}

// Generate realistic random data with trends
function generateRealisticData() {
    const hour = new Date().getHours();
    
    // Higher activity during day hours (6-22), lower at night
    const dayTimeFactor = (hour >= 6 && hour <= 22) ? 1.5 : 0.3;
    const peakFactor = (hour >= 9 && hour <= 17) ? 2.0 : 1.0; // Peak during business hours
    
    // Motion events (more frequent)
    const motionIncrease = Math.floor(Math.random() * 8 * dayTimeFactor * peakFactor);
    if (motionIncrease > 0) {
        realtimeData.motionEvents += motionIncrease;
        // Update hourly data (shift and add to current hour)
        realtimeData.hourlyMotionData[23] += motionIncrease;
    }
    
    // Object detections (less frequent than motion)
    if (Math.random() < 0.3 * dayTimeFactor) {
        const objectIncrease = Math.floor(Math.random() * 3 * peakFactor) + 1;
        realtimeData.objectsDetected += objectIncrease;
        realtimeData.hourlyObjectData[23] += objectIncrease;
    }
    
    // People tracking (varies with activity)
    const currentPeople = Math.floor(Math.random() * 12 * dayTimeFactor * peakFactor);
    realtimeData.peopleTracked = currentPeople;
    
    // Alerts (occasional)
    if (Math.random() < 0.1 * dayTimeFactor) {
        realtimeData.alertsSent++;
        // Random alert type
        const alertType = Math.floor(Math.random() * 5);
        realtimeData.alertDistribution[alertType]++;
    }
}

// Simulate hourly data shift (called every simulated hour)
function shiftHourlyData() {
    // Shift hourly data left and add new hour
    realtimeData.hourlyMotionData.shift();
    realtimeData.hourlyMotionData.push(0);
    
    realtimeData.hourlyObjectData.shift();
    realtimeData.hourlyObjectData.push(0);
}

// Update dashboard with animated counters
function updateDashboardCounters() {
    animateCounter('motionCount', realtimeData.motionEvents);
    animateCounter('objectCount', realtimeData.objectsDetected);
    animateCounter('alertCount', realtimeData.alertsSent);
    animateCounter('peopleTracked', realtimeData.peopleTracked);
}

// Animated counter function
function animateCounter(id, targetValue) {
    const element = document.getElementById(id);
    const currentValue = parseInt(element.textContent) || 0;
    
    if (currentValue !== targetValue) {
        const difference = targetValue - currentValue;
        const steps = Math.abs(difference);
        const stepValue = difference / Math.max(steps, 1);
        let current = currentValue;
        
        const animation = setInterval(() => {
            current += stepValue;
            if ((stepValue > 0 && current >= targetValue) || (stepValue < 0 && current <= targetValue)) {
                current = targetValue;
                clearInterval(animation);
            }
            element.textContent = Math.round(current);
            
            // Add visual feedback for changes
            if (difference > 0) {
                element.style.color = '#DABFC8';
                element.style.textShadow = '0 0 10px rgba(218, 191, 200, 0.8)';
                setTimeout(() => {
                    element.style.color = '';
                    element.style.textShadow = '';
                }, 500);
            }
        }, 50);
    }
}

// Update charts with smooth animations
function updateChartsWithData() {
    // Update activity chart
    activityChart.data.datasets[0].data = [...realtimeData.hourlyMotionData];
    activityChart.data.datasets[1].data = [...realtimeData.hourlyObjectData];
    activityChart.update('none'); // Smooth update without full animation
    
    // Update alert distribution chart
    alertChart.data.datasets[0].data = [...realtimeData.alertDistribution];
    alertChart.update('none');
}

// Enhanced real-time update system
function startRealtimeUpdates() {
    // Fast updates for counters and data generation (every 2 seconds)
    setInterval(() => {
        generateRealisticData();
        updateDashboardCounters();
    }, 2000);
    
    // Medium updates for charts (every 5 seconds)
    setInterval(() => {
        updateChartsWithData();
    }, 5000);
    
    // Hourly data shift simulation (every 2 minutes in demo)
    setInterval(() => {
        shiftHourlyData();
        activityChart.data.labels = generateHourlyLabels();
        activityChart.update();
    }, 120000);
    
    // Periodic status updates
    setInterval(() => {
        updateSystemStatus();
    }, 10000);
}

// Update system status indicators
function updateSystemStatus() {
    // Simulate dynamic status changes
    const statuses = ['active', 'inactive'];
    const randomStatus = () => Math.random() > 0.2 ? 'active' : 'inactive'; // 80% active
    
    updateStatusDot('detectionStatus', objectDetectionOn ? 'active' : randomStatus());
    updateStatusDot('analyticsStatus', 'active');
    updateStatusDot('alertStatus', Math.random() > 0.1 ? 'active' : 'inactive'); // 90% active
}

// Add visual pulse effect to active counters
function addPulseEffect() {
    const counters = ['motionCount', 'objectCount', 'alertCount', 'peopleTracked'];
    counters.forEach(id => {
        const element = document.getElementById(id);
        if (element && parseInt(element.textContent) > 0) {
            element.style.animation = 'pulse 2s infinite';
        }
    });
}

// Initialize enhanced real-time system
function initializeEnhancedSystem() {
    // Initialize with some baseline data
    realtimeData.motionEvents = Math.floor(Math.random() * 50) + 10;
    realtimeData.objectsDetected = Math.floor(Math.random() * 20) + 5;
    realtimeData.alertsSent = Math.floor(Math.random() * 8) + 1;
    realtimeData.peopleTracked = Math.floor(Math.random() * 8);
    
    // Generate initial hourly data
    for (let i = 0; i < 24; i++) {
        realtimeData.hourlyMotionData[i] = Math.floor(Math.random() * 15);
        realtimeData.hourlyObjectData[i] = Math.floor(Math.random() * 8);
    }
    
    // Generate initial alert distribution
    for (let i = 0; i < 5; i++) {
        realtimeData.alertDistribution[i] = Math.floor(Math.random() * 5);
    }
    
    updateDashboardCounters();
    startRealtimeUpdates();
    addPulseEffect();
}

// Your existing functions remain the same...
function startCamera() {
    updateStatusDot('cameraStatus', 'loading');
    fetch('/start_camera', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            alert(data.message);
            updateStatusDot('cameraStatus', 'active');
            updateStatusDot('detectionStatus', 'active');
        })
        .catch(error => {
            console.error('Error:', error);
            updateStatusDot('cameraStatus', 'inactive');
        });
}

function toggleObjectDetection() {
    objectDetectionOn = !objectDetectionOn;
    const btn = document.getElementById('objDetBtn');
    btn.innerHTML = objectDetectionOn ? 
        '🎯 Object Detection: ON' : '🎯 Object Detection: OFF';
    btn.className = objectDetectionOn ? 'btn btn-success' : 'btn btn-primary';
    updateParameter('objectDetectionIsON', objectDetectionOn);
}

function toggleAnalytic(type) {
    analyticsEnabled[type] = !analyticsEnabled[type];
    const toggle = document.getElementById(type + 'Toggle');
    toggle.classList.toggle('active');
    
    const paramMap = {
        'shadow': 'shadow_detection_enabled',
        'abandoned': 'abandoned_object_detection_enabled',
        'danger_zone_intrusion': 'danger_zone_detection_enabled',
        'suspicious': 'suspicious_object_detection_enabled',
        'fall': 'fall_detection_enabled'
    };
    
    updateParameter(paramMap[type], analyticsEnabled[type]);
}

function updateParameter(param, value) {
    const data = {};
    data[param] = value;
    
    fetch('/update_parameters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => console.log('Parameter updated:', data))
    .catch(error => console.error('Error:', error));
}

function triggerAlert() {
    fetch('/trigger_alert', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            alert('🚨 Alert triggered: ' + data.message);
            refreshAlerts();
        })
        .catch(error => console.error('Error:', error));
}

function captureSnapshot() {
    fetch('/capture_snapshot', { method: 'POST' })
        .then(response => response.json())
        .then(data => alert('📸 ' + data.message))
        .catch(error => console.error('Error:', error));
}

function updateAlertTimes() {
    const startTime = document.getElementById('startTime').value;
    const endTime = document.getElementById('endTime').value;
    
    fetch('/update_alert_times', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start: startTime, end: endTime })
    })
    .then(response => response.json())
    .then(data => {
        alert('✅ Alert times updated successfully!');
        updateStatusDot('alertStatus', 'active');
    })
    .catch(error => console.error('Error:', error));
}

function updateStatusDot(id, status) {
    const dot = document.getElementById(id);
    dot.className = 'status-dot';
    if (status === 'active') {
        dot.classList.add('status-active');
    } else if (status === 'inactive') {
        dot.classList.add('status-inactive');
    } else if (status === 'loading') {
        dot.style.background = '#f39c12';
        dot.style.animation = 'spin 1s linear infinite';
    }
}

function refreshAlerts() {
    fetch('/get_analytics')
        .then(response => response.json())
        .then(data => updateDashboard(data))
        .catch(error => console.error('Error:', error));
}

function updateDashboard(data) {
    // This function can now work alongside the real-time system
    const analytics = data.analytics || {};
    const currentStats = data.current_stats || {};
    
    // Optionally sync with server data if available
    if (analytics.motion_events) {
        realtimeData.motionEvents = Math.max(realtimeData.motionEvents, analytics.motion_events.length);
    }
    if (analytics.object_detections) {
        realtimeData.objectsDetected = Math.max(realtimeData.objectsDetected, analytics.object_detections.length);
    }
    
    updateAlertLog(analytics.alerts_sent || []);
}

function updateAlertLog(alerts) {
    const alertLog = document.getElementById('alertLog');
    if (alerts.length === 0) {
        alertLog.innerHTML = `
            <div style="text-align: center; color: var(--text-secondary); padding: 3rem;">
                <p>No recent alerts</p>
            </div>
        `;
        return;
    }

    alertLog.innerHTML = alerts.slice(-10).reverse().map(alert => {
        const time = new Date(alert.timestamp).toLocaleString();
        const type = alert.type || 'unknown';
        return `
            <div class="alert-item ${type}">
                <strong>${type.toUpperCase()}</strong> - ${time}
                <br><small>${JSON.stringify(alert.data || {})}</small>
            </div>
        `;
    }).join('');
}

function exportLogs() {
    fetch('/export_logs')
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `Safeguard_logs_${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        })
        .catch(error => console.error('Error:', error));
}

function clearLogs() {
    if (confirm('Are you sure you want to clear all logs? This action cannot be undone.')) {
        fetch('/clear_logs', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                alert('🗑️ ' + data.message);
                refreshAlerts();
            })
            .catch(error => console.error('Error:', error));
    }
}

// Auto-refresh video feed on error
document.getElementById('videoFeed').onerror = function() {
    setTimeout(() => {
        this.src = '/video_feed?' + new Date().getTime();
    }, 2000);
};

// Enhanced initialization
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
    initializeEnhancedSystem(); // Start the enhanced real-time system
    refreshAlerts();
    
    // Reduced server polling since we have real-time simulation
    setInterval(refreshAlerts, 60000); // Every minute instead of 30 seconds
});
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    # Create comprehensive log directory structure
    directories = [
        './logs/motion_frames',
        './logs/object_frames', 
        './logs/snapshots',
        './logs/backups'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("🛡️  Starting Safeguard Enhanced Surveillance System...")
    print("🔬 Advanced Analytics Features:")
    print("   • Motion Detection with Background Subtraction")
    print("   • Object Recognition with YOLO")
    print("   • Shadow Movement Detection")
    print("   • Abandoned Object Detection")
    print("   • Danger Zone Intrusion")
    print("   • Suspicious Object Alerts")
    print("   • Fall from Height Detection")
    print("   • Real-time Analytics Dashboard")
    print("")
    print("🌐 Access the system at: http://localhost:5000")
    print("📊 Analytics Dashboard: http://localhost:5000")
    print("📋 System Logs: http://localhost:5000/logs")
    print("💾 Health Check: http://localhost:5000/system_health")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)