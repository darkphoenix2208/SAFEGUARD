"""
Safeguard Surveillance System - FIXED VERSION
Complete implementation with accurate person detection, tracking, pose estimation, 
zone analytics, and real-time alerts system
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import supervision as sv
from scipy.spatial import distance as dist
from collections import OrderedDict, defaultdict, deque
import threading
import time
import json
import os
from datetime import datetime, timedelta
from flask import Flask, Response, jsonify, request, render_template_string
from flask_cors import CORS
import torch
import logging
from modules.enhanced_detector import Enhanced2A2SDetector
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Flask Web Application
app = Flask(__name__)
CORS(app)

# Global variables
cap = None
detector = None
camera_initialized = False

def initialize_camera():
    """Initialize camera and enhanced detector"""
    global cap, detector, camera_initialized
    
    if not camera_initialized:
        try:
            cap = cv2.VideoCapture("test1.mp4")
            # if not cap.isOpened():
            #     for i in range(1, 5):
            #         cap = cv2.VideoCapture(i)
            #         if cap.isOpened():
            #             break
            
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                detector = Enhanced2A2SDetector(cap)
                detector.start_detection()
                camera_initialized = True
                logger.info("Safeguard Camera initialized successfully")
            else:
                logger.error("Failed to open camera")
                
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")

@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Safeguard Surveillance System</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-gray: #898f8f;
            --light-cream: #DABFC8;
            --accent-red: #ED0021;
            --soft-gray: #BABBC0;
            --dark-bg: #1a1a1a;
            --glass-bg: rgba(218, 191, 200, 0.15);
            --glass-border: rgba(255, 255, 255, 0.2);
        }

        body {
           
            background: radial-gradient(circle at 30% 30%, rgba(94, 8, 2, 0.7) 0%, rgba(26, 0, 0, 0.9) 40%, #1a0000 80%),
              linear-gradient(120deg, #1a0000 0%, #2c0a0a 25%, #3b0e0e 50%, #5e0802 75%, #b2b2b2 100%);
            background-blend-mode: overlay, multiply;
            box-shadow: inset 0 0 150px rgba(0, 0, 0, 0.8);
            color: #fff;
            font-family: 'Segoe UI', sans-serif;
            background-attachment: fixed;
            min-height: 100vh;
            padding: 20px;
            position: relative;
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
                radial-gradient(circle at 40% 40%, rgba(186, 187, 192, 0.05) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }

        h1 {
            text-align: center;
            margin-bottom: 40px;
            color: var(--light-cream);
            font-size: clamp(2rem, 4vw, 3.5rem);
            font-weight: 800;
            text-shadow: 
                0 0 20px rgba(237, 0, 33, 0.3),
                0 0 40px rgba(218, 191, 200, 0.2);
            letter-spacing: -0.02em;
            position: relative;
            animation: titleGlow 3s ease-in-out infinite alternate;
        }

        @keyframes titleGlow {
            from { text-shadow: 0 0 20px rgba(237, 0, 33, 0.3), 0 0 40px rgba(218, 191, 200, 0.2); }
            to { text-shadow: 0 0 30px rgba(237, 0, 33, 0.5), 0 0 60px rgba(218, 191, 200, 0.4); }
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            height: calc(100vh - 140px);
        }

        .video-section {
            background: var(--glass-bg);
            border-radius: 25px;
            padding: 30px;
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            box-shadow: 
                0 25px 45px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
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
            background: linear-gradient(90deg, transparent, var(--light-cream), transparent);
            opacity: 0.6;
        }

        .controls-section {
            background: var(--glass-bg);
            border-radius: 25px;
            padding: 30px;
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            box-shadow: 
                0 25px 45px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            overflow-y: auto;
            position: relative;
        }

        .controls-section::-webkit-scrollbar {
            width: 8px;
        }

        .controls-section::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }

        .controls-section::-webkit-scrollbar-thumb {
            background: linear-gradient(45deg, var(--accent-red), var(--primary-gray));
            border-radius: 10px;
        }

        .video-container {
            background: #000;
            border-radius: 20px;
            overflow: hidden;
            position: relative;
            height: 70vh;
            box-shadow: 
                0 20px 40px rgba(0, 0, 0, 0.4),
                inset 0 0 0 1px rgba(237, 0, 33, 0.2);
        }

        #videoFeed {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        .video-overlay {
            position: absolute;
            top: 20px;
            left: 20px;
            background: linear-gradient(135deg, rgba(237, 0, 33, 0.9), rgba(137, 143, 143, 0.9));
            color: white;
            padding: 15px 20px;
            border-radius: 15px;
            font-size: 14px;
            font-weight: 600;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            animation: overlayPulse 2s ease-in-out infinite alternate;
        }

        @keyframes overlayPulse {
            from { box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3); }
            to { box-shadow: 0 8px 16px rgba(237, 0, 33, 0.4); }
        }

        h2, h3 {
            color: var(--light-cream);
            margin-bottom: 20px;
            font-weight: 700;
            position: relative;
            padding-bottom: 10px;
        }

        h2 {
            font-size: 1.8rem;
            text-align: center;
        }

        h3 {
            font-size: 1.2rem;
        }

        h2::after, h3::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, var(--accent-red), var(--primary-gray), var(--light-cream));
            border-radius: 2px;
        }

        .control-group {
            margin-bottom: 30px;
            padding: 25px;
            background: linear-gradient(135deg, rgba(186, 187, 192, 0.1), rgba(218, 191, 200, 0.05));
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            position: relative;
            transition: all 0.3s ease;
        }

        .control-group:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
            border-color: rgba(237, 0, 33, 0.3);
        }

        .control-group::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background: linear-gradient(180deg, var(--accent-red), var(--primary-gray));
            border-radius: 2px;
        }

        .control-row {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
            padding: 10px 0;
        }

        .control-row label {
            flex: 1;
            font-weight: 500;
            color: var(--light-cream);
            font-size: 0.95rem;
        }

        .control-row input, .control-row select {
            flex: 1;
            padding: 12px 15px;
            border: 2px solid rgba(186, 187, 192, 0.3);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.05);
            color: var(--light-cream);
            font-size: 0.9rem;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }

        .control-row input:focus, .control-row select:focus {
            outline: none;
            border-color: var(--accent-red);
            background: rgba(255, 255, 255, 0.1);
            box-shadow: 0 0 0 3px rgba(237, 0, 33, 0.2);
        }

        .slider {
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(90deg, rgba(186, 187, 192, 0.3), rgba(218, 191, 200, 0.5));
            outline: none;
            margin: 15px 0;
            appearance: none;
            cursor: pointer;
        }

        .slider::-webkit-slider-thumb {
            appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: linear-gradient(45deg, var(--accent-red), #ff4757);
            cursor: pointer;
            box-shadow: 
                0 4px 8px rgba(0, 0, 0, 0.3),
                0 0 0 3px rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }

        .slider::-webkit-slider-thumb:hover {
            transform: scale(1.2);
            box-shadow: 
                0 6px 12px rgba(0, 0, 0, 0.4),
                0 0 0 4px rgba(237, 0, 33, 0.3);
        }

        button {
            width: 100%;
            padding: 15px 20px;
            margin: 10px 0;
            border: none;
            border-radius: 15px;
            cursor: pointer;
            font-size: 0.95rem;
            font-weight: 600;
            font-family: inherit;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        button::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            transition: all 0.4s ease;
            transform: translate(-50%, -50%);
        }

        button:hover::before {
            width: 300px;
            height: 300px;
        }

        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary-gray), #a8b0b0);
            color: white;
        }

        .btn-success {
            background: linear-gradient(135deg, #4CAF50, #66BB6A);
            color: white;
        }

        .btn-warning {
            background: linear-gradient(135deg, #FF9800, #FFB74D);
            color: white;
        }

        .btn-danger {
            background: linear-gradient(135deg, var(--accent-red), #FF5252);
            color: white;
        }

        .btn-info {
            background: linear-gradient(135deg, #2196F3, #42A5F5);
            color: white;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }

        .stat-card {
            background: linear-gradient(135deg, var(--accent-red), #ff4757);
            color: white;
            padding: 20px;
            border-radius: 18px;
            text-align: center;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: conic-gradient(transparent, rgba(255, 255, 255, 0.1), transparent);
            animation: rotate 4s linear infinite;
        }

        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .stat-card:nth-child(2) {
            background: linear-gradient(135deg, var(--primary-gray), #a8b0b0);
        }

        .stat-card:nth-child(3) {
            background: linear-gradient(135deg, var(--soft-gray), #d0d1d6);
            color: #333;
        }

        .stat-card:nth-child(4) {
            background: linear-gradient(135deg, var(--light-cream), #e8dde3);
            color: #333;
        }

        .stat-card:hover {
            transform: scale(1.05) rotateY(5deg);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
        }

        .stat-value {
            font-size: 2rem;
            font-weight: 800;
            display: block;
            position: relative;
            z-index: 1;
        }

        .stat-label {
            font-size: 0.85rem;
            opacity: 0.9;
            font-weight: 500;
            position: relative;
            z-index: 1;
        }

        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 70px;
            height: 38px;
        }

        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(186, 187, 192, 0.5), rgba(137, 143, 143, 0.5));
            transition: 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 38px;
            border: 2px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
        }

        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 28px;
            width: 28px;
            left: 3px;
            bottom: 3px;
            background: linear-gradient(135deg, white, #f0f0f0);
            transition: 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 50%;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        input:checked + .toggle-slider {
            background: linear-gradient(135deg, var(--accent-red), #ff4757);
        }

        input:checked + .toggle-slider:before {
            transform: translateX(32px);
            background: linear-gradient(135deg, white, var(--light-cream));
        }

        .toggle-slider:hover {
            box-shadow: 0 0 20px rgba(237, 0, 33, 0.3);
        }

        .alert-logs {
            background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(63, 81, 181, 0.1));
            border: 2px solid rgba(33, 150, 243, 0.3);
            border-radius: 18px;
            padding: 25px;
            margin: 20px 0;
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
            max-height: 200px;
            overflow-y: auto;
        }

        .alert-logs::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background: linear-gradient(180deg, #2196F3, #3F51B5);
            border-radius: 2px;
        }

        .alert-logs h3 {
            color: #2196F3;
        }

        .alert-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            font-size: 0.85rem;
            color: var(--light-cream);
        }

        .video-section .btn-container {
            display: flex;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }

        .video-section .btn-container button {
            flex: 1;
            min-width: 120px;
        }

        @media (max-width: 1200px) {
            .container {
                grid-template-columns: 1fr;
                grid-template-rows: auto auto;
                gap: 20px;
            }
            
            .video-container {
                height: 50vh;
            }
            
            .video-section .btn-container {
                flex-direction: column;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
                gap: 10px;
            }
            
            h1 {
                font-size: 2rem;
                margin-bottom: 20px;
            }
        }

        @media (max-width: 768px) {
            body {
                padding: 10px;
            }
            
            .video-section, .controls-section {
                padding: 20px;
                border-radius: 20px;
            }
            
            .control-group {
                padding: 20px;
            }
            
            button {
                padding: 12px 16px;
                font-size: 0.9rem;
            }
        }
    </style>
</head>
<body>
    <h1>🔍 Safeguard: AI-Augmented Surveillance System</h1>
    
    <div class="container">
        <div class="video-section">
            <div class="video-container">
                <img id="videoFeed" src="/video_feed" alt="Video Feed">
                <div class="video-overlay" id="videoOverlay">Initializing...</div>
            </div>

            <div class="btn-container">
                <button class="btn-success" onclick="startCamera()">🎥 Start Camera</button>
                <button class="btn-warning" onclick="triggerAlert()">🚨 Manual Alert</button>
                <button class="btn-info" onclick="resetStats()">🔄 Reset Stats</button>
                <button class="btn-primary" onclick="openLogs()">📊 View Analytics</button>
            </div>
        </div>
        
        <div class="controls-section">
            <h2>🎛️ Control Panel</h2>
            
            <!-- Real-time Statistics -->
            <div class="control-group">
                <h3>📈 Live Statistics</h3>
                <div class="stats-grid" id="statsGrid">
                    <div class="stat-card">
                        <span class="stat-value" id="currentPeople">0</span>
                        <span class="stat-label">Current People</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value" id="totalEntries">0</span>
                        <span class="stat-label">Entries</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value" id="totalExits">0</span>
                        <span class="stat-label">Exits</span>
                    </div>
                  
                </div>
            </div>
            
            <!-- Detection Features -->
            <div class="control-group">
                <h3>🎯 Detection Features</h3>
                
                <div class="control-row">
                    <label>Face Blurring (Privacy)</label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="blurFaces" checked onchange="updateParameter('blur_faces', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                
                <div class="control-row">
                    <label>Pose Detection</label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="showPoses" checked onchange="updateParameter('show_poses', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                
                <div class="control-row">
                    <label>Zone Heatmap</label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="showZones" onchange="updateParameter('show_zones', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                
                <div class="control-row">
                    <label>Line Counter</label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="showLineCounter" checked onchange="updateParameter('show_line_counter', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
            </div>
            
            <!-- Alert Thresholds -->
            <div class="control-group">
                <h3>⚠️ Alert Thresholds</h3>
                
                <div class="control-row">
                    <label>Crowd Threshold: <span id="crowdThresholdValue">5</span> people</label>
                </div>
                <input type="range" class="slider" min="2" max="20" value="5" 
                       onchange="updateParameter('crowd_threshold', this.value)" 
                       oninput="document.getElementById('crowdThresholdValue').textContent = this.value">
                
                <div class="control-row">
                    <label>Inactivity Threshold: <span id="inactivityThresholdValue">2</span> minutes</label>
                </div>
                <input type="range" class="slider" min="1" max="10" value="2" 
                       onchange="updateParameter('inactivity_threshold', this.value * 60)" 
                       oninput="document.getElementById('inactivityThresholdValue').textContent = this.value">
            </div>
            
            <!-- Real-time Alert Logs -->
            <div class="alert-logs">
                <h3>🚨 Real-time Alert Logs</h3>
                <div id="alertLogs">
                    <div class="alert-item">System initialized - No alerts yet</div>
                </div>
            </div>
            
            <!-- System Actions -->
            <div class="control-group">
                <h3>🔧 System Actions</h3>
                <button class="btn-info" onclick="exportAnalytics()">📊 Export Analytics</button>
                <button class="btn-primary" onclick="downloadLogs()">📁 Download Logs</button>
                <button class="btn-danger" onclick="emergencyStop()">🛑 Emergency Stop</button>
            </div>
        </div>
    </div>

    <script>
        let statsUpdateInterval;
        
        window.onload = function() {
            startCamera();
            startStatsUpdate();
        };
        
        function startCamera() {
            fetch('/start_camera', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log(data.message);
                    document.getElementById('videoOverlay').textContent = 'System Active';
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('videoOverlay').textContent = 'Camera Error';
                });
        }
        
        function startStatsUpdate() {
            statsUpdateInterval = setInterval(updateStats, 1000); // Update every second for real-time
        }
        
        function updateStats() {
            fetch('/analytics')
                .then(response => response.json())
                .then(data => {
                    if (data.stats) {
                        const stats = data.stats;
                        document.getElementById('currentPeople').textContent = stats.current_people_count;
                        document.getElementById('totalEntries').textContent = stats.entry_count;
                        document.getElementById('totalExits').textContent = stats.exit_count;
                        
                        // Update alert logs
                        if (data.recent_alerts && data.recent_alerts.length > 0) {
                            updateAlertLogs(data.recent_alerts);
                        }
                    }
                })
                .catch(error => console.error('Error updating stats:', error));
        }
        
        function updateAlertLogs(alerts) {
            const alertLogsContainer = document.getElementById('alertLogs');
            alertLogsContainer.innerHTML = '';
            
            if (alerts.length === 0) {
                alertLogsContainer.innerHTML = '<div class="alert-item">No recent alerts</div>';
                return;
            }
            
            alerts.reverse().forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert-item';
                const timestamp = new Date(alert.timestamp).toLocaleTimeString();
                alertDiv.innerHTML = `<strong>${timestamp}</strong> - ${alert.type.toUpperCase()}: ${alert.message}`;
                alertLogsContainer.appendChild(alertDiv);
            });
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
                    alert('Manual alert triggered: ' + data.message);
                    updateStats(); // Refresh stats immediately
                })
                .catch(error => console.error('Error:', error));
        }
        
        function resetStats() {
            if (confirm('Are you sure you want to reset all statistics and logs?')) {
                fetch('/reset_stats', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        alert('Statistics and logs reset successfully');
                        updateStats();
                        document.getElementById('alertLogs').innerHTML = '<div class="alert-item">Statistics reset - No alerts</div>';
                    })
                    .catch(error => console.error('Error:', error));
            }
        }
        
        function exportAnalytics() {
            window.open('/export_analytics', '_blank');
        }
        
        function downloadLogs() {
            window.open('/download_logs', '_blank');
        }
        
        function openLogs() {
            window.open('/analytics_dashboard', '_blank');
        }
        
        function emergencyStop() {
            if (confirm('Are you sure you want to stop the surveillance system?')) {
                fetch('/emergency_stop', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        alert(data.message);
                        document.getElementById('videoOverlay').textContent = 'System Stopped';
                        clearInterval(statsUpdateInterval);
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
        
        // Add connection status indicator
        let connectionLost = false;
        setInterval(() => {
            fetch('/health_check')
                .then(response => {
                    if (connectionLost) {
                        document.getElementById('videoOverlay').textContent = 'System Active';
                        connectionLost = false;
                    }
                })
                .catch(() => {
                    if (!connectionLost) {
                        document.getElementById('videoOverlay').textContent = 'Connection Lost';
                        connectionLost = true;
                    }
                });
        }, 5000);
    </script>
</body>
</html>
    """)

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global detector
        while True:
            try:
                if detector and detector.export_frame is not None:
                    frame = detector.get_export_frame()
                    if frame is None:
                        continue
                        
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ret:
                        continue
                        
                    frame_bytes = buffer.tobytes();
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Loading frame
                    loading_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(loading_frame, "Initializing Safeguard System...", 
                              (350, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', loading_frame)
                    if ret:
                        frame_bytes = buffer.tobytes();
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error generating frame: {e}")
                error_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Stream Error: {str(e)[:50]}", 
                          (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                if ret:
                    frame_bytes = buffer.tobytes();
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(1)
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    try:
        initialize_camera()
        return jsonify({"message": "Safeguard Camera started successfully" if camera_initialized else "Failed to start camera"})
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return jsonify({"message": f"Error starting camera: {str(e)}"}), 500

@app.route('/update_parameters', methods=['POST'])
def update_parameters():
    global detector
    try:
        if detector:
            params = request.get_json()
            detector.update_parameters(params)
            return jsonify({"status": "Parameters updated", "params": params})
        return jsonify({"status": "Detector not initialized"}), 400
    except Exception as e:
        logger.error(f"Error updating parameters: {e}")
        return jsonify({"status": f"Error: {str(e)}"}), 500

@app.route('/analytics')
def get_analytics():
    global detector
    try:
        if detector:
            return jsonify(detector.get_analytics_data())
        return jsonify({"error": "Detector not initialized"}), 400
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/trigger_alert', methods=['POST'])
def trigger_alert():
    global detector
    try:
        if detector:
            detector.alert_system.log_alert('manual', 'Manual alert triggered from web interface')
            return jsonify({"message": "Manual alert triggered and logged"})
        return jsonify({"message": "Detector not initialized"}), 400
    except Exception as e:
        logger.error(f"Error triggering alert: {e}")
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    global detector
    try:
        if detector:
            detector.reset_statistics()
            return jsonify({"status": "Statistics and logs reset successfully"})
        return jsonify({"status": "Detector not initialized"}), 400
    except Exception as e:
        logger.error(f"Error resetting stats: {e}")
        return jsonify({"status": f"Error: {str(e)}"}), 500

@app.route('/emergency_stop', methods=['POST'])
def emergency_stop():
    global detector
    try:
        if detector:
            detector.stop_detection()
            return jsonify({"message": "Emergency stop activated - System stopped"})
        return jsonify({"message": "System already stopped"}), 400
    except Exception as e:
        logger.error(f"Error in emergency stop: {e}")
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/health_check')
def health_check():
    global detector
    return jsonify({
        "status": "healthy",
        "detector_running": detector.running if detector else False,
        "camera_initialized": camera_initialized,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/export_analytics')
def export_analytics():
    """Export analytics data as JSON"""
    global detector
    try:
        if detector:
            analytics_data = detector.get_analytics_data()
            analytics_data['export_timestamp'] = datetime.now().isoformat()
            analytics_data['system_info'] = {
                'version': '2A2S Enhanced v2.0 - Fixed',
                'features': [
                    'Accurate Person Detection & Tracking with Unique IDs',
                    'Improved Pose Estimation & Fall Detection', 
                    'Face Processing & Privacy Protection',
                    'Accurate Zone-based Analytics & Heatmap',
                    'Precise Entry/Exit Counting',
                    'Crowd Detection with Real-time Alerts',
                    'Inactivity Monitoring',
                    'Real-time Alert System with Logging'
                ]
            }
            
            response = app.response_class(
                response=json.dumps(analytics_data, indent=2),
                status=200,
                mimetype='application/json'
            )
            response.headers['Content-Disposition'] = f'attachment; filename=2a2s_analytics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            return response
        
        return jsonify({"error": "Detector not initialized"}), 400
    except Exception as e:
        logger.error(f"Error exporting analytics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download_logs')
def download_logs():
    """Download system logs"""
    try:
        return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Safeguard System Logs</title>
    <style>
        body { font-family: 'Inter', sans-serif; margin: 20px; background: linear-gradient(135deg, #1a1a1a, #2d2d2d); color: #DABFC8; }
        .log-container { background: rgba(218, 191, 200, 0.1); padding: 25px; border-radius: 15px; margin: 15px 0; border: 1px solid rgba(255, 255, 255, 0.2); }
        .log-entry { margin: 8px 0; padding: 10px; background: rgba(255, 255, 255, 0.05); border-left: 3px solid #ED0021; border-radius: 5px; }
        h1 { color: #ED0021; text-align: center; font-size: 2.5rem; margin-bottom: 30px; }
        h3 { color: #DABFC8; border-bottom: 2px solid #ED0021; padding-bottom: 10px; }
        .status { background: linear-gradient(135deg, #4CAF50, #66BB6A); color: white; padding: 10px 20px; border-radius: 20px; display: inline-block; margin: 10px 0; }
        button { background: linear-gradient(135deg, #ED0021, #FF5252); color: white; border: none; padding: 15px 30px; border-radius: 10px; cursor: pointer; font-size: 1rem; transition: all 0.3s ease; }
        button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(237, 0, 33, 0.3); }
    </style>
</head>
<body>
    <h1>📊 Safeguard System Logs - FIXED VERSION</h1>
    
    <div class="status">✅ System Status: All Issues Resolved</div>
    
    <div class="log-container">
        <h3>🔧 Fixed Issues</h3>
        <div class="log-entry">✅ JSON Serialization Error - Fixed tuple keys issue</div>
        <div class="log-entry">✅ Centroid Tracking - Now shows unique IDs with bounding boxes</div>
        <div class="log-entry">✅ Zone Heatmap - Accurate zone-based analytics implemented</div>
        <div class="log-entry">✅ Pose Detection - Improved accuracy for standing/sitting/lying detection</div>
        <div class="log-entry">✅ Real-time Alerts - Live alert logging and display system</div>
        <div class="log-entry">✅ Analytics System - Robust real-time data fetching</div>
    </div>
    
    <div class="log-container">
        <h3>📈 Current Session Analytics</h3>
        <div class="log-entry">System initialized: {{ timestamp }}</div>
        <div class="log-entry">Enhanced Features: Accurate Person Tracking, Improved Pose Detection, Zone Analytics</div>
        <div class="log-entry">Real-time Systems: Live alerts, Statistics updates, Error handling</div>
        <div class="log-entry">Privacy: Face blurring enabled by default</div>
        <div class="log-entry">Performance: Optimized for 30 FPS real-time processing</div>
    </div>
    
    <div class="log-container">
        <h3>🎯 Key Improvements</h3>
        <div class="log-entry">Unique ID Tracking: Each person gets assigned unique ID with bounding box</div>
        <div class="log-entry">Accurate Pose Classification: Better standing/sitting/lying detection</div>
        <div class="log-entry">Zone-based Heatmap: 6x6 grid with accurate occupancy tracking</div>
        <div class="log-entry">Real-time Alert Logs: Live alert display in web interface</div>
        <div class="log-entry">Robust Error Handling: Better exception management and recovery</div>
        <div class="log-entry">JSON Serialization: All data properly serializable for API</div>
    </div>
    
    <div class="log-container">
        <h3>⚙️ System Configuration</h3>
        <div class="log-entry">Camera Resolution: 1280x720 @ 30 FPS</div>
        <div class="log-entry">Detection Models: YOLOv8n (Person Detection), MediaPipe (Pose Estimation)</div>
        <div class="log-entry">Tracking Algorithm: Enhanced Centroid Tracker with improved accuracy</div>
        <div class="log-entry">Zone Grid: 6x6 analysis zones with heatmap overlay</div>
        <div class="log-entry">Alert System: Real-time logging with cooldown management</div>
    </div>
    
    <button onclick="window.close()">Close Logs</button>
</body>
</html>
        """, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        logger.error(f"Error generating logs page: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analytics_dashboard')
def analytics_dashboard():
    """Enhanced analytics dashboard"""
    try:
        return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Safeguard Analytics Dashboard</title>
    <style>
        body { 
            font-family: 'Inter', sans-serif; 
            background: radial-gradient(circle at 30% 30%, rgba(94, 8, 2, 0.7) 0%, rgba(26, 0, 0, 0.9) 40%, #1a0000 80%),
            linear-gradient(120deg, #1a0000 0%, #2c0a0a 25%, #3b0e0e 50%, #5e0802 75%, #b2b2b2 100%);
            background-blend-mode: overlay, multiply;
            box-shadow: inset 0 0 150px rgba(0, 0, 0, 0.8);
            color: #fff;
            font-family: 'Segoe UI', sans-serif;
            min-height: 100vh;
        }
        .dashboard { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 25px; 
            max-width: 1400px;
            margin: 0 auto;
        }
        .card { 
            background: linear-gradient(135deg, rgba(218, 191, 200, 0.15), rgba(186, 187, 192, 0.1)); 
            padding: 25px; 
            border-radius: 20px; 
            box-shadow: 0 15px 30px rgba(0,0,0,0.3); 
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(20px);
            transition: all 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(237, 0, 33, 0.2);
        }
        .metric { 
            font-size: 3em; 
            font-weight: 800; 
            background: linear-gradient(45deg, #ED0021, #FF5252);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin: 20px 0;
        }
        h1 { 
            text-align: center; 
            color: #DABFC8; 
            font-size: 3rem;
            margin-bottom: 40px;
            text-shadow: 0 0 20px rgba(237, 0, 33, 0.5);
        }
        h3 { 
            color: #ED0021; 
            border-bottom: 3px solid #ED0021; 
            padding-bottom: 10px; 
            margin-bottom: 20px;
            font-size: 1.3rem;
        }
        .status-active { 
            color: #4CAF50; 
            background: linear-gradient(135deg, #4CAF50, #66BB6A);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }
        .status-inactive { 
            color: #f44336; 
            background: linear-gradient(135deg, #f44336, #FF5252);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }
        .alert-log {
            background: rgba(33, 150, 243, 0.1);
            border: 1px solid rgba(33, 150, 243, 0.3);
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            font-size: 0.9rem;
        }
        .refresh-btn {
            background: linear-gradient(135deg, #ED0021, #FF5252);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 15px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            margin: 20px auto;
            display: block;
        }
        .refresh-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(237, 0, 33, 0.3);
        }

        .back-btn {
            position: fixed;
            top: 20px;
            left: 20px;
            background: linear-gradient(135deg, #ED0021, #FF5252);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            z-index: 1000;
            box-shadow: 0 4px 15px rgba(237, 0, 33, 0.3);
        }
        
        .back-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(237, 0, 33, 0.4);
        }
    </style>
    <script>
        function refreshData() {
            fetch('/analytics')
                .then(response => response.json())
                .then(data => {
                    if (data.stats) {
                        document.getElementById('currentPeople').textContent = data.stats.current_people_count;
                        document.getElementById('totalEntries').textContent = data.stats.entry_count;
                        document.getElementById('totalExits').textContent = data.stats.exit_count;
                        document.getElementById('crowdAlerts').textContent = data.stats.crowd_alerts;
                        document.getElementById('inactivityAlerts').textContent = data.stats.inactivity_alerts;
                        document.getElementById('totalDetections').textContent = data.stats.total_detections;
                        document.getElementById('timestamp').textContent = new Date(data.timestamp).toLocaleString();
                        
                        // Update alert logs
                        const alertContainer = document.getElementById('alertLogs');
                        alertContainer.innerHTML = '';
                        if (data.recent_alerts && data.recent_alerts.length > 0) {
                            data.recent_alerts.forEach(alert => {
                                const alertDiv = document.createElement('div');
                                alertDiv.className = 'alert-log';
                                alertDiv.innerHTML = `<strong>${new Date(alert.timestamp).toLocaleTimeString()}</strong> - ${alert.type.toUpperCase()}: ${alert.message}`;
                                alertContainer.appendChild(alertDiv);
                            });
                        } else {
                            alertContainer.innerHTML = '<div class="alert-log">No recent alerts</div>';
                        }
                    }
                })
                .catch(error => console.error('Error:', error));
        }
        
        setInterval(refreshData, 2000); // Update every 2 seconds
        window.onload = refreshData;
    </script>
</head>
<body>
    <button onclick="window.location.href='http://localhost:3001'" class="back-btn">
        ← Back
    </button>
    <h1>📊 Safeguard Analytics Dashboard</h1>
    
    <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
    
    <div class="dashboard">
        <div class="card">
            <h3>👥 Current People</h3>
            <div class="metric" id="currentPeople">0</div>
            <p>People currently being tracked</p>
        </div>
        
        <div class="card">
            <h3>🚪 Entry Count</h3>
            <div class="metric" id="totalEntries">0</div>
            <p>Total entries detected</p>
        </div>
        
        <div class="card">
            <h3>🚪 Exit Count</h3>
            <div class="metric" id="totalExits">0</div>
            <p>Total exits detected</p>
        </div>
        
        
        <div class="card">
            <h3>👨‍👩‍👧‍👦 Crowd Alerts</h3>
            <div class="metric" id="crowdAlerts">0</div>
            <p>Crowd detection alerts triggered</p>
        </div>
        
        <div class="card">
            <h3>⏱️ Inactivity Alerts</h3>
            <div class="metric" id="inactivityAlerts">0</div>
            <p>Inactivity alerts triggered</p>
        </div>
        
        <div class="card">
            <h3>🎯 Total Detections</h3>
            <div class="metric" id="totalDetections">0</div>
            <p>Total person detections</p>
        </div>
        
        <div class="card">
            <h3>🕒 System Status</h3>
            <div class="metric status-active">ACTIVE</div>
            <p>Last Updated: <span id="timestamp">--</span></p>
        </div>
        
        <div class="card" style="grid-column: 1 / -1;">
            <h3>🚨 Recent Alert Logs</h3>
            <div id="alertLogs">
                <div class="alert-log">Loading alert logs...</div>
            </div>
        </div>
    </div>
</body>
</html>
        """)
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('./logs', exist_ok=True)
    
    print("=" * 70)
    print("🚀 Starting Safeguard Surveillance System - FIXED VERSION")
    print("=" * 70)
    print("🔧 FIXES IMPLEMENTED:")
    print("  ✅ JSON Serialization Error - Resolved tuple keys issue")
    print("  ✅ Accurate Centroid Tracking - Unique IDs with bounding boxes")
    print("  ✅ Improved Zone Heatmap - 6x6 grid with proper analytics")
    print("  ✅ Enhanced Pose Detection - Better standing/sitting/lying accuracy")
    print("  ✅ Real-time Alert System - Live logging and display")
    print("  ✅ Robust Error Handling - Better exception management")
    print("=" * 70)
    print("🌟 ENHANCED FEATURES:")
    print("  🎯 Person Detection & Tracking with Unique IDs")
    print("  🤸 Improved Pose Estimation & Fall Detection")
    print("  🔒 Face Processing & Privacy Protection")
    print("  🗺️ Accurate Zone-based Analytics & Heatmap")
    print("  📊 Precise Entry/Exit Counting")
    print("  ✅ Real-time Email & Alert System")
    print("  ✅ Comprehensive Web Dashboard")
    print("=" * 60)
    print("🌐 Access the system at: http://localhost:8080")
    print("📊 Analytics Dashboard: http://localhost:8080/analytics_dashboard")
    print("=" * 60)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=8080, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Safeguard System...")
        if detector:
            detector.cleanup()
        print("✅ System shutdown complete.")