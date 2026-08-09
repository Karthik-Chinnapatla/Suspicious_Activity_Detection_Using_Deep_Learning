import os
import cv2
import numpy as np
from django.conf import settings
from .models import Detection

def get_dashboard_statistics():
    """
    Computes real summary statistics based on SQLite database records.
    Returns: dict of stats (total, suspicious, normal, latest_detection, activity_breakdown).
    """
    all_detections = Detection.objects.all()
    total_count = all_detections.count()
    suspicious_count = all_detections.filter(is_suspicious=True).count()
    normal_count = all_detections.filter(is_suspicious=False).count()
    latest = all_detections.first()

    # Category breakdown for charts / dashboard cards
    activity_counts = {}
    for d in all_detections:
        activity_counts[d.activity] = activity_counts.get(d.activity, 0) + 1

    return {
        'total_count': total_count,
        'suspicious_count': suspicious_count,
        'normal_count': normal_count,
        'latest': latest,
        'activity_breakdown': activity_counts,
    }


def create_demo_surveillance_video(output_path, activity_type="Fighting", duration_sec=4, fps=25):
    """
    Generates a synthetic surveillance video for immediate testing & evaluation.
    Outputs a valid MP4 file with sample CCTV footage animation.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 480
    total_frames = duration_sec * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    bg_color = (25, 30, 35)
    
    for i in range(total_frames):
        # Dark security camera background
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # Grid lines simulating CCTV grid
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (40, 45, 50), 1)
        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height), (40, 45, 50), 1)

        # Dynamic motion shapes
        cx = int(width / 2 + np.sin(i / 10.0) * 150)
        cy = int(height / 2 + np.cos(i / 10.0) * 80)
        
        if activity_type.upper() in ['FIGHTING', 'BURGLARY', 'SHOOTING']:
            # Fast red flashing circle
            color = (30, 30, 220) if (i % 6 < 3) else (60, 60, 255)
            cv2.circle(frame, (cx, cy), 45, color, -1)
            cv2.putText(frame, f"TEST SURVEILLANCE: {activity_type.upper()}", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif activity_type.upper() == 'FIRE':
            # Expanding orange/yellow fire simulation
            cv2.circle(frame, (cx, cy), 30 + (i % 25), (0, 140, 255), -1)
            cv2.putText(frame, "TEST SURVEILLANCE: FIRE HAZARD", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Smooth green normal motion
            cv2.circle(frame, (cx, cy), 35, (50, 200, 50), -1)
            cv2.putText(frame, "TEST SURVEILLANCE: NORMAL ACTIVITY", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # CCTV timestamp stamp
        cv2.putText(frame, f"CAM-01 LIVE | FRAME {i+1}/{total_frames}", (40, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        out.write(frame)

    out.release()
    return output_path
