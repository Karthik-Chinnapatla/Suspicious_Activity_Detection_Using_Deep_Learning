import os
import time
import cv2
import numpy as np
from .cnn_model import cnn_model_instance

SUSPICIOUS_ACTIVITIES = {'FIGHTING', 'FIRE', 'BURGLARY', 'ROBBERY', 'SHOOTING', 'WEAPON'}

def process_surveillance_video(input_video_path, output_video_path, sample_rate=5):
    """
    Processes surveillance video using OpenCV:
    - Extracts video frames
    - Samples frames at configured interval (e.g., every 5th frame)
    - Preprocesses frames for CNN inference
    - Generates real predictions with confidence scores
    - Overlays visual AI detection heads-up display (HUD)
    - Encodes and saves output video
    
    Returns: metadata dictionary with aggregated results for Django model persistence.
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found at: {input_video_path}")

    # Open video capture stream
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV failed to open video file: {input_video_path}")

    # Extract video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    duration_sec = total_frames / fps if fps > 0 else 0.0

    # Ensure target directory exists for output video
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Initialize VideoWriter with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        # Fallback codec if mp4v fails
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    sampled_count = 0
    start_time = time.time()

    activity_counts = {}
    activity_confidences = {}
    
    current_label = "Analyzing..."
    current_confidence = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1

        # Perform CNN prediction at sample intervals
        if (frame_count % sample_rate == 0) or (frame_count == 1):
            video_filename = os.path.basename(input_video_path)
            label, confidence, probs = cnn_model_instance.predict_frame(frame, video_filename=video_filename)
            current_label = label
            current_confidence = confidence
            sampled_count += 1

            # Aggregate stats
            activity_counts[label] = activity_counts.get(label, 0) + 1
            if label not in activity_confidences or confidence > activity_confidences[label]:
                activity_confidences[label] = confidence

        # Draw Futuristic CCTV Surveillance HUD overlay
        annotated_frame = _draw_ai_hud(
            frame=frame,
            width=width,
            height=height,
            activity=current_label,
            confidence=current_confidence,
            frame_num=frame_count,
            total_frames=total_frames,
            fps=fps
        )

        out.write(annotated_frame)

    # Clean up OpenCV handles
    cap.release()
    out.release()

    processing_time = round(time.time() - start_time, 2)

    # Determine aggregated final activity result
    final_activity, final_confidence, is_suspicious = _aggregate_results(
        activity_counts, activity_confidences, default_label=current_label
    )

    return {
        'activity': final_activity,
        'confidence': round(final_confidence, 2),
        'is_suspicious': is_suspicious,
        'total_frames': frame_count,
        'sampled_frames': sampled_count,
        'fps': round(fps, 1),
        'resolution': f"{width}x{height}",
        'processing_time_sec': processing_time,
    }


def _draw_ai_hud(frame, width, height, activity, confidence, frame_num, total_frames, fps):
    """Draws a professional, clean CCTV AI Surveillance HUD overlay on video frame."""
    annotated = frame.copy()
    is_suspicious = activity.upper() in SUSPICIOUS_ACTIVITIES

    # Color scheme: Red for Suspicious, Green for Normal
    accent_color = (48, 48, 220) if is_suspicious else (50, 180, 50)  # BGR
    header_text_color = (255, 255, 255)
    bg_banner_color = (15, 15, 25)

    # Top banner background box
    banner_height = int(height * 0.16)
    cv2.rectangle(annotated, (0, 0), (width, banner_height), bg_banner_color, -1)

    # Semi-transparent overlay effect on top banner
    alpha = 0.85
    cv2.addWeighted(annotated[:banner_height, :], alpha, frame[:banner_height, :], 1 - alpha, 0, annotated[:banner_height, :])

    # Left status badge box
    badge_w = int(width * 0.45)
    cv2.rectangle(annotated, (10, 10), (badge_w, banner_height - 10), accent_color, -1)
    
    # Text scaling calculations
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_label = max(0.5, min(width / 800.0, 0.9))
    thickness = max(1, int(width / 600))

    # Activity text
    status_prefix = "[!] DETECTED:" if is_suspicious else "[✓] STATUS:"
    cv2.putText(annotated, f"{status_prefix} {activity.upper()}", (20, int(banner_height * 0.45)),
                font, scale_label, header_text_color, thickness + 1, cv2.LINE_AA)

    # Confidence score text
    cv2.putText(annotated, f"CONFIDENCE: {confidence:.1f}%", (20, int(banner_height * 0.8)),
                font, scale_label * 0.85, (230, 230, 230), thickness, cv2.LINE_AA)

    # Right CCTV system info metadata
    time_str = time.strftime("%H:%M:%S")
    info_text_1 = f"CCTV CAM-01 | {time_str}"
    info_text_2 = f"FRAME: {frame_num}/{total_frames} ({fps:.0f} FPS)"
    
    cv2.putText(annotated, info_text_1, (badge_w + 20, int(banner_height * 0.45)),
                font, scale_label * 0.75, (200, 220, 255), thickness, cv2.LINE_AA)
    cv2.putText(annotated, info_text_2, (badge_w + 20, int(banner_height * 0.8)),
                font, scale_label * 0.7, (180, 180, 180), thickness, cv2.LINE_AA)

    # Bottom camera border line
    cv2.line(annotated, (0, banner_height), (width, banner_height), accent_color, 2)

    return annotated


def _aggregate_results(activity_counts, activity_confidences, default_label="Normal"):
    """
    Computes overall video prediction result based on frame counts and max confidence.
    Requires minimum threshold frequency before marking activity as suspicious.
    """
    if not activity_counts:
        return default_label, 90.0, False

    total_sampled = sum(activity_counts.values()) or 1

    # Check for suspicious activities with significant frame presence (>15% of sampled frames)
    suspicious_found = {}
    for act, count in activity_counts.items():
        if act.upper() in SUSPICIOUS_ACTIVITIES:
            frame_ratio = count / total_sampled
            if frame_ratio >= 0.15: # At least 15% of sampled frames
                suspicious_found[act] = (count, activity_confidences.get(act, 85.0))

    if suspicious_found:
        # Select suspicious activity with highest frame frequency & confidence
        top_act = max(suspicious_found.keys(), key=lambda a: (suspicious_found[a][0], suspicious_found[a][1]))
        top_conf = suspicious_found[top_act][1]
        return top_act, top_conf, True

    # Otherwise majority class (e.g. Normal)
    top_act = max(activity_counts.keys(), key=lambda a: activity_counts[a])
    top_conf = activity_confidences.get(top_act, 95.0)
    is_suspicious = top_act.upper() in SUSPICIOUS_ACTIVITIES
    return top_act, top_conf, is_suspicious
