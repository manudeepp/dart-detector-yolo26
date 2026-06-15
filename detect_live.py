import cv2
import os
import time
from ultralytics import YOLO

def run_live_detection():
    # Set the model path relative to the script location (so it runs correctly from anywhere)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(script_dir, "models", "pretrained", "best.pt")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: YOLO model not found at {MODEL_PATH}")
        print("Please ensure the default model is at 'models/pretrained/best.pt',")
        print("or copy your custom trained model there.")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows stability
    
    # Set standard widescreen resolution (1280x720) to ensure high frame rates (typically 30 FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Query the actual resolved width and height
    max_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    max_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate the square dimensions once at startup (symmetrical FOV)
    square_size = min(max_w, max_h)
    x_start = (max_w - square_size) // 2
    y_start = (max_h - square_size) // 2

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Attempt to extract class names from the trained model
    try:
        classNames = model.names
    except AttributeError:
        classNames = {0: "dart"}

    # Confidence threshold (interactive)
    confidence_threshold = 0.50
    
    # FPS variables
    prev_frame_time = 0
    
    # Create window and set it to be resizable
    window_name = 'Dart Detector (YOLO26s) - Symmetrical Square FOV'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        success, img = cap.read()
        if not success:
            break

        # Symmetrically crop to center square (maximum field of view with 1:1 aspect ratio)
        cropped_img = img[y_start:y_start+square_size, x_start:x_start+square_size]

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time

        # Run YOLO object detection silently on the cropped square
        results = model(cropped_img, stream=True, verbose=False)

        # Count detected darts
        dart_count = 0

        # Process detected objects
        for r in results:
            for box in r.boxes:
                confidence = float(box.conf[0])

                if confidence >= confidence_threshold:
                    dart_count += 1
                    
                    # Get bounding box coordinates relative to the cropped square
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calculate center point on the square
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Compute relative (normalized) center coordinates on 1:1 scale
                    norm_cx = cx / square_size
                    norm_cy = cy / square_size

                    # Color scheme
                    box_color = (0, 255, 0)      # Neon Green for detection
                    dot_color = (0, 0, 255)      # Bright Red for target dot
                    text_color = (0, 0, 0)       # Black text

                    # Draw neat bounding box with corner overlays
                    # 1. Base thin rectangle
                    cv2.rectangle(cropped_img, (x1, y1), (x2, y2), box_color, 1)
                    
                    # 2. Tech corner ticks
                    tick_length = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
                    thickness = 3
                    # Top-Left
                    cv2.line(cropped_img, (x1, y1), (x1 + tick_length, y1), box_color, thickness)
                    cv2.line(cropped_img, (x1, y1), (x1, y1 + tick_length), box_color, thickness)
                    # Top-Right
                    cv2.line(cropped_img, (x2, y1), (x2 - tick_length, y1), box_color, thickness)
                    cv2.line(cropped_img, (x2, y1), (x2, y1 + tick_length), box_color, thickness)
                    # Bottom-Left
                    cv2.line(cropped_img, (x1, y2), (x1 + tick_length, y2), box_color, thickness)
                    cv2.line(cropped_img, (x1, y2), (x1, y2 - tick_length), box_color, thickness)
                    # Bottom-Right
                    cv2.line(cropped_img, (x2, y2), (x2 - tick_length, y2), box_color, thickness)
                    cv2.line(cropped_img, (x2, y2), (x2, y2 - tick_length), box_color, thickness)

                    # 3. Draw target center "dot" and fine crosshair
                    cv2.circle(cropped_img, (cx, cy), 5, dot_color, -1)
                    cv2.line(cropped_img, (cx - 10, cy), (cx + 10, cy), dot_color, 1)
                    cv2.line(cropped_img, (cx, cy - 10), (cx, cy + 10), dot_color, 1)

                    # 4. Draw text label showing coordinates near the dot
                    coord_label = f"({cx}, {cy}) [{norm_cx:.2f}, {norm_cy:.2f}]"
                    cv2.putText(cropped_img, coord_label, (cx + 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(cropped_img, coord_label, (cx + 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

                    # 5. Bounding box label
                    cls_id = int(box.cls[0])
                    class_name = classNames.get(cls_id, "dart")
                    label = f"{class_name.upper()} {confidence:.2f}"
                    
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # Label background
                    cv2.rectangle(cropped_img, (x1, y1 - 20), (x1 + w + 10, y1), box_color, -1)
                    # Label text
                    cv2.putText(cropped_img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        # Draw Heads-Up Display (HUD) overlay
        # Dark transparent banner top background for telemetry
        overlay = cropped_img.copy()
        cv2.rectangle(overlay, (0, 0), (square_size, 45), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.4, cropped_img, 0.6, 0, cropped_img)

        # Draw HUD text (compact sizes to fit square window)
        hud_font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cropped_img, "DART DETECTOR", (10, 28), hud_font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(cropped_img, f"DARTS: {dart_count}", (160, 28), hud_font, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(cropped_img, f"THR: {confidence_threshold:.2f} (w/s) | Q/ESC to Exit", (260, 28), hud_font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(cropped_img, f"FPS: {int(fps)}", (square_size - 75, 28), hud_font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # Show cropped square feed
        cv2.imshow(window_name, cropped_img)

        # Listen for keystrokes
        key = cv2.waitKey(1) & 0xFF
        
        # Interactive Controls
        if key == ord('q') or key == ord('Q') or key == 27:  # Exit on 'q', 'Q', or ESC
            break
        elif key == ord('w') or key == ord('+'):  # Raise confidence threshold
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
        elif key == ord('s') or key == ord('-'):  # Lower confidence threshold
            confidence_threshold = max(0.05, confidence_threshold - 0.05)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_detection()
