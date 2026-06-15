import os
import sys
import shutil

# Check if opencv is installed
try:
    import cv2
except ImportError:
    print("ERROR: OpenCV is not installed. Please run:")
    print("  pip install opencv-python")
    sys.exit(1)

# Configuration paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw_images")
IMG_TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "images", "train")
LBL_TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "labels", "train")

# Ensure required folders exist
for folder in [RAW_DIR, IMG_TRAIN_DIR, LBL_TRAIN_DIR]:
    os.makedirs(folder, exist_ok=True)

# State variables for mouse callback
drawing = False
ix, iy = -1, -1
cx, cy = -1, -1
selected_box = None  # (x1, y1, x2, y2) in display coordinates
box_confirmed = False

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, cx, cy, selected_box, box_confirmed

    # Left button down: start drawing box
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        cx, cy = x, y
        selected_box = None

    # Mouse move: update coordinates if drawing
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cx, cy = x, y

    # Left button up: finalize box
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Only set if we actually dragged a minimum distance
        if abs(ix - x) > 3 or abs(iy - y) > 3:
            x1, x2 = min(ix, x), max(ix, x)
            y1, y2 = min(iy, y), max(iy, y)
            selected_box = (x1, y1, x2, y2)
        else:
            # Single click (not a drag) doesn't set a box
            selected_box = None

    # Double click: shortcut to place a standard 30x30 bounding box centered on the click
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        half_size = 15
        selected_box = (x - half_size, y - half_size, x + half_size, y + half_size)
        drawing = False

def label_single_image(image_path, idx, total):
    global selected_box, drawing, ix, iy, cx, cy
    
    selected_box = None
    drawing = False
    
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return 'skip'
        
    h_orig, w_orig = img_orig.shape[:2]
    
    # Calculate scale factor to fit standard desktop screens
    max_w, max_h = 1280, 720
    scale = min(max_w / w_orig, max_h / h_orig, 1.0)
    w_disp = int(w_orig * scale)
    h_disp = int(h_orig * scale)
    
    window_name = f"YOLO Dart Labeler - Image {idx}/{total}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print(f"\n[{idx}/{total}] Labeling: {os.path.basename(image_path)}")
    print("  -> Drag Mouse: draw bounding box around dart")
    print("  -> Double Click: place a default 30x30 bounding box at cursor")
    print("  -> Enter or Space: Save label & copy image to dataset")
    print("  -> S or Esc: Skip/ignore this image (No dart present)")
    print("  -> Q: Quit labeling application")

    while True:
        # Clone image for display overlay
        img_disp = cv2.resize(img_orig, (w_disp, h_disp))
        
        # Draw dynamic bounding box while dragging
        if drawing:
            cv2.rectangle(img_disp, (ix, iy), (cx, cy), (0, 255, 0), 2)
        # Draw confirmed bounding box
        elif selected_box:
            cv2.rectangle(img_disp, (selected_box[0], selected_box[1]), (selected_box[2], selected_box[3]), (0, 255, 0), 2)
            # Add a crosshair at center
            mx = (selected_box[0] + selected_box[2]) // 2
            my = (selected_box[1] + selected_box[3]) // 2
            cv2.drawMarker(img_disp, (mx, my), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)

        # Draw UI overlay panel at the top
        overlay = img_disp.copy()
        cv2.rectangle(overlay, (0, 0), (w_disp, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img_disp, 0.3, 0, img_disp)
        
        # Display instructions
        text1 = f"[{idx}/{total}] {os.path.basename(image_path)} ({w_orig}x{h_orig})"
        text2 = "Drag: Draw | DblClick: Quick Box | Enter: Save | S/Esc: Skip (No Dart) | Q: Quit"
        cv2.putText(img_disp, text1, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_disp, text2, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(window_name, img_disp)
        key = cv2.waitKey(30) & 0xFF
        
        # ESC or 's'/'S' -> Skip image
        if key == 27 or key == ord('s') or key == ord('S'):
            cv2.destroyWindow(window_name)
            return 'skip'
            
        # 'q'/'Q' -> Quit program
        elif key == ord('q') or key == ord('Q'):
            cv2.destroyWindow(window_name)
            return 'quit'
            
        # Enter (13) or Space (32) -> Save box
        elif key == 13 or key == 32:
            if selected_box:
                x1, y1, x2, y2 = selected_box
                # Clamp coordinates to display boundaries
                x1 = max(0, min(x1, w_disp))
                x2 = max(0, min(x2, w_disp))
                y1 = max(0, min(y1, h_disp))
                y2 = max(0, min(y2, h_disp))
                
                # Convert back to original scale
                x1_orig = x1 / scale
                x2_orig = x2 / scale
                y1_orig = y1 / scale
                y2_orig = y2 / scale
                
                # Calculate YOLO normalized coordinates
                w_orig_box = abs(x2_orig - x1_orig)
                h_orig_box = abs(y2_orig - y1_orig)
                x_center = (min(x1_orig, x2_orig) + w_orig_box / 2.0) / w_orig
                y_center = (min(y1_orig, y2_orig) + h_orig_box / 2.0) / h_orig
                yolo_w = w_orig_box / w_orig
                yolo_h = h_orig_box / h_orig
                
                # Save label text file
                filename = os.path.basename(image_path)
                filename_no_ext, _ = os.path.splitext(filename)
                label_filename = f"{filename_no_ext}.txt"
                label_path = os.path.join(LBL_TRAIN_DIR, label_filename)
                
                with open(label_path, "w") as f:
                    # Class ID is 0 for 'dart'
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")
                
                # Copy image file to train folder
                dest_image_path = os.path.join(IMG_TRAIN_DIR, filename)
                shutil.copy2(image_path, dest_image_path)
                
                print(f"  -> SAVED: {filename} and label text file.")
                cv2.destroyWindow(window_name)
                return 'next'
            else:
                print("  -> Please draw a box first, or press 'S' to skip.")

def main():
    print("=" * 60)
    print("YOLO DART IMAGE LABELING UTILITY")
    print("=" * 60)
    
    # Check if raw folder is empty
    raw_images = [
        f for f in os.listdir(RAW_DIR) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
    ]
    
    if not raw_images:
        print(f"\nNo raw images found in '{RAW_DIR}'.")
        print("Please follow these instructions:")
        print(f"  1. Put all your raw unlabeled images in: {RAW_DIR}")
        print("  2. Run this script again.")
        print(f"\nCreated folder structure for you at: {RAW_DIR}")
        return

    print(f"Found {len(raw_images)} raw image(s) in '{RAW_DIR}'. Starting labeling...")
    
    labeled_count = 0
    skipped_count = 0
    
    for idx, img_name in enumerate(raw_images, start=1):
        img_path = os.path.join(RAW_DIR, img_name)
        result = label_single_image(img_path, idx, len(raw_images))
        
        if result == 'quit':
            print("\nLabeling utility closed by user.")
            break
        elif result == 'skip':
            print(f"  -> SKIPPED/IGNORED: {img_name}")
            skipped_count += 1
        elif result == 'next':
            labeled_count += 1
            
    print("\n" + "=" * 60)
    print("LABELING SESSION SUMMARY")
    print("=" * 60)
    print(f"  - Images successfully labeled: {labeled_count}")
    print(f"  - Images skipped/ignored:     {skipped_count}")
    print(f"  - Output Images:               {IMG_TRAIN_DIR}")
    print(f"  - Output Labels:               {LBL_TRAIN_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
