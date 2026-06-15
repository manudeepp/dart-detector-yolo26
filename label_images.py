import os
import sys
import shutil
import hashlib

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
IGNORED_DIR = os.path.join(RAW_DIR, "ignored")

# Train & Validation Directories
IMG_TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "images", "train")
LBL_TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "labels", "train")
IMG_VAL_DIR = os.path.join(BASE_DIR, "dataset", "images", "val")
LBL_VAL_DIR = os.path.join(BASE_DIR, "dataset", "labels", "val")

# Ensure required folders exist
for folder in [RAW_DIR, IGNORED_DIR, IMG_TRAIN_DIR, LBL_TRAIN_DIR, IMG_VAL_DIR, LBL_VAL_DIR]:
    os.makedirs(folder, exist_ok=True)

# State variables for mouse callback
drawing = False
ix, iy = -1, -1
cx, cy = -1, -1
selected_box = None  # (x1, y1, x2, y2) in display coordinates

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, cx, cy, selected_box

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
            selected_box = None

    # Double click: shortcut to place a standard 30x30 bounding box centered on the click
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        half_size = 15
        selected_box = (x - half_size, y - half_size, x + half_size, y + half_size)
        drawing = False

def get_split_assignment(filename, val_ratio=0.20):
    """
    Uses a deterministic MD5 hash of the filename to decide the train/val split.
    This guarantees the same image always maps to the same split.
    """
    hash_val = int(hashlib.md5(filename.encode('utf-8')).hexdigest(), 16)
    if (hash_val % 100) < (val_ratio * 100):
        return 'val'
    return 'train'

def label_single_image(image_path, idx, total):
    global selected_box, drawing, ix, iy, cx, cy
    
    selected_box = None
    drawing = False
    
    filename = os.path.basename(image_path)
    split = get_split_assignment(filename)
    
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
    
    print(f"\n[{idx}/{total}] Labeling: {filename} -> Target Split: {split.upper()}")
    print("  -> Drag Mouse: draw bounding box around dart")
    print("  -> Double Click: place a default 30x30 bounding box at cursor")
    print("  -> Enter or Space (with box drawn): Save labeled image to dataset")
    print("  -> Enter or Space (no box drawn):   Save as background image (no dart) to dataset")
    print("  -> Esc or I: Ignore/discard this image (Move to raw_images/ignored/)")
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
        
        # Display instructions with split details
        text1 = f"[{idx}/{total}] {filename} ({w_orig}x{h_orig}) -> [{split.upper()}]"
        text2 = "Drag: Draw | Enter: Save Labeled/Background | Esc/I: Ignore | Q: Quit"
        cv2.putText(img_disp, text1, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100) if split == 'train' else (255, 200, 100), 1, cv2.LINE_AA)
        cv2.putText(img_disp, text2, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(window_name, img_disp)
        key = cv2.waitKey(30) & 0xFF
        
        # ESC or 'i'/'I' -> Ignore/Skip image
        if key == 27 or key == ord('i') or key == ord('I'):
            cv2.destroyWindow(window_name)
            # Move ignored image to ignored folder
            dest_ignored_path = os.path.join(IGNORED_DIR, filename)
            shutil.move(image_path, dest_ignored_path)
            print(f"  -> IGNORED: {filename} (Moved to raw_images/ignored/)")
            return 'skip'
            
        # 'q'/'Q' -> Quit program
        elif key == ord('q') or key == ord('Q'):
            cv2.destroyWindow(window_name)
            return 'quit'
            
        # Enter (13) or Space (32) -> Save
        elif key == 13 or key == 32:
            # Determine directory destinations based on train/val split
            if split == 'train':
                dest_img_dir = IMG_TRAIN_DIR
                dest_lbl_dir = LBL_TRAIN_DIR
            else:
                dest_img_dir = IMG_VAL_DIR
                dest_lbl_dir = LBL_VAL_DIR
            
            filename_no_ext, _ = os.path.splitext(filename)
            label_filename = f"{filename_no_ext}.txt"
            label_path = os.path.join(dest_lbl_dir, label_filename)
            
            if selected_box:
                # 1. SAVE AS LABELED IMAGE
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
                
                # Save coordinates
                with open(label_path, "w") as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")
                
                # Move image file
                dest_image_path = os.path.join(dest_img_dir, filename)
                shutil.move(image_path, dest_image_path)
                
                print(f"  -> SAVED LABELED to [{split.upper()}]: {filename} with box coordinates.")
            else:
                # 2. SAVE AS BACKGROUND IMAGE (Empty label file)
                with open(label_path, "w") as f:
                    pass  # Create empty file
                
                # Move image file
                dest_image_path = os.path.join(dest_img_dir, filename)
                shutil.move(image_path, dest_image_path)
                
                print(f"  -> SAVED BACKGROUND to [{split.upper()}]: {filename} (no darts, empty label).")
            
            cv2.destroyWindow(window_name)
            return 'next'

def main():
    print("=" * 60)
    print("YOLO DART IMAGE LABELING UTILITY")
    print("=" * 60)
    
    # Check if raw folder is empty (only files, ignoring directories like 'ignored')
    raw_images = [
        f for f in os.listdir(RAW_DIR) 
        if os.path.isfile(os.path.join(RAW_DIR, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
    ]
    
    if not raw_images:
        print(f"\nNo raw images found in '{RAW_DIR}'.")
        print("Please follow these instructions:")
        print(f"  1. Put all your raw unlabeled images in: {RAW_DIR}")
        print("  2. Run this script again.")
        print(f"\nCreated folder structure for you at: {RAW_DIR}")
        return

    print(f"Found {len(raw_images)} raw image(s) in queue. Starting labeling...")
    
    labeled_count = 0
    skipped_count = 0
    
    for idx, img_name in enumerate(raw_images, start=1):
        img_path = os.path.join(RAW_DIR, img_name)
        result = label_single_image(img_path, idx, len(raw_images))
        
        if result == 'quit':
            print("\nLabeling utility closed by user.")
            break
        elif result == 'skip':
            skipped_count += 1
        elif result == 'next':
            labeled_count += 1
            
    print("\n" + "=" * 60)
    print("LABELING SESSION SUMMARY")
    print("=" * 60)
    print(f"  - Images successfully added to dataset: {labeled_count}")
    print(f"  - Images ignored/skipped:              {skipped_count}")
    print(f"  - Train Set Output:                    {IMG_TRAIN_DIR}")
    print(f"  - Val Set Output:                      {IMG_VAL_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
