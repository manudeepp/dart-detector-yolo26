import os
import sys
import random
import numpy as np

# Check if opencv and numpy are installed
try:
    import cv2
except ImportError:
    print("ERROR: OpenCV is not installed. Please run:")
    print("  pip install opencv-python")
    sys.exit(1)

# Configuration paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw_images")

def apply_rotation(img, angle_range=(-20, 20)):
    """Applies a random rotation using reflection borders to prevent black edges."""
    h, w = img.shape[:2]
    angle = random.uniform(angle_range[0], angle_range[1])
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Use BORDER_REFLECT to keep the background natural-looking
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated

def apply_perspective_skew(img, max_distortion=0.05):
    """Applies a random perspective warp to simulate camera angle distortion."""
    h, w = img.shape[:2]
    
    # Define source corners of the image
    src_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]
    ])
    
    # Apply a random shift to corners (up to max_distortion of image size)
    max_shift_x = int(w * max_distortion)
    max_shift_y = int(h * max_distortion)
    
    dst_pts = np.float32([
        [random.randint(-max_shift_x, max_shift_x), random.randint(-max_shift_y, max_shift_y)],
        [w - 1 + random.randint(-max_shift_x, max_shift_x), random.randint(-max_shift_y, max_shift_y)],
        [random.randint(-max_shift_x, max_shift_x), h - 1 + random.randint(-max_shift_y, max_shift_y)],
        [w - 1 + random.randint(-max_shift_x, max_shift_x), h - 1 + random.randint(-max_shift_y, max_shift_y)]
    ])
    
    P = cv2.getPerspectiveTransform(src_pts, dst_pts)
    skewed = cv2.warpPerspective(img, P, (w, h), borderMode=cv2.BORDER_REFLECT)
    return skewed

def apply_brightness_contrast(img, alpha_range=(0.75, 1.25), beta_range=(-30, 30)):
    """Applies a random brightness and contrast scaling adjustment."""
    alpha = random.uniform(alpha_range[0], alpha_range[1])  # Contrast scaling
    beta = random.randint(beta_range[0], beta_range[1])     # Brightness offset
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

def augment_image(image_path):
    """Loads an image, applies the deformation pipeline, and returns the augmented image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Pipeline: Rotation -> Skew -> Brightness/Contrast
    img = apply_rotation(img)
    img = apply_perspective_skew(img)
    img = apply_brightness_contrast(img)
    
    return img

def main():
    print("=" * 60)
    print("YOLO DATASET AUGMENTATION UTILITY")
    print("=" * 60)
    
    if not os.path.exists(RAW_DIR):
        print(f"ERROR: Raw images directory not found at: {RAW_DIR}")
        return
        
    # Get all original images in raw_images (ignore already augmented files starting with 'aug_')
    all_files = os.listdir(RAW_DIR)
    original_images = [
        f for f in all_files 
        if os.path.isfile(os.path.join(RAW_DIR, f))
        and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        and not f.lower().startswith('aug_')
    ]
    
    if not original_images:
        print(f"No original raw images found in '{RAW_DIR}'.")
        print("Please place original images there first (non-augmented).")
        return
        
    print(f"Found {len(original_images)} original raw image(s).")
    
    # Augment approximately 50% of the images
    num_to_augment = max(1, int(len(original_images) * 0.50))
    print(f"Selecting {num_to_augment} random image(s) for augmentation (50% subset)...")
    
    # Seed random for reproducibility per execution run, or let it be stochastic
    random.seed()
    images_to_augment = random.sample(original_images, num_to_augment)
    
    augmented_count = 0
    for img_name in images_to_augment:
        src_path = os.path.join(RAW_DIR, img_name)
        aug_img = augment_image(src_path)
        
        if aug_img is not None:
            dest_name = f"aug_{img_name}"
            dest_path = os.path.join(RAW_DIR, dest_name)
            cv2.imwrite(dest_path, aug_img)
            print(f"  -> Augmented: {img_name} -> {dest_name}")
            augmented_count += 1
        else:
            print(f"  -> Error: Could not process {img_name}")
            
    print("\n" + "=" * 60)
    print("AUGMENTATION UTILITY SUMMARY")
    print("=" * 60)
    print(f"  - Original images scanned:  {len(original_images)}")
    print(f"  - Images selected:          {num_to_augment}")
    print(f"  - Successfully augmented:   {augmented_count}")
    print(f"  - Output folder:            {RAW_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
