import torch
from ultralytics import YOLO
import os
from datetime import datetime

def check_device_and_train():
    print("=" * 60)
    print("SYSTEM AND HARDWARE CHECK")
    print("=" * 60)
    
    # Check PyTorch and CUDA status
    pytorch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    
    print(f"PyTorch version: {pytorch_version}")
    print(f"CUDA (GPU acceleration) available: {cuda_available}")
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        device_id = 0
        print(f"Target GPU: {device_name}")
        print(f"Active training device: CUDA Device {device_id}")
    else:
        device_id = "cpu"
        print("WARNING: CUDA is not available. Training will fallback to CPU (highly inefficient).")
        print("Active training device: CPU")
        
    print("=" * 60)
    print("STARTING YOLO26 TRAINING")
    print("=" * 60)
    
    # Get the directory of this script to ensure absolute/portable paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to dataset yaml configuration
    yaml_path = os.path.join(script_dir, "dataset", "dataset.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"ERROR: Dataset YAML configuration not found at: {yaml_path}")
        return

    # Dedicated project folder inside the package directory
    project_dir = os.path.join(script_dir, "trained_models")
    
    # Generate timestamped run folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"dart_yolo26s_{timestamp}"
    
    print(f"Training outputs will be saved to: {os.path.join(project_dir, run_name)}")

    # Load the latest YOLO26 Small model
    model = YOLO("yolo26s.pt")
    
    # Start the training process
    model.train(
        data=yaml_path,
        epochs=100,
        batch=16,
        imgsz=640,
        device=device_id,
        workers=4,
        plots=True,
        project=project_dir,  # Dedicated folder
        name=run_name        # Timestamped folder name
    )

if __name__ == "__main__":
    check_device_and_train()
