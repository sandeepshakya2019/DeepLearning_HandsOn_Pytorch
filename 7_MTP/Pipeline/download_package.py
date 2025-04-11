import subprocess
import sys

# List of packages to install
required_packages = [
    "torch",
    "torchvision",
    "ultralytics",
    "opencv-python",
    "scikit-learn",
    "matplotlib",
    "tqdm",
    "pillow",
    "requests",
]

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install each package
for package in required_packages:
    try:
        __import__(package if package != "opencv-python" else "cv2")
        print(f"✅ {package} is already installed")
    except ImportError:
        print(f"📦 Installing {package}...")
        install(package)

# Environment variable fix for OpenMP issue on Windows (if needed)
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


