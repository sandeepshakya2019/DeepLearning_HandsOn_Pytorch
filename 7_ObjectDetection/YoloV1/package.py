import subprocess
import sys

def install(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def try_import(package, import_name=None):
    try:
        __import__(import_name if import_name else package)
        return True
    except ImportError:
        install(package)
        return False

# List of (package_name, optional_import_name)
packages = [
    ("torch",),
    ("torchvision",),
    ("torchinfo",),
]

for pkg in packages:
    try_import(*pkg)

print("All packages installed. Done ✅")
