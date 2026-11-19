"""
LeafCheck - Environment & Dependency Verification Script
Run this script to verify that your environment is properly configured.
"""

import sys
import os
import platform

# Ensure matplotlib cache is within project workspace
os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib")

print("=" * 60)
print(" LeafCheck Environment Verification")
print("=" * 60)

print(f"Python Version  : {platform.python_version()}")
print(f"Python Executable: {sys.executable}")
print(f"Operating System : {platform.system()} {platform.release()} ({platform.machine()})")
print("-" * 60)

modules_to_check = [
    ("tensorflow", "TensorFlow"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("sklearn", "Scikit-Learn"),
    ("matplotlib", "Matplotlib"),
    ("PIL", "Pillow"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("pydantic", "Pydantic"),
    ("mlflow", "MLflow"),
    ("dvc", "DVC"),
    ("evidently", "Evidently AI"),
]

all_passed = True

for module_name, display_name in modules_to_check:
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "Installed (version attr missing)")
        print(f" [PASS] {display_name:<15}: v{version}")
    except ImportError as e:
        print(f" [FAIL] {display_name:<15}: NOT INSTALLED ({e})")
        all_passed = False
    except Exception as e:
        print(f" [WARN] {display_name:<15}: Error during import ({e})")
        all_passed = False

print("-" * 60)

# Check TensorFlow device availability (CPU/GPU)
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f" Hardware Acceleration: {len(gpus)} GPU(s) detected: {[gpu.name for gpu in gpus]}")
    else:
        print(" Hardware Acceleration: Running on CPU (Normal for local dev & evaluation)")
except Exception as e:
    print(f" Hardware Acceleration: Unable to probe devices ({e})")

print("=" * 60)

if all_passed:
    print(" ALL DEPENDENCIES VERIFIED SUCCESSFULLY! You are ready for Phase 2.")
else:
    print(" SOME DEPENDENCIES ARE MISSING OR FAILED TO LOAD.")
    print(" Run: pip install -r requirements.txt")

print("=" * 60)
