#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""

import sys
import os

# Ensure we're in the correct directory
os.chdir('/home/green/code/thinker/flask_server')

print("Testing imports...")

try:
    print("1. Testing worker imports...")
    import worker
    print("   ✓ worker imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import worker: {e}")
    sys.exit(1)

try:
    print("2. Testing hf_backend imports...")
    import hf_backend
    print("   ✓ hf_backend imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import hf_backend: {e}")
    sys.exit(1)

try:
    print("3. Testing loss_functions imports...")
    import loss_functions
    print("   ✓ loss_functions imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import loss_functions: {e}")
    sys.exit(1)

try:
    print("4. Testing tasks imports...")
    import tasks
    print("   ✓ tasks imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import tasks: {e}")
    sys.exit(1)

try:
    print("5. Testing storage imports...")
    import storage
    print("   ✓ storage imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import storage: {e}")
    sys.exit(1)

try:
    print("6. Testing app imports...")
    import app
    print("   ✓ app imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All imports successful!")
print("\nYou can now run the server with:")
print("  python app.py")
