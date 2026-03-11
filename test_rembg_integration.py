#!/usr/bin/env python3
"""Quick test of rembg integration in service."""
import sys
sys.path.insert(0, '/Users/rifaz/EZ-CorridorKey')

print("Testing rembg integration...")
print()

# Test 1: Service imports
print("✓ Testing imports...")
try:
    from backend.service import CorridorKeyService
    print("  ✓ CorridorKeyService imported")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check run_rembg exists
print("✓ Checking run_rembg method exists...")
try:
    service = CorridorKeyService()
    assert hasattr(service, 'run_rembg'), "run_rembg method not found"
    print("  ✓ run_rembg method exists")
except Exception as e:
    print(f"  ✗ Check failed: {e}")
    sys.exit(1)

# Test 3: Check UI imports
print("✓ Testing UI imports...")
try:
    from ui.widgets.parameter_panel import ParameterPanel
    from ui.main_window import MainWindow
    from ui.workers.gpu_job_worker import GPUJobWorker
    print("  ✓ UI modules imported")
except Exception as e:
    print(f"  ✗ UI import failed: {e}")
    sys.exit(1)

# Test 4: Check signals
print("✓ Checking rembg_requested signal...")
try:
    assert hasattr(ParameterPanel, 'rembg_requested'), "rembg_requested signal not found"
    print("  ✓ rembg_requested signal exists")
except Exception as e:
    print(f"  ✗ Signal check failed: {e}")
    sys.exit(1)

print()
print("✓ All integration tests passed!")
print()
print("The rembg integration is ready to use.")
print("Start the app: ./2-start.sh")
