#!/usr/bin/env python3
"""
PiVot Installation Test Script
PiVot インストール確認スクリプト

This script tests if all PiVot components are properly installed and working.
このスクリプトはPiVotのすべてのコンポーネントが正しくインストールされ動作しているかをテストします。
"""

import sys
import subprocess
import importlib
import os
import platform
from pathlib import Path

def print_header(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print('='*60)

def test_python_version():
    """Test Python version compatibility"""
    print_header("Python Version Check")
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible (3.8+)")
        return True
    else:
        print("❌ Python version is too old (3.8+ required)")
        return False

def test_package_import(package_name, optional=False):
    """Test if a package can be imported"""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name}")
        return True
    except ImportError as e:
        if optional:
            print(f"⚠️  {package_name} (optional) - {str(e)}")
        else:
            print(f"❌ {package_name} - {str(e)}")
        return False

def test_required_packages():
    """Test all required packages"""
    print_header("Required Packages Test")
    
    required_packages = [
        'numpy',
        'opencv-python',
        'fastapi', 
        'uvicorn',
        'requests',
        'pillow'
    ]
    
    results = []
    for package in required_packages:
        # Handle package name differences
        import_name = package
        if package == 'opencv-python':
            import_name = 'cv2'
        elif package == 'pillow':
            import_name = 'PIL'
        
        results.append(test_package_import(import_name))
    
    return all(results)

def test_optional_packages():
    """Test optional packages"""
    print_header("Optional Packages Test")
    
    optional_packages = [
        ('pyaudio', 'Audio input/output'),
        ('picamera', 'Raspberry Pi Camera (Pi only)'),
        ('openvino', 'Intel NPU support (Windows only)')
    ]
    
    for package, description in optional_packages:
        import_name = package
        if package == 'picamera':
            # Try both picamera and picamera2
            try:
                importlib.import_module('picamera')
                print(f"✅ picamera - {description}")
            except ImportError:
                try:
                    importlib.import_module('picamera2')
                    print(f"✅ picamera2 - {description}")
                except ImportError:
                    print(f"⚠️  picamera/picamera2 - {description}")
        else:
            test_package_import(import_name, optional=True)

def test_system_info():
    """Display system information"""
    print_header("System Information")
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Check if running on Raspberry Pi
    try:
        with open('/proc/cpuinfo', 'r') as f:
            if 'Raspberry Pi' in f.read():
                print("🥧 Raspberry Pi detected")
    except FileNotFoundError:
        pass
    
    # Check available camera devices
    if platform.system() == "Linux":
        video_devices = list(Path('/dev').glob('video*'))
        if video_devices:
            print(f"📷 Camera devices: {[str(d) for d in video_devices]}")
        else:
            print("📷 No camera devices found")

def test_network_connectivity():
    """Test network connectivity"""
    print_header("Network Connectivity Test")
    
    try:
        import requests
        response = requests.get('https://www.google.com', timeout=5)
        if response.status_code == 200:
            print("✅ Internet connectivity: OK")
        else:
            print(f"⚠️  Internet connectivity: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Internet connectivity: {str(e)}")

def test_file_permissions():
    """Test file permissions"""
    print_header("File Permissions Test")
    
    current_dir = Path.cwd()
    
    # Test read permission
    if os.access(current_dir, os.R_OK):
        print("✅ Read permission: OK")
    else:
        print("❌ Read permission: Failed")
    
    # Test write permission
    if os.access(current_dir, os.W_OK):
        print("✅ Write permission: OK")
    else:
        print("❌ Write permission: Failed")
    
    # Test execute permission
    if os.access(current_dir, os.X_OK):
        print("✅ Execute permission: OK")
    else:
        print("❌ Execute permission: Failed")

def test_config_files():
    """Test configuration files"""
    print_header("Configuration Files Test")
    
    config_files = [
        'config.py',
        '../config.py',
        'requirements.txt',
        's_compornents/requirements.txt'
    ]
    
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            print(f"✅ {config_file}: Found")
        else:
            print(f"⚠️  {config_file}: Not found")

def run_quick_functionality_test():
    """Run a quick functionality test"""
    print_header("Quick Functionality Test")
    
    try:
        # Test numpy
        import numpy as np
        arr = np.array([1, 2, 3])
        print(f"✅ NumPy: {arr.sum()}")
        
        # Test OpenCV
        import cv2
        print(f"✅ OpenCV: Version {cv2.__version__}")
        
        # Test FastAPI import
        from fastapi import FastAPI
        app = FastAPI()
        print("✅ FastAPI: Import OK")
        
        # Test requests
        import requests
        print("✅ Requests: Import OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {str(e)}")
        return False

def generate_report(results):
    """Generate final test report"""
    print_header("Test Summary Report")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! PiVot is ready to use.")
        print("\nNext steps:")
        print("  1. Start Windows PC PiVot-Server: python main.py")
        print("  2. Start Raspberry Pi PiVot: bash start_pivot_assistant.sh")
    elif passed >= total * 0.8:
        print("\n⚠️  Most tests passed. Some optional components may be missing.")
        print("Check the TROUBLESHOOTING.md file for help with failed tests.")
    else:
        print("\n❌ Multiple tests failed. Please check your installation.")
        print("Run the setup script again or check TROUBLESHOOTING.md for help.")
    
    return passed / total

def main():
    """Main test function"""
    print("🚀 PiVot Installation Test Script")
    print("=================================")
    print("This script will test your PiVot installation...")
    
    # Run all tests
    results = {}
    
    results['python_version'] = test_python_version()
    results['required_packages'] = test_required_packages()
    results['functionality'] = run_quick_functionality_test()
    
    # Optional tests (don't affect pass/fail)
    test_optional_packages()
    test_system_info()
    test_network_connectivity()
    test_file_permissions()
    test_config_files()
    
    # Generate final report
    success_rate = generate_report(results)
    
    # Exit with appropriate code
    sys.exit(0 if success_rate >= 0.8 else 1)

if __name__ == "__main__":
    main()