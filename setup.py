#!/usr/bin/env python3
"""
HRM Setup Script

Sets up the development environment and runs initial tests.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} is not compatible. Requires Python 3.8+")
        return False


def install_dependencies():
    """Install required dependencies"""
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("💡 Try: pip install --user -r requirements.txt")
        return False
    return True


def run_tests():
    """Run the test suite"""
    return run_command("python test_hrm.py", "Running HRM tests")


def main():
    """Main setup routine"""
    print("🧠 HRM Development Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Dependency installation failed. You may need to:")
        print("   1. Create a virtual environment: python -m venv hrm_env")
        print("   2. Activate it: source hrm_env/bin/activate")
        print("   3. Try installation again: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run tests
    print("\n🧪 Running verification tests...")
    if run_tests():
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("   1. Test training: python train_hrm.py --dataset sudoku --epochs 5 --num_samples 50")
        print("   2. Full training: python train_hrm.py --dataset sudoku --epochs 100 --num_samples 1000")
        print("   3. Monitor with W&B: python train_hrm.py --use_wandb --project_name hrm-test")
    else:
        print("\n⚠️  Tests failed. Check the error messages above.")
        print("💡 Common issues:")
        print("   - Missing PyTorch: pip install torch")
        print("   - Import errors: Check Python path and dependencies")


if __name__ == '__main__':
    main()
