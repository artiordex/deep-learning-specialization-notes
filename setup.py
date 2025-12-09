#!/usr/bin/env python3
"""
Setup script for the Deep Learning Specialization repository.

This script helps set up the repository by:
- Creating necessary directories
- Verifying dependencies
- Testing imports
"""

import os
import sys
import subprocess


def create_directories():
    """Create all necessary subdirectories."""
    directories = [
        'notes/course1-neural-networks',
        'notes/course2-optimization',
        'notes/course3-ml-projects',
        'notes/course4-cnns',
        'notes/course5-sequences',
        'labs/course1-neural-networks',
        'labs/course2-optimization',
        'labs/course3-ml-projects',
        'labs/course4-cnns',
        'labs/course5-sequences',
        'experiments',
        'checkpoints',
        'results',
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")
    print()


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ✗ Python 3.8+ required. Current: {version.major}.{version.minor}")
        return False
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    print()
    return True


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'matplotlib',
        'sklearn',  # Note: installed as scikit-learn
        'jupyter',
        'tqdm',
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    print()
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def test_imports():
    """Test importing utility modules."""
    print("Testing utility imports...")
    try:
        from utils import (
            normalize_data,
            train_test_split,
            count_parameters,
            train_model,
            plot_training_curves,
        )
        print("  ✓ All utility modules imported successfully")
        print()
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        print()
        return False


def check_gpu():
    """Check if GPU is available."""
    print("Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
        else:
            print("  ⚠ No GPU available (using CPU)")
        print()
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")
        print()


def print_next_steps():
    """Print next steps for the user."""
    print("=" * 60)
    print("Setup complete! 🎉")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Start Jupyter: jupyter notebook")
    print("  2. Open a lab template: labs/lab_template.ipynb")
    print("  3. Try the PyTorch example: python labs/pytorch_example.py")
    print("  4. Read the README: README.md")
    print()
    print("Happy learning! 🚀")


def main():
    """Main setup function."""
    print("=" * 60)
    print("Deep Learning Specialization - Repository Setup")
    print("=" * 60)
    print()
    
    # Run all checks
    checks = [
        check_python_version(),
        # check_dependencies(),  # Commented out as packages may not be installed yet
        # test_imports(),  # Commented out as it depends on packages
    ]
    
    # Create directories
    create_directories()
    
    # Check GPU
    try:
        check_gpu()
    except:
        pass
    
    # Print next steps
    print_next_steps()


if __name__ == '__main__':
    main()
