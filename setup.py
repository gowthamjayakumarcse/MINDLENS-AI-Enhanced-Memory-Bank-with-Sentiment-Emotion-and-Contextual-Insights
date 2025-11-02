#!/usr/bin/env python3
"""
Setup script for MindLens application.
This script will install all required dependencies and verify the setup.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install all required dependencies."""
    print("\n" + "="*50)
    print("Installing Dependencies")
    print("="*50)
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        if not run_command(f"{sys.executable} -m pip install -r {requirements_file}", "Installing requirements"):
            return False
    else:
        print("✗ requirements.txt not found")
        return False
    
    return True

def verify_model_paths():
    """Verify that all model paths exist."""
    print("\n" + "="*50)
    print("Verifying Model Paths")
    print("="*50)
    
    model_paths = {
        "Emotion Model": Path(__file__).parent / "BERT_FINE_TURNED_EMOTION_DECTION_USING_TEXT",
        "Spacy Model": Path(__file__).parent / "spacy_model_context" / "model",
        "SBERT Model": Path(__file__).parent / "bert_model_offilne"
    }
    
    all_exist = True
    for name, path in model_paths.items():
        if Path(path).exists():
            print(f"✓ {name} found at: {path}")
        else:
            print(f"✗ {name} not found at: {path}")
            all_exist = False
    
    return all_exist

def test_imports():
    """Test if all required modules can be imported."""
    print("\n" + "="*50)
    print("Testing Imports")
    print("="*50)
    
    modules_to_test = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("spacy", "SpaCy"),
        ("faiss", "FAISS"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn")
    ]
    
    all_imported = True
    for module, name in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
            all_imported = False
    
    return all_imported

def create_data_directory():
    """Create data directory for storing vectors and entries."""
    print("\n" + "="*50)
    print("Setting up Data Directory")
    print("="*50)
    
    data_dir = Path(__file__).parent / "data"
    try:
        data_dir.mkdir(exist_ok=True)
        print(f"✓ Data directory created at: {data_dir}")
        return True
    except Exception as e:
        print(f"✗ Failed to create data directory: {e}")
        return False

def main():
    """Main setup function."""
    print("MindLens Setup Script")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Setup failed during dependency installation")
        sys.exit(1)
    
    # Verify model paths
    if not verify_model_paths():
        print("\n⚠ Warning: Some model paths are missing. Please check the paths in config.py")
    
    # Test imports
    if not test_imports():
        print("\n✗ Setup failed during import testing")
        sys.exit(1)
    
    # Create data directory
    if not create_data_directory():
        print("\n✗ Setup failed during data directory creation")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("✓ Setup completed successfully!")
    print("="*50)
    print("\nYou can now run the application with:")
    print("  python main.py")
    print("\nOr use the individual modules in your own scripts.")

if __name__ == "__main__":
    main()