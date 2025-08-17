#!/usr/bin/env python3
"""
Setup Script for Audio Noise Removal Pipeline
==============================================

This script helps set up all dependencies and requirements for the audio noise removal pipeline.
It handles package installation and provides guidance for system dependencies.

Usage:
    python setup_environment.py
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
import zipfile
import shutil

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def run_command(command, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        return False


def install_python_packages():
    """Install Python packages from requirements.txt."""
    print("\n" + "="*80)
    print("INSTALLING PYTHON PACKAGES")
    print("="*80)
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install packages
    packages = [
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
        "noisereduce>=2.0.0",
        "pydub>=0.25.0",
        "torch>=1.10.0",
        "torchaudio>=0.10.0",
        "moviepy>=1.0.3",
        "openai-whisper>=20230918",
        "demucs>=4.0.0",
        "ffmpeg-python>=0.2.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0"
    ]
    
    for package in packages:
        success = run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}")
        if not success:
            print(f"Warning: Failed to install {package}")
    
    print("\nPython package installation completed!")


def setup_ffmpeg_windows():
    """Set up FFmpeg on Windows."""
    print("\n" + "="*80)
    print("SETTING UP FFMPEG FOR WINDOWS")
    print("="*80)
    
    # Check if FFmpeg is already available in PATH
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("FFmpeg is already available in your PATH!")
            return True
    except FileNotFoundError:
        pass
    
    ffmpeg_dir = Path("ffmpeg")
    
    if (ffmpeg_dir / "bin" / "ffmpeg.exe").exists():
        print("FFmpeg already installed locally!")
        # Add to PATH for current session
        ffmpeg_bin = str(ffmpeg_dir / "bin")
        os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]
        return True
    
    if not REQUESTS_AVAILABLE:
        print("❌ Cannot download FFmpeg automatically (requests module not available)")
        print("Please download FFmpeg manually:")
        print("1. Go to: https://www.gyan.dev/ffmpeg/builds/")
        print("2. Download 'ffmpeg-release-essentials.zip'")
        print("3. Extract it to this directory and rename the folder to 'ffmpeg'")
        print("4. Or add FFmpeg to your system PATH")
        return False
    
    print("Downloading FFmpeg for Windows...")
    
    try:
        # Download FFmpeg
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save to zip file
        zip_file = "ffmpeg.zip"
        with open(zip_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting FFmpeg...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Find the extracted folder and rename it
        for item in Path(".").iterdir():
            if item.is_dir() and item.name.startswith("ffmpeg-"):
                item.rename("ffmpeg")
                break
        
        # Clean up
        os.remove(zip_file)
        
        # Add to PATH for current session
        ffmpeg_bin = str(Path("ffmpeg") / "bin")
        os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]
        
        print("FFmpeg setup completed!")
        print(f"FFmpeg installed to: {Path('ffmpeg').absolute()}")
        print("Note: You may need to add the FFmpeg bin directory to your system PATH permanently.")
        
        return True
        
    except Exception as e:
        print(f"Error setting up FFmpeg: {e}")
        print("Please download FFmpeg manually from: https://ffmpeg.org/download.html")
        return False


def check_dependencies():
    """Check if all dependencies are properly installed."""
    print("\n" + "="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)
    
    # Check Python packages
    packages_to_check = [
        "numpy", "scipy", "librosa", "soundfile", "noisereduce",
        "pydub", "torch", "moviepy", "whisper", "demucs"
    ]
    
    missing_packages = []
    
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT FOUND")
            missing_packages.append(package)
    
    # Check FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg")
        else:
            print("✗ FFmpeg - NOT WORKING")
    except FileNotFoundError:
        print("✗ FFmpeg - NOT FOUND")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\nAll dependencies are installed!")
        return True


def create_test_script():
    """Create a simple test script to verify the installation."""
    test_script = '''#!/usr/bin/env python3
"""
Test script to verify audio processing setup.
"""

def test_imports():
    """Test all imports."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy")
        
        import librosa
        print("✓ Librosa")
        
        import soundfile as sf
        print("✓ SoundFile")
        
        import noisereduce as nr
        print("✓ NoiseReduce")
        
        from pydub import AudioSegment
        print("✓ PyDub")
        
        import torch
        print("✓ PyTorch")
        
        from moviepy.editor import VideoFileClip
        print("✓ MoviePy")
        
        import whisper
        print("✓ Whisper")
        
        try:
            from demucs.pretrained import get_model
            print("✓ Demucs")
        except ImportError:
            print("⚠ Demucs - Available but not loaded (normal)")
        
        print("\\nAll imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_ffmpeg():
    """Test FFmpeg availability."""
    import subprocess
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg is available")
            return True
        else:
            print("✗ FFmpeg not working properly")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg not found in PATH")
        return False


if __name__ == "__main__":
    print("="*60)
    print("AUDIO PROCESSING SETUP TEST")
    print("="*60)
    
    imports_ok = test_imports()
    ffmpeg_ok = test_ffmpeg()
    
    if imports_ok and ffmpeg_ok:
        print("\\n🎉 Setup verification completed successfully!")
        print("You can now run the audio noise removal pipeline.")
    else:
        print("\\n❌ Setup verification failed.")
        print("Please check the errors above and install missing dependencies.")
'''
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    print("Created test_setup.py - run this to verify your installation")


def main():
    """Main setup function."""
    print("="*80)
    print("AUDIO NOISE REMOVAL PIPELINE SETUP")
    print("="*80)
    print("This script will set up all dependencies for the audio noise removal pipeline.")
    
    # Install Python packages
    install_python_packages()
    
    # Set up FFmpeg (Windows only for now)
    if platform.system() == "Windows":
        setup_ffmpeg_windows()
    else:
        print("\n" + "="*80)
        print("FFMPEG SETUP")
        print("="*80)
        print("Please install FFmpeg using your system package manager:")
        if platform.system() == "Darwin":  # macOS
            print("macOS: brew install ffmpeg")
        else:  # Linux
            print("Linux: sudo apt-get install ffmpeg")
    
    # Create test script
    create_test_script()
    
    # Check dependencies
    all_good = check_dependencies()
    
    print("\n" + "="*80)
    print("SETUP COMPLETE")
    print("="*80)
    
    if all_good:
        print("✅ All dependencies are installed and ready!")
        print("\nNext steps:")
        print("1. Run 'python test_setup.py' to verify everything works")
        print("2. Run 'python audio_noise_removal.py' to process your video")
    else:
        print("❌ Some dependencies are missing.")
        print("Please install the missing packages and try again.")
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
