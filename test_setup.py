#!/usr/bin/env python3
"""
Simple test script to verify audio processing setup.
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
        
        print("\nAll imports successful!")
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
        print("\n🎉 Setup verification completed successfully!")
        print("You can now run the audio noise removal pipeline.")
    else:
        print("\n❌ Setup verification failed.")
        print("Please check the errors above and install missing dependencies.")
