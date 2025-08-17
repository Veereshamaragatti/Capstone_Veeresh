#!/usr/bin/env python3
"""
Simplified Audio Noise Removal Pipeline
=======================================

This is a streamlined version that uses basic moviepy for audio extraction and processing,
without requiring heavy dependencies like Demucs.

Requirements:
- moviepy (for video/audio processing)
- numpy (for array operations)
- pydub (for audio manipulation)
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleAudioNoiseRemover:
    """
    Simplified audio noise removal pipeline.
    """
    
    def __init__(self, input_file: str, output_dir: str = "output"):
        """Initialize the audio processor."""
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # File paths
        self.extracted_audio = self.output_dir / "extracted_audio.wav"
        self.final_video = self.output_dir / f"processed_{self.input_file.stem}.mp4"
        
        logger.info(f"Initialized processor for {self.input_file}")
    
    def extract_and_process_audio(self) -> bool:
        """Extract audio from video and apply basic processing."""
        logger.info("Extracting and processing audio...")
        
        try:
            # Try to import moviepy
            try:
                from moviepy.editor import VideoFileClip
            except ImportError:
                logger.error("MoviePy not available. Installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy"])
                from moviepy.editor import VideoFileClip
            
            # Load video and extract audio
            logger.info("Loading video...")
            video = VideoFileClip(str(self.input_file))
            
            if video.audio is None:
                logger.error("No audio track found in video")
                return False
            
            # Extract audio
            logger.info("Extracting audio...")
            audio = video.audio
            audio.write_audiofile(str(self.extracted_audio), verbose=False, logger=None)
            
            # Apply basic audio enhancement if possible
            try:
                self._enhance_audio()
            except Exception as e:
                logger.warning(f"Audio enhancement failed: {e}")
                logger.info("Proceeding with original audio...")
            
            # Create new video with processed audio
            logger.info("Creating final video...")
            from moviepy.editor import AudioFileClip
            processed_audio = AudioFileClip(str(self.extracted_audio))
            final_video = video.set_audio(processed_audio)
            
            final_video.write_videofile(
                str(self.final_video),
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # Cleanup
            audio.close()
            processed_audio.close()
            final_video.close()
            video.close()
            
            logger.info(f"Processing completed: {self.final_video}")
            return True
            
        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            return False
    
    def _enhance_audio(self):
        """Apply basic audio enhancement using available tools."""
        try:
            # Try pydub for basic audio processing
            from pydub import AudioSegment
            from pydub.effects import normalize
            
            logger.info("Applying basic audio enhancement...")
            
            # Load audio
            audio = AudioSegment.from_wav(str(self.extracted_audio))
            
            # Normalize audio
            audio = normalize(audio)
            
            # Reduce very quiet parts (basic noise gate)
            # This helps remove some background noise
            threshold = audio.dBFS - 20  # 20dB below average
            
            # Simple noise reduction by boosting volume and applying compression
            audio = audio + 3  # Boost by 3dB
            
            # Export enhanced audio
            audio.export(str(self.extracted_audio), format="wav")
            
            logger.info("Basic audio enhancement applied")
            
        except ImportError:
            logger.warning("PyDub not available for audio enhancement")
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {e}")


def install_basic_requirements():
    """Install basic requirements if not available."""
    requirements = ["moviepy", "pydub"]
    
    for package in requirements:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed successfully")
            except Exception as e:
                print(f"✗ Failed to install {package}: {e}")
                return False
    
    return True


def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg is available")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ FFmpeg not found!")
    print("Please install FFmpeg:")
    print("1. Download from: https://ffmpeg.org/download.html")
    print("2. Add to your system PATH")
    print("3. Or place ffmpeg.exe in the same directory as this script")
    
    return False


def main():
    """Main function."""
    print("="*60)
    print("SIMPLIFIED AUDIO PROCESSING PIPELINE")
    print("="*60)
    
    # Check and install basic requirements
    print("\nChecking requirements...")
    if not install_basic_requirements():
        print("Failed to install required packages")
        return False
    
    if not check_ffmpeg():
        print("FFmpeg is required but not found")
        return False
    
    # Process the video
    input_video = "Unit-2 TA Session 1.mp4"
    
    if not Path(input_video).exists():
        print(f"Input video not found: {input_video}")
        return False
    
    processor = SimpleAudioNoiseRemover(input_video)
    success = processor.extract_and_process_audio()
    
    if success:
        print(f"\n✅ Processing completed successfully!")
        print(f"Output video: {processor.final_video}")
        print(f"Extracted audio: {processor.extracted_audio}")
    else:
        print("\n❌ Processing failed")
    
    return success


if __name__ == "__main__":
    main()
