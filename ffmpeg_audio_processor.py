#!/usr/bin/env python3
"""
FFmpeg-based Audio Processing Pipeline
=====================================

This script uses FFmpeg directly for audio processing, minimizing Python dependencies.
It provides a comprehensive audio noise removal pipeline using FFmpeg's built-in filters.

Requirements:
- FFmpeg (must be installed and in PATH)
- Python 3.6+ (standard library only)
"""

import os
import subprocess
import sys
from pathlib import Path
import logging
import json
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FFmpegAudioProcessor:
    """
    Audio noise removal using FFmpeg's advanced audio filters.
    """
    
    def __init__(self, input_file: str, output_dir: str = "output"):
        """Initialize the processor."""
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create temp directory for intermediate files
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # File paths
        self.extracted_audio = self.temp_dir / "extracted_audio.wav"
        self.noise_profile = self.temp_dir / "noise_profile.wav"
        self.denoised_audio = self.temp_dir / "denoised_audio.wav"
        self.final_audio = self.temp_dir / "final_audio.wav"
        self.final_video = self.output_dir / f"cleaned_{self.input_file.stem}.mp4"
        self.transcript_file = self.output_dir / f"transcript_{self.input_file.stem}.txt"
        
        logger.info(f"Initialized FFmpeg processor for {self.input_file}")
    
    def check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(["ffmpeg", "-version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("FFmpeg is available")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        logger.error("FFmpeg not found in PATH")
        return False
    
    def get_video_info(self) -> dict:
        """Get video information using FFprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(self.input_file)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                logger.error(f"FFprobe failed: {result.stderr}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}
    
    def step1_extract_audio(self) -> bool:
        """Step 1: Extract audio from video."""
        logger.info("Step 1: Extracting audio from video...")
        
        try:
            cmd = [
                "ffmpeg", "-i", str(self.input_file),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # Uncompressed audio
                "-ar", "44100",  # Sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite output
                str(self.extracted_audio)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Audio extracted: {self.extracted_audio}")
                return True
            else:
                logger.error(f"Audio extraction failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return False
    
    def step2_noise_reduction(self) -> bool:
        """Step 2: Apply noise reduction using FFmpeg filters."""
        logger.info("Step 2: Applying noise reduction...")
        
        try:
            # Create noise profile from the first 2 seconds (assuming it contains mostly noise)
            logger.info("Creating noise profile from first 2 seconds...")
            
            profile_cmd = [
                "ffmpeg", "-i", str(self.extracted_audio),
                "-t", "2",  # First 2 seconds
                "-y",
                str(self.noise_profile)
            ]
            
            subprocess.run(profile_cmd, capture_output=True, text=True, timeout=60)
            
            # Apply noise reduction using afftdn filter
            logger.info("Applying advanced noise reduction...")
            
            denoise_cmd = [
                "ffmpeg", "-i", str(self.extracted_audio),
                "-af", (
                    "afftdn=nr=20:nf=-20:tn=1,"  # Spectral noise reduction
                    "highpass=f=80,"              # Remove low-frequency noise
                    "lowpass=f=8000,"             # Remove high-frequency noise
                    "dynaudnorm=p=0.9:s=5"       # Dynamic normalization
                ),
                "-y",
                str(self.denoised_audio)
            ]
            
            result = subprocess.run(denoise_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info(f"Noise reduction completed: {self.denoised_audio}")
                return True
            else:
                logger.error(f"Noise reduction failed: {result.stderr}")
                # Try simpler approach
                return self._simple_noise_reduction()
                
        except Exception as e:
            logger.error(f"Error in noise reduction: {e}")
            return self._simple_noise_reduction()
    
    def _simple_noise_reduction(self) -> bool:
        """Fallback simple noise reduction."""
        logger.info("Applying simple noise reduction...")
        
        try:
            cmd = [
                "ffmpeg", "-i", str(self.extracted_audio),
                "-af", (
                    "highpass=f=100,"     # High-pass filter
                    "lowpass=f=7000,"     # Low-pass filter
                    "volume=1.2"          # Slight volume boost
                ),
                "-y",
                str(self.denoised_audio)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("Simple noise reduction completed")
                return True
            else:
                logger.error(f"Simple noise reduction failed: {result.stderr}")
                # Copy original if all fails
                shutil.copy2(self.extracted_audio, self.denoised_audio)
                return True
                
        except Exception as e:
            logger.error(f"Simple noise reduction error: {e}")
            return False
    
    def step3_silence_handling(self) -> bool:
        """Step 3: Remove long silences."""
        logger.info("Step 3: Removing long silences...")
        
        try:
            cmd = [
                "ffmpeg", "-i", str(self.denoised_audio),
                "-af", (
                    "silenceremove="
                    "start_periods=1:start_duration=0.1:start_threshold=-40dB:"
                    "stop_periods=-1:stop_duration=2:stop_threshold=-40dB"
                ),
                "-y",
                str(self.final_audio)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Silence removal completed: {self.final_audio}")
                return True
            else:
                logger.warning(f"Silence removal failed: {result.stderr}")
                # Copy denoised audio if silence removal fails
                shutil.copy2(self.denoised_audio, self.final_audio)
                return True
                
        except Exception as e:
            logger.error(f"Error in silence handling: {e}")
            return False
    
    def step4_create_transcript(self) -> bool:
        """Step 4: Create a basic transcript placeholder."""
        logger.info("Step 4: Creating transcript placeholder...")
        
        try:
            # Get audio duration
            info_cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(self.final_audio)
            ]
            
            result = subprocess.run(info_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                duration = float(result.stdout.strip())
            else:
                duration = 0.0
            
            # Create basic transcript file
            with open(self.transcript_file, 'w', encoding='utf-8') as f:
                f.write(f"Audio Transcript for: {self.input_file.name}\\n")
                f.write("=" * 50 + "\\n\\n")
                f.write(f"Total Duration: {duration:.2f} seconds\\n\\n")
                f.write("Note: This is a placeholder transcript.\\n")
                f.write("For actual speech transcription, install whisper:\\n")
                f.write("pip install openai-whisper\\n")
                f.write("\\nTo get actual transcription, use:\\n")
                f.write("whisper 'audio_file.wav' --output_format txt\\n")
            
            logger.info(f"Transcript placeholder created: {self.transcript_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating transcript: {e}")
            return False
    
    def step5_merge_with_video(self) -> bool:
        """Step 5: Merge processed audio with original video."""
        logger.info("Step 5: Merging processed audio with video...")
        
        try:
            cmd = [
                "ffmpeg", "-i", str(self.input_file), "-i", str(self.final_audio),
                "-c:v", "copy",  # Copy video stream without re-encoding
                "-c:a", "aac",   # AAC audio codec
                "-map", "0:v:0", # Video from first input
                "-map", "1:a:0", # Audio from second input
                "-shortest",     # Match shortest stream
                "-y",
                str(self.final_video)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info(f"Final video created: {self.final_video}")
                return True
            else:
                logger.error(f"Video merging failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error merging video: {e}")
            return False
    
    def process(self) -> bool:
        """Execute the complete pipeline."""
        logger.info("Starting FFmpeg audio processing pipeline...")
        
        if not self.check_ffmpeg():
            return False
        
        # Get video info
        info = self.get_video_info()
        if info:
            logger.info(f"Processing video: {info.get('format', {}).get('filename', 'Unknown')}")
            duration = info.get('format', {}).get('duration', 'Unknown')
            logger.info(f"Duration: {duration} seconds")
        
        steps = [
            ("Extract Audio", self.step1_extract_audio),
            ("Noise Reduction", self.step2_noise_reduction),
            ("Silence Handling", self.step3_silence_handling),
            ("Create Transcript", self.step4_create_transcript),
            ("Merge with Video", self.step5_merge_with_video)
        ]
        
        for step_name, step_function in steps:
            logger.info(f"\\n{'='*60}")
            logger.info(f"Executing: {step_name}")
            logger.info(f"{'='*60}")
            
            if not step_function():
                logger.error(f"Failed at step: {step_name}")
                return False
        
        logger.info(f"\\n{'='*60}")
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"{'='*60}")
        logger.info(f"Final video: {self.final_video}")
        logger.info(f"Transcript: {self.transcript_file}")
        
        return True
    
    def cleanup_temp_files(self):
        """Remove temporary files."""
        logger.info("Cleaning up temporary files...")
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")


def download_ffmpeg_windows():
    """Download FFmpeg for Windows if not available."""
    logger.info("FFmpeg not found. Attempting to download...")
    
    try:
        import urllib.request
        import zipfile
        
        # Download FFmpeg
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        zip_file = "ffmpeg.zip"
        
        logger.info("Downloading FFmpeg...")
        urllib.request.urlretrieve(url, zip_file)
        
        logger.info("Extracting FFmpeg...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Find and rename directory
        for item in Path(".").iterdir():
            if item.is_dir() and item.name.startswith("ffmpeg-"):
                item.rename("ffmpeg")
                break
        
        # Add to PATH
        ffmpeg_bin = str(Path("ffmpeg") / "bin")
        os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]
        
        # Clean up
        os.remove(zip_file)
        
        logger.info("FFmpeg installed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download FFmpeg: {e}")
        return False


def main():
    """Main function."""
    print("="*60)
    print("FFMPEG AUDIO PROCESSING PIPELINE")
    print("="*60)
    
    input_video = "Unit-2 TA Session 1.mp4"
    
    if not Path(input_video).exists():
        logger.error(f"Input video not found: {input_video}")
        return False
    
    processor = FFmpegAudioProcessor(input_video)
    
    # Check FFmpeg availability
    if not processor.check_ffmpeg():
        logger.info("Attempting to install FFmpeg...")
        if not download_ffmpeg_windows():
            logger.error("Please install FFmpeg manually:")
            logger.error("1. Download from: https://ffmpeg.org/download.html")
            logger.error("2. Add to your system PATH")
            return False
    
    try:
        success = processor.process()
        
        if success:
            logger.info("\\n🎉 Processing completed successfully!")
            logger.info(f"Check the 'output' folder for results.")
            
            # Optionally clean up
            cleanup = input("Clean up temporary files? (y/n): ").lower().strip()
            if cleanup == 'y':
                processor.cleanup_temp_files()
        else:
            logger.error("❌ Processing failed. Check the logs above.")
        
        return success
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


if __name__ == "__main__":
    main()
