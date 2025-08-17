#!/usr/bin/env python3
"""
Adaptive Audio Noise Removal Pipeline
====================================

This script automatically detects available libraries and uses the best processing
method available. It falls back gracefully from advanced to basic processing.

Processing Methods (in order of preference):
1. Full Pipeline: Demucs + Noisereduce + Whisper + MoviePy
2. Hybrid Pipeline: FFmpeg + Noisereduce + Whisper
3. FFmpeg Pipeline: Pure FFmpeg processing
4. Basic Pipeline: MoviePy + PyDub

Author: Audio Processing Expert
Date: August 17, 2025
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path
import logging
import json
import tempfile
import shutil

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaptiveAudioProcessor:
    """
    Adaptive audio processor that uses the best available tools.
    """
    
    def __init__(self, input_file: str, output_dir: str = "output"):
        """Initialize the processor."""
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create temp directory
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # File paths
        self.extracted_audio = self.temp_dir / "extracted_audio.wav"
        self.vocals_separated = self.temp_dir / "vocals_separated.wav"
        self.denoised_audio = self.temp_dir / "denoised_audio.wav"
        self.final_audio = self.temp_dir / "final_audio.wav"
        self.final_video = self.output_dir / f"cleaned_{self.input_file.stem}.mp4"
        self.transcript_file = self.output_dir / f"transcript_{self.input_file.stem}.txt"
        
        # Check available tools
        self.available_tools = self._check_available_tools()
        logger.info(f"Available tools: {', '.join(self.available_tools)}")
        
        # Select processing method
        self.processing_method = self._select_processing_method()
        logger.info(f"Selected processing method: {self.processing_method}")
    
    def _check_available_tools(self) -> list:
        """Check which tools are available."""
        tools = []
        
        # Check FFmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
            tools.append("ffmpeg")
        except:
            pass
        
        # Check Python libraries
        libs_to_check = [
            ("numpy", "numpy"),
            ("librosa", "librosa"),
            ("soundfile", "soundfile"),
            ("noisereduce", "noisereduce"),
            ("pydub", "pydub"),
            ("torch", "torch"),
            ("moviepy", "moviepy"),
            ("whisper", "whisper"),
            ("demucs", "demucs")
        ]
        
        for lib_name, import_name in libs_to_check:
            try:
                __import__(import_name)
                tools.append(lib_name)
            except ImportError:
                pass
        
        return tools
    
    def _select_processing_method(self) -> str:
        """Select the best available processing method."""
        if all(tool in self.available_tools for tool in ["demucs", "noisereduce", "whisper", "moviepy"]):
            return "full"
        elif all(tool in self.available_tools for tool in ["ffmpeg", "noisereduce", "whisper"]):
            return "hybrid"
        elif "ffmpeg" in self.available_tools:
            return "ffmpeg"
        elif "moviepy" in self.available_tools:
            return "basic"
        else:
            return "minimal"
    
    def extract_audio(self) -> bool:
        """Extract audio using the best available method."""
        logger.info("Extracting audio...")
        
        if "moviepy" in self.available_tools:
            return self._extract_audio_moviepy()
        elif "ffmpeg" in self.available_tools:
            return self._extract_audio_ffmpeg()
        else:
            logger.error("No audio extraction method available")
            return False
    
    def _extract_audio_moviepy(self) -> bool:
        """Extract audio using MoviePy."""
        try:
            from moviepy.editor import VideoFileClip
            
            video = VideoFileClip(str(self.input_file))
            if video.audio is None:
                logger.error("No audio track found")
                return False
            
            audio = video.audio
            audio.write_audiofile(str(self.extracted_audio), verbose=False, logger=None)
            
            audio.close()
            video.close()
            
            logger.info("Audio extracted with MoviePy")
            return True
            
        except Exception as e:
            logger.error(f"MoviePy extraction failed: {e}")
            return False
    
    def _extract_audio_ffmpeg(self) -> bool:
        """Extract audio using FFmpeg."""
        try:
            cmd = [
                "ffmpeg", "-i", str(self.input_file),
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                "-y", str(self.extracted_audio)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("Audio extracted with FFmpeg")
                return True
            else:
                logger.error(f"FFmpeg extraction failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"FFmpeg extraction error: {e}")
            return False
    
    def separate_vocals(self) -> bool:
        """Separate vocals using available method."""
        if "demucs" in self.available_tools:
            return self._separate_vocals_demucs()
        else:
            # Skip vocal separation, copy original audio
            shutil.copy2(self.extracted_audio, self.vocals_separated)
            logger.info("Vocal separation skipped (Demucs not available)")
            return True
    
    def _separate_vocals_demucs(self) -> bool:
        """Separate vocals using Demucs."""
        try:
            import torch
            import librosa
            import soundfile as sf
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            import numpy as np
            
            logger.info("Loading Demucs model...")
            model = get_model('htdemucs')
            model.eval()
            
            # Load audio
            waveform, sample_rate = librosa.load(str(self.extracted_audio), sr=None, mono=False)
            
            # Ensure stereo
            if waveform.ndim == 1:
                waveform = np.stack([waveform, waveform])
            elif waveform.shape[0] == 1:
                waveform = np.vstack([waveform, waveform])
            
            # Convert to tensor
            waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
            
            # Apply separation
            logger.info("Separating vocals...")
            with torch.no_grad():
                sources = apply_model(model, waveform_tensor)
            
            # Extract vocals
            vocals = sources[0, 3].numpy()  # vocals track
            
            # Save
            sf.write(str(self.vocals_separated), vocals.T, sample_rate)
            
            logger.info("Vocal separation completed with Demucs")
            return True
            
        except Exception as e:
            logger.error(f"Demucs separation failed: {e}")
            # Fallback
            shutil.copy2(self.extracted_audio, self.vocals_separated)
            return True
    
    def denoise_audio(self) -> bool:
        """Denoise audio using available method."""
        if "noisereduce" in self.available_tools:
            return self._denoise_noisereduce()
        elif "ffmpeg" in self.available_tools:
            return self._denoise_ffmpeg()
        else:
            shutil.copy2(self.vocals_separated, self.denoised_audio)
            logger.info("Denoising skipped (no tools available)")
            return True
    
    def _denoise_noisereduce(self) -> bool:
        """Denoise using noisereduce library."""
        try:
            import librosa
            import soundfile as sf
            import noisereduce as nr
            
            logger.info("Applying noise reduction...")
            
            # Load audio
            audio_data, sample_rate = librosa.load(str(self.vocals_separated), sr=None)
            
            # Apply noise reduction
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=sample_rate,
                stationary=True,
                prop_decrease=0.8
            )
            
            # Save
            sf.write(str(self.denoised_audio), reduced_noise, sample_rate)
            
            logger.info("Noise reduction completed with noisereduce")
            return True
            
        except Exception as e:
            logger.error(f"Noisereduce failed: {e}")
            return self._denoise_ffmpeg()
    
    def _denoise_ffmpeg(self) -> bool:
        """Denoise using FFmpeg filters."""
        try:
            cmd = [
                "ffmpeg", "-i", str(self.vocals_separated),
                "-af", (
                    "afftdn=nr=20:nf=-20:tn=1,"
                    "highpass=f=80,"
                    "lowpass=f=8000,"
                    "dynaudnorm=p=0.9:s=5"
                ),
                "-y", str(self.denoised_audio)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("Noise reduction completed with FFmpeg")
                return True
            else:
                logger.error(f"FFmpeg denoising failed: {result.stderr}")
                shutil.copy2(self.vocals_separated, self.denoised_audio)
                return True
                
        except Exception as e:
            logger.error(f"FFmpeg denoising error: {e}")
            return False
    
    def handle_silence(self) -> bool:
        """Handle silence using available method."""
        if "pydub" in self.available_tools:
            return self._handle_silence_pydub()
        elif "ffmpeg" in self.available_tools:
            return self._handle_silence_ffmpeg()
        else:
            shutil.copy2(self.denoised_audio, self.final_audio)
            logger.info("Silence handling skipped (no tools available)")
            return True
    
    def _handle_silence_pydub(self) -> bool:
        """Handle silence using PyDub."""
        try:
            from pydub import AudioSegment
            from pydub.silence import split_on_silence
            
            logger.info("Handling silence with PyDub...")
            
            # Load audio
            audio = AudioSegment.from_wav(str(self.denoised_audio))
            
            # Split on silence
            chunks = split_on_silence(
                audio,
                min_silence_len=2000,
                silence_thresh=audio.dBFS - 16,
                keep_silence=500
            )
            
            if chunks:
                # Concatenate chunks
                trimmed_audio = AudioSegment.empty()
                for i, chunk in enumerate(chunks):
                    trimmed_audio += chunk
                    if i < len(chunks) - 1:
                        trimmed_audio += AudioSegment.silent(duration=200)
                
                # Export
                trimmed_audio.export(str(self.final_audio), format="wav")
                
                original_duration = len(audio) / 1000
                trimmed_duration = len(trimmed_audio) / 1000
                logger.info(f"Silence handling completed. Duration: {original_duration:.2f}s → {trimmed_duration:.2f}s")
            else:
                shutil.copy2(self.denoised_audio, self.final_audio)
            
            return True
            
        except Exception as e:
            logger.error(f"PyDub silence handling failed: {e}")
            return self._handle_silence_ffmpeg()
    
    def _handle_silence_ffmpeg(self) -> bool:
        """Handle silence using FFmpeg."""
        try:
            cmd = [
                "ffmpeg", "-i", str(self.denoised_audio),
                "-af", (
                    "silenceremove="
                    "start_periods=1:start_duration=0.1:start_threshold=-40dB:"
                    "stop_periods=-1:stop_duration=2:stop_threshold=-40dB"
                ),
                "-y", str(self.final_audio)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("Silence handling completed with FFmpeg")
                return True
            else:
                logger.warning("FFmpeg silence removal failed, using original")
                shutil.copy2(self.denoised_audio, self.final_audio)
                return True
                
        except Exception as e:
            logger.error(f"FFmpeg silence handling error: {e}")
            return False
    
    def transcribe_audio(self) -> bool:
        """Transcribe audio using available method."""
        if "whisper" in self.available_tools:
            return self._transcribe_whisper()
        else:
            return self._create_basic_transcript()
    
    def _transcribe_whisper(self) -> bool:
        """Transcribe using Whisper."""
        try:
            import whisper
            
            logger.info("Transcribing with Whisper...")
            
            # Load model
            model = whisper.load_model("base")
            
            # Transcribe
            result = model.transcribe(
                str(self.final_audio),
                word_timestamps=True,
                verbose=False
            )
            
            # Save transcript
            with open(self.transcript_file, 'w', encoding='utf-8') as f:
                f.write(f"Transcript for: {self.input_file.name}\\n")
                f.write("=" * 50 + "\\n\\n")
                
                # Full transcript
                f.write("FULL TRANSCRIPT:\\n")
                f.write(result["text"] + "\\n\\n")
                
                # Segments
                f.write("SEGMENTS WITH TIMESTAMPS:\\n")
                for segment in result["segments"]:
                    start_time = segment["start"]
                    end_time = segment["end"]
                    text = segment["text"]
                    f.write(f"[{start_time:.2f}s - {end_time:.2f}s]: {text}\\n")
                
                # Words if available
                if "words" in result:
                    f.write("\\nWORD-LEVEL TIMESTAMPS:\\n")
                    for word_info in result["words"]:
                        word = word_info["word"]
                        start = word_info["start"]
                        end = word_info["end"]
                        f.write(f"{word} [{start:.2f}s - {end:.2f}s]\\n")
            
            logger.info("Transcription completed with Whisper")
            return True
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return self._create_basic_transcript()
    
    def _create_basic_transcript(self) -> bool:
        """Create basic transcript placeholder."""
        try:
            # Get duration
            if "ffmpeg" in self.available_tools:
                cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                       "-of", "csv=p=0", str(self.final_audio)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                duration = float(result.stdout.strip()) if result.returncode == 0 else 0.0
            else:
                duration = 0.0
            
            # Create transcript
            with open(self.transcript_file, 'w', encoding='utf-8') as f:
                f.write(f"Audio Transcript for: {self.input_file.name}\\n")
                f.write("=" * 50 + "\\n\\n")
                f.write(f"Total Duration: {duration:.2f} seconds\\n\\n")
                f.write("Note: This is a placeholder transcript.\\n")
                f.write("For actual speech transcription, install whisper:\\n")
                f.write("pip install openai-whisper\\n")
            
            logger.info("Basic transcript created")
            return True
            
        except Exception as e:
            logger.error(f"Error creating transcript: {e}")
            return False
    
    def merge_with_video(self) -> bool:
        """Merge processed audio with video."""
        if "moviepy" in self.available_tools:
            return self._merge_moviepy()
        elif "ffmpeg" in self.available_tools:
            return self._merge_ffmpeg()
        else:
            logger.error("No video merging method available")
            return False
    
    def _merge_moviepy(self) -> bool:
        """Merge using MoviePy."""
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            
            logger.info("Merging with MoviePy...")
            
            # Load video and audio
            video = VideoFileClip(str(self.input_file))
            audio = AudioFileClip(str(self.final_audio))
            
            # Set audio
            final_video = video.set_audio(audio)
            
            # Write
            final_video.write_videofile(
                str(self.final_video),
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # Cleanup
            audio.close()
            final_video.close()
            video.close()
            
            logger.info("Video merging completed with MoviePy")
            return True
            
        except Exception as e:
            logger.error(f"MoviePy merging failed: {e}")
            return self._merge_ffmpeg()
    
    def _merge_ffmpeg(self) -> bool:
        """Merge using FFmpeg."""
        try:
            cmd = [
                "ffmpeg", "-i", str(self.input_file), "-i", str(self.final_audio),
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-y", str(self.final_video)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("Video merging completed with FFmpeg")
                return True
            else:
                logger.error(f"FFmpeg merging failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"FFmpeg merging error: {e}")
            return False
    
    def process(self) -> bool:
        """Execute the complete pipeline."""
        logger.info(f"Starting adaptive audio processing pipeline...")
        logger.info(f"Processing method: {self.processing_method}")
        
        steps = [
            ("Extract Audio", self.extract_audio),
            ("Separate Vocals", self.separate_vocals),
            ("Denoise Audio", self.denoise_audio),
            ("Handle Silence", self.handle_silence),
            ("Transcribe Audio", self.transcribe_audio),
            ("Merge with Video", self.merge_with_video)
        ]
        
        for step_name, step_function in steps:
            logger.info(f"\\n{'='*60}")
            logger.info(f"Executing: {step_name}")
            logger.info(f"{'='*60}")
            
            if not step_function():
                logger.error(f"Failed at step: {step_name}")
                return False
        
        logger.info(f"\\n{'='*60}")
        logger.info("ADAPTIVE PIPELINE COMPLETED SUCCESSFULLY!")
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


def install_missing_tools():
    """Try to install missing essential tools."""
    logger.info("Checking for missing tools...")
    
    # Check basic packages
    basic_packages = ["moviepy", "pydub"]
    missing_basic = []
    
    for package in basic_packages:
        try:
            __import__(package)
        except ImportError:
            missing_basic.append(package)
    
    if missing_basic:
        logger.info(f"Installing basic packages: {', '.join(missing_basic)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_basic)
            logger.info("Basic packages installed successfully")
        except Exception as e:
            logger.warning(f"Could not install basic packages: {e}")
    
    # Check FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        logger.info("FFmpeg is available")
    except:
        logger.info("FFmpeg not found, attempting to download...")
        try:
            # Download FFmpeg for Windows
            import urllib.request
            import zipfile
            
            url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            zip_file = "ffmpeg.zip"
            
            logger.info("Downloading FFmpeg...")
            urllib.request.urlretrieve(url, zip_file)
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Rename directory
            for item in Path(".").iterdir():
                if item.is_dir() and item.name.startswith("ffmpeg-"):
                    item.rename("ffmpeg")
                    break
            
            # Add to PATH
            ffmpeg_bin = str(Path("ffmpeg") / "bin")
            os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]
            
            os.remove(zip_file)
            logger.info("FFmpeg installed successfully")
            
        except Exception as e:
            logger.warning(f"Could not install FFmpeg: {e}")


def main():
    """Main function."""
    print("="*60)
    print("ADAPTIVE AUDIO PROCESSING PIPELINE")
    print("="*60)
    
    input_video = "Unit-2 TA Session 1.mp4"
    
    if not Path(input_video).exists():
        logger.error(f"Input video not found: {input_video}")
        return False
    
    # Try to install missing tools
    install_missing_tools()
    
    # Create processor
    processor = AdaptiveAudioProcessor(input_video)
    
    try:
        success = processor.process()
        
        if success:
            logger.info("\\n🎉 Processing completed successfully!")
            logger.info(f"Processing method used: {processor.processing_method}")
            logger.info(f"Available tools: {', '.join(processor.available_tools)}")
            logger.info(f"Check the 'output' folder for results.")
            
            # Cleanup option
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
