#!/usr/bin/env python3
"""
Audio Noise Removal Pipeline for Video Files
============================================

This script implements a comprehensive audio denoising pipeline following the specified methodology:
1. Audio extraction using FFmpeg
2. Noise removal using Demucs and Noisereduce
3. Silence handling with PyDub
4. Speech transcription with Whisper
5. Output synchronization with MoviePy

Author: Audio Processing Expert
Date: August 17, 2025
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path
from typing import Tuple, Optional
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Audio processing libraries
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import noisereduce as nr
import torch

# Video processing
from moviepy.editor import VideoFileClip, AudioFileClip

# Speech recognition
import whisper

# Demucs for source separation
try:
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    print("Warning: Demucs not available. Will skip vocal separation step.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioNoiseRemover:
    """
    Comprehensive audio noise removal pipeline for video files.
    """
    
    def __init__(self, input_file: str, output_dir: str = "output"):
        """
        Initialize the audio noise remover.
        
        Args:
            input_file (str): Path to input video/audio file
            output_dir (str): Directory for output files
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for intermediate files
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # File paths for different processing stages
        self.extracted_audio = self.temp_dir / "extracted_audio.wav"
        self.demucs_vocals = self.temp_dir / "vocals_separated.wav"
        self.denoised_audio = self.temp_dir / "denoised_audio.wav"
        self.trimmed_audio = self.temp_dir / "trimmed_audio.wav"
        self.final_video = self.output_dir / f"cleaned_{self.input_file.stem}.mp4"
        self.transcript_file = self.output_dir / f"transcript_{self.input_file.stem}.txt"
        
        logger.info(f"Initialized AudioNoiseRemover for {self.input_file}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def step1_extract_audio(self) -> bool:
        """
        Step 1: Extract audio from video using FFmpeg while preserving timestamps.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Step 1: Extracting audio from video...")
        
        try:
            # Use MoviePy for reliable audio extraction
            video = VideoFileClip(str(self.input_file))
            audio = video.audio
            
            if audio is None:
                logger.error("No audio track found in the video file")
                return False
            
            # Write audio to file
            audio.write_audiofile(
                str(self.extracted_audio),
                verbose=False,
                logger=None
            )
            
            # Clean up
            audio.close()
            video.close()
            
            logger.info(f"Audio extracted successfully: {self.extracted_audio}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            return False
    
    def step2_vocal_separation_demucs(self) -> bool:
        """
        Step 2a: Use Demucs to separate vocals from background noise.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Step 2a: Separating vocals using Demucs...")
        
        if not DEMUCS_AVAILABLE:
            logger.warning("Demucs not available, copying original audio...")
            # If Demucs is not available, copy the original audio
            import shutil
            shutil.copy2(self.extracted_audio, self.demucs_vocals)
            return True
        
        try:
            # Load pre-trained Demucs model
            logger.info("Loading Demucs model...")
            model = get_model('htdemucs')
            model.eval()
            
            # Load audio
            waveform, sample_rate = librosa.load(str(self.extracted_audio), sr=None, mono=False)
            
            # Ensure stereo format for Demucs
            if waveform.ndim == 1:
                waveform = np.stack([waveform, waveform])
            elif waveform.shape[0] == 1:
                waveform = np.vstack([waveform, waveform])
            
            # Convert to torch tensor
            waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
            
            # Apply source separation
            logger.info("Applying vocal separation...")
            with torch.no_grad():
                sources = apply_model(model, waveform_tensor)
            
            # Extract vocals (usually index 3 in htdemucs: drums, bass, other, vocals)
            vocals = sources[0, 3].numpy()  # Shape: (channels, samples)
            
            # Save separated vocals
            sf.write(str(self.demucs_vocals), vocals.T, sample_rate)
            
            logger.info(f"Vocal separation completed: {self.demucs_vocals}")
            return True
            
        except Exception as e:
            logger.error(f"Error in vocal separation: {str(e)}")
            # Fallback: copy original audio
            import shutil
            shutil.copy2(self.extracted_audio, self.demucs_vocals)
            return True
    
    def step3_spectral_denoising(self) -> bool:
        """
        Step 2b: Apply spectral gating using Noisereduce for stationary noise.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Step 2b: Applying spectral denoising...")
        
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(str(self.demucs_vocals), sr=None)
            
            # Apply noise reduction
            logger.info("Applying noise reduction...")
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=sample_rate,
                stationary=True,
                prop_decrease=0.8  # Reduce noise by 80%
            )
            
            # Save denoised audio
            sf.write(str(self.denoised_audio), reduced_noise, sample_rate)
            
            logger.info(f"Spectral denoising completed: {self.denoised_audio}")
            return True
            
        except Exception as e:
            logger.error(f"Error in spectral denoising: {str(e)}")
            return False
    
    def step4_silence_handling(self) -> bool:
        """
        Step 3: Handle silence using PyDub - trim silence longer than 2 seconds.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Step 3: Handling silence and trimming...")
        
        try:
            # Load audio with PyDub
            audio = AudioSegment.from_wav(str(self.denoised_audio))
            
            # Split on silence
            logger.info("Analyzing silence intervals...")
            chunks = split_on_silence(
                audio,
                min_silence_len=2000,  # 2 seconds
                silence_thresh=audio.dBFS - 16,  # Silence threshold
                keep_silence=500  # Keep 500ms of silence at the edges
            )
            
            if not chunks:
                logger.warning("No audio chunks found after silence splitting")
                # Copy original if splitting fails
                import shutil
                shutil.copy2(self.denoised_audio, self.trimmed_audio)
                return True
            
            # Concatenate chunks
            logger.info(f"Concatenating {len(chunks)} audio chunks...")
            trimmed_audio = AudioSegment.empty()
            for chunk in chunks:
                trimmed_audio += chunk
                # Add small gap between chunks for natural flow
                if chunk != chunks[-1]:  # Don't add gap after last chunk
                    trimmed_audio += AudioSegment.silent(duration=200)  # 200ms gap
            
            # Export trimmed audio
            trimmed_audio.export(str(self.trimmed_audio), format="wav")
            
            original_duration = len(audio) / 1000
            trimmed_duration = len(trimmed_audio) / 1000
            logger.info(f"Silence handling completed: {self.trimmed_audio}")
            logger.info(f"Duration reduced from {original_duration:.2f}s to {trimmed_duration:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in silence handling: {str(e)}")
            return False
    
    def step5_speech_transcription(self) -> bool:
        """
        Step 4: Transcribe speech using Whisper ASR with word-level timestamps.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Step 4: Transcribing speech with Whisper...")
        
        try:
            # Load Whisper model
            logger.info("Loading Whisper model...")
            model = whisper.load_model("base")  # You can use "small", "medium", "large" for better accuracy
            
            # Transcribe audio
            logger.info("Transcribing audio...")
            result = model.transcribe(
                str(self.trimmed_audio),
                word_timestamps=True,
                verbose=False
            )
            
            # Save transcript with timestamps
            with open(self.transcript_file, 'w', encoding='utf-8') as f:
                f.write(f"Transcript for: {self.input_file.name}\n")
                f.write("=" * 50 + "\n\n")
                
                # Full transcript
                f.write("FULL TRANSCRIPT:\n")
                f.write(result["text"] + "\n\n")
                
                # Segment-level timestamps
                f.write("SEGMENTS WITH TIMESTAMPS:\n")
                for segment in result["segments"]:
                    start_time = segment["start"]
                    end_time = segment["end"]
                    text = segment["text"]
                    f.write(f"[{start_time:.2f}s - {end_time:.2f}s]: {text}\n")
                
                # Word-level timestamps if available
                if "words" in result:
                    f.write("\nWORD-LEVEL TIMESTAMPS:\n")
                    for word_info in result["words"]:
                        word = word_info["word"]
                        start = word_info["start"]
                        end = word_info["end"]
                        f.write(f"{word} [{start:.2f}s - {end:.2f}s]\n")
            
            logger.info(f"Transcription completed: {self.transcript_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error in speech transcription: {str(e)}")
            return False
    
    def step6_output_synchronization(self) -> bool:
        """
        Step 5: Merge cleaned audio with original video using MoviePy.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Step 5: Synchronizing cleaned audio with video...")
        
        try:
            # Load original video
            logger.info("Loading original video...")
            video = VideoFileClip(str(self.input_file))
            
            # Load cleaned audio
            logger.info("Loading cleaned audio...")
            cleaned_audio = AudioFileClip(str(self.trimmed_audio))
            
            # Set the cleaned audio to the video
            logger.info("Merging audio and video...")
            final_video = video.set_audio(cleaned_audio)
            
            # Write final video
            logger.info("Writing final video...")
            final_video.write_videofile(
                str(self.final_video),
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # Clean up
            cleaned_audio.close()
            final_video.close()
            video.close()
            
            logger.info(f"Final video created: {self.final_video}")
            return True
            
        except Exception as e:
            logger.error(f"Error in output synchronization: {str(e)}")
            return False
    
    def process(self) -> bool:
        """
        Execute the complete audio noise removal pipeline.
        
        Returns:
            bool: True if all steps completed successfully
        """
        logger.info("Starting complete audio noise removal pipeline...")
        
        steps = [
            ("Audio Extraction", self.step1_extract_audio),
            ("Vocal Separation (Demucs)", self.step2_vocal_separation_demucs),
            ("Spectral Denoising", self.step3_spectral_denoising),
            ("Silence Handling", self.step4_silence_handling),
            ("Speech Transcription", self.step5_speech_transcription),
            ("Output Synchronization", self.step6_output_synchronization)
        ]
        
        for step_name, step_function in steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"Executing: {step_name}")
            logger.info(f"{'='*60}")
            
            if not step_function():
                logger.error(f"Failed at step: {step_name}")
                return False
                
        logger.info(f"\n{'='*60}")
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"{'='*60}")
        logger.info(f"Final video: {self.final_video}")
        logger.info(f"Transcript: {self.transcript_file}")
        
        return True
    
    def cleanup_temp_files(self):
        """Remove temporary files to save space."""
        logger.info("Cleaning up temporary files...")
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {str(e)}")


def main():
    """
    Main function to execute the audio noise removal pipeline.
    """
    # Configuration
    input_video = "Unit-2 TA Session 1.mp4"
    output_directory = "cleaned_output"
    
    # Verify input file exists
    if not Path(input_video).exists():
        logger.error(f"Input file not found: {input_video}")
        return False
    
    # Initialize and run the pipeline
    noise_remover = AudioNoiseRemover(input_video, output_directory)
    
    try:
        success = noise_remover.process()
        
        if success:
            logger.info("\nProcessing completed successfully!")
            logger.info(f"Check the '{output_directory}' folder for results.")
            
            # Optionally clean up temporary files
            cleanup = input("Clean up temporary files? (y/n): ").lower().strip()
            if cleanup == 'y':
                noise_remover.cleanup_temp_files()
        else:
            logger.error("Processing failed. Check the logs above for details.")
            
        return success
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    main()
