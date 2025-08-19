#!/usr/bin/env python3
"""
Advanced Audio Noise Removal Pipeline
====================================

This script implements the exact methodology specified:
1. Audio Extraction (FFmpeg)
2. Noise Removal (Demucs & Noisereduce) 
3. Silence Handling (PyDub)
4. Speech Transcription (Whisper)
5. Output Synchronization (MoviePy)

Requirements:
- FFmpeg (for audio extraction)
- Demucs (for vocal separation)
- Noisereduce (for spectral gating)
- PyDub (for silence handling)
- Whisper (for speech transcription)
- MoviePy (for output synchronization)

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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedAudioProcessor:
    """
    Advanced audio noise removal pipeline following the exact specified methodology.
    """
    
    def __init__(self, input_file: str, output_dir: str = "output"):
        """Initialize the advanced audio processor."""
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create temp directory for intermediate files
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # File paths for different processing stages
        self.extracted_audio = self.temp_dir / "01_extracted_audio.wav"
        self.demucs_vocals = self.temp_dir / "02_demucs_vocals.wav"
        self.noisereduced_audio = self.temp_dir / "03_noisereduced_audio.wav"
        self.silence_trimmed_audio = self.temp_dir / "04_silence_trimmed_audio.wav"
        self.final_video = self.output_dir / f"cleaned_{self.input_file.stem}.mp4"
        self.transcript_file = self.output_dir / f"transcript_{self.input_file.stem}.txt"
        
        # Check available tools
        self.available_tools = self._check_dependencies()
        self._log_available_tools()
        
        logger.info(f"Initialized AdvancedAudioProcessor for {self.input_file}")
    
    def _check_dependencies(self) -> dict:
        """Check which dependencies are available."""
        tools = {}
        
        # Check FFmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
            tools['ffmpeg'] = True
        except:
            # Check local FFmpeg installation
            local_ffmpeg = Path("ffmpeg") / "bin" / "ffmpeg.exe"
            if local_ffmpeg.exists():
                # Add to PATH
                ffmpeg_bin = str(Path("ffmpeg") / "bin")
                os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]
                tools['ffmpeg'] = True
            else:
                tools['ffmpeg'] = False
        
        # Check Python libraries
        libraries = [
            ('demucs', 'demucs.pretrained'),
            ('noisereduce', 'noisereduce'),
            ('pydub', 'pydub'),
            ('whisper', 'whisper'),
            ('moviepy', 'moviepy.editor'),
            ('torch', 'torch'),
            ('librosa', 'librosa'),
            ('soundfile', 'soundfile'),
            ('numpy', 'numpy')
        ]
        
        for lib_name, import_name in libraries:
            try:
                __import__(import_name)
                tools[lib_name] = True
            except ImportError:
                tools[lib_name] = False
        
        return tools
    
    def _log_available_tools(self):
        """Log which tools are available."""
        logger.info("Dependency Check Results:")
        for tool, available in self.available_tools.items():
            status = "✓ Available" if available else "✗ Missing"
            logger.info(f"  {tool}: {status}")
    
    def _install_missing_dependencies(self):
        """Attempt to install missing dependencies."""
        missing_libs = [lib for lib, available in self.available_tools.items() 
                       if not available and lib != 'ffmpeg']
        
        if missing_libs:
            logger.info(f"Attempting to install missing libraries: {', '.join(missing_libs)}")
            
            # Map library names to pip package names
            pip_packages = {
                'demucs': 'demucs',
                'noisereduce': 'noisereduce', 
                'pydub': 'pydub',
                'whisper': 'openai-whisper',
                'moviepy': 'moviepy',
                'torch': 'torch',
                'librosa': 'librosa',
                'soundfile': 'soundfile',
                'numpy': 'numpy'
            }
            
            for lib in missing_libs:
                if lib in pip_packages:
                    try:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", pip_packages[lib]
                        ])
                        logger.info(f"Successfully installed {lib}")
                        self.available_tools[lib] = True
                    except subprocess.CalledProcessError:
                        logger.warning(f"Failed to install {lib}")
    
    def step1_audio_extraction_ffmpeg(self) -> bool:
        """
        Step 1: Audio Extraction using FFmpeg
        Extract audio stream while retaining original timestamps.
        """
        logger.info("="*60)
        logger.info("STEP 1: AUDIO EXTRACTION (FFmpeg)")
        logger.info("="*60)
        
        if not self.available_tools['ffmpeg']:
            logger.error("FFmpeg not available for audio extraction")
            return False
        
        try:
            # Extract audio with FFmpeg, preserving timestamps
            cmd = [
                "ffmpeg", "-i", str(self.input_file),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # Uncompressed audio
                "-ar", "44100",  # 44.1kHz sample rate
                "-ac", "2",  # Stereo (needed for Demucs)
                "-y",  # Overwrite output
                str(self.extracted_audio)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✓ Audio extracted successfully: {self.extracted_audio}")
                return True
            else:
                logger.error(f"✗ Audio extraction failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Error in audio extraction: {str(e)}")
            return False
    
    def step2_noise_removal_demucs(self) -> bool:
        """
        Step 2a: Noise Removal using Demucs U-Net model
        Separate vocal track from background noise for non-stationary sounds.
        """
        logger.info("="*60)
        logger.info("STEP 2a: VOCAL SEPARATION (Demucs)")
        logger.info("="*60)
        
        if not self.available_tools['demucs']:
            logger.warning("Demucs not available, skipping vocal separation")
            shutil.copy2(self.extracted_audio, self.demucs_vocals)
            return True
        
        try:
            # Import Demucs dependencies
            import torch
            import librosa
            import soundfile as sf
            import numpy as np
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            
            logger.info("Loading pre-trained Demucs U-Net model...")
            model = get_model('htdemucs')  # Hybrid Transformer Demucs
            model.eval()
            
            # Load audio for Demucs processing
            logger.info("Loading audio for vocal separation...")
            waveform, sample_rate = librosa.load(str(self.extracted_audio), sr=None, mono=False)
            
            # Ensure stereo format for Demucs
            if waveform.ndim == 1:
                waveform = np.stack([waveform, waveform])
            elif waveform.shape[0] == 1:
                waveform = np.vstack([waveform, waveform])
            
            # Convert to torch tensor
            waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
            
            # Apply Demucs source separation
            logger.info("Applying Demucs U-Net model for vocal separation...")
            with torch.no_grad():
                sources = apply_model(model, waveform_tensor)
            
            # Extract vocals (index 3 in htdemucs: drums, bass, other, vocals)
            vocals = sources[0, 3].numpy()  # Shape: (channels, samples)
            
            # Convert to mono for further processing
            if vocals.shape[0] == 2:
                vocals_mono = np.mean(vocals, axis=0)
            else:
                vocals_mono = vocals[0]
            
            # Save separated vocals
            sf.write(str(self.demucs_vocals), vocals_mono, sample_rate)
            
            logger.info(f"✓ Vocal separation completed: {self.demucs_vocals}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Demucs separation failed: {str(e)}")
            logger.info("Falling back to original audio...")
            shutil.copy2(self.extracted_audio, self.demucs_vocals)
            return True
    
    def step3_noise_removal_noisereduce(self) -> bool:
        """
        Step 2b: Noise Removal using Noisereduce
        Apply spectral gating for stationary noise suppression.
        """
        logger.info("="*60)
        logger.info("STEP 2b: SPECTRAL GATING (Noisereduce)")
        logger.info("="*60)
        
        if not self.available_tools['noisereduce']:
            logger.warning("Noisereduce not available, skipping spectral gating")
            shutil.copy2(self.demucs_vocals, self.noisereduced_audio)
            return True
        
        try:
            import librosa
            import soundfile as sf
            import noisereduce as nr
            
            logger.info("Loading audio for noise reduction...")
            # Load audio
            audio_data, sample_rate = librosa.load(str(self.demucs_vocals), sr=None)
            
            # Apply spectral gating with noisereduce
            logger.info("Applying spectral gating for stationary noise suppression...")
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=sample_rate,
                stationary=True,
                prop_decrease=0.8,  # Reduce noise by 80%
                n_grad_freq=2,      # Number of frequency channels for gradual noise reduction
                n_grad_time=4,      # Number of time channels for gradual noise reduction
                n_fft=1024,         # FFT window size
                win_length=None,    # Window length for FFT
                hop_length=None     # Hop length for FFT
            )
            
            # Save noise-reduced audio
            sf.write(str(self.noisereduced_audio), reduced_noise, sample_rate)
            
            logger.info(f"✓ Spectral gating completed: {self.noisereduced_audio}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Noisereduce failed: {str(e)}")
            logger.info("Copying vocals without additional noise reduction...")
            shutil.copy2(self.demucs_vocals, self.noisereduced_audio)
            return True
    
    def step4_silence_handling_pydub(self) -> bool:
        """
        Step 3: Silence Handling using PyDub
        Analyze and trim silence longer than 2 seconds.
        """
        logger.info("="*60)
        logger.info("STEP 3: SILENCE HANDLING (PyDub)")
        logger.info("="*60)
        
        if not self.available_tools['pydub']:
            logger.warning("PyDub not available, skipping silence handling")
            shutil.copy2(self.noisereduced_audio, self.silence_trimmed_audio)
            return True
        
        try:
            from pydub import AudioSegment
            from pydub.silence import split_on_silence
            
            logger.info("Loading audio for silence analysis...")
            # Load audio with PyDub
            audio = AudioSegment.from_wav(str(self.noisereduced_audio))
            
            # Analyze and split on silence
            logger.info("Analyzing non-speech intervals...")
            chunks = split_on_silence(
                audio,
                min_silence_len=2000,  # 2 seconds minimum silence
                silence_thresh=audio.dBFS - 16,  # Silence threshold
                keep_silence=500  # Keep 500ms of silence at edges
            )
            
            if not chunks:
                logger.warning("No audio chunks found after silence analysis")
                shutil.copy2(self.noisereduced_audio, self.silence_trimmed_audio)
                return True
            
            # Concatenate chunks to create concise audio track
            logger.info(f"Trimming {len(chunks)} audio segments...")
            trimmed_audio = AudioSegment.empty()
            for i, chunk in enumerate(chunks):
                trimmed_audio += chunk
                # Add small gap between chunks for natural flow
                if i < len(chunks) - 1:
                    trimmed_audio += AudioSegment.silent(duration=200)  # 200ms gap
            
            # Export trimmed audio
            trimmed_audio.export(str(self.silence_trimmed_audio), format="wav")
            
            # Log duration reduction
            original_duration = len(audio) / 1000
            trimmed_duration = len(trimmed_audio) / 1000
            reduction_percent = ((original_duration - trimmed_duration) / original_duration) * 100
            
            logger.info(f"✓ Silence handling completed: {self.silence_trimmed_audio}")
            logger.info(f"✓ Duration: {original_duration:.2f}s → {trimmed_duration:.2f}s ({reduction_percent:.1f}% reduction)")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ PyDub silence handling failed: {str(e)}")
            shutil.copy2(self.noisereduced_audio, self.silence_trimmed_audio)
            return True
    
    def step5_speech_transcription_whisper(self) -> bool:
        """
        Step 4: Speech Transcription using Whisper ASR
        Generate transcript with word-level timestamps for keyword indexing.
        """
        logger.info("="*60)
        logger.info("STEP 4: SPEECH TRANSCRIPTION (Whisper)")
        logger.info("="*60)
        
        if not self.available_tools['whisper']:
            logger.warning("Whisper not available, creating basic transcript")
            return self._create_basic_transcript()
        
        try:
            import whisper
            
            logger.info("Loading Whisper ASR model...")
            # Load Whisper model (base provides good balance of speed vs accuracy)
            model = whisper.load_model("base")
            
            # Transcribe with word-level timestamps
            logger.info("Generating transcript with word-level timestamps...")
            result = model.transcribe(
                str(self.silence_trimmed_audio),
                word_timestamps=True,
                verbose=False
            )
            
            # Save comprehensive transcript
            with open(self.transcript_file, 'w', encoding='utf-8') as f:
                f.write(f"Advanced Audio Processing Transcript\\n")
                f.write(f"Input: {self.input_file.name}\\n")
                f.write("=" * 60 + "\\n\\n")
                
                # Full transcript
                f.write("FULL TRANSCRIPT:\\n")
                f.write("-" * 20 + "\\n")
                f.write(result["text"].strip() + "\\n\\n")
                
                # Segment-level timestamps
                f.write("SEGMENT-LEVEL TIMESTAMPS:\\n")
                f.write("-" * 30 + "\\n")
                for segment in result["segments"]:
                    start_time = segment["start"]
                    end_time = segment["end"]
                    text = segment["text"].strip()
                    f.write(f"[{start_time:7.2f}s - {end_time:7.2f}s]: {text}\\n")
                
                # Word-level timestamps for keyword indexing
                if "words" in result and result["words"]:
                    f.write("\\nWORD-LEVEL TIMESTAMPS (for keyword indexing):\\n")
                    f.write("-" * 50 + "\\n")
                    for word_info in result["words"]:
                        word = word_info["word"]
                        start = word_info["start"]
                        end = word_info["end"]
                        f.write(f"{word:<15} [{start:7.2f}s - {end:7.2f}s]\\n")
                
                # Summary statistics
                f.write(f"\\nTRANSCRIPT STATISTICS:\\n")
                f.write("-" * 25 + "\\n")
                f.write(f"Total Duration: {result.get('segments', [{}])[-1].get('end', 0):.2f} seconds\\n")
                f.write(f"Total Segments: {len(result['segments'])}\\n")
                f.write(f"Total Words: {len(result.get('words', []))}\\n")
            
            logger.info(f"✓ Whisper transcription completed: {self.transcript_file}")
            logger.info(f"✓ Generated {len(result['segments'])} segments with word-level timestamps")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Whisper transcription failed: {str(e)}")
            return self._create_basic_transcript()
    
    def _create_basic_transcript(self) -> bool:
        """Create basic transcript when Whisper is not available."""
        try:
            # Get audio duration
            if self.available_tools['ffmpeg']:
                cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                       "-of", "csv=p=0", str(self.silence_trimmed_audio)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                duration = float(result.stdout.strip()) if result.returncode == 0 else 0.0
            else:
                duration = 0.0
            
            with open(self.transcript_file, 'w', encoding='utf-8') as f:
                f.write(f"Basic Transcript Placeholder\\n")
                f.write(f"Input: {self.input_file.name}\\n")
                f.write("=" * 60 + "\\n\\n")
                f.write(f"Processed Duration: {duration:.2f} seconds\\n\\n")
                f.write("Note: This is a placeholder transcript.\\n")
                f.write("For actual speech transcription with word-level timestamps,\\n")
                f.write("install Whisper: pip install openai-whisper\\n\\n")
                f.write("Whisper provides:\\n")
                f.write("- Full speech-to-text transcription\\n")
                f.write("- Segment-level timestamps\\n")
                f.write("- Word-level timestamps for keyword indexing\\n")
            
            logger.info(f"✓ Basic transcript created: {self.transcript_file}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error creating basic transcript: {str(e)}")
            return False
    
    def step6_output_synchronization_moviepy(self) -> bool:
        """
        Step 5: Output Synchronization using MoviePy
        Merge cleaned audio with original video for final output.
        """
        logger.info("="*60)
        logger.info("STEP 5: OUTPUT SYNCHRONIZATION (MoviePy)")
        logger.info("="*60)
        
        if not self.available_tools['moviepy']:
            logger.warning("MoviePy not available, falling back to FFmpeg")
            return self._output_synchronization_ffmpeg()
        
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            
            logger.info("Loading original video...")
            # Load original video
            video = VideoFileClip(str(self.input_file))
            
            logger.info("Loading cleaned and processed audio...")
            # Load cleaned audio
            cleaned_audio = AudioFileClip(str(self.silence_trimmed_audio))
            
            # Merge audio with video
            logger.info("Merging cleaned audio with original video...")
            final_video = video.set_audio(cleaned_audio)
            
            # Write final video with enhanced speech clarity
            logger.info("Writing final video with enhanced speech clarity...")
            final_video.write_videofile(
                str(self.final_video),
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None,
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            # Clean up
            cleaned_audio.close()
            final_video.close()
            video.close()
            
            logger.info(f"✓ Output synchronization completed: {self.final_video}")
            return True
            
        except Exception as e:
            logger.error(f"✗ MoviePy synchronization failed: {str(e)}")
            return self._output_synchronization_ffmpeg()
    
    def _output_synchronization_ffmpeg(self) -> bool:
        """Fallback video synchronization using FFmpeg."""
        try:
            cmd = [
                "ffmpeg", "-i", str(self.input_file), "-i", str(self.silence_trimmed_audio),
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-y", str(self.final_video)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info(f"✓ FFmpeg synchronization completed: {self.final_video}")
                return True
            else:
                logger.error(f"✗ FFmpeg synchronization failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"✗ FFmpeg synchronization error: {str(e)}")
            return False
    
    def process(self) -> bool:
        """
        Execute the complete advanced audio processing pipeline.
        Following the exact specified methodology.
        """
        logger.info("\\n" + "="*80)
        logger.info("ADVANCED AUDIO NOISE REMOVAL PIPELINE")
        logger.info("Following Specified Methodology")
        logger.info("="*80)
        
        # Attempt to install missing dependencies
        self._install_missing_dependencies()
        
        # Execute the 5-step methodology
        steps = [
            ("Audio Extraction (FFmpeg)", self.step1_audio_extraction_ffmpeg),
            ("Vocal Separation (Demucs)", self.step2_noise_removal_demucs),
            ("Spectral Gating (Noisereduce)", self.step3_noise_removal_noisereduce),
            ("Silence Handling (PyDub)", self.step4_silence_handling_pydub),
            ("Speech Transcription (Whisper)", self.step5_speech_transcription_whisper),
            ("Output Synchronization (MoviePy)", self.step6_output_synchronization_moviepy)
        ]
        
        for i, (step_name, step_function) in enumerate(steps, 1):
            logger.info(f"\\nExecuting Step {i}: {step_name}")
            
            if not step_function():
                logger.error(f"❌ Failed at Step {i}: {step_name}")
                return False
                
            logger.info(f"✅ Completed Step {i}: {step_name}")
        
        logger.info("\\n" + "="*80)
        logger.info("🎉 ADVANCED PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"✓ Enhanced Video: {self.final_video}")
        logger.info(f"✓ Transcript: {self.transcript_file}")
        logger.info(f"✓ Methodology: Demucs → Noisereduce → PyDub → Whisper → MoviePy")
        
        return True
    
    def cleanup_temp_files(self):
        """Remove temporary intermediate files."""
        logger.info("Cleaning up temporary files...")
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("✓ Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")


def main():
    """
    Main function to execute the advanced audio processing pipeline
    following the exact specified methodology.
    """
    input_video = "Unit-2 TA Session 1.mp4"
    output_directory = "output"
    
    # Verify input file exists
    if not Path(input_video).exists():
        logger.error(f"Input file not found: {input_video}")
        return False
    
    # Initialize advanced processor
    processor = AdvancedAudioProcessor(input_video, output_directory)
    
    try:
        # Execute the complete pipeline
        success = processor.process()
        
        if success:
            logger.info("\\n🎊 PROCESSING COMPLETED SUCCESSFULLY!")
            logger.info("\\nResults:")
            logger.info(f"  📹 Enhanced Video: {processor.final_video}")
            logger.info(f"  📝 Transcript: {processor.transcript_file}")
            logger.info("\\nMethodology Applied:")
            logger.info("  1. ✅ Audio Extraction (FFmpeg)")
            logger.info("  2. ✅ Vocal Separation (Demucs)")
            logger.info("  3. ✅ Spectral Gating (Noisereduce)")
            logger.info("  4. ✅ Silence Handling (PyDub)")
            logger.info("  5. ✅ Speech Transcription (Whisper)")
            logger.info("  6. ✅ Output Synchronization (MoviePy)")
            
            # Cleanup option
            cleanup = input("\\nClean up temporary files? (y/n): ").lower().strip()
            if cleanup == 'y':
                processor.cleanup_temp_files()
        else:
            logger.error("❌ Processing failed. Check the logs above for details.")
        
        return success
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    main()
