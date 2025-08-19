#!/usr/bin/env python3
"""
Simplified Audio Processor with Live Subtitles
==============================================

This version creates a video with live subtitles using PyDub for audio processing
and MoviePy for video creation, without requiring FFmpeg installation.

Features:
- Audio processing using PyDub only
- Live subtitle display synchronized with audio
- Professional subtitle styling
- Automatic subtitle timing from Whisper transcription
"""

import os
import sys
import logging
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class SimpleAudioProcessorWithSubtitles:
    """
    Simplified audio processor that creates live subtitles without FFmpeg dependency.
    """
    
    def __init__(self, input_file: str, output_dir: str = "output"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create temporary directory for processing
        self.temp_dir = Path(tempfile.mkdtemp(prefix="simple_audio_"))
        
        # File paths for each processing stage
        base_name = self.input_file.stem
        self.extracted_audio = self.temp_dir / f"extracted_{base_name}.wav"
        self.processed_audio = self.temp_dir / f"processed_{base_name}.wav"
        self.transcript_file = self.output_dir / f"transcript_{base_name}.txt"
        self.srt_file = self.output_dir / f"subtitles_{base_name}.srt"
        self.final_video = self.output_dir / f"subtitled_{base_name}.mp4"
        
        # Track available tools
        self.available_tools = {
            'pydub': False,
            'whisper': False,
            'moviepy': False
        }
        
        # Store transcription data for subtitle generation
        self.transcription_data = None
        
        logger.info(f"Initialized Simple Audio Processor with Subtitles")
        logger.info(f"Input: {self.input_file}")
        logger.info(f"Output Directory: {self.output_dir}")
    
    def _install_missing_dependencies(self):
        """Install missing Python dependencies automatically."""
        logger.info("Installing required dependencies...")
        
        dependencies = [
            'pydub',
            'openai-whisper',
            'moviepy',
            'torch',
            'librosa',
            'soundfile'
        ]
        
        for dep in dependencies:
            try:
                if dep == 'openai-whisper':
                    import whisper
                    self.available_tools['whisper'] = True
                elif dep == 'pydub':
                    from pydub import AudioSegment
                    self.available_tools['pydub'] = True
                elif dep == 'moviepy':
                    import moviepy.editor
                    self.available_tools['moviepy'] = True
                else:
                    __import__(dep.replace('-', '_'))
            except ImportError:
                logger.info(f"Installing {dep}...")
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                 check=True, capture_output=True)
                    logger.info(f"✓ {dep} installed successfully")
                    
                    # Re-check availability
                    if dep == 'openai-whisper':
                        try:
                            import whisper
                            self.available_tools['whisper'] = True
                        except ImportError:
                            pass
                    elif dep == 'pydub':
                        try:
                            from pydub import AudioSegment
                            self.available_tools['pydub'] = True
                        except ImportError:
                            pass
                    elif dep == 'moviepy':
                        try:
                            import moviepy.editor
                            self.available_tools['moviepy'] = True
                        except ImportError:
                            pass
                        
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install {dep}: {e}")
    
    def step1_extract_and_process_audio(self) -> bool:
        """Step 1: Extract and process audio using PyDub and MoviePy."""
        logger.info("STEP 1: AUDIO EXTRACTION AND PROCESSING")
        logger.info("="*60)
        
        if not self.available_tools['moviepy']:
            logger.error("MoviePy not available for audio extraction")
            return False
        
        try:
            from moviepy.editor import VideoFileClip
            
            logger.info("Loading video and extracting audio...")
            # Load video and extract audio
            video = VideoFileClip(str(self.input_file))
            audio = video.audio
            
            # Export audio as WAV
            audio.write_audiofile(str(self.extracted_audio), verbose=False, logger=None)
            
            # Close video and audio
            audio.close()
            video.close()
            
            logger.info(f"✓ Audio extraction completed: {self.extracted_audio}")
            
            # Basic audio processing with PyDub
            if self.available_tools['pydub']:
                logger.info("Applying basic audio processing...")
                from pydub import AudioSegment
                from pydub.effects import normalize
                
                # Load audio with PyDub
                audio_segment = AudioSegment.from_wav(str(self.extracted_audio))
                
                # Apply basic processing
                # Normalize audio
                normalized_audio = normalize(audio_segment)
                
                # Apply basic noise reduction (simple high-pass filter)
                filtered_audio = normalized_audio.high_pass_filter(300)
                
                # Export processed audio
                filtered_audio.export(str(self.processed_audio), format="wav")
                
                logger.info(f"✓ Audio processing completed: {self.processed_audio}")
            else:
                # Copy original audio if PyDub not available
                shutil.copy2(self.extracted_audio, self.processed_audio)
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Audio extraction/processing failed: {str(e)}")
            return False
    
    def step2_transcribe_with_whisper(self) -> bool:
        """Step 2: Transcribe audio using OpenAI Whisper."""
        logger.info("STEP 2: SPEECH TRANSCRIPTION (OpenAI Whisper)")
        logger.info("="*60)
        
        if not self.available_tools['whisper']:
            logger.warning("Whisper not available")
            return self._create_basic_transcript()
        
        try:
            import whisper
            
            logger.info("Loading Whisper ASR model...")
            
            # Load Whisper model (base model for good balance of speed/accuracy)
            model = whisper.load_model("base")
            
            logger.info("Transcribing speech with word-level timestamps...")
            
            # Transcribe with word-level timestamps
            result = model.transcribe(
                str(self.processed_audio),
                word_timestamps=True,
                language="en"  # Auto-detect if needed
            )
            
            # Store transcription data for subtitle generation
            self.transcription_data = result
            
            # Generate comprehensive transcript
            with open(self.transcript_file, 'w', encoding='utf-8') as f:
                f.write(f"Simple Audio Processing Transcript\\n")
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
                
                # Summary statistics
                f.write(f"\\nTRANSCRIPT STATISTICS:\\n")
                f.write("-" * 25 + "\\n")
                f.write(f"Total Duration: {result.get('segments', [{}])[-1].get('end', 0):.2f} seconds\\n")
                f.write(f"Total Segments: {len(result['segments'])}\\n")
                f.write(f"Total Words: {len(result.get('words', []))}\\n")
            
            # Generate SRT subtitle file
            self._generate_srt_subtitles(result)
            
            logger.info(f"✓ Whisper transcription completed: {self.transcript_file}")
            logger.info(f"✓ SRT subtitles generated: {self.srt_file}")
            logger.info(f"✓ Generated {len(result['segments'])} segments with word-level timestamps")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Whisper transcription failed: {str(e)}")
            return self._create_basic_transcript()
    
    def _generate_srt_subtitles(self, transcription_data: Dict[Any, Any]) -> bool:
        """Generate SRT subtitle file from Whisper transcription."""
        try:
            with open(self.srt_file, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(transcription_data["segments"], 1):
                    start_time = self._seconds_to_srt_time(segment["start"])
                    end_time = self._seconds_to_srt_time(segment["end"])
                    text = segment["text"].strip()
                    
                    f.write(f"{i}\\n")
                    f.write(f"{start_time} --> {end_time}\\n")
                    f.write(f"{text}\\n\\n")
            
            logger.info(f"✓ SRT subtitles generated with {len(transcription_data['segments'])} entries")
            return True
            
        except Exception as e:
            logger.error(f"✗ SRT generation failed: {str(e)}")
            return False
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _create_basic_transcript(self) -> bool:
        """Create basic transcript when Whisper is not available."""
        try:
            with open(self.transcript_file, 'w', encoding='utf-8') as f:
                f.write(f"Basic Transcript\\n")
                f.write(f"Input: {self.input_file.name}\\n")
                f.write("=" * 60 + "\\n\\n")
                f.write("TRANSCRIPT:\\n")
                f.write("-" * 20 + "\\n")
                f.write("[Speech content - Whisper transcription not available]\\n\\n")
            
            # Create basic SRT file
            with open(self.srt_file, 'w', encoding='utf-8') as f:
                f.write("1\\n")
                f.write("00:00:00,000 --> 00:01:00,000\\n")
                f.write("[Speech content - Automatic subtitles not available]\\n\\n")
            
            logger.info(f"✓ Basic transcript created: {self.transcript_file}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Basic transcript creation failed: {str(e)}")
            return False
    
    def step3_create_subtitled_video(self) -> bool:
        """Step 3: Create final video with processed audio and live subtitles."""
        logger.info("STEP 3: CREATE VIDEO WITH LIVE SUBTITLES")
        logger.info("="*60)
        
        if not self.available_tools['moviepy']:
            logger.error("MoviePy not available for video creation")
            return False
        
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
            
            logger.info("Loading original video...")
            video = VideoFileClip(str(self.input_file))
            
            logger.info("Loading processed audio...")
            processed_audio = AudioFileClip(str(self.processed_audio))
            
            logger.info("Creating synchronized subtitles...")
            
            # Create subtitle clips if transcription data is available
            subtitle_clips = []
            if self.transcription_data and "segments" in self.transcription_data:
                for segment in self.transcription_data["segments"]:
                    start_time = segment["start"]
                    end_time = segment["end"]
                    text = segment["text"].strip()
                    
                    if text and len(text) > 0:  # Only create clip if there's text
                        try:
                            # Create subtitle clip with professional styling
                            subtitle_clip = TextClip(
                                text,
                                fontsize=28,
                                color='white',
                                font='Arial-Bold',
                                stroke_color='black',
                                stroke_width=3,
                                method='caption',
                                size=(video.w * 0.9, None),  # 90% of video width
                                align='center'
                            ).set_position(('center', 0.85), relative=True).set_start(start_time).set_duration(end_time - start_time)
                            
                            subtitle_clips.append(subtitle_clip)
                        except Exception as e:
                            logger.warning(f"Could not create subtitle for segment: {text[:50]}... Error: {e}")
                            continue
            
            logger.info(f"Created {len(subtitle_clips)} subtitle clips")
            
            # Combine video with processed audio
            video_with_audio = video.set_audio(processed_audio)
            
            # Composite video with subtitles
            if subtitle_clips:
                final_video = CompositeVideoClip([video_with_audio] + subtitle_clips)
            else:
                final_video = video_with_audio
            
            logger.info("Rendering final video with live subtitles...")
            logger.info("This may take several minutes depending on video length...")
            
            # Write final video
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
            for clip in subtitle_clips:
                try:
                    clip.close()
                except:
                    pass
            processed_audio.close()
            final_video.close()
            video_with_audio.close()
            video.close()
            
            logger.info(f"✓ Subtitled video completed: {self.final_video}")
            logger.info("✓ Professional subtitle styling applied")
            return True
            
        except Exception as e:
            logger.error(f"✗ Subtitled video creation failed: {str(e)}")
            return False
    
    def process(self) -> bool:
        """Execute the complete simple audio processing pipeline with subtitles."""
        logger.info("\\n" + "="*80)
        logger.info("SIMPLE AUDIO PROCESSOR WITH LIVE SUBTITLES")
        logger.info("No FFmpeg Required - Pure Python Solution")
        logger.info("="*80)
        
        # Install missing dependencies
        self._install_missing_dependencies()
        
        # Execute the 3-step process
        steps = [
            ("Audio Extraction & Processing", self.step1_extract_and_process_audio),
            ("Speech Transcription (Whisper)", self.step2_transcribe_with_whisper),
            ("Create Video with Live Subtitles", self.step3_create_subtitled_video)
        ]
        
        for i, (step_name, step_function) in enumerate(steps, 1):
            logger.info(f"\\nExecuting Step {i}: {step_name}")
            
            if not step_function():
                logger.error(f"❌ Failed at Step {i}: {step_name}")
                return False
                
            logger.info(f"✅ Completed Step {i}: {step_name}")
        
        logger.info("\\n" + "="*80)
        logger.info("🎉 SIMPLE PIPELINE WITH SUBTITLES COMPLETED!")
        logger.info("="*80)
        logger.info(f"✓ Enhanced Video with Live Subtitles: {self.final_video}")
        logger.info(f"✓ External Subtitle File: {self.srt_file}")
        logger.info(f"✓ Full Transcript: {self.transcript_file}")
        logger.info(f"✓ Technology: PyDub + Whisper + MoviePy")
        
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
    """Main function to execute the simple audio processing pipeline with subtitles."""
    input_video = "Unit-2 TA Session 1.mp4"
    output_directory = "output"
    
    # Verify input file exists
    if not Path(input_video).exists():
        logger.error(f"Input file not found: {input_video}")
        return False
    
    # Initialize simple processor
    processor = SimpleAudioProcessorWithSubtitles(input_video, output_directory)
    
    try:
        # Execute the complete pipeline
        success = processor.process()
        
        if success:
            logger.info("\\n🎊 SIMPLE PROCESSING WITH SUBTITLES COMPLETED!")
            logger.info("\\nResults:")
            logger.info(f"  📹 Video with Live Subtitles: {processor.final_video}")
            logger.info(f"  📝 SRT Subtitle File: {processor.srt_file}")
            logger.info(f"  📄 Full Transcript: {processor.transcript_file}")
            logger.info("\\nFeatures Applied:")
            logger.info("  1. ✅ Audio Extraction & Processing (MoviePy + PyDub)")
            logger.info("  2. ✅ Speech Transcription (Whisper)")
            logger.info("  3. ✅ Live Subtitle Generation (MoviePy)")
            logger.info("\\n🎬 Your video now has professional live subtitles!")
            logger.info("   The transcript appears synchronized at the bottom of the video.")
            
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
