#!/usr/bin/env python3
"""
Burned-In Subtitle Video Creator
===============================

This script creates a video with burned-in subtitles that are permanently
visible on the video frames. No manual subtitle loading required!

The subtitles are rendered directly onto the video, ensuring they work
on ANY player, platform, or device.
"""

import os
import sys
import logging
import subprocess
import re
import tempfile
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class BurnedSubtitleCreator:
    """Creates video with burned-in subtitles using FFmpeg or MoviePy."""
    
    def __init__(self, video_file: str, transcript_file: str, output_dir: str = "output"):
        self.video_file = Path(video_file)
        self.transcript_file = Path(transcript_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="burned_subs_"))
        
        base_name = self.video_file.stem.replace("cleaned_", "").replace("subtitled_", "")
        self.srt_file = self.temp_dir / f"subtitles_{base_name}.srt"
        self.final_video = self.output_dir / f"burned_subtitles_{base_name}.mp4"
        
        # Check available tools
        self.has_ffmpeg = self._check_ffmpeg()
        self.has_moviepy = self._check_moviepy()
        
        logger.info(f"Initialized Burned Subtitle Creator")
        logger.info(f"Video: {self.video_file}")
        logger.info(f"Transcript: {self.transcript_file}")
        logger.info(f"FFmpeg available: {self.has_ffmpeg}")
        logger.info(f"MoviePy available: {self.has_moviepy}")
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(["ffmpeg", "-version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def _check_moviepy(self) -> bool:
        """Check if MoviePy is available."""
        try:
            import moviepy.editor
            return True
        except ImportError:
            return False
    
    def _install_moviepy(self) -> bool:
        """Install MoviePy if needed."""
        try:
            logger.info("Installing MoviePy...")
            subprocess.run([sys.executable, "-m", "pip", "install", "moviepy"], 
                         check=True, capture_output=True)
            logger.info("✓ MoviePy installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ MoviePy installation failed: {e}")
            return False
    
    def parse_transcript_timestamps(self) -> list:
        """Parse the transcript file to extract timestamp information."""
        try:
            with open(self.transcript_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace literal \n with actual newlines for proper parsing
            content = content.replace('\\n', '\n')
            
            # Find the segment-level timestamps section
            timestamp_section = re.search(
                r'SEGMENT-LEVEL TIMESTAMPS:\n-+\n(.*?)\n\n', 
                content, 
                re.DOTALL
            )
            
            if not timestamp_section:
                logger.error("Could not find timestamp section in transcript")
                return []
            
            # Parse individual timestamp lines
            timestamp_lines = timestamp_section.group(1).strip().split('\n')
            segments = []
            
            for line in timestamp_lines:
                # Parse format: [   0.00s -    1.98s]: text
                match = re.match(r'\[\s*(\d+\.\d+)s\s*-\s*(\d+\.\d+)s\]:\s*(.*)', line)
                if match:
                    start_time = float(match.group(1))
                    end_time = float(match.group(2))
                    text = match.group(3).strip()
                    
                    if text and len(text) > 0:  # Only add non-empty text
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'text': text
                        })
            
            logger.info(f"✓ Parsed {len(segments)} subtitle segments from transcript")
            return segments
            
        except Exception as e:
            logger.error(f"✗ Failed to parse transcript: {str(e)}")
            return []
    
    def generate_srt_file(self, segments: list) -> bool:
        """Generate SRT subtitle file from segments."""
        try:
            with open(self.srt_file, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments, 1):
                    start_time = self._seconds_to_srt_time(segment['start'])
                    end_time = self._seconds_to_srt_time(segment['end'])
                    text = segment['text']
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
            
            logger.info(f"✓ SRT file generated: {self.srt_file}")
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
    
    def burn_subtitles_with_ffmpeg(self) -> bool:
        """Burn subtitles into video using FFmpeg (most reliable method)."""
        try:
            if not self.has_ffmpeg:
                logger.error("FFmpeg not available for burning subtitles")
                return False
            
            logger.info("Burning subtitles into video using FFmpeg...")
            logger.info("This creates permanent subtitles that work on any player!")
            
            # FFmpeg command to burn subtitles
            cmd = [
                "ffmpeg",
                "-i", str(self.video_file),
                "-vf", f"subtitles={str(self.srt_file)}:force_style='FontName=Arial,FontSize=24,PrimaryColour=&Hffffff,OutlineColour=&H000000,BorderStyle=3,Outline=2,Shadow=1,MarginV=30'",
                "-c:a", "copy",  # Copy audio without re-encoding
                "-y", str(self.final_video)
            ]
            
            logger.info("FFmpeg is processing the video with burned-in subtitles...")
            logger.info("This may take several minutes depending on video length...")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                logger.info(f"✓ FFmpeg subtitle burning completed: {self.final_video}")
                return True
            else:
                logger.error(f"✗ FFmpeg subtitle burning failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("✗ FFmpeg process timed out (video too long)")
            return False
        except Exception as e:
            logger.error(f"✗ FFmpeg subtitle burning error: {str(e)}")
            return False
    
    def burn_subtitles_with_moviepy(self, segments: list) -> bool:
        """Burn subtitles into video using MoviePy (fallback method)."""
        try:
            if not self.has_moviepy:
                if not self._install_moviepy():
                    return False
                # Try importing again after installation
                try:
                    import moviepy.editor
                    self.has_moviepy = True
                except ImportError:
                    logger.error("MoviePy still not available after installation")
                    return False
            
            from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
            
            logger.info("Loading video for MoviePy subtitle burning...")
            video = VideoFileClip(str(self.video_file))
            
            logger.info("Creating burned-in subtitle clips...")
            subtitle_clips = []
            
            for segment in segments:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text']
                duration = end_time - start_time
                
                if duration > 0.1:  # Only create clips for reasonable durations
                    try:
                        # Create subtitle with professional styling
                        subtitle_clip = TextClip(
                            text,
                            fontsize=28,
                            color='white',
                            font='Arial-Bold',
                            stroke_color='black',
                            stroke_width=3,
                            method='caption',
                            size=(video.w * 0.9, None),
                            align='center'
                        ).set_position(('center', 0.85), relative=True).set_start(start_time).set_duration(duration)
                        
                        subtitle_clips.append(subtitle_clip)
                    except Exception as e:
                        logger.warning(f"Skipping subtitle segment: {text[:30]}... Error: {e}")
                        continue
            
            logger.info(f"Compositing video with {len(subtitle_clips)} burned-in subtitle clips...")
            
            # Composite video with burned-in subtitles
            if subtitle_clips:
                final_video = CompositeVideoClip([video] + subtitle_clips)
            else:
                logger.warning("No subtitle clips created, using original video")
                final_video = video
            
            logger.info("Rendering video with permanently burned-in subtitles...")
            logger.info("This will take several minutes but ensures universal compatibility...")
            
            # Write final video with burned subtitles
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
            final_video.close()
            video.close()
            
            logger.info(f"✓ MoviePy subtitle burning completed: {self.final_video}")
            return True
            
        except Exception as e:
            logger.error(f"✗ MoviePy subtitle burning failed: {str(e)}")
            return False
    
    def process(self) -> bool:
        """Execute the burned subtitle creation process."""
        logger.info("\n" + "="*70)
        logger.info("CREATING VIDEO WITH BURNED-IN SUBTITLES")
        logger.info("Subtitles will be permanently visible - no manual loading!")
        logger.info("="*70)
        
        # Step 1: Parse transcript timestamps
        logger.info("Step 1: Parsing transcript timestamps...")
        segments = self.parse_transcript_timestamps()
        if not segments:
            logger.error("❌ Could not parse transcript timestamps")
            return False
        
        # Step 2: Generate SRT file (for FFmpeg)
        logger.info("Step 2: Generating SRT subtitle file...")
        if not self.generate_srt_file(segments):
            logger.error("❌ Could not generate SRT file")
            return False
        
        # Step 3: Burn subtitles into video
        logger.info("Step 3: Burning subtitles permanently into video...")
        
        # Try FFmpeg first (more reliable and faster)
        if self.has_ffmpeg:
            logger.info("Using FFmpeg for subtitle burning (recommended method)...")
            if self.burn_subtitles_with_ffmpeg():
                logger.info("✅ FFmpeg subtitle burning successful!")
                return True
            else:
                logger.warning("FFmpeg failed, trying MoviePy...")
        
        # Fallback to MoviePy
        logger.info("Using MoviePy for subtitle burning (fallback method)...")
        if self.burn_subtitles_with_moviepy(segments):
            logger.info("✅ MoviePy subtitle burning successful!")
            return True
        else:
            logger.error("❌ All subtitle burning methods failed")
            return False
    
    def cleanup_temp_files(self):
        """Remove temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("✓ Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")


def main():
    """Main function to create burned-in subtitle video."""
    
    # File paths
    video_file = "output/cleaned_Unit-2 TA Session 1.mp4"
    transcript_file = "output/transcript_Unit-2 TA Session 1.txt"
    
    # Check if files exist
    if not Path(video_file).exists():
        logger.error(f"Video file not found: {video_file}")
        return False
    
    if not Path(transcript_file).exists():
        logger.error(f"Transcript file not found: {transcript_file}")
        return False
    
    # Create burned subtitle creator
    creator = BurnedSubtitleCreator(video_file, transcript_file)
    
    try:
        # Execute the process
        success = creator.process()
        
        if success:
            logger.info("\n" + "="*70)
            logger.info("🎉 BURNED-IN SUBTITLE CREATION SUCCESSFUL!")
            logger.info("="*70)
            logger.info(f"✓ Video with Burned Subtitles: {creator.final_video}")
            logger.info("\n🎬 FEATURES:")
            logger.info("  ✅ Subtitles permanently visible on video")
            logger.info("  ✅ Works on ANY video player or platform")
            logger.info("  ✅ No manual subtitle loading required")
            logger.info("  ✅ Professional styling and positioning")
            logger.info("  ✅ Perfect timing synchronization")
            logger.info("\n🚀 Your video is now ready for universal playback!")
            logger.info("   Simply double-click to play - subtitles are always visible!")
            
            # Cleanup option
            cleanup = input("\nClean up temporary files? (y/n): ").lower().strip()
            if cleanup == 'y':
                creator.cleanup_temp_files()
        else:
            logger.error("❌ Burned subtitle creation failed")
        
        return success
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    main()
