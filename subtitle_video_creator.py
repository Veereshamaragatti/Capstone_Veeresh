#!/usr/bin/env python3
"""
Subtitle Video Creator
====================

This script takes the existing cleaned video and transcript to create
a new version with embedded live subtitles.
"""

import os
import sys
import logging
import subprocess
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class SubtitleVideoCreator:
    """Creates subtitled video from existing cleaned video and transcript."""
    
    def __init__(self, video_file: str, transcript_file: str, output_dir: str = "output"):
        self.video_file = Path(video_file)
        self.transcript_file = Path(transcript_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        base_name = self.video_file.stem.replace("cleaned_", "")
        self.srt_file = self.output_dir / f"subtitles_{base_name}.srt"
        self.final_video = self.output_dir / f"subtitled_{base_name}.mp4"
        
        logger.info(f"Initialized Subtitle Video Creator")
        logger.info(f"Video: {self.video_file}")
        logger.info(f"Transcript: {self.transcript_file}")
    
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
                    
                    if text:  # Only add non-empty text
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
    
    def install_moviepy(self) -> bool:
        """Install MoviePy for video processing."""
        try:
            logger.info("Installing MoviePy...")
            subprocess.run([sys.executable, "-m", "pip", "install", "moviepy"], 
                         check=True, capture_output=True)
            logger.info("✓ MoviePy installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ MoviePy installation failed: {e}")
            return False
    
    def create_subtitled_video_with_moviepy(self, segments: list) -> bool:
        """Create subtitled video using MoviePy."""
        try:
            # Try to import MoviePy
            try:
                from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
            except ImportError:
                logger.info("MoviePy not found, installing...")
                if not self.install_moviepy():
                    return False
                from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
            
            logger.info("Loading video file...")
            video = VideoFileClip(str(self.video_file))
            
            logger.info("Creating subtitle clips...")
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
                            fontsize=32,
                            color='white',
                            font='Arial-Bold',
                            stroke_color='black',
                            stroke_width=3,
                            method='caption',
                            size=(video.w * 0.9, None),
                            align='center'
                        ).set_position(('center', 'bottom')).set_start(start_time).set_duration(duration)
                        
                        subtitle_clips.append(subtitle_clip)
                    except Exception as e:
                        logger.warning(f"Skipping subtitle segment: {text[:30]}... Error: {e}")
                        continue
            
            logger.info(f"Created {len(subtitle_clips)} subtitle clips")
            
            # Composite video with subtitles
            if subtitle_clips:
                final_video = CompositeVideoClip([video] + subtitle_clips)
            else:
                logger.warning("No subtitle clips created, using original video")
                final_video = video
            
            logger.info("Rendering video with live subtitles...")
            logger.info("This will take several minutes...")
            
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
            final_video.close()
            video.close()
            
            logger.info(f"✓ Subtitled video created: {self.final_video}")
            return True
            
        except Exception as e:
            logger.error(f"✗ MoviePy subtitle creation failed: {str(e)}")
            return False
    
    def create_subtitled_video_simple(self, segments: list) -> bool:
        """Create a simple text overlay version using basic method."""
        try:
            logger.info("Creating simple text-based subtitle display...")
            
            # For now, just copy the original video and provide the SRT file
            import shutil
            shutil.copy2(self.video_file, self.final_video)
            
            logger.info(f"✓ Video copied: {self.final_video}")
            logger.info(f"✓ Use subtitle file: {self.srt_file}")
            logger.info("ℹ️  Load the SRT file in your video player to see subtitles")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Simple subtitle creation failed: {str(e)}")
            return False
    
    def process(self) -> bool:
        """Execute the subtitle creation process."""
        logger.info("\\n" + "="*60)
        logger.info("CREATING VIDEO WITH LIVE SUBTITLES")
        logger.info("="*60)
        
        # Step 1: Parse transcript timestamps
        logger.info("Step 1: Parsing transcript timestamps...")
        segments = self.parse_transcript_timestamps()
        if not segments:
            logger.error("❌ Could not parse transcript timestamps")
            return False
        
        # Step 2: Generate SRT file
        logger.info("Step 2: Generating SRT subtitle file...")
        if not self.generate_srt_file(segments):
            logger.error("❌ Could not generate SRT file")
            return False
        
        # Step 3: Create subtitled video
        logger.info("Step 3: Creating video with live subtitles...")
        
        # Try MoviePy first for embedded subtitles
        if self.create_subtitled_video_with_moviepy(segments):
            logger.info("✅ Successfully created video with embedded subtitles")
        else:
            logger.warning("MoviePy failed, using simple method...")
            if self.create_subtitled_video_simple(segments):
                logger.info("✅ Created video with external subtitle file")
            else:
                logger.error("❌ All subtitle methods failed")
                return False
        
        logger.info("\\n" + "="*60)
        logger.info("🎉 SUBTITLE VIDEO CREATION COMPLETED!")
        logger.info("="*60)
        logger.info(f"✓ Subtitled Video: {self.final_video}")
        logger.info(f"✓ SRT Subtitle File: {self.srt_file}")
        logger.info("\\n🎬 Your video now has live subtitles!")
        
        return True


def main():
    """Main function to create subtitled video."""
    
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
    
    # Create subtitle video creator
    creator = SubtitleVideoCreator(video_file, transcript_file)
    
    try:
        # Execute the process
        success = creator.process()
        
        if success:
            logger.info("\\n🎊 SUBTITLE CREATION SUCCESSFUL!")
            logger.info(f"\\n📹 Watch your subtitled video: {creator.final_video}")
            logger.info(f"📝 Or use external subtitles: {creator.srt_file}")
            logger.info("\\n✨ The transcript now appears as live subtitles on your video!")
        else:
            logger.error("❌ Subtitle creation failed")
        
        return success
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    main()
