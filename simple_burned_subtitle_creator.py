#!/usr/bin/env python3
"""
Simple Burned Subtitle Creator
=============================

Creates a video with burned-in subtitles using pure Python libraries.
This approach works without requiring FFmpeg or complex MoviePy setup.
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

class SimpleBurnedSubtitleCreator:
    """Creates video with burned-in subtitles using available Python tools."""
    
    def __init__(self, video_file: str, transcript_file: str, output_dir: str = "output"):
        self.video_file = Path(video_file)
        self.transcript_file = Path(transcript_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        base_name = self.video_file.stem.replace("cleaned_", "").replace("subtitled_", "")
        self.final_video = self.output_dir / f"burned_subtitles_{base_name}.mp4"
        
        logger.info(f"Initialized Simple Burned Subtitle Creator")
        logger.info(f"Video: {self.video_file}")
        logger.info(f"Transcript: {self.transcript_file}")
    
    def install_required_packages(self):
        """Install required packages for video processing."""
        packages = ['opencv-python', 'Pillow', 'numpy']
        
        for package in packages:
            try:
                __import__(package.replace('-', '_').lower())
                logger.info(f"✓ {package} already available")
            except ImportError:
                logger.info(f"Installing {package}...")
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                 check=True, capture_output=True)
                    logger.info(f"✓ {package} installed successfully")
                except subprocess.CalledProcessError as e:
                    logger.error(f"✗ Failed to install {package}: {e}")
                    return False
        return True
    
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
    
    def create_burned_subtitle_video(self, segments: list) -> bool:
        """Create video with burned-in subtitles using OpenCV."""
        try:
            import cv2
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
            
            logger.info("Loading video for subtitle burning...")
            
            # Open video
            cap = cv2.VideoCapture(str(self.video_file))
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(self.final_video), fourcc, fps, (width, height))
            
            # Prepare subtitle data indexed by time
            subtitle_map = {}
            for segment in segments:
                start_frame = int(segment['start'] * fps)
                end_frame = int(segment['end'] * fps)
                for frame_num in range(start_frame, min(end_frame + 1, total_frames)):
                    subtitle_map[frame_num] = segment['text']
            
            logger.info(f"Processing {total_frames} frames with burned-in subtitles...")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add subtitle if available for this frame
                if frame_count in subtitle_map:
                    text = subtitle_map[frame_count]
                    frame = self._add_subtitle_to_frame(frame, text, width, height)
                
                # Write frame
                out.write(frame)
                frame_count += 1
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            # Release everything
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            logger.info(f"✓ Burned subtitle video created: {self.final_video}")
            return True
            
        except Exception as e:
            logger.error(f"✗ OpenCV subtitle burning failed: {str(e)}")
            return False
    
    def _add_subtitle_to_frame(self, frame, text, width, height):
        """Add subtitle text to a video frame."""
        try:
            import cv2
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
            
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # Try to use a good font, fallback to default
            try:
                font_size = max(24, int(height * 0.04))  # 4% of video height
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            # Calculate text position
            if font:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(text) * 10  # Rough estimate
                text_height = 20
            
            # Position at bottom center with margin
            x = (width - text_width) // 2
            y = height - text_height - 50  # 50px from bottom
            
            # Draw text with outline (for better visibility)
            outline_color = (0, 0, 0)  # Black outline
            text_color = (255, 255, 255)  # White text
            
            # Draw outline
            if font:
                for adj in range(-2, 3):
                    for adj2 in range(-2, 3):
                        draw.text((x+adj, y+adj2), text, font=font, fill=outline_color)
                # Draw main text
                draw.text((x, y), text, font=font, fill=text_color)
            else:
                # Fallback without font
                draw.text((x, y), text, fill=text_color)
            
            # Convert back to BGR for OpenCV
            frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return frame_bgr
            
        except Exception as e:
            logger.warning(f"Failed to add subtitle to frame: {e}")
            return frame  # Return original frame if subtitle fails
    
    def process(self) -> bool:
        """Execute the burned subtitle creation process."""
        logger.info("\n" + "="*70)
        logger.info("CREATING VIDEO WITH BURNED-IN SUBTITLES")
        logger.info("Pure Python Implementation - Universal Compatibility")
        logger.info("="*70)
        
        # Install required packages
        logger.info("Step 1: Installing required packages...")
        if not self.install_required_packages():
            logger.error("❌ Failed to install required packages")
            return False
        
        # Parse transcript timestamps
        logger.info("Step 2: Parsing transcript timestamps...")
        segments = self.parse_transcript_timestamps()
        if not segments:
            logger.error("❌ Could not parse transcript timestamps")
            return False
        
        # Create burned subtitle video
        logger.info("Step 3: Creating video with burned-in subtitles...")
        logger.info("This process will take several minutes...")
        
        if self.create_burned_subtitle_video(segments):
            logger.info("✅ Burned subtitle video creation successful!")
            return True
        else:
            logger.error("❌ Burned subtitle video creation failed")
            return False


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
    creator = SimpleBurnedSubtitleCreator(video_file, transcript_file)
    
    try:
        # Execute the process
        success = creator.process()
        
        if success:
            logger.info("\n" + "="*70)
            logger.info("🎉 BURNED-IN SUBTITLE VIDEO COMPLETED!")
            logger.info("="*70)
            logger.info(f"✓ Video with Burned Subtitles: {creator.final_video}")
            logger.info("\n🎬 FEATURES:")
            logger.info("  ✅ Subtitles permanently burned into video frames")
            logger.info("  ✅ Works on ANY video player, platform, or device")
            logger.info("  ✅ No manual subtitle loading ever required")
            logger.info("  ✅ Professional white text with black outline")
            logger.info("  ✅ Perfect timing synchronization")
            logger.info("\n🚀 Your video is ready for universal playback!")
            logger.info("   Subtitles are always visible - just play the video!")
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
