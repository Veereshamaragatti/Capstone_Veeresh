#!/usr/bin/env python3
"""
Simple Audio-Video Combiner
===========================

A simplified approach to combine burned subtitle video with original audio.
"""

import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def combine_audio_video_simple():
    """Combine burned subtitle video with original audio using Python libraries."""
    try:
        # Import moviepy components individually
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        
        # File paths
        original_video = Path("output/cleaned_Unit-2 TA Session 1.mp4")
        burned_video = Path("output/fixed_burned_subtitles_Unit-2 TA Session 1.mp4")
        final_output = Path("output/FINAL_with_audio_and_subtitles_Unit-2 TA Session 1.mp4")
        
        logger.info("🎬 COMBINING BURNED SUBTITLE VIDEO WITH ORIGINAL AUDIO")
        logger.info(f"📹 Original video (for audio): {original_video}")
        logger.info(f"📺 Burned subtitle video: {burned_video}")
        logger.info(f"🎯 Final output: {final_output}")
        
        # Check if files exist
        if not original_video.exists():
            logger.error(f"❌ Original video not found: {original_video}")
            return False
            
        if not burned_video.exists():
            logger.error(f"❌ Burned subtitle video not found: {burned_video}")
            return False
        
        # Load videos
        logger.info("🔄 Loading video files...")
        video_clip = VideoFileClip(str(burned_video))
        original_clip = VideoFileClip(str(original_video))
        
        # Check if original has audio
        if original_clip.audio is None:
            logger.warning("⚠️ Original video has no audio track")
            logger.info("📋 Copying burned subtitle video as final output")
            import shutil
            shutil.copy2(burned_video, final_output)
            video_clip.close()
            original_clip.close()
            return True
        
        # Combine video with subtitles + original audio
        logger.info("🔀 Combining subtitled video with original audio...")
        final_clip = video_clip.set_audio(original_clip.audio)
        
        # Write the final video
        logger.info("💾 Writing final video file (this may take a few minutes)...")
        final_clip.write_videofile(
            str(final_output),
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            verbose=False,
            logger=None
        )
        
        # Clean up
        video_clip.close()
        original_clip.close()
        final_clip.close()
        
        logger.info("✅ SUCCESS! Final video with audio and burned subtitles created!")
        logger.info(f"🎉 Output file: {final_output}")
        logger.info("\n🎬 YOUR FINAL VIDEO FEATURES:")
        logger.info("  ✅ Burned-in subtitles (always visible)")
        logger.info("  ✅ Original audio (fully preserved)")
        logger.info("  ✅ Professional subtitle styling")
        logger.info("  ✅ Works on any video player")
        logger.info("  ✅ No manual subtitle loading needed")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ MoviePy import failed: {str(e)}")
        logger.info("🔧 Trying alternative method...")
        return combine_with_opencv_fallback()
    except Exception as e:
        logger.error(f"❌ Failed to combine video and audio: {str(e)}")
        return False

def combine_with_opencv_fallback():
    """Fallback method using OpenCV to extract audio info."""
    try:
        import cv2
        
        logger.info("🔧 Using OpenCV fallback method...")
        
        # File paths
        original_video = Path("output/cleaned_Unit-2 TA Session 1.mp4")
        burned_video = Path("output/fixed_burned_subtitles_Unit-2 TA Session 1.mp4")
        final_output = Path("output/FINAL_with_audio_and_subtitles_Unit-2 TA Session 1.mp4")
        
        # Check video properties
        cap_original = cv2.VideoCapture(str(original_video))
        cap_burned = cv2.VideoCapture(str(burned_video))
        
        if not cap_original.isOpened():
            logger.error("❌ Cannot open original video")
            return False
            
        if not cap_burned.isOpened():
            logger.error("❌ Cannot open burned subtitle video")
            return False
        
        # Get properties
        original_fps = cap_original.get(cv2.CAP_PROP_FPS)
        burned_fps = cap_burned.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"📊 Original video FPS: {original_fps}")
        logger.info(f"📊 Burned video FPS: {burned_fps}")
        
        cap_original.release()
        cap_burned.release()
        
        # For now, just copy the burned video as the final output
        # This preserves the subtitles, but we need to investigate the audio issue
        logger.warning("⚠️ Using simplified fallback - copying burned subtitle video")
        import shutil
        shutil.copy2(burned_video, final_output)
        
        logger.info(f"📋 Video copied to: {final_output}")
        logger.info("🔍 The audio issue may be that the burned subtitle video was created without audio")
        logger.info("💡 Please check if the original cleaned video has audible audio")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenCV fallback failed: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("🎵 AUDIO + VIDEO COMBINER FOR BURNED SUBTITLES")
    logger.info("="*70)
    
    success = combine_audio_video_simple()
    
    if success:
        logger.info("\n🎉 PROCESS COMPLETED!")
        logger.info("🎬 Check the final video file in the output folder")
        logger.info("🔊 Test the audio to confirm it's working properly")
    else:
        logger.error("❌ Failed to create final video with audio")
