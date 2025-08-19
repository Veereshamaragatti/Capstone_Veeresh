#!/usr/bin/env python3
"""
Audio Combiner for Burned Subtitle Video
========================================

Combines the burned subtitle video with original audio using MoviePy.
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def add_audio_to_burned_video():
    """Add original audio to the burned subtitle video."""
    try:
        import moviepy.editor as mp
        
        # File paths
        original_video = Path("output/cleaned_Unit-2 TA Session 1.mp4")
        burned_video = Path("output/fixed_burned_subtitles_Unit-2 TA Session 1.mp4")
        final_output = Path("output/final_with_audio_and_subtitles_Unit-2 TA Session 1.mp4")
        
        logger.info("Adding audio to burned subtitle video...")
        logger.info(f"Original video (for audio): {original_video}")
        logger.info(f"Burned subtitle video: {burned_video}")
        logger.info(f"Final output: {final_output}")
        
        # Check if files exist
        if not original_video.exists():
            logger.error(f"Original video not found: {original_video}")
            return False
            
        if not burned_video.exists():
            logger.error(f"Burned subtitle video not found: {burned_video}")
            return False
        
        # Load videos
        logger.info("Loading video files...")
        video_with_subtitles = mp.VideoFileClip(str(burned_video))
        original_with_audio = mp.VideoFileClip(str(original_video))
        
        # Check if original has audio
        if original_with_audio.audio is None:
            logger.warning("Original video has no audio track")
            logger.info("Copying burned subtitle video as final output")
            import shutil
            shutil.copy2(burned_video, final_output)
            return True
        
        # Combine video with subtitles + original audio
        logger.info("Combining subtitled video with original audio...")
        final_video = video_with_subtitles.set_audio(original_with_audio.audio)
        
        # Write the final video
        logger.info("Writing final video file...")
        final_video.write_videofile(
            str(final_output),
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )
        
        # Clean up
        video_with_subtitles.close()
        original_with_audio.close()
        final_video.close()
        
        logger.info("✅ SUCCESS! Final video with audio and burned subtitles created!")
        logger.info(f"✓ Output file: {final_output}")
        logger.info("\n🎉 Your video now has:")
        logger.info("  ✅ Burned-in subtitles (always visible)")
        logger.info("  ✅ Original audio (fully audible)")
        logger.info("  ✅ Professional subtitle styling")
        logger.info("  ✅ Works on any video player")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to add audio: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("ADDING AUDIO TO BURNED SUBTITLE VIDEO")
    logger.info("="*60)
    
    success = add_audio_to_burned_video()
    
    if success:
        logger.info("\n🎬 FINAL VIDEO READY!")
        logger.info("You can now play the video and hear the audio clearly!")
    else:
        logger.error("❌ Failed to create final video with audio")
