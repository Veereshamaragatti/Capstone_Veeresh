#!/usr/bin/env python3
"""
Audio Test and Final Video Creator
=================================

Tests audio in original video and creates final video with proper audio.
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_and_create_final_video():
    """Test audio and create final video."""
    try:
        # Import MoviePy properly
        import moviepy
        from moviepy import VideoFileClip
        
        logger.info("🎬 TESTING AUDIO AND CREATING FINAL VIDEO")
        
        # File paths
        original_video = Path("output/cleaned_Unit-2 TA Session 1.mp4")
        burned_video = Path("output/fixed_burned_subtitles_Unit-2 TA Session 1.mp4")
        final_output = Path("output/FINAL_Unit-2_TA_Session_with_audio_and_subtitles.mp4")
        
        logger.info(f"📹 Original video: {original_video}")
        logger.info(f"📺 Burned subtitle video: {burned_video}")
        
        # Test original video audio
        logger.info("🔍 Testing original video for audio...")
        original_clip = VideoFileClip(str(original_video))
        
        if original_clip.audio is None:
            logger.warning("⚠️  Original video has NO AUDIO")
            logger.info("📋 The issue is that your cleaned video doesn't have audio")
            logger.info("🔧 Let's check the very original video file...")
            original_clip.close()
            
            # Check if original unprocessed video exists
            original_raw = Path("Unit-2 TA Session 1.mp4")
            if original_raw.exists():
                logger.info(f"🎥 Found original raw video: {original_raw}")
                raw_clip = VideoFileClip(str(original_raw))
                if raw_clip.audio is not None:
                    logger.info("✅ Original raw video HAS AUDIO!")
                    logger.info("🔀 Using raw video audio with burned subtitle video...")
                    
                    # Load burned subtitle video
                    burned_clip = VideoFileClip(str(burned_video))
                    
                    # Combine burned video with raw audio
                    final_clip = burned_clip.with_audio(raw_clip.audio)
                    
                    logger.info("💾 Writing final video with audio...")
                    final_clip.write_videofile(
                        str(final_output),
                        codec='libx264',
                        audio_codec='aac'
                    )
                    
                    # Cleanup
                    raw_clip.close()
                    burned_clip.close()
                    final_clip.close()
                    
                    logger.info("✅ SUCCESS! Final video created with audio!")
                    return True
                else:
                    raw_clip.close()
                    logger.error("❌ Even the original raw video has no audio")
                    return False
            else:
                logger.error("❌ Original raw video not found")
                return False
        else:
            logger.info("✅ Original cleaned video HAS AUDIO!")
            duration = original_clip.duration
            logger.info(f"📊 Audio duration: {duration:.2f} seconds")
            
            # Load burned subtitle video
            logger.info("🔄 Loading burned subtitle video...")
            burned_clip = VideoFileClip(str(burned_video))
            
            # Combine
            logger.info("🔀 Combining burned subtitles with audio...")
            final_clip = burned_clip.with_audio(original_clip.audio)
            
            logger.info("💾 Writing final video...")
            final_clip.write_videofile(
                str(final_output),
                codec='libx264',
                audio_codec='aac'
            )
            
            # Cleanup
            original_clip.close()
            burned_clip.close()
            final_clip.close()
            
            logger.info("✅ SUCCESS! Final video created!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("="*60)
    success = test_and_create_final_video()
    
    if success:
        logger.info("\n🎉 FINAL VIDEO IS READY!")
        logger.info("🎬 Check the output folder for:")
        logger.info("   FINAL_Unit-2_TA_Session_with_audio_and_subtitles.mp4")
        logger.info("\n✅ This video has:")
        logger.info("   🔊 Working audio")
        logger.info("   📝 Burned-in subtitles")
        logger.info("   🎯 Ready to play anywhere!")
    else:
        logger.error("❌ Failed to create final video")
        logger.info("🔧 The issue might be with the audio in your original files")
