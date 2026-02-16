"""
Feature 5: Demo Recording System
=================================
Automated GIF/video generation for thesis presentations and publications.

Problem:
    Thesis defense needs polished recordings of generation process and gameplay.
    Manual screen recording is time-consuming and inconsistent.

Solution:
    - Automated frame capture during generation pipeline
    - Configurable recording modes (fast/slow, full/key-frames)
    - GIF generation with optimal compression
    - MP4 video export with annotations
    - Side-by-side comparisons (before/after repair, etc.)

Integration Point: Throughout pipeline, orchestrated by robust_pipeline.py
"""

import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class RecordingMode(Enum):
    """Recording quality/speed modes."""
    FAST = "fast"  # Key frames only (start, end, milestones)
    NORMAL = "normal"  # Regular interval sampling
    SLOW = "slow"  # Every frame (for detailed analysis)
    COMPARISON = "comparison"  # Side-by-side (before/after)


@dataclass
class FrameMetadata:
    """Metadata for a single recorded frame."""
    timestamp: float  # Seconds since recording start
    frame_index: int
    pipeline_block: str  # Which block generated this (e.g., "VQ-VAE", "WFC")
    description: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordingConfig:
    """Configuration for demo recording."""
    mode: RecordingMode = RecordingMode.NORMAL
    output_dir: str = "outputs/recordings"
    fps: int = 10  # Frames per second for video
    gif_duration: float = 3.0  # Seconds per frame in GIF
    max_frames: int = 100  # Limit for SLOW mode
    include_annotations: bool = True  # Overlay text on frames
    font_size: int = 16
    comparison_layout: str = "horizontal"  # "horizontal" or "vertical"
    save_individual_frames: bool = False


@dataclass
class Recording:
    """A complete recording session."""
    recording_id: str
    frames: List[np.ndarray]  # (H, W, 3) RGB frames
    metadata: List[FrameMetadata]
    config: RecordingConfig
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None


# ============================================================================
# FRAME RECORDER
# ============================================================================

class DemoRecorder:
    """
    Records pipeline execution for demo/thesis videos.
    
    Features:
    - Capture frames at key pipeline stages
    - Annotate frames with block names and metrics
    - Generate GIFs and MP4 videos
    - Create comparison views (before/after)
    
    Usage:
        recorder = DemoRecorder(config=RecordingConfig(mode=RecordingMode.NORMAL))
        
        recorder.start_recording("dungeon_generation_demo")
        recorder.capture_frame(
            frame=visual_grid,
            block="VQ-VAE",
            description="Latent decoding"
        )
        recorder.capture_frame(
            frame=repaired_grid,
            block="WFC",
            description="Symbolic repair"
        )
        recorder.stop_recording()
        
        recorder.export_gif("demo.gif")
        recorder.export_video("demo.mp4")
    """
    
    def __init__(self, config: Optional[RecordingConfig] = None):
        self.config = config or RecordingConfig()
        self.current_recording: Optional[Recording] = None
        self.is_recording = False
        self._frame_count = 0
        self._start_time = 0.0
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def start_recording(self, recording_id: str):
        """Start a new recording session."""
        import time
        
        self.current_recording = Recording(
            recording_id=recording_id,
            frames=[],
            metadata=[],
            config=self.config,
            start_time=datetime.now()
        )
        self.is_recording = True
        self._frame_count = 0
        self._start_time = time.time()
        
        logger.info(f"Started recording: {recording_id}")
    
    def capture_frame(
        self,
        frame: np.ndarray,
        block: str,
        description: str,
        metrics: Optional[Dict] = None
    ):
        """
        Capture a single frame.
        
        Args:
            frame: (H, W) or (H, W, 3) numpy array
            block: Pipeline block name (e.g., "Latent Diffusion")
            description: Frame description
            metrics: Optional metrics to overlay
        """
        if not self.is_recording or self.current_recording is None:
            logger.warning("capture_frame called but not recording")
            return
        
        # Check frame limit for SLOW mode
        if self.config.mode == RecordingMode.SLOW:
            if self._frame_count >= self.config.max_frames:
                return
        
        # Convert grayscale to RGB if needed
        if frame.ndim == 2:
            frame_rgb = self._grid_to_rgb(frame)
        else:
            frame_rgb = frame.copy()
        
        # Add annotations if enabled
        if self.config.include_annotations:
            frame_rgb = self._annotate_frame(frame_rgb, block, description, metrics or {})
        
        # Store frame
        import time
        timestamp = time.time() - self._start_time
        
        self.current_recording.frames.append(frame_rgb)
        self.current_recording.metadata.append(FrameMetadata(
            timestamp=timestamp,
            frame_index=self._frame_count,
            pipeline_block=block,
            description=description,
            metrics=metrics or {}
        ))
        
        self._frame_count += 1
        
        # Save individual frame if requested
        if self.config.save_individual_frames:
            self._save_frame(frame_rgb, self._frame_count)
    
    def capture_comparison(
        self,
        frame_before: np.ndarray,
        frame_after: np.ndarray,
        block: str,
        description: str
    ):
        """
        Capture side-by-side comparison frame.
        
        Args:
            frame_before: (H, W) or (H, W, 3) "before" frame
            frame_after: (H, W) or (H, W, 3) "after" frame
            block: Pipeline block
            description: Comparison description
        """
        # Convert to RGB
        before_rgb = self._grid_to_rgb(frame_before) if frame_before.ndim == 2 else frame_before
        after_rgb = self._grid_to_rgb(frame_after) if frame_after.ndim == 2 else frame_after
        
        # Create comparison layout
        if self.config.comparison_layout == "horizontal":
            comparison = np.hstack([before_rgb, after_rgb])
        else:  # vertical
            comparison = np.vstack([before_rgb, after_rgb])
        
        # Annotate
        if self.config.include_annotations:
            comparison = self._annotate_comparison(
                comparison, block, description, self.config.comparison_layout
            )
        
        self.capture_frame(comparison, block, description)
    
    def stop_recording(self):
        """Stop recording and finalize."""
        if not self.is_recording or self.current_recording is None:
            return
        
        self.current_recording.end_time = datetime.now()
        self.is_recording = False
        
        logger.info(
            f"Stopped recording: {self.current_recording.recording_id} "
            f"({len(self.current_recording.frames)} frames)"
        )
    
    def export_gif(
        self,
        output_path: Optional[str] = None,
        optimize: bool = True
    ) -> str:
        """
        Export recording as GIF.
        
        Args:
            output_path: Output file path (auto-generated if None)
            optimize: Apply GIF optimization
        
        Returns:
            Path to saved GIF
        """
        if self.current_recording is None or not self.current_recording.frames:
            raise RuntimeError("No recording to export")
        
        if output_path is None:
            output_path = str(Path(self.config.output_dir) / 
                            f"{self.current_recording.recording_id}.gif")
        
        try:
            from PIL import Image
            
            # Convert frames to PIL Images
            pil_frames = [Image.fromarray(frame) for frame in self.current_recording.frames]
            
            # Calculate duration per frame (ms)
            duration_ms = int(self.config.gif_duration * 1000 / len(pil_frames))
            
            # Save as GIF
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=optimize
            )
            
            logger.info(f"Exported GIF: {output_path}")
            return output_path
        
        except ImportError:
            logger.error("PIL not available, cannot export GIF")
            return ""
    
    def export_video(
        self,
        output_path: Optional[str] = None,
        codec: str = 'mp4v'
    ) -> str:
        """
        Export recording as MP4 video.
        
        Args:
            output_path: Output file path
            codec: Video codec (mp4v, h264, etc.)
        
        Returns:
            Path to saved video
        """
        if self.current_recording is None or not self.current_recording.frames:
            raise RuntimeError("No recording to export")
        
        if output_path is None:
            output_path = str(Path(self.config.output_dir) / 
                            f"{self.current_recording.recording_id}.mp4")
        
        try:
            import cv2
            
            # Get frame dimensions
            height, width = self.current_recording.frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, self.config.fps, (width, height))
            
            # Write frames
            for frame in self.current_recording.frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            logger.info(f"Exported video: {output_path}")
            return output_path
        
        except ImportError:
            logger.error("OpenCV not available, cannot export video")
            return ""
    
    def export_metadata(self, output_path: Optional[str] = None) -> str:
        """Export frame metadata as JSON."""
        if self.current_recording is None:
            raise RuntimeError("No recording to export")
        
        if output_path is None:
            output_path = str(Path(self.config.output_dir) / 
                            f"{self.current_recording.recording_id}_metadata.json")
        
        metadata_list = [
            {
                'timestamp': meta.timestamp,
                'frame_index': meta.frame_index,
                'pipeline_block': meta.pipeline_block,
                'description': meta.description,
                'metrics': meta.metrics
            }
            for meta in self.current_recording.metadata
        ]
        
        with open(output_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        logger.info(f"Exported metadata: {output_path}")
        return output_path
    
    def _grid_to_rgb(self, grid: np.ndarray) -> np.ndarray:
        """Convert semantic grid to RGB for visualization."""
        # Simple colormap (expand for full palette)
        colormap = {
            0: (0, 0, 0),       # VOID - black
            1: (197, 179, 88),  # FLOOR - beige
            2: (88, 48, 0),     # WALL - brown
            3: (128, 128, 128), # BLOCK - gray
            10: (0, 255, 0),    # DOOR - green
        }
        
        H, W = grid.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        
        for tile_id, color in colormap.items():
            mask = (grid == tile_id)
            rgb[mask] = color
        
        return rgb
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        block: str,
        description: str,
        metrics: Dict
    ) -> np.ndarray:
        """Add text annotations to frame."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Convert to PIL
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            
            # Try to load font, fall back to default
            try:
                font = ImageFont.truetype("arial.ttf", self.config.font_size)
            except:
                font = ImageFont.load_default()
            
            # Draw block name (top-left)
            text = f"{block}: {description}"
            draw.text((10, 10), text, fill=(255, 255, 255), font=font)
            
            # Draw metrics (top-right)
            y = 10
            for key, value in metrics.items():
                metric_text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
                draw.text((frame.shape[1] - 200, y), metric_text, fill=(255, 255, 0), font=font)
                y += 20
            
            return np.array(img)
        
        except ImportError:
            logger.warning("PIL not available, skipping annotations")
            return frame
    
    def _annotate_comparison(
        self,
        comparison: np.ndarray,
        block: str,
        description: str,
        layout: str
    ) -> np.ndarray:
        """Add labels to comparison frame."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            img = Image.fromarray(comparison)
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", self.config.font_size)
            except:
                font = ImageFont.load_default()
            
            # Title
            draw.text((10, 10), f"{block}: {description}", fill=(255, 255, 255), font=font)
            
            # Before/After labels
            if layout == "horizontal":
                mid_x = comparison.shape[1] // 2
                draw.text((10, 30), "Before", fill=(255, 0, 0), font=font)
                draw.text((mid_x + 10, 30), "After", fill=(0, 255, 0), font=font)
            else:  # vertical
                mid_y = comparison.shape[0] // 2
                draw.text((10, 30), "Before", fill=(255, 0, 0), font=font)
                draw.text((10, mid_y + 30), "After", fill=(0, 255, 0), font=font)
            
            return np.array(img)
        
        except ImportError:
            return comparison
    
    def _save_frame(self, frame: np.ndarray, frame_index: int):
        """Save individual frame to disk."""
        try:
            from PIL import Image
            
            frame_path = Path(self.config.output_dir) / self.current_recording.recording_id / f"frame_{frame_index:04d}.png"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            
            img = Image.fromarray(frame)
            img.save(str(frame_path))
        
        except ImportError:
            pass


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

"""
# In src/pipeline/robust_pipeline.py:

from src.utils.demo_recorder import DemoRecorder, RecordingConfig, RecordingMode

class PipelineBlock:
    def __init__(self, ..., recorder: Optional[DemoRecorder] = None):
        # ... existing init ...
        self.recorder = recorder
    
    def execute(self, state: Dict[str, Any]) -> BlockResult:
        # ... existing execution ...
        
        # Capture frame if recording
        if self.recorder and self.recorder.is_recording:
            self.recorder.capture_frame(
                frame=output_grid,
                block=self.name,
                description=f"Output from {self.name}",
                metrics={'execution_time': execution_time}
            )
        
        return result


# In gui_runner.py (record button):

from src.utils.demo_recorder import DemoRecorder, RecordingConfig, RecordingMode

class ZeldaGUI:
    def __init__(self, ...):
        # ... existing init ...
        self.recorder = DemoRecorder(RecordingConfig(
            mode=RecordingMode.NORMAL,
            fps=10,
            include_annotations=True
        ))
        self.is_recording_demo = False
    
    def _handle_record_toggle(self, event):
        '''Toggle demo recording.'''
        if event.key == pygame.K_r and (pygame.key.get_mods() & pygame.KMOD_CTRL):
            if not self.is_recording_demo:
                # Start recording
                recording_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.recorder.start_recording(recording_id)
                self.is_recording_demo = True
                self._show_toast("Recording started", (0, 255, 0))
            else:
                # Stop and export
                self.recorder.stop_recording()
                gif_path = self.recorder.export_gif()
                video_path = self.recorder.export_video()
                self.recorder.export_metadata()
                self.is_recording_demo = False
                self._show_toast(f"Saved: {gif_path}", (0, 255, 0))
    
    def _render(self):
        # ... existing rendering ...
        
        # Capture frame if recording
        if self.is_recording_demo and self.dungeon_grid is not None:
            self.recorder.capture_frame(
                frame=self.dungeon_grid,
                block="GUI Render",
                description="Current view",
                metrics={'timestamp': time.time()}
            )


# In src/pipeline/dungeon_pipeline.py:

from src.utils.demo_recorder import DemoRecorder

class NeuralSymbolicDungeonPipeline:
    def generate_room(self, ..., recorder: Optional[DemoRecorder] = None):
        # ... existing generation ...
        
        # Record neural output
        if recorder and recorder.is_recording:
            recorder.capture_frame(
                frame=neural_grid,
                block="Latent Diffusion",
                description=f"Room {room_id} - Neural output",
                metrics={'entropy': float(np.mean(entropy))}
            )
        
        # ... symbolic repair ...
        
        # Record comparison
        if recorder and recorder.is_recording and was_repaired:
            recorder.capture_comparison(
                frame_before=neural_grid,
                frame_after=final_grid,
                block="WFC Repair",
                description=f"Room {room_id} - Repair"
            )
        
        return result
"""
