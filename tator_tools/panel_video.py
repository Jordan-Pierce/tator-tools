import argparse
import os
import subprocess
import sys
import math


class VideoPaneler:
    """A class for creating paneled videos from multiple input video files."""
    
    def __init__(self, video_files, output_file, rows=None, cols=None, 
                 scale_width=None, scale_height=None, verbose=False):
        """
        Initialize the VideoPaneler.
        
        Args:
            video_files (list): List of input video file paths
            output_file (str): Output video file path
            rows (int, optional): Number of rows in the grid
            cols (int, optional): Number of columns in the grid
            scale_width (int, optional): Scale videos to this width before paneling
            scale_height (int, optional): Scale videos to this height before paneling
            verbose (bool): Show FFmpeg output during processing
        """
        self.video_files = video_files
        self.output_file = output_file
        self.scale_width = scale_width
        self.scale_height = scale_height
        self.verbose = verbose
        
        # Calculate grid dimensions
        self.rows, self.cols = self._calculate_grid_dimensions(
            len(video_files), rows, cols
        )
    
    def validate_video_files(self):
        """Validate that all video files exist and are accessible."""
        missing_files = []
        for video_file in self.video_files:
            if not os.path.exists(video_file):
                missing_files.append(video_file)
        
        if missing_files:
            print("ERROR: The following video files do not exist:")
            for file in missing_files:
                print(f"  - {file}")
            sys.exit(1)
    
    def get_video_info(self, video_file):
        """Get basic video information using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_file
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            # For this script, we'll assume videos are compatible
            # In a production script, you'd parse the JSON and validate dimensions/duration
            return True
        except subprocess.CalledProcessError:
            print(f"ERROR: Could not get video information for {video_file}")
            return False
        except FileNotFoundError:
            print("ERROR: ffprobe not found. Please install FFmpeg.")
            sys.exit(1)
    
    def _calculate_grid_dimensions(self, num_videos, rows=None, cols=None):
        """Calculate optimal grid dimensions for the given number of videos."""
        if rows and cols:
            if rows * cols < num_videos:
                print(f"ERROR: Grid size {rows}x{cols} is too small for {num_videos} videos")
                sys.exit(1)
            return rows, cols
        
        if rows:
            cols = math.ceil(num_videos / rows)
            return rows, cols
        if cols:
            rows = math.ceil(num_videos / cols)
            return rows, cols
            
        # Auto-calculate optimal dimensions
        sqrt_videos = math.sqrt(num_videos)
        cols = math.ceil(sqrt_videos)
        rows = math.ceil(num_videos / cols)
        
        return rows, cols
    
    def build_ffmpeg_command(self):
        """Build the FFmpeg command for creating a paneled video."""
        
        # Start with ffmpeg command
        cmd = ['ffmpeg', '-y']  # -y to overwrite output file
        
        # Add input files
        for video_file in self.video_files:
            cmd.extend(['-i', video_file])
        
        # Build complex filter for video paneling
        filter_parts = []
        
        # Scale videos if specified
        if self.scale_width and self.scale_height:
            for i in range(len(self.video_files)):
                filter_parts.append(f"[{i}:v]scale={self.scale_width}:{self.scale_height}[v{i}]")
            video_labels = [f"[v{i}]" for i in range(len(self.video_files))]
        else:
            video_labels = [f"[{i}:v]" for i in range(len(self.video_files))]
        
        # Create rows by horizontally stacking videos
        row_labels = []
        for row in range(self.rows):
            start_idx = row * self.cols
            end_idx = min(start_idx + self.cols, len(self.video_files))
            row_videos = video_labels[start_idx:end_idx]
            
            if len(row_videos) == 1:
                # Single video in row
                row_label = row_videos[0]
            else:
                # Multiple videos in row - horizontal stack
                row_label = f"[row{row}]"
                hstack_filter = f"{''.join(row_videos)}hstack=inputs={len(row_videos)}{row_label}"
                filter_parts.append(hstack_filter)
        
            row_labels.append(row_label if len(row_videos) > 1 else row_videos[0])
        
        # Vertically stack all rows
        if len(row_labels) == 1:
            # Single row
            final_filter = f"{row_labels[0][1:-1]}:v" if row_labels[0].startswith('[') else row_labels[0]
        else:
            # Multiple rows - vertical stack
            vstack_filter = f"{''.join(row_labels)}vstack=inputs={len(row_labels)}[out]"
            filter_parts.append(vstack_filter)
            final_filter = "[out]"
        
        # Combine all filter parts
        if filter_parts:
            filter_complex = ';'.join(filter_parts)
            cmd.extend(['-filter_complex', filter_complex])
            
            # Map video output
            if final_filter != "[out]":
                cmd.extend(['-map', final_filter])
            else:
                cmd.extend(['-map', '[out]'])
            
            # No audio output
            cmd.extend(['-an'])
        else:
            # No video filtering, just copy streams
            cmd.extend(['-c', 'copy'])
        
        # Output file
        cmd.append(self.output_file)
        
        return cmd
    
    def create_panel(self, dry_run=False):
        """Create the paneled video."""
        # Validate video files
        self.validate_video_files()
        
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except FileNotFoundError:
            print("ERROR: ffmpeg not found. Please install FFmpeg and add it to your PATH.")
            sys.exit(1)
        
        # Validate video files
        print(f"Validating {len(self.video_files)} video files...")
        for video_file in self.video_files:
            if not self.get_video_info(video_file):
                sys.exit(1)
        
        print(f"Creating {self.rows}x{self.cols} panel from {len(self.video_files)} videos")
        
        # Build FFmpeg command
        cmd = self.build_ffmpeg_command()
        
        if dry_run:
            print("FFmpeg command that would be executed:")
            print(' '.join(cmd))
            return
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Execute FFmpeg command
        print(f"Creating paneled video: {self.output_file}")
        print("This may take a while depending on video size and duration...")
        
        try:
            if self.verbose:
                subprocess.run(cmd, check=True)
            else:
                subprocess.run(cmd, capture_output=True, check=True)
            
            print(f"Successfully created paneled video: {self.output_file}")
            
        except subprocess.CalledProcessError as e:
            print(f"ERROR: FFmpeg failed with return code {e.returncode}")
            if hasattr(e, 'stderr') and e.stderr:
                print("FFmpeg error output:")
                print(e.stderr.decode('utf-8'))
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Create paneled videos from multiple input video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """
                Examples:
                # Create 2x2 panel from 4 videos
                python cac/panel_video.py video1.mp4 video2.mp4 video3.mp4 video4.mp4 -o output.mp4

                # Create 3x2 panel (3 rows, 2 columns)
                python panel_video.py *.mp4 -o output.mp4 --rows 3 --cols 2

                # Scale videos to specific size before paneling
                python panel_video.py *.mp4 -o output.mp4 --scale-width 640 --scale-height 480

                # Auto-calculate grid dimensions
                python panel_video.py video*.mp4 -o output.mp4
    """
    )
    
    parser.add_argument(
        'video_files',
        nargs='+',
        help='Input video files to panel together'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output video file path'
    )
    
    parser.add_argument(
        '--rows',
        type=int,
        help='Number of rows in the grid (auto-calculated if not specified)'
    )
    
    parser.add_argument(
        '--cols',
        type=int,
        help='Number of columns in the grid (auto-calculated if not specified)'
    )
    
    parser.add_argument(
        '--scale-width',
        type=int,
        help='Scale individual videos to this width before paneling'
    )
    
    parser.add_argument(
        '--scale-height',
        type=int,
        help='Scale individual videos to this height before paneling'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the FFmpeg command without executing it'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show FFmpeg output during processing'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.video_files) < 2:
        print("ERROR: At least 2 video files are required for paneling")
        sys.exit(1)
    
    # Create VideoPaneler instance
    paneler = VideoPaneler(
        video_files=args.video_files,
        output_file=args.output,
        rows=args.rows,
        cols=args.cols,
        scale_width=args.scale_width,
        scale_height=args.scale_height,
        verbose=args.verbose
    )
    
    # Create the paneled video
    paneler.create_panel(dry_run=args.dry_run)


if __name__ == "__main__":
    main()