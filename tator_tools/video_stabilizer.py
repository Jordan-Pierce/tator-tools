import os
import argparse
from tqdm import tqdm

import cv2
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class VideoStabilizer:
    """
    A class to stabilize videos by reducing camera shake and movement.
    
    This class implements video stabilization techniques using optical flow and
    affine transformations to create a smoother video output.
    
    Attributes:
        input_path (str): Path to the input video file.
        output_path (str): Path where the stabilized video will be saved.
        ext (str): File extension of the input video.
    """
    
    def __init__(self, input_path, output_path):
        """
        Initialize the VideoStabilizer with input and output paths.
        
        Args:
            input_path (str): Path to the input video file.
            output_path (str): Path where the stabilized video will be saved. If None,
                               a default name will be generated based on the input filename.
                               
        Raises:
            ValueError: If the input file doesn't exist or if the output file already exists.
        """
        self.input_path = input_path
        self.ext = os.path.splitext(input_path)[1]
        
        # Validate the input and output paths
        if not os.path.exists(input_path):
            raise ValueError(f"ERROR: Input video file not found at {input_path}")
        
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + "_stabilized" + self.ext
        elif os.path.exists(output_path):
            raise ValueError(f"ERROR: Output video file already exists at {output_path}")
        
        # Set the output path
        self.output_path = output_path

    @staticmethod
    def movingAverage(self, curve, radius):
        """
        Apply a moving average filter to smooth a curve.
        
        Args:
            curve (numpy.ndarray): The input curve to smooth.
            radius (int): The radius of the moving average window.
                          Window size will be 2 * radius + 1.
                          
        Returns:
            numpy.ndarray: The smoothed curve.
        """
        window_size = 2 * radius + 1
        f = np.ones(window_size) / window_size
        curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
        curve_smoothed = np.convolve(curve_pad, f, mode='same')
        curve_smoothed = curve_smoothed[radius:-radius]
        return curve_smoothed

    @staticmethod
    def smooth(self, trajectory, radius=50):
        """
        Smooth the trajectory of camera movement.
        
        Applies a moving average filter to each dimension of the trajectory.
        
        Args:
            trajectory (numpy.ndarray): The camera trajectory to smooth.
            radius (int, optional): Radius for the moving average filter. Defaults to 50.
            
        Returns:
            numpy.ndarray: The smoothed trajectory.
        """
        smoothed_trajectory = np.copy(trajectory)
        for i in range(3):
            smoothed_trajectory[:, i] = self.movingAverage(trajectory[:, i], radius)
        return smoothed_trajectory

    @staticmethod
    def fixBorder(self, frame):
        """
        Fix the border effects after applying transformations.
        
        This method slightly zooms in on the frame to remove black borders
        that may appear after stabilization.
        
        Args:
            frame (numpy.ndarray): The input frame to process.
            
        Returns:
            numpy.ndarray: The processed frame with fixed borders.
        """
        s = frame.shape
        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame

    def stabilize_video(self):
        """
        Perform video stabilization on the input video.
        
        This method:
        1. Tracks feature points across frames using optical flow
        2. Estimates and accumulates transformations between consecutive frames
        3. Smooths the resulting trajectory
        4. Applies smoothed transformations to create a stabilized video
        
        Raises:
            ValueError: If the video cannot be opened or has fewer than 2 frames.
        """
        # Read input video
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"ERROR: Could not open video at {self.input_path}")

        # Get frame count
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames <= 1:
            raise ValueError("ERROR: Video must have more than one frame.")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

        # Read first frame
        _, prev = cap.read()
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

        transforms = np.zeros((max(n_frames - 1, 0), 3), np.float32)
        frames = []

        for i in tqdm(range(n_frames - 1), desc="Stabilizing Video"):
            success, curr = cap.read()
            if not success:
                break

            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

            idx = np.where(status == 1)[0]
            good_old = prev_pts[idx]
            good_new = curr_pts[idx]

            m, _ = cv2.estimateAffine2D(good_old, good_new)
            if m is None:
                continue

            dx = m[0, 2]
            dy = m[1, 2]
            da = np.arctan2(m[1, 0], m[0, 0])

            transforms[i] = [dx, dy, da]

            mask = np.zeros_like(prev)
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 1)
                curr = cv2.circle(curr, (int(a), int(b)), 5, (0, 0, 255), -1)

            img = cv2.add(curr, mask)
            frames.append(img)

            prev_pts = good_new.reshape(-1, 1, 2)
            prev_gray = curr_gray.copy()

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = self.smooth(trajectory)
        difference = smoothed_trajectory - trajectory
        transforms_smooth = transforms + difference

        cap = cv2.VideoCapture(self.input_path)
        out1 = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

        for i in tqdm(range(n_frames - 1), desc="Applying Transformations"):
            success, frame = cap.read()
            if not success:
                break

            dx = transforms_smooth[i, 0]
            dy = transforms_smooth[i, 1]
            da = transforms_smooth[i, 2]

            m = np.zeros((2, 3), np.float32)
            m[0, 0] = np.cos(da)
            m[0, 1] = -np.sin(da)
            m[1, 0] = np.sin(da)
            m[1, 1] = np.cos(da)
            m[0, 2] = dx
            m[1, 2] = dy

            frame_stabilized = cv2.warpAffine(frame, m, (w, h))
            frame_stabilized = self.fixBorder(frame_stabilized)

            out1.write(frame_stabilized)
            cv2.waitKey(10)

        cap.release()
        out1.release()
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Stabilization")
    
    parser.add_argument("input_video", type=str, help="Path to the input video file",
                        required=True)
    
    parser.add_argument("--output_video", type=str, help="Path to the output video file", 
                        default=None)
    
    args = parser.parse_args()
    
    try:
        # Create the VideoStabilizer object
        stabilizer = VideoStabilizer(args.input_video, 
                                     args.output_video)
        
        # Stabilize the video
        stabilizer.stabilize_video()
        print("Drone.")
        
    except Exception as e:
        print(f"Error: {e}")