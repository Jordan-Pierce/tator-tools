import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed

def smooth(trajectory, radius=50):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius)
    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def stabilize_video(input_path, output_path):
    # Read input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video at {input_path}")

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 1:
        print("Error: Video must have more than one frame.")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Read first frame
    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    transforms = np.zeros((max(n_frames - 1, 0), 3), np.float32)
    frames = []

    for i in range(n_frames - 1):
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

        # Comment out or remove the following line to skip writing the optical flow visualization
        # out.write(img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap = cv2.VideoCapture(input_path)
    out1 = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i in range(n_frames - 1):
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
        frame_stabilized = fixBorder(frame_stabilized)

        out1.write(frame_stabilized)
        cv2.waitKey(10)

    cap.release()
    out1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Stabilization")
    parser.add_argument("input_video", type=str, help="Path to the input video file")
    args = parser.parse_args()

    input_video = args.input_video
    output_video = os.path.splitext(input_video)[0] + "_stabilized.mp4"

    stabilize_video(input_video, output_video)