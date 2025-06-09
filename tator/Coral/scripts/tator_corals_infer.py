import argparse
import datetime
import os
import sys
import yaml
from pathlib import Path

import tator

current_dir = Path.cwd()
parent_dir = current_dir.parent
utilities_path = parent_dir.joinpath("utilities")
sys.path.insert(1, str(utilities_path))

import tator_utilities

current_dir = Path.cwd()
parent_dir = current_dir.parents[1]
algorithm_path = parent_dir.joinpath("Algorithms")
algorithm_path = algorithm_path.joinpath("Coral")
sys.path.insert(1, str(algorithm_path))

from coral_algorithm import CoralAlgorithm

def parse_args() -> argparse.Namespace:
    """ Process script arguments
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--host", type=str, help="Tator URL", required=True)
    parser.add_argument("--token", type=str, help="User's API token", required=True)
    parser.add_argument("--media-id", type=int, help="Media ID to process", required=True)
    parser.add_argument("--frame", type=int, help="Frame to create detection boxes", required=True)
    parser.add_argument("--version-id", type=int, help="Version ID to put localizations in", required=True)
    parser.add_argument("--algorithm-config", type=str, required=True)
    parser.add_argument("--tator-config", type=str, help="Tator configuration .yaml file", required=True)
    parser.add_argument("--work-folder", type=str, default="/tmp/")
    return parser.parse_args()

def script_main() -> None:
    """ Main entrypoint to script
    """

    # Get script arguments
    args = parse_args()

    # Set up API to Tator
    tator_api = tator.get_api(host=args.host, token=args.token)

    # Get the algorithm configuration and tator configuration information
    with open(args.algorithm_config, "r") as file_handle:
        algo_config = yaml.safe_load(file_handle)

    with open(args.tator_config, "r") as file_handle:
        tator_config = yaml.safe_load(file_handle)

    # Gather media and frame information, perform checks on them
    media = tator_api.get_media(id=args.media_id)
    assert media.project == tator_config["project"]

    media_types = tator_api.get_media_type_list(project=media.project)
    is_video = False
    for media_type in media_types:
        if media_type.dtype == "video" and media.type == media_type.id:
            is_video = True
            break

    frame = 0
    if is_video:
        assert args.frame is not None, "Frame is required for video media"
        frame = args.frame
        assert frame >= 0 and frame < media.num_frames, "Frame is out of the video bounds"

    print(f"[{datetime.datetime.now()}] Processing media {media.name} ({media.id} - Frame {frame}")

    # Check the version exists
    version = tator_api.get_version(id=args.version_id)
    assert version.project == tator_config["project"]

    # Check work folder
    work_folder = args.work_folder
    assert os.path.exists(work_folder), "Work folder does not exist"

    # Get the image
    print(f"[{datetime.datetime.now()}] Downloading frame {frame} for {media.name} ({media.id})")

    frame_buffer = tator_utilities.FrameBuffer(
        tator_api=tator_api,
        media=media,
        work_folder=work_folder)

    print(f"[{datetime.datetime.now()}] Finished downloading frame {frame}.")

    # Initialize the algorithm
    print(f"[{datetime.datetime.now()}] Initializing model")
    algo = CoralAlgorithm(config=algo_config)
    algo.initialize()
    print(f"[{datetime.datetime.now()}] Finished initialization.")

    # Run the algorithm on the frame
    print(f"[{datetime.datetime.now()}] Pulling frame")
    image = frame_buffer.get_single_frame(frame=frame)
    print(f"[{datetime.datetime.now()}] Running inference")
    all_bboxes, all_confidences = algo.infer(image)
    print(f"[{datetime.datetime.now()}] Finished inference.")

    # Create localization specs and upload to Tator
    localization_specs = []
    for bbox, conf in zip(all_bboxes, all_confidences):
        spec = {
            "type": tator_config["coral_box_type"],
            "media_id": media.id,
            "version_id": version.id,
            "frame": frame,
            "x": float(bbox[0]),
            "y": float(bbox[1]),
            "width": float(bbox[2]),
            "height": float(bbox[3]),
            "attributes": {
                "AlgorithmName": f"Algorithm Generated - CARL {algo_config['version']}",
                "AlgorithmScore": float(conf)}
        }
        localization_specs.append(spec)

    answer = input(f"[{datetime.datetime.now()}]  Upload {len(localization_specs)} to Tator? (Y/n): ")
    if answer == "Y":
        print(f"[{datetime.datetime.now()}] Uploading {len(localization_specs)} localizations to Tator")
        for response in tator.util.chunked_create(
                func=tator_api.create_localization_list,
                project=media.project,
                body=localization_specs):
            print(f"[{datetime.datetime.now()}] Uploaded {len(response.id)} of {len(localization_specs)} localizations")

if __name__ == "__main__":
    script_main()