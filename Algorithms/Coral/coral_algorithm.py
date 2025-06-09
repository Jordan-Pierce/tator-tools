import os

from torch.cuda import is_available

import supervision as sv

from ultralytics import YOLO
from ultralytics import RTDETR

# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def polygons_to_points(bboxes, image):
    """
    Convert bounding boxes to normalized x, y, w, h format.

    Args:
    bboxes (list): List of bounding boxes, each represented as a numpy array of shape (N, 4)
    image (numpy.ndarray): The image array, used for normalization.

    Returns:
    list: List of normalized bounding boxes in [x, y, w, h] format.
    """
    # To hold the normalized bounding boxes
    normalized_bboxes = []

    for bbox in bboxes:
        # Extract min and max x, y coordinates of the bounding box
        min_x, min_y, max_x, max_y = bbox

        # Calculate width and height
        width = max_x - min_x
        height = max_y - min_y

        # Normalize x, y, width, and height
        normalized_x = min_x / image.shape[1]
        normalized_y = min_y / image.shape[0]
        normalized_w = width / image.shape[1]
        normalized_h = height / image.shape[0]

        # Ensure normalized coordinates are within bounds
        normalized_x = float(max(0, min(normalized_x, 1)))
        normalized_y = float(max(0, min(normalized_y, 1)))
        normalized_w = float(max(0, min(normalized_w, 1)))
        normalized_h = float(max(0, min(normalized_h, 1)))

        # Append normalized bounding box to the list
        normalized_bboxes.append([normalized_x, normalized_y, normalized_w, normalized_h])

    return normalized_bboxes


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class CoralAlgorithm:
    """
    Coral detection algorithm

    Utilizes a configuration file containing the various model/algorithm parameters.
    """

    def __init__(self, config: dict):
        """

        :param config:
        """
        self.yolo_model = None

        self.config = config
        self.device = 'cuda:0' if is_available() else 'cpu'
        print("NOTE: Using device", self.device)

    def initialize(self):
        """
        Initializes the model

        :return:
        """
        # Get the rock detection model path as specified in the config file
        model_path = self.config["model_path"]

        if not os.path.exists(model_path):
            raise Exception(f"ERROR: Model weights not found in ({model_path})!")

        try:
            if "yolo" in self.config['model_type'].lower():
                # Load the model weights
                self.yolo_model = YOLO(model_path)
            elif "rtdetr" in self.config['model_type'].lower():
                self.yolo_model = RTDETR(model_path)
            else:
                raise Exception(f"ERROR: Model type {self.config['model_type']} not recognized!")

        except Exception as e:
            raise Exception(f"ERROR: Could not load model!\n{e}")

        print(f"NOTE: Successfully loaded weights {model_path}")

    def infer(self, original_frame):
        """
        Performs inference on a single frame; if using smol mode, will use the
        slicer callback function to perform SAHI using supervision (detections will
        be aggregated together).

        Detections will be used with SAM to create instance segmentation masks.

        :param original_frame:
        """
        # Parameters in the config file
        iou = self.config["iou_threshold"]
        conf = self.config["model_confidence_threshold"]

        # Perform detection normally
        detections = self.yolo_model(original_frame,
                                     iou=iou,
                                     conf=conf,
                                     device=self.device,
                                     verbose=False)[0]

        # Convert results to supervision standards
        detections = sv.Detections.from_ultralytics(detections)

        # Do NMM / NMS with all detections (bboxes)
        detections = detections.with_nms(iou, class_agnostic=True)

        # Get the boxes
        bboxes = detections.xyxy
        conf = detections.confidence.tolist()

        # Convert to points
        polygon_points = polygons_to_points(bboxes, original_frame)

        return polygon_points, conf
