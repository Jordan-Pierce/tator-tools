import os
import re
import shutil
import argparse
from tqdm.auto import tqdm

import concurrent.futures

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import supervision as sv


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class YOLODataset:
    """
    Class to create a YOLO-formatted dataset for image clasissification, object detection, instance segmentation.
    Takes a pandas DataFrame with annotation data and generates the necessary directory structure,
    labels, and configuration files.

    DataFrame columns:
    - label: class name for the object (optional)
    - x, y, width, height: bounding box coordinates (optional, can be absolute or normalized)
    - polygon: list of polygon coordinates (optional, for segmentation)
    - image_path: path to the image file
    - image_name: filename of the image
    """

    def __init__(self, data, output_dir, dataset_name="yolo_dataset", train_ratio=0.8, test_ratio=0, task='detect',
                 format_class_names=False):
        """
        Initialize the YOLO dataset.

        :param data: pandas DataFrame containing annotation data
        :param output_dir: directory where datasets will be created
        :param dataset_name: name of the dataset (subdirectory will be created)
        :param train_ratio: train/val split ratio (default: 0.8)
        :param test_ratio: test split ratio (default: 0) - if > 0, train_ratio becomes train ratio
        :param task: task type ('classify', 'detect', 'segment')
        :param format_class_names: Whether to format class names (removes special characters)
        """
        # Define the minimal set of required columns
        needed_columns = ['label', 'x', 'y', 'width', 'height', 'image_path', 'image_name', 'polygon']
        
        # Validate that all required columns exist in the dataframe
        missing_columns = [col for col in needed_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
        
        has_labels = data['label'].notnull().any()
        has_polygon = data['polygon'].notnull().any()
        has_x = data['x'].notnull().any()
        has_y = data['y'].notnull().any()
        has_width = data['width'].notnull().any()
        has_height = data['height'].notnull().any()
        
        tasks = []
        
        if not has_labels:
            raise ValueError("DataFrame must contain 'label' column for class names")
        
        # Check if polygon values are present
        if has_polygon:
            tasks.extend(['classify', 'detect', 'segment'])
            
        # Check if bounding box values are present
        elif has_x and has_y and has_width and has_height:
            tasks.extend(['classify', 'detect'])
            
        # There are only class labels
        elif has_labels:
            tasks.append('classify')
        
        # Set the task based on the available data
        if task in tasks:
            self.task = task
        else:
            raise ValueError(f"Only the following tasks are available for this dataset: {tasks}")

        self.data = data
        
        # Convert to absolute paths
        self.output_dir = os.path.abspath(output_dir)
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.abspath(os.path.join(self.output_dir, dataset_name))
        
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.classes = None
        self.class_to_id = None
        
        if format_class_names:
            # Format class names to remove special characters
            # Replace special characters in all category labels with empty string
            self.data['label'] = self.data['label'].apply(lambda x: re.sub(r'[<>:"/\\|?* ]', '', x))
            
        # Check if and label values are empty
        if self.data['label'].isnull().any():
            # Replace with "Unverified" if label is missing
            self.data['label'].fillna("Unverified", inplace=True)

        # Create dataset directories based on task
        if self.task == 'classify':
            # For classification, we need a different directory structure with class folders
            # We'll create these folders after determining classes in split_dataset
            os.makedirs(os.path.join(self.dataset_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_dir, "val"), exist_ok=True)
            if self.test_ratio > 0:
                os.makedirs(os.path.join(self.dataset_dir, "test"), exist_ok=True)
        else:
            # Original directory structure for detection/segmentation
            os.makedirs(os.path.join(self.dataset_dir, "train", "images"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_dir, "train", "labels"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_dir, "valid", "images"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_dir, "valid", "labels"), exist_ok=True)
            if self.test_ratio > 0:
                os.makedirs(os.path.join(self.dataset_dir, "test", "images"), exist_ok=True)
                os.makedirs(os.path.join(self.dataset_dir, "test", "labels"), exist_ok=True)
                        
    def split_dataset(self):
        """
        Split the dataset into training, validation, and test sets based on train_ratio and test_ratio.

        :return: None
        """
        # Get unique images
        unique_images = self.data['image_name'].unique()

        # Shuffle
        np.random.shuffle(unique_images)
        
        # Calculate split indices
        if self.test_ratio > 0:
            # If test_ratio is provided, split into train/val/test
            test_idx = int(len(unique_images) * self.test_ratio)
            train_idx = int(len(unique_images) * self.train_ratio)
            
            test_images = unique_images[:test_idx]
            train_images = unique_images[test_idx:test_idx + train_idx]
            val_images = unique_images[test_idx + train_idx:]
        else:
            # Original behavior: split into train/val only
            split_idx = int(len(unique_images) * self.train_ratio)
            train_images = unique_images[:split_idx]
            val_images = unique_images[split_idx:]
            test_images = []  # Empty, no test set

        # Initialize yolo path columns if they don't exist
        if 'yolo_image_path' not in self.data.columns:
            self.data['yolo_image_path'] = ''
        if 'yolo_label_path' not in self.data.columns:
            self.data['yolo_label_path'] = ''

        # Update paths in the DataFrame with absolute paths
        for image in train_images:
            # Get the image extension   
            ext = os.path.splitext(image)[-1]
            # Create a mask for the image
            mask = self.data['image_name'] == image
            
            # Update paths based on task
            if self.task == 'classify':
                # For classification, images go in class-specific folders
                class_name = self.data.loc[mask, 'label'].iloc[0]  # Get class name for this image
                class_dir = os.path.join(self.dataset_dir, "train", class_name)
                os.makedirs(class_dir, exist_ok=True)
                image_paths = os.path.join(class_dir, image)
                self.data.loc[mask, 'yolo_image_path'] = image_paths
                # No label files for classification
                self.data.loc[mask, 'yolo_label_path'] = ''
            else:
                # Original behavior for detection/segmentation
                image_paths = os.path.join(self.dataset_dir, "train", "images", image)
                self.data.loc[mask, 'yolo_image_path'] = image_paths
                label_paths = os.path.join(self.dataset_dir, "train", "labels", image.replace(ext, '.txt'))
                self.data.loc[mask, 'yolo_label_path'] = label_paths

        for image in val_images:
            # Get the image extension
            ext = os.path.splitext(image)[-1]
            # Create a mask for the image
            mask = self.data['image_name'] == image
            
            # Update paths based on task
            if self.task == 'classify':
                # For classification, images go in class-specific folders
                class_name = self.data.loc[mask, 'label'].iloc[0]  # Get class name for this image
                class_dir = os.path.join(self.dataset_dir, "val", class_name)
                os.makedirs(class_dir, exist_ok=True)
                image_paths = os.path.join(class_dir, image)
                self.data.loc[mask, 'yolo_image_path'] = image_paths
                # No label files for classification
                self.data.loc[mask, 'yolo_label_path'] = ''
            else:
                # Original behavior for detection/segmentation
                image_paths = os.path.join(self.dataset_dir, "valid", "images", image)
                self.data.loc[mask, 'yolo_image_path'] = image_paths
                label_paths = os.path.join(self.dataset_dir, "valid", "labels", image.replace(ext, '.txt'))
                self.data.loc[mask, 'yolo_label_path'] = label_paths
            
        for image in test_images:
            # Get the image extension
            ext = os.path.splitext(image)[-1]
            # Create a mask for the image
            mask = self.data['image_name'] == image
            
            # Update paths based on task
            if self.task == 'classify':
                # For classification, images go in class-specific folders
                class_name = self.data.loc[mask, 'label'].iloc[0]  # Get class name for this image
                class_dir = os.path.join(self.dataset_dir, "test", class_name)
                os.makedirs(class_dir, exist_ok=True)
                image_paths = os.path.join(class_dir, image)
                self.data.loc[mask, 'yolo_image_path'] = image_paths
                # No label files for classification
                self.data.loc[mask, 'yolo_label_path'] = ''
            else:
                # Original behavior for detection/segmentation
                image_paths = os.path.join(self.dataset_dir, "test", "images", image)
                self.data.loc[mask, 'yolo_image_path'] = image_paths
                label_paths = os.path.join(self.dataset_dir, "test", "labels", image.replace(ext, '.txt'))
                self.data.loc[mask, 'yolo_label_path'] = label_paths
            
        # Print split information
        print(f"Dataset split: {len(train_images)} train, {len(val_images)} valid, {len(test_images)} test images")

    def write_yaml(self):
        """
        Writes a YOLO-formatted dataset yaml file

        :return: None
        """
        # Get the class names
        self.classes = sorted(self.data['label'].unique().tolist())
        
        # Check if there are any special characters in the class names
        if any(re.search(r'[<>:"/\\|?*]', name) for name in self.classes):
            raise ValueError("Class names cannot contain special characters; use format_class_names to format them.")
        
        self.class_to_id = {class_name: i for i, class_name in enumerate(self.classes)}

        # Create data.yaml with support for test set and absolute paths
        yaml_path = os.path.join(self.dataset_dir, "data.yaml")
        with open(yaml_path, 'w') as f:
            f.write(f"path: {self.dataset_dir}\n")
            f.write(f"train: {os.path.join(self.dataset_dir, 'train')}\n")
            f.write(f"val: {os.path.join(self.dataset_dir, 'valid')}\n")
            
            # Add test path if test_ratio > 0
            if self.test_ratio > 0:
                f.write(f"test: {os.path.join(self.dataset_dir, 'test')}\n")
                
            f.write(f"nc: {len(self.classes)}\n")
            
            # Write class names as dictionary for better yaml format
            f.write("names:\n")
            for i, class_name in enumerate(self.classes):
                f.write(f"  {i}: {class_name}\n")

    def write_labels(self):
        """
        Write YOLO-formatted labels based on the task type.

        :return: None
        """
        if self.task == 'detect':
            self.write_detection_labels()
        elif self.task == 'segment':
            self.write_segmentation_labels()
        elif self.task == 'classify':
            self.write_classification_dataset()
        else:
            raise ValueError(f"Unsupported task type: {self.task}")

    def write_detection_labels(self):
        """
        Write YOLO-formatted bounding box labels to text files for object detection.

        :return: None
        """
        # Group annotations by label path
        for label_path, label_group in tqdm(self.data.groupby('yolo_label_path'), desc="Writing detection labels"):
            # Ensure the directory exists
            os.makedirs(os.path.dirname(label_path), exist_ok=True)

            yolo_annotations = []
            for _, row in label_group.iterrows():
                
                if not os.path.exists(row['image_path']):
                    print(f"Warning: Image path {row['image_path']} does not exist. Skipping annotation.")
                    continue
                
                class_id = self.class_to_id[row['label']]

                # Extract image width and height
                image = Image.open(row['image_path'])
                img_width, img_height = image.size

                # Convert to YOLO format (normalized center coordinates) from normalized coordinates
                x_center = row['x'] + row['width'] / 2
                y_center = row['y'] + row['height'] / 2
                w = row['width']
                h = row['height']

                # Create YOLO-formatted annotation string
                yolo_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                yolo_annotations.append(yolo_annotation)

            # Save annotations
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

    def write_segmentation_labels(self):
        """
        Write YOLO-formatted polygon labels to text files for instance segmentation.

        :return: None
        """
        # Group annotations by label path
        for label_path, label_group in tqdm(self.data.groupby('yolo_label_path'), desc="Writing segmentation labels"):
            # Ensure the directory exists
            os.makedirs(os.path.dirname(label_path), exist_ok=True)

            yolo_annotations = []
            for _, row in label_group.iterrows():
                
                if not os.path.exists(row['image_path']):
                    print(f"Warning: Image path {row['image_path']} does not exist. Skipping annotation.")
                    continue
                
                class_id = self.class_to_id[row['label']]

                # Extract image width and height
                image = Image.open(row['image_path'])
                img_width, img_height = image.size

                # Check if polygon data exists
                if len(row['polygon']) == 0:
                    continue

                try:
                    # Polygon data stored as list of [x,y] coordinates
                    normalized_points = row['polygon']
                    
                    # Flatten the list of coordinates and format for YOLO segmentation
                    flattened_points = []
                    for point in normalized_points:
                        for coord in point:
                            # Convert to float if it's a string
                            if isinstance(coord, str):
                                flattened_points.append(float(coord))
                            else:
                                flattened_points.append(coord)
                    
                    points_str = " ".join([f"{p:.6f}" for p in flattened_points])
                    yolo_annotation = f"{class_id} {points_str}"
                    yolo_annotations.append(yolo_annotation)
                    
                except Exception as e:
                    print(f"Error processing polygon for {row['image_name']}: {e}")
                    continue

            # Save annotations
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
                
    def write_classification_dataset(self):
        """
        Create directories for each class in train, valid, and test splits.
        For classification, we don't write label files, just organize images into class folders.

        :return: None
        """
        # Get unique class names
        self.classes = sorted(self.data['label'].unique().tolist())
        self.class_to_id = {class_name: i for i, class_name in enumerate(self.classes)}
        
        print(f"Creating classification dataset with {len(self.classes)} classes")
        
        # Create directories for each class in each split
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(self.dataset_dir, split)
            if os.path.exists(split_dir):
                for class_name in self.classes:
                    os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        # Count images per class for reporting
        class_counts = {class_name: {'train': 0, 'val': 0, 'test': 0} for class_name in self.classes}
        
        # Count images in each class/split based on destination paths
        for _, row in self.data.drop_duplicates(subset=['image_path']).iterrows():
            if 'train' in row['yolo_image_path']:
                class_counts[row['label']]['train'] += 1
            elif 'valid' in row['yolo_image_path']:
                class_counts[row['label']]['val'] += 1
            elif 'test' in row['yolo_image_path']:
                class_counts[row['label']]['test'] += 1
        
        # Print class distribution
        for class_name, counts in class_counts.items():
            print(f"  {class_name}: train={counts['train']}, val={counts['val']}, test={counts['test']}")
        
    def copy_images(self, move_instead_of_copy=False, num_threads=None):
        """
        Copy or move images to the appropriate train/val/test directories based on the split.

        :param move_instead_of_copy: If True, move images instead of copying them
        :param num_threads: Number of threads to use. If None, uses default (typically CPU count)
        :return: None
        """        
        # Check if the DataFrame has the required columns
        if not all(col in self.data.columns for col in ['image_path', 'image_name', 'yolo_image_path']):
            raise ValueError("DataFrame must contain 'image_path', 'image_name' and 'yolo_image_path' columns")

        action_name = "Moving" if move_instead_of_copy else "Copying"
        
        # Define worker function to copy/move a single file
        def copy_image(src_path, dst_path):
            # Make sure source path is absolute
            if not os.path.isabs(src_path):
                src_path = os.path.abspath(src_path)
                
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            # Check if source exists and isn't already at destination
            if os.path.exists(src_path) and src_path != dst_path:
                if move_instead_of_copy:
                    shutil.move(src_path, dst_path)
                else:
                    shutil.copy(src_path, dst_path)
        
        # Get unique image paths and their destinations
        image_mapping = self.data[['image_path', 'yolo_image_path']].drop_duplicates()
        total = len(image_mapping)
        
        # Create a thread pool executor
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit unique file operations to the thread pool
            futures = [executor.submit(copy_image, row['image_path'], row['yolo_image_path']) 
                      for _, row in image_mapping.iterrows()]
            
            # Use tqdm to show progress
            for _ in tqdm(concurrent.futures.as_completed(futures), total=total, desc=f"{action_name} images"):
                pass
        
    def render_examples(self, output_dir, num_examples=10, include_boxes=True, include_labels=True):
        """
        Render examples from the dataset with annotations (boxes or polygons) and save the images.

        :param output_dir: Directory to save rendered images
        :param num_examples: Number of examples to render (default: 10)
        :param include_boxes: Whether to include annotations in the rendering
        :param include_labels: Whether to include class labels in the rendering
        :return: None
        """
        os.makedirs(output_dir, exist_ok=True)

        # Get unique image paths
        unique_images = self.data['image_path'].unique()

        # Select a subset of images
        if len(unique_images) > num_examples:
            import random
            selected_images = random.sample(list(unique_images), num_examples)
        else:
            selected_images = unique_images

        # Create the annotation objects
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        polygon_annotator = sv.PolygonAnnotator()

        # Process each selected image
        for image_path in tqdm(selected_images, desc="Rendering Examples"):
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Get image dimensions
                img_height, img_width = image.shape[:2]

                # Get annotations for this image
                annotations_df = self.data[self.data['image_path'] == image_path]
                
                if self.task == 'classify':
                    detections = None
                    labels = None
                    
                    # Show an image with the label
                    image_name = os.path.basename(image_path)
                    label = annotations_df['label'].iloc[0]
                    title = f"{image_name} - {label}"
                    image = cv2.putText(image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                elif self.task == 'detect':
                    # Create detections for bounding boxes
                    boxes = []
                    class_ids = []

                    for _, row in annotations_df.iterrows():
                        
                        if 'x' not in row or 'y' not in row or 'width' not in row or 'height' not in row:
                            print(f"Warning: No bounding box data for {row['image_name']}")
                            continue
                        
                        # Convert from normalized to pixel coordinates
                        x1 = int(row['x'] * img_width)
                        y1 = int(row['y'] * img_height)
                        x2 = int((row['x'] + row['width']) * img_width)
                        y2 = int((row['y'] + row['height']) * img_height)

                        boxes.append([x1, y1, x2, y2])
                        class_ids.append(self.class_to_id[row['label']])

                    detections = sv.Detections(
                        xyxy=np.array(boxes),
                        class_id=np.array(class_ids)
                    )

                    # Annotate image with boxes
                    if include_boxes:
                        image = box_annotator.annotate(scene=image, detections=detections)

                else:  
                    # Create polygon annotations
                    class_ids = []
                    masks = []
                    boxes = []
    
                    for _, row in annotations_df.iterrows():
                        
                        if 'polygon' not in row:
                            print(f"Warning: No polygon data for {row['image_name']}")
                            continue
                        
                        elif len(row['polygon']) < 3:
                            print(f"Warning: Not enough points for polygon in {row['image_name']}")
                            continue

                        # Convert list of coordinate pairs to numpy array 
                        polygon_pixels = [[p[0] * img_width, p[1] * img_height] for p in row['polygon']]
                        polygon_pixels = np.array(polygon_pixels)
                        
                        # Create bounding box for the polygon
                        x1 = int(row['x'] * img_width)
                        y1 = int(row['y'] * img_height)
                        x2 = int((row['x'] + row['width']) * img_width)
                        y2 = int((row['y'] + row['height']) * img_height)

                        box = ([x1, y1, x2, y2])
                        mask = sv.polygon_to_mask(polygon_pixels, (img_width, img_height))
                        
                        class_ids.append(self.class_to_id[row['label']])
                        masks.append(mask)
                        boxes.append(box)

                    if include_boxes:
                        detections = sv.Detections(
                            xyxy=np.array(boxes),
                            mask=np.array(masks),
                            class_id=np.array(class_ids)
                        )
                            
                    # Draw polygons on the image
                    image = polygon_annotator.annotate(
                        scene=image,
                        detections=detections
                    )

                # Add labels if requested
                if include_labels:
                    if self.task == 'detect':
                        # Map class IDs back to class names for bounding boxes
                        id_to_class = {v: k for k, v in self.class_to_id.items()}
                        labels = [id_to_class[class_id] for class_id in detections.class_id]
                        image = label_annotator.annotate(scene=image, detections=detections, labels=labels)

                # Save output image
                output_file = os.path.join(output_dir, os.path.basename(image_path))
                cv2.imwrite(output_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        print(f"Rendered {len(selected_images)} examples to {output_dir}")

    def process_dataset(self, move_images=False):
        """
        Process the entire dataset: split, write labels and generate YAML.

        :param move_images: If True, move images instead of copying them
        :return: None
        """
        print(f"Processing YOLO dataset with {len(self.data)} annotations...")
        # Split the dataset, create paths
        self.split_dataset()
        # Write YAML file for dataset
        self.write_yaml()
        # Write labels based on task type
        self.write_labels()
        # Move or copy images 
        self.copy_images(move_instead_of_copy=move_images)
        print(f"Dataset created at {self.dataset_dir}")
        print(f"Classes: {self.classes}")

        # Optional: Render examples
        examples_dir = os.path.join(self.dataset_dir, "examples")
        self.render_examples(examples_dir)
        
        
# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    """
    Main function to process the YOLO dataset from command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Create a YOLO-formatted dataset for object detection")
    
    parser.add_argument("--dataframe", "-d", type=str, required=True, 
                        help="Path to the CSV file containing annotation data")
    
    parser.add_argument("--output_dir", "-o", type=str, required=True, 
                        help="Base directory where datasets will be created/stored")
    
    parser.add_argument("--dataset_name", "-n", type=str, default="yolo_dataset",
                        help="Name of the dataset (subdirectory inside output_dir)")
    
    parser.add_argument("--train_ratio", "-s", type=float, default=0.8, 
                        help="Train/val split ratio (default: 0.8). If test_ratio > 0, this becomes train ratio.")
    
    parser.add_argument("--test_ratio", "-t", type=float, default=0.0,
                        help="Test split ratio (default: 0.0). If > 0, data will be split into train/val/test.")
    
    parser.add_argument("--task", "-k", type=str, default="detect",
                        help="Task type ('detect' or 'segment')")
    
    parser.add_argument("--move_images", "-m", action="store_true", 
                        help="Move images instead of copying them")
                        
    args = parser.parse_args()

    # Load dataframe
    print(f"Loading annotations from {args.dataframe}")
    try:
        df = pd.read_csv(args.dataframe)
    except Exception as e:
        print(f"Error loading dataframe: {e}")
        return
        
    print(f"Loaded dataframe with {len(df)} rows")
    
    # Create and process dataset
    dataset = YOLODataset(
        data=df,
        output_dir=os.path.abspath(args.output_dir),  # Convert to absolute path
        dataset_name=args.dataset_name,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        task=args.task
    )
    
    # Process the dataset
    dataset.process_dataset(move_images=args.move_images)
    print(f"YOLO dataset processed successfully. Output directory: {dataset.dataset_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
