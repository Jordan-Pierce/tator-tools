import os
import re
import yaml
import random
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import supervision as sv


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class YOLORegionCropper:
    def __init__(self, dataset_path, output_dir=None, dataset_name=None, format_class_names=False):
        """
        Initialize the YOLORegionCropper with a dataset path. YOLORegionCropper is used to create a classification dataset from an existing detection or segmentation dataset.
        
        :param dataset_path: Path to the YAML dataset configuration file
        :param output_dir: Directory where the classification dataset will be saved
        :param dataset_name: Name for the output dataset folder (default: "Cropped_Dataset")
        :param format_class_names: Whether to format class names (removes special characters)
        """
        # Path to YAML dataset configuration file
        self.dataset_path = dataset_path
        # Dataset (parsed from YAML)
        self.dataset_data = None
        # List of class names
        self.classes = None
        # Whether to format class names
        self.format_class_names = format_class_names
        
        # Paths to train, validation, and test folders in the dataset
        self.train_path = None
        self.valid_path = None
        self.test_path = None
        
        # Source dataset (Supervision Detection Dataset)
        self.src_dataset = None

        # Determine output directory
        folder_name = dataset_name if dataset_name else "Cropped_Dataset"
        
        if output_dir:
            self.output_dir = f"{output_dir}/{folder_name}"
        else:
            # Use the directory containing the dataset.yaml
            dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
            self.output_dir = f"{dataset_dir}/{folder_name}"

    def load_dataset(self):
        """
        Load the YAML dataset configuration file and create output directories.
        
        :return: None
        """
        print("NOTE: Loading dataset...")

        special_characters = False

        # Check if the dataset file exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError("Dataset not found.")

        with open(self.dataset_path, 'r') as file:
            self.dataset_data = yaml.safe_load(file)

        # Get the class names (handle dict format: {0: name0, 1: name1, ...})
        names = self.dataset_data['names']
        if isinstance(names, dict):
            # Sort by key to ensure correct order
            self.classes = [names[i] for i in sorted(names, key=lambda x: int(x))]
        else:
            self.classes = names

        # Check if there are any special characters in the class names
        if any(re.search(r'[<>:"/\\|?*]', name) for name in self.classes):
            special_characters = True

        # Format class names if specified
        if special_characters:
            if self.format_class_names:
                self.classes = [re.sub(r'[<>:"/\\|?*]', '', name).strip() for name in self.classes]
            else:
                raise ValueError("Class names cannot contain special characters use format_class_names to format them.")

        # Process train path
        if isinstance(self.dataset_data.get('train'), str):
            self.train_path = self.dataset_data['train']
        elif isinstance(self.dataset_data.get('train'), list) and len(self.dataset_data['train']) > 0:
            self.train_path = self.dataset_data['train'][0]  # Take the first train path if multiple exist

        # Process validation path - check both 'val' and 'valid' keys
        if isinstance(self.dataset_data.get('val'), str):
            self.valid_path = self.dataset_data['val']
        elif isinstance(self.dataset_data.get('val'), list) and len(self.dataset_data['val']) > 0:
            self.valid_path = self.dataset_data['val'][0]
        elif isinstance(self.dataset_data.get('valid'), str):
            self.valid_path = self.dataset_data['valid']
        elif isinstance(self.dataset_data.get('valid'), list) and len(self.dataset_data['valid']) > 0:
            self.valid_path = self.dataset_data['valid'][0]

        # Process test path
        if isinstance(self.dataset_data.get('test'), str):
            self.test_path = self.dataset_data['test']
        elif isinstance(self.dataset_data.get('test'), list) and len(self.dataset_data['test']) > 0:
            self.test_path = self.dataset_data['test'][0]

        # Create all the sub folders (YOLO Image Classification Dataset format)
        for split in ['train', 'val', 'test']:
            for name in self.classes:
                os.makedirs(f"{self.output_dir}/{split}/{name}", exist_ok=True)

    def convert_dataset(self):
        """
        Load detection dataset from train, validation, and test paths.

        :return: None
        """
        print("NOTE: Converting dataset...")
        
        # Dictionary to store all dataset components
        dataset_parts = []

        # Load train dataset if available
        if self.train_path:
            images_path = f"{self.train_path}/images" if not self.train_path.endswith("images") else self.train_path
            labels_path = images_path.replace('images', 'labels')
            try:
                # Create a Supervision Detection Dataset from YOLO format
                train = sv.DetectionDataset.from_yolo(
                    images_directory_path=images_path,
                    annotations_directory_path=labels_path,
                    data_yaml_path=self.dataset_path,
                )
                dataset_parts.append(train)
                # Count the number of images
                image_count = sum(1 for _ in train)
                print(f"Added {image_count} images from train dataset")
                
            except Exception as e:
                print(f"Warning: Failed to load train dataset. Error: {str(e)}")

        # Load validation dataset if available
        if self.valid_path:
            # Handle paths that might use 'valid' or 'val'
            images_path = f"{self.valid_path}/images" if not self.valid_path.endswith("images") else self.valid_path
            labels_path = images_path.replace('images', 'labels')
            try:
                # Create a Supervision Detection Dataset from YOLO format
                valid = sv.DetectionDataset.from_yolo(
                    images_directory_path=images_path,
                    annotations_directory_path=labels_path,
                    data_yaml_path=self.dataset_path,
                )
                dataset_parts.append(valid)
                # Count the number of images
                image_count = sum(1 for _ in valid)
                print(f"Added {image_count} images from validation dataset")
            except Exception as e:
                print(f"Warning: Failed to load validation dataset. Error: {str(e)}")

        # Load test dataset if available
        if self.test_path:
            images_path = f"{self.test_path}/images" if not self.test_path.endswith("images") else self.test_path
            labels_path = images_path.replace('images', 'labels')
            try:
                # Create a Supervision Detection Dataset from YOLO format
                test = sv.DetectionDataset.from_yolo(
                    images_directory_path=images_path,
                    annotations_directory_path=labels_path,
                    data_yaml_path=self.dataset_path,
                )
                dataset_parts.append(test)
                # Count the number of images
                image_count = sum(1 for _ in test)
                print(f"Added {image_count} images from test dataset")
            except Exception as e:
                print(f"Warning: Failed to load test dataset. Error: {str(e)}")

        # Check if any data was loaded
        if not dataset_parts:
            raise ValueError("No data could be loaded. Please check your dataset paths.")

        # Merge all datasets into a single dataset
        # If there's only one part, use it directly
        if len(dataset_parts) == 1:
            self.src_dataset = dataset_parts[0]
        else:
            # Merge multiple datasets - first, collect all data
            merged_images = {}
            merged_annotations = {}
            
            for dataset in dataset_parts:
                for img_path, image, annotation in dataset:
                    merged_images[img_path] = image
                    merged_annotations[img_path] = annotation
            
            # Create a new merged dataset
            self.src_dataset = sv.DetectionDataset(
                classes=self.classes,
                images=merged_images,
                annotations=merged_annotations
            )

        # Count total detections
        detection_count = 0
        for _, _, annotation in self.src_dataset:
            detection_count += len(annotation.xyxy)

        print(f"NOTE: Loaded dataset - {detection_count} detections found")

    def extract_crop(self, image, xyxy):
        """
        Extract a crop (sub-image) from a NumPy array image based on bounding box coordinates.

        :param image: NumPy array representing the image
        :param xyxy: List or tuple of bounding box coordinates [x1, y1, x2, y2]
        :return: NumPy array of the extracted crop, or None if the crop has no area
        """
        x1, y1, x2, y2 = map(int, xyxy)

        # Ensure coordinates are within image boundaries
        height, width = image.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)

        # Extract the crop
        crop = image[y1:y2, x1:x2]

        if crop.shape[0] > 0 and crop.shape[1] > 0:
            return crop

        return None

    def save_crop(self, split, class_name, crop_name, crop):
        """
        Save a crop as an RGB image.

        :param split: Dataset split (e.g., 'train', 'val', 'test')
        :param class_name: Name of the class
        :param crop_name: Name of the crop
        :param crop: Numpy array representing the image
        :return: Path to the saved crop
        """
        crop_dir = f"{self.output_dir}/{split}/{class_name}"
        os.makedirs(crop_dir, exist_ok=True)

        crop_path = f"{crop_dir}/{crop_name}"

        # Ensure the crop is in RGB format
        if len(crop.shape) == 2:  # If it's a grayscale image
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 4:  # If it's RGBA
            crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
        elif crop.shape[2] == 3:
            # If it's already RGB, ensure it's in the correct order (OpenCV uses BGR by default)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Ensure the data type is uint8
        if crop.dtype != np.uint8:
            crop = (crop * 255).astype(np.uint8)

        # Save the image
        Image.fromarray(crop).save(crop_path)

        return str(crop_path)

    def create_crops(self):
        """
        Creates classification crops from detection bounding boxes and 
        organizes them into train/val/test splits by class.

        :return: None
        """
        # Count to track crops created
        total_crops = 0
        total_images = 0

        # Loop through all the images in the detection dataset using the recommended approach
        for image_path, image, annotations in tqdm(self.src_dataset, desc="Creating crops"):
            total_images += 1

            # Get the image basename
            image_name = os.path.basename(image_path).split(".")[0]

            # Loop through detections, crop, and then save in split folder
            for i, (xyxy, class_id) in enumerate(zip(annotations.xyxy, annotations.class_id)):

                # Randomly assign the crop to train, valid or test
                split = random.choices(['train', 'val', 'test'], weights=[70, 20, 10])[0]

                # Get the crop
                crop = self.extract_crop(image, xyxy)

                if crop is not None:
                    class_name = self.src_dataset.classes[class_id]
                    crop_name = f"{class_name}_{i}_{image_name}.jpg"
                    self.save_crop(split, class_name, crop_name, crop)
                    total_crops += 1

        print(f"Created {total_crops} crops from {total_images} images")

    def write_classification_yaml(self):
        """
        Create a YOLO-formatted classification dataset YAML file.
        
        :return: None
        """
        print("NOTE: Writing classification dataset YAML...")
        
        # Count classes and items per class
        class_counts = {class_name: {'train': 0, 'val': 0, 'test': 0} for class_name in self.classes}
        
        for split in ['train', 'val', 'test']:
            for class_name in self.classes:
                class_dir = f"{self.output_dir}/{split}/{class_name}"
                if os.path.exists(class_dir):
                    count = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
                    class_counts[class_name][split] = count
        
        # Create the YAML content
        yaml_content = {
            'path': self.output_dir,
            'train': f"{self.output_dir}/train",
            'val': f"{self.output_dir}/val",
            'test': f"{self.output_dir}/test",
            'nc': len(self.classes),
            'names': {i: name for i, name in enumerate(self.classes)},
        }
        
        # Write the YAML file
        yaml_path = f"{self.output_dir}/data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            
        # Print summary
        print(f"Classification dataset YAML written to {yaml_path}")
        print("Class distribution:")
        for class_name, counts in class_counts.items():
            print(f"  {class_name}: train={counts['train']}, val={counts['val']}, test={counts['test']}")

    def process_dataset(self):
        """
        Main execution method that loads dataset and creates classification crops.

        :return: None
        """
        # Load the data.yaml file
        self.load_dataset()

        # Convert the dataset to Supervision format
        self.convert_dataset()
        
        # Create crops from the Supervision dataset
        self.create_crops()

        # Create the classification dataset YAML file
        self.write_classification_yaml()

        print(f"NOTE: Created classification dataset in {self.output_dir}")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    """
    Parse command line arguments and run the crop creation process.

    :return: None
    """
    parser = argparse.ArgumentParser(description="Create a classification from an existing detection dataset")

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the detection dataset's data.yaml file")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to the output directory for the classification dataset (optional)")
                        
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name for the output dataset folder (default: 'Cropped_Dataset')")

    args = parser.parse_args()

    try:
        # Run the conversion process
        converter = YOLORegionCropper(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name
        )
        # Run the conversion process
        converter.process_dataset()
        print("Done.")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except ValueError as e:
        print(f"Error: Invalid value - {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")


if __name__ == "__main__":
    main()
