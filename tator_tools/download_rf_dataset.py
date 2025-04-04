import os
import re
import yaml
import shutil
import random
import argparse
from typing import Optional, Tuple, Dict, List, Union

from roboflow import Roboflow
import supervision as sv


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class RoboflowDatasetDownloader:
    """Class for downloading datasets from Roboflow."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Roboflow dataset downloader.
        
        Args:
            api_key: Roboflow API key. If None, will look for ROBOFLOW_API_KEY env variable.
        """
        self.version = None
        self.project_name = None
        self.dataset_path = None
        self.dataset = None

        # Get API key from environment variable if not provided
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "No API key provided. Please provide an API key or set ROBOFLOW_API_KEY environment variable."
            )
            
        self.rf = Roboflow(api_key=self.api_key)

    def download_dataset(self, workspace: str, project_name: str, version: int, 
                         format: str = "yolov8", output_dir: Optional[str] = None) -> str:
        """Download a dataset from Roboflow.
        
        Args:
            workspace: Roboflow workspace name
            project_name: Roboflow project name
            version: Dataset version number
            format: Dataset format (default: yolov8)
            output_dir: Directory to download dataset to (default: current directory)
            
        Returns:
            Path to the downloaded dataset
        """
        project = self.rf.workspace(workspace).project(project_name)
        version_obj = project.version(version)
        
        # Set output_dir if provided
        kwargs = {}
        if output_dir:
            kwargs["location"] = f"{output_dir}/{project_name}"
            
        # Download dataset
        dataset = version_obj.download(format, **kwargs)
        
        if os.path.exists(dataset.location):    
            print(f"Dataset downloaded to: {dataset.location}")
            self.dataset = dataset
            self.version = version
            self.project_name = project_name
            self.dataset_path = dataset.location
        else:
            print("Dataset download failed.")
            self.dataset = None
            
        return self.dataset
    
    @staticmethod
    def parse_roboflow_url(url: str) -> Tuple[str, str, int]:
        """Parse a Roboflow Universe URL to extract workspace, project, and version.
        
        Args:
            url: Roboflow Universe URL in the format:
                 https://universe.roboflow.com/workspace/project/dataset/version
                 
        Returns:
            Tuple of (workspace, project_name, version)
            
        Raises:
            ValueError: If the URL is not in the expected format
        """
        # Match pattern: workspace/project/dataset/version
        pattern = r"roboflow\.com/([^/]+)/([^/]+)(?:/dataset)?/(\d+)"
        match = re.search(pattern, url)
        
        if not match:
            raise ValueError(
                "Invalid Roboflow URL format. Expected: "
                "https://universe.roboflow.com/workspace/project/dataset/version"
            )
            
        workspace = match.group(1)
        project_name = match.group(2)
        version = int(match.group(3))
        
        return workspace, project_name, version
    
    def download_from_url(self, url: str, format: str = "yolov8", output_dir: Optional[str] = None) -> str:
        """Download a dataset from a Roboflow Universe URL.
        
        Args:
            url: Roboflow Universe URL
            format: Dataset format (default: yolov8)
            output_dir: Directory to download dataset to (default: current directory)
            
        Returns:
            Path to the downloaded dataset
        """
        workspace, project_name, version = self.parse_roboflow_url(url)
        return self.download_dataset(
            workspace=workspace,
            project_name=project_name,
            version=version,
            format=format,
            output_dir=output_dir
        )
    
    def get_supervision_dataset(self):
        """Returns a merged supervision dataset from the downloaded YOLO dataset.
        
        This method merges train, valid, and test datasets if they exist.
        
        Returns:
            A supervision DetectionDataset with all data merged
            
        Raises:
            ImportError: If supervision package is not installed
            ValueError: If no dataset has been downloaded yet or if no valid folders are found
        """
        if not self.dataset_path:
            raise ValueError("No dataset has been downloaded. Call download_dataset first.")
            
        # Check which folders exist
        folders = ["train", "valid", "test"]
        datasets = []
        data_yaml_path = os.path.join(self.dataset_path, "data.yaml")
        
        for folder in folders:
            images_dir = os.path.join(self.dataset_path, folder, "images")
            labels_dir = os.path.join(self.dataset_path, folder, "labels")
            
            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                try:
                    dataset = sv.DetectionDataset.from_yolo(
                        images_directory_path=images_dir,
                        annotations_directory_path=labels_dir,
                        data_yaml_path=data_yaml_path,
                    )
                    datasets.append(dataset)
                    print(f"Loaded {folder} dataset with {len(dataset)} samples")
                except Exception as e:
                    print(f"Warning: Could not load {folder} dataset: {e}")
        
        if not datasets:
            raise ValueError("No valid dataset folders found in the downloaded data")
            
        # Merge all datasets
        if len(datasets) == 1:
            return datasets[0]
            
        merged_dataset = sv.DetectionDataset.merge(datasets)
        print(f"Merged dataset contains {len(merged_dataset)} samples")
        return merged_dataset

    def export_to_yolo(self, dataset, output_dir: str, 
                       train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
        """Export a Supervision DetectionDataset to YOLO format with train/valid/test split.
        
        Args:
            dataset: A supervision.DetectionDataset
            output_dir: Directory to save the YOLO dataset
            train_ratio: Ratio of data to use for training (default: 0.7)
            val_ratio: Ratio of data to use for validation (default: 0.2)
            test_ratio: Ratio of data to use for testing (default: 0.1)
                        
        Returns:
            Path to the exported dataset directory
            
        Raises:
            ValueError: If the ratios don't sum to 1.0 or if dataset is empty
        """            
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
            
        if len(dataset) == 0:
            raise ValueError("Dataset is empty, nothing to export")
            
        # Create the output directory structure
        os.makedirs(output_dir, exist_ok=True)
        
        train_dir = os.path.abspath(os.path.join(output_dir, 'train', 'images'))
        val_dir = os.path.abspath(os.path.join(output_dir, 'valid', 'images'))
        test_dir = os.path.abspath(os.path.join(output_dir, 'test', 'images'))
        
        # Create data.yaml file
        data_yaml = {
            'path': os.path.abspath(output_dir),
            'train': train_dir,
            'val': val_dir,
            'test': test_dir,
            'names': dataset.classes,
            'nc': len(dataset.classes),
        }
        
        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
            
        # Create directory structure
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
            
        # Split dataset based on ratios
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        train_size = int(len(dataset) * train_ratio)
        val_size = int(len(dataset) * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Map indices to splits for easy lookup
        split_map = {}
        for idx in train_indices:
            split_map[idx] = 'train'
        for idx in val_indices:
            split_map[idx] = 'valid'
        for idx in test_indices:
            split_map[idx] = 'test'
            
        print(f"Exporting dataset with {len(train_indices)} train, "
              f"{len(val_indices)} validation, and {len(test_indices)} test samples")
            
        # Export each sample to the appropriate split folder
        for image_path, annotations in dataset.annotations.items():
            # Get the index of the current image_path
            idx = list(dataset.annotations.keys()).index(image_path)
            split = split_map[idx]
            
            # Get filename without path
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            
            # Copy image
            dst_image_path = os.path.join(output_dir, split, 'images', filename)
            shutil.copy2(image_path, dst_image_path)
            
            # Create YOLO annotation file
            label_path = os.path.join(output_dir, split, 'labels', f"{base_name}.txt")
            
            # Convert annotations to YOLO format and write to file
            self._write_yolo_annotations(annotations, label_path)
                
        print(f"Dataset exported to {output_dir}")
        return output_dir
    
    def _write_yolo_annotations(self, annotations, label_path):
        """Write detection annotations in YOLO format.
        
        Args:
            annotations: Supervision Detections object
            label_path: Path to save the YOLO label file
        """        
        with open(label_path, 'w') as f:
            if len(annotations.xyxy) == 0:
                return
                
            # Get image dimensions from the first bounding box
            img_height, img_width = 1.0, 1.0  # Normalized coordinates
            
            # Process each detection
            for i in range(len(annotations.xyxy)):
                # Get bounding box in YOLO format (normalized)
                x1, y1, x2, y2 = annotations.xyxy[i]
                
                # Convert to YOLO format: [class_id, x_center, y_center, width, height]
                x_center = (x1 + x2) / 2.0 / img_width
                y_center = (y1 + y2) / 2.0 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                class_id = int(annotations.class_id[i])
                
                # Sanity check to ensure values are in [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # Write to file in YOLO format: class_id center_x center_y width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Download datasets from Roboflow")
    
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"),
                        help="Roboflow API key (can also use ROBOFLOW_API_KEY env variable)")
    
    # Create mutually exclusive group for URL or individual parameters
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--url", 
                             help="Roboflow Universe URL")
    
    # Keep the original parameters as an alternative
    input_group.add_argument("--workspace", 
                             help="Roboflow workspace name")
    
    parser.add_argument("--project",
                        help="Roboflow project name")
    
    parser.add_argument("--version", type=int, 
                        help="Dataset version number")
    
    # Common parameters
    parser.add_argument("--format", default="yolov8", 
                        help="Dataset format (default: yolov8)")
    
    parser.add_argument("--output_dir", 
                        help="Directory to download dataset to")
    
    parser.add_argument("--export", 
                        help="Export merged dataset to YOLO format (specify output directory)")
    
    parser.add_argument("--train-ratio", type=float, default=0.7, 
                        help="Ratio of data to use for training (default: 0.7)")
    
    parser.add_argument("--val-ratio", type=float, default=0.2, 
                        help="Ratio of data to use for validation (default: 0.2)")
    
    parser.add_argument("--test-ratio", type=float, default=0.1, 
                        help="Ratio of data to use for testing (default: 0.1)")
    
    args = parser.parse_args()
    
    # Validate that if workspace is provided, project and version are also provided
    if args.workspace and (not args.project or not args.version):
        parser.error("--project and --version are required when --workspace is specified")
    
    downloader = RoboflowDatasetDownloader(api_key=args.api_key)
    
    # Choose download method based on input type
    if args.url:
        downloader.download_from_url(
            url=args.url,
            format=args.format,
            output_dir=args.output_dir
        )
    else:
        downloader.download_dataset(
            workspace=args.workspace,
            project_name=args.project,
            version=args.version,
            format=args.format,
            output_dir=args.output_dir
        )
    
    # After downloading, export if requested
    if args.export:
        try:
            dataset = downloader.get_supervision_dataset()
            downloader.export_to_yolo(
                dataset=dataset,
                output_dir=args.export,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio
            )
        except Exception as e:
            print(f"Failed to export dataset: {e}")


if __name__ == "__main__":
    main()
