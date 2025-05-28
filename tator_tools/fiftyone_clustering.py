import os
import glob
import argparse
from tqdm import tqdm
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

import torch
from ultralytics import YOLO

import fiftyone as fo
import fiftyone.brain as fob


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def calculate_embeddings(df: pd.DataFrame, model_weights: str, imgsz: int, image_column: str = "Path") -> np.ndarray:
    """
    Calculate embeddings for a given dataframe using a YOLO model.
    """
    # Load the model
    model = YOLO(model_weights)
    # Get the image size
    imgsz = model.__dict__['overrides']['imgsz']

    # Get the device
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    print(f"NOTE: Using device {device}")

    # Run a blank image through the model to load the weights
    _ = model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), device=device) 

    embeddings_list = []

    # Use the length of df as the total for tqdm
    total_items = len(df)
    for path in tqdm(df['Path'].tolist(), total=total_items, desc="Calculating embeddings"):
        embeddings = model.embed(path, imgsz=imgsz, stream=False, device=device, verbose=False)
        embeddings_list.append(embeddings[0].cpu().numpy())
    
    # Convert the embeddings list to a numpy array
    embeddings = np.array(embeddings_list)
     
    # Clear the cache
    torch.cuda.empty_cache()  
    
    return embeddings


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class FiftyOneDatasetViewer:
    def __init__(self, 
                 image_dir=None, 
                 dataframe=None, 
                 image_path_column=None, 
                 feature_columns=None, 
                 dataset_name=None, 
                 nickname=None, 
                 custom_embeddings=None, 
                 clustering_method='umap',
                 num_dims=2):
        """
        Initialize the FiftyOneDatasetViewer object
        
        Args:
            param: image_dir (str): Path to directory containing images
            param: dataframe (pd.DataFrame): Dataframe containing image paths
            param: image_path_column (str): Column name in dataframe that contains image paths
            param: feature_columns (list): Column names in dataframe to add as features
            param: dataset_name (str): Name of existing FiftyOne dataset to load
            param: nickname (str): Optional nickname for the dataset
            param: custom_embeddings (np.ndarray): Custom embeddings to use for visualization
            param: clustering_method (str): Clustering method for visualization (umap, pca, tsne)
            param: num_dims (int): Number of dimensions for UMAP visualization
        """  
        self.image_dir = image_dir
        self.dataframe = dataframe
        self.image_path_column = image_path_column
        self.feature_columns = feature_columns or []
        self.custom_embeddings = custom_embeddings
        self.clustering_method = clustering_method
        self.num_dims = num_dims
        
        # Determine dataset name
        if dataset_name:
            self.dataset_name = dataset_name
        elif self.image_dir:
            self.dataset_name = os.path.basename(os.path.normpath(image_dir))
        else:
            self.dataset_name = "dataframe_dataset"
            
        self.nickname = nickname or self.dataset_name  # Use dataset_name as default nickname
        self.dataset = None
        self.brain_key = None

    def create_or_load_dataset(self):
        """Creates a new dataset or loads existing one"""
        if self.nickname in fo.list_datasets():
            overwrite = input(f"Dataset with nickname '{self.nickname}' already exists. Overwrite? (y/n): ").lower()
            if overwrite == 'y':
                print(f"Overwriting existing dataset: {self.nickname}")
                fo.delete_dataset(self.nickname)
                self.dataset = fo.Dataset(self.nickname)
            else:
                print("Loading existing dataset.")
                self.dataset = fo.load_dataset(self.nickname)
                return
        else:
            print(f"Creating new dataset: {self.nickname}")
            self.dataset = fo.Dataset(self.nickname)

        # Get filepaths either from directory or dataframe
        filepaths = self._get_filepaths()
        if not filepaths:
            raise ValueError("No valid image files found")

        # Add standard fields to dataset schema
        self.dataset.add_sample_field("file_extension", fo.StringField)
        self.dataset.add_sample_field("relative_path", fo.StringField)
        self.dataset.add_sample_field("creation_date", fo.DateTimeField)
        self.dataset.add_sample_field("modification_date", fo.DateTimeField)
        self.dataset.add_sample_field("mean_color", fo.VectorField)
        self.dataset.add_sample_field("mean_brightness", fo.VectorField)
        if self.custom_embeddings is not None:
            self.dataset.add_sample_field("embedding", fo.VectorField)

        # Add dataframe feature fields to schema
        if self.dataframe is not None and self.feature_columns:
            for col in self.feature_columns:
                if col in self.dataframe.columns:
                    dtype = self.dataframe[col].dtype
                    if np.issubdtype(dtype, np.number):
                        if np.issubdtype(dtype, np.integer):
                            self.dataset.add_sample_field(col, fo.IntField)
                        else:
                            self.dataset.add_sample_field(col, fo.FloatField)
                    elif np.issubdtype(dtype, np.datetime64):
                        self.dataset.add_sample_field(col, fo.DateTimeField)
                    elif np.issubdtype(dtype, np.bool_):
                        self.dataset.add_sample_field(col, fo.BooleanField)
                    else:
                        self.dataset.add_sample_field(col, fo.StringField)

        samples = []
        for idx, filepath in enumerate(tqdm(filepaths, desc="Processing images")):
            img = cv2.imread(filepath)
            if img is None:
                continue

            filename = os.path.basename(filepath)
            file_stats = os.stat(filepath)

            sample = fo.Sample(filepath=filepath)
            sample.metadata = fo.ImageMetadata(
                size_bytes=file_stats.st_size,
                mime_type=f"image/{os.path.splitext(filename)[1][1:]}",
                width=img.shape[1],
                height=img.shape[0],
                num_channels=img.shape[2] if len(img.shape) > 2 else 1,
            )

            sample["file_extension"] = os.path.splitext(filename)[1]
            sample["relative_path"] = os.path.relpath(filepath, os.path.dirname(filepath))
            sample["creation_date"] = datetime.fromtimestamp(file_stats.st_ctime)
            sample["modification_date"] = datetime.fromtimestamp(file_stats.st_mtime)
            sample["mean_color"] = img.mean(axis=(0, 1))
            sample["mean_brightness"] = [float(img.mean())]
            
            # Add features from dataframe if available
            if self.dataframe is not None and self.feature_columns:
                if len(filepaths) == len(self.dataframe):
                    df_idx = idx
                else:
                    df_idx = self.dataframe[self.dataframe[self.image_path_column] == filepath].index[0]

                for col in self.feature_columns:
                    if col in self.dataframe.columns:
                        value = self.dataframe.iloc[df_idx][col]
                        sample[col] = value
                        
            # Add embedding if available
            if self.custom_embeddings is not None and idx < len(self.custom_embeddings):
                sample["embedding"] = self.custom_embeddings[idx]

            samples.append(sample)

        self.dataset.add_samples(samples)

    def _get_filepaths(self):
        """Get filepaths from either directory or dataframe"""
        if self.image_dir:
            return glob.glob(os.path.join(self.image_dir, "*.*"))
        elif self.dataframe is not None and self.image_path_column:
            # Extract image paths from dataframe
            filepaths = self.dataframe[self.image_path_column].tolist()
            # Filter out any invalid paths
            return [path for path in filepaths if os.path.isfile(path)]
        return []

    def compute_embeddings(self):
        """Compute embeddings for all images in the dataset"""
        # If custom embeddings are provided, use them
        if self.custom_embeddings is not None:
            if isinstance(self.custom_embeddings, np.ndarray) and len(self.custom_embeddings) == len(self.dataset):
                print("Using provided custom embeddings")
                return self.custom_embeddings
            else:
                print("Warning: Custom embeddings don't match dataset size. Computing default embeddings instead.")

        # Otherwise compute default embeddings
        filepaths = [sample.filepath for sample in self.dataset]
        return np.array([
            cv2.resize(cv2.imread(f, cv2.IMREAD_UNCHANGED), (64, 64),
                       interpolation=cv2.INTER_AREA).ravel()
            for f in filepaths
        ])

    def create_visualization(self, embeddings):
        """Create UMAP visualization"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.brain_key = f"{self.dataset_name}_{self.clustering_method}_{timestamp}"

        return fob.compute_visualization(
            self.dataset,
            embeddings=embeddings,
            num_dims=self.num_dims,
            method=self.clustering_method,  # umap, pca, tsne
            brain_key=self.brain_key,
            verbose=True,
        )

    def process_dataset(self):
        """Main processing method"""
        # Create or load dataset
        self.create_or_load_dataset()

        print("Computing embeddings...")
        # Compute embeddings
        embeddings = self.compute_embeddings()

        print("Computing UMAP visualization...")
        # Create UMAP visualization
        self.create_visualization(embeddings)
        self.dataset.load_brain_results(self.brain_key)
        
    def visualize(self):
        """visualize the dataset"""
        # Process the dataset
        self.process_dataset()

        print(f"Launching FiftyOne App with visualization '{self.brain_key}'")
        # Launch FiftyOne App
        session = fo.launch_app(self.dataset)
        session.wait()


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Process images and create a FiftyOne dataset with UMAP visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
        Examples:
        python FiftyOne_Clustering.py --image_dir /path/to/images
        python FiftyOne_Clustering.py --dataset_name existing_dataset
        python FiftyOne_Clustering.py --dataframe data.csv --image_path_column path
        python FiftyOne_Clustering.py --embeddings embeddings.npy --image_dir /path/to/images
        python FiftyOne_Clustering.py --list_datasets
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_dir', type=str,
                       help='Path to directory containing images')
                       
    group.add_argument('--dataframe', type=str,
                       help='Path to CSV/Excel file containing image paths')

    group.add_argument('--dataset_name', type=str,
                       help='Name of existing FiftyOne dataset to load')

    group.add_argument('--list_datasets', action='store_true',
                       help='List all available FiftyOne datasets and exit')
    
    group.add_argument('--delete_dataset', type=str,
                       help='Delete an existing dataset.')

    parser.add_argument('--nickname', type=str, 
                        help='Optional: A nickname for the dataset.')
                        
    parser.add_argument('--image_path_column', type=str, default='filepath',
                        help='Column name in dataframe that contains image paths')
                        
    parser.add_argument('--feature_columns', type=str, nargs='+',
                        help='Column names in dataframe to add as features')
    
    parser.add_argument('--clustering_method', type=str, default='umap', choices=['umap', 'pca', 'tsne'],
                        help='Clustering method for visualization (umap, pca, tsne)')
    
    parser.add_argument('--num_dims', type=int, default=2, choices=[2, 3],
                        help='Number of dimensions for UMAP visualization')

    args = parser.parse_args()
    
    if args.delete_dataset:
        if args.delete_dataset in fo.list_datasets():
            print(f"Deleting dataset: {args.delete_dataset}")
            fo.delete_dataset(args.delete_dataset)
        else:
            print(f"Dataset not found: {args.delete_dataset}")
        return

    if args.list_datasets:
        datasets = fo.list_datasets()
        if datasets:
            print("Available datasets:")
            for dataset in datasets:
                print(f"  - {dataset}")
        else:
            print("No FiftyOne datasets available.")
        return

    if args.image_dir:
        if not os.path.isdir(args.image_dir):
            raise ValueError(f"Directory not found: {args.image_dir}")
        
        # Load images from directory
        viewer = FiftyOneDatasetViewer(image_dir=args.image_dir, 
                                       nickname=args.nickname,
                                       num_dims=args.num_dims,
                                       clustering_method=args.clustering_method)
        
    elif args.dataframe:
        # Load dataframe from file
        if not os.path.isfile(args.dataframe):
            raise ValueError(f"Dataframe file not found: {args.dataframe}")
        
        # Determine file type and load accordingly
        if args.dataframe.endswith('.csv'):
            df = pd.read_csv(args.dataframe)
        elif args.dataframe.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(args.dataframe)
        else:
            raise ValueError("Dataframe must be a CSV or Excel file")
            
        if args.image_path_column not in df.columns:
            raise ValueError(f"Image path column '{args.image_path_column}' not found in dataframe")
        
        # Load dataframe
        viewer = FiftyOneDatasetViewer(
            dataframe=df,
            image_path_column=args.image_path_column,
            feature_columns=args.feature_columns,
            nickname=args.nickname,
            clustering_method=args.clustering_method,
            num_dims=args.num_dims)
    else:
        if args.dataset_name not in fo.list_datasets():
            raise ValueError(f"Dataset not found: {args.dataset_name}")
        
        # Load existing dataset
        viewer = FiftyOneDatasetViewer(dataset_name=args.dataset_name, 
                                       nickname=args.nickname)

    viewer.visualize()


if __name__ == "__main__":
    main()
