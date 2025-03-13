# 🥔 tator-tools 🛠️

A library for automating detection within benthic habitats (for finding rocks, coral, and other benthic features). This library revolves around Tator.

## Tator Algorithms

<details>
<summary>For production deployment in Tator (Mark T.)</summary>

### Installation

```bash
# cmd

conda create --name tt python==3.10 -y
conda activate tt

pip install uv

uv pip install -r requirements.txt

# Install cuda nvcc, toolkit for your version (example: 11.8)
conda install cuda-nvcc -c nvidia/label/cuda-11.8.0 -y
conda install cudatoolkit=11.8 -c nvidia/label/cuda-11.8.0 -y

# Example for torch 2.0.0 and torchvision 0.15.1 w/ CUDA 11.8
uv pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

Test out the algorithms using the `app.py` script (`gradio`):

```bash
# cmd

python Algorithms/app.py
```

</details>

# `tator_tools`

For local testing and debugging algorithms before deployment in Tator. Also useful for data visualization.

### Installation

```bash
# cmd

conda create --name tt python==3.10 -y
conda activate tt

pip install uv

uv pip install -e .

# Install cuda nvcc, toolkit for your version (example: 11.8)
conda install cuda-nvcc -c nvidia/label/cuda-11.8.0 -y
conda install cudatoolkit=11.8 -c nvidia/label/cuda-11.8.0 -y

# Example for torch 2.0.0 and torchvision 0.15.1 w/ CUDA 11.8
uv pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# ffmpeg for converting videos, extracting frames
conda install ffmpeg

# FiftyOne Plugins
fiftyone plugins download https://github.com/jacobmarks/clustering-plugin
```

### Classes

<details>
<summary>MediaDownloader</summary>

The `MediaDownloader` class is used to download, convert, and extract frames from videos in TATOR.

```python
from tator_tools.download_media import MediaDownloader

# Initialize the downloader with the required parameters
downloader = MediaDownloader(
    api_token=os.getenv("TATOR_TOKEN"),
    project_id=123,
    output_dir="path/to/output"
)

# Download the media
media_ids = ["123456", "78910"]
downloader.download_data(media_ids, convert=False, extract=True, every_n_seconds=1.0)
```
</details>

<details>
<summary>QueryDownloader</summary>

The `QueryDownloader` class is used to download frames / images and their labels from TATOR, which can later be used to
create YOLO-formatted datasets. This class expects the encoded search string obtained from the Export Data utility 
offered in Tator's UI.

```python
from tator_tools.download_query_data import QueryDownloader

# Initialize the downloader with the required parameters
downloader = QueryDownloader(
    api_token="your_api_token",
    project_id=123,
    search_string="your_encoded_search_string",     # See Tator Metadata -> Export Data utility
    frac=1.0,                                       # Sample dataset, if applicable
    dataset_name="your_dataset_name",               # Output Directory Name
    output_dir="path/to/output",                    # Output Directory
    label_field="your_label_field",                 # "ScientificName", "Label", (or a list of fields)
    download_width=1024,                            # Width of downloaded image (maintains aspect ratio)
)

# Download the data and create the dataset
downloader.download_data()

# View a sample
downloader.display_sample()

df = downloader.as_dataframe()  # as_dict()
```
</details>

<details>
<summary>YOLODataset</summary>

The `YOLODataset` class is used to create a YOLO-formatted dataset for object detection. It takes a pandas DataFrame 
with annotation data and generates the necessary directory structure, labels, and configuration files.

```python
import pandas as pd
from tator_tools.yolo_dataset import YOLODataset

# Load your annotation data into a pandas DataFrame
df = pd.read_csv("path/to/annotations.csv")

# Initialize the YOLODataset with the DataFrame and the output directory
dataset = YOLODataset(
    data=df,
    output_dir="path/to/output",                    # Output Directoy
    dataset_name="YOLODataset_Detection",           # Output Directoy /Dataset Name -> train/valid/test, data.yaml 
    train_ratio=0.8                                 # Training ratio -> train / valid
    test_ratio=0.1,                                 # Testing ratio -> (train / valid) / test
    task='detect'                                   # 'classify', 'detect' or 'segment'
)

# Process the dataset to create the YOLO-formatted dataset
dataset.process_dataset(move_images=False)  # Makes a copy of the images instead of moving them
```
</details>

<details>
<summary>YOLORegionCropper</summary>

The `YOLORegionCropper` class is used to convert detection datasets into classification datasets by extracting crops from detection bounding boxes and organizing them into train/val/test splits by class.

```python
from tator_tools.yolo_crop_regions import YOLORegionCropper

# Initialize the converter with the path to the detection / segmentation dataset's data.yaml file and the 
# desired output directory. The class will create a YOLO-formatted image classification dataset.
cropper = YOLORegionCropper(dataset_path="path/to/detection/data.yaml", 
                            output_dir="path/to/output",
                            dataset_name="Cropped_Dataset")

# Process the dataset to create classification crops
cropper.process_dataset()
```
</details>

<details>
<summary>FiftyOneDatasetViewer</summary>

The `FiftyOneDatasetViewer` class is used to create a FiftyOne dataset from a directory of images and generate a UMAP 
visualization of the dataset. This can be run from command line or in a notebook.

```python
from tator_tools.fiftyone_clustering import FiftyOneDatasetViewer

# Initialize the viewer with the path to the directory containing images
viewer = FiftyOneDatasetViewer(image_dir="path/to/images")

# Or, initialize the viewer with a pandas dataframe
viewer = FiftyOneDatasetViewer(dataframe=pandas_df,
                               image_path_column='Path',
                               feature_columns=['feature 1', 'feature 2'],
                               nickname='my_dataset',
                               custom_embeddings=embeddings,  # Pass the pre-calculated embeddings, or None
                               clustering_method='umap',      # umap, pca, tsne
                               num_dims=2)                    # Number of dimensions for UMAP (2 or 3)

# Process the dataset to create the FiftyOne dataset and generate the UMAP visualization
viewer.process_dataset()
```
</details>

<details>
<summary>ModelTrainer</summary>

The `ModelTrainer` class is used to train a model using a YOLO-formatted dataset.

```python
from tator_tools.model_training import ModelTrainer

# Initialize the trainer with the required parameters
trainer = ModelTrainer(
    training_data="path/to/training_data/",                    # Can be a classification dataset, or data.yaml
    weights="yolov8n.pt",                                      # Model to start with, see ultralytics docs
    output_dir="path/to/output_dir",
    name="results",
    task='classify',
    epochs=100,
    patience=10,
    half=True,
    imgsz=640,
    single_cls=False,
    plots=True,
    batch=0.5,
)

# Train the model
trainer.train_model()
trainer.evaluate_model()
```
</details>

<details>
<summary>VideoInferencer</summary>

The `VideoInferencer` class is used to perform inference on video files using a pre-trained model.

```python
from tator_tools.inference_video import VideoInferencer

# Initialize the inferencer with the required parameters
inferencer = VideoInferencer(
    model_path="path/to/model.pt",
    video_path="path/to/video.mp4",
    output_dir="path/to/output"
)

# Perform inference on the video
inferencer.inference()
```
</details>

<details>
<summary>VideoStabilizer</summary>

The `VideoStabilizer` class is used to create a stabilized version of the input video.

```python
from tator_tools.stabilize_video import VideoStabilizer

# Initialize the stabilizer with the required parameters
stabilizer = VideoStabilizer(
    input_path="path/to/video.mp4",
    output_path="path/to/output.mp4"
)

# Perform stabilization
stabilizer.stabilize_video()
```
</details>
