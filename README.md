# tator-tools

A library for automating detection within benthic habitats (for finding rocks, coral, and other benthic features). This library revolves around Tator.

## Tator Algorithms

<details>
<summary>For production deployment in Tator</summary>

### Installation

```bash
# cmd

conda create --name tt python==3.10 -y
conda activate tt

pip install uv

uv pip install -r requirements.txt

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

## `tator_tools`

For local testing and debugging algorithms before deployment in Tator. Also useful for data visualization.

### Installation

```bash
# cmd

conda create --name tt python==3.10 -y
conda activate tt

pip install uv

uv pip install -e .

conda install cuda-nvcc -c nvidia/label/cuda-11.8.0 -y
conda install cudatoolkit=11.8 -c nvidia/label/cuda-11.8.0 -y

# Example for torch 2.0.0 and torchvision 0.15.1 w/ CUDA 11.8
uv pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install ffmpeg
```

### Classes

#### MediaDownloader

The `MediaDownloader` class is used to download, convert, and extract frames from videos in TATOR.

##### Example Usage

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

#### DatasetDownloader

The `DatasetDownloader` class is used to download frames / images and their labels from TATOR, which can be used to
create YOLO-formatted datasets. This class expects the encoded search string obtained from the Export Data utility 
offered in Tator's UI.

##### Example Usage

```python
from tator_tools.download_labeled_data import DatasetDownloader

# Initialize the downloader with the required parameters
downloader = DatasetDownloader(
    api_token="your_api_token",
    project_id=123,
    search_string="your_encoded_search_string",     # See Tator Metadata -> Export Data utility
    frac=1.0,                                       # Sample dataset, if applicable
    dataset_name="your_dataset_name",               # Output Directory Name
    output_dir="path/to/output",                    # Output Directory
    label_field="your_label_field",                 # "ScientificName", "Label", (or a list of fields)
)

# Download the data and create the dataset
downloader.download_data()

# View a sample
downloader.display_sample()

df = downloader.as_dataframe()  # as_dict()
```

#### YOLODataset

The `YOLODataset` class is used to create a YOLO-formatted dataset for object detection. It takes a pandas DataFrame 
with annotation data and generates the necessary directory structure, labels, and configuration files.

##### Example Usage

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
    task='detect'                                   # 'detect' or 'segment' (the latter needs polygons)
)

# Process the dataset to create the YOLO-formatted dataset
dataset.process_dataset(move_images=False)  # Makes a copy of the images instead of moving them
```

#### DetectionToClassifier

The `DetectionToClassifier` class is used to convert detection datasets into classification datasets by extracting crops from detection bounding boxes and organizing them into train/val/test splits by class.

##### Example Usage

```python
from tator_tools.detection_to_classification import DetectionToClassifier

# Initialize the converter with the path to the detection dataset's data.yaml file and the output directory
converter = DetectionToClassifier(dataset_path="path/to/detection/data.yaml", output_dir="path/to/output")

# Process the dataset to create classification crops
converter.process_dataset()
```

#### FiftyOneDatasetViewer

The `FiftyOneDatasetViewer` class is used to create a FiftyOne dataset from a directory of images and generate a UMAP 
visualization of the dataset. This can be run from command line or in a notebook.

##### Example Usage

```python
from tator_tools.fiftyone_clustering import FiftyOneDatasetViewer

# Initialize the viewer with the path to the directory containing images
viewer = FiftyOneDatasetViewer(image_dir="path/to/images")

# Process the dataset to create the FiftyOne dataset and generate the UMAP visualization
viewer.process_dataset()

```

#### ModelTrainer

The `ModelTrainer` class is used to train a model using a YOLO-formatted dataset.

##### Example Usage

```python
from tator_tools.model_training import ModelTrainer

# Initialize the trainer with the required parameters
trainer = ModelTrainer(
    training_data=f"{dataset.dataset_dir}\\data.yaml",
    weights="yolov8n.pt",                                       # See ultralytics website for models (8.3.0)
    output_dir=f"{dataset.dataset_dir}\\Training",
    task=dataset.task,
    epochs=10,
    half=True,
    imgsz=640,
    single_cls=True,
    plots=True,
    batch=0.5,
)

# Train the model
trainer.train_model()
trainer.evaluate_model()
```

#### VideoInferencer

The `VideoInferencer` class is used to perform inference on video files using a pre-trained model.

##### Example Usage

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
