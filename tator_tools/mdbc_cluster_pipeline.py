import os
import cv2
import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

class MDBCClusterPipeline:
    """
    A pipeline for clustering images or video frames based on object embeddings using YOLO and CLIP models.
    """

    def __init__(self, yolo_model_path="yolov8n.pt", clip_model_name="openai/clip-vit-base-patch32", device="cuda"):
        """
        Initializes the pipeline with YOLO and CLIP models.

        Args:
            yolo_model_path (str): Path to the YOLO model file.
            clip_model_name (str): Name of the CLIP model to use.
            device (str): Device to use for computation ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLO(yolo_model_path)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.eval()

    def get_clip_embeddings(self, image):
        """
        Computes CLIP embeddings for a given image.

        Args:
            image (PIL.Image.Image): Input image.

        Returns:
            torch.Tensor: Normalized CLIP embeddings.
        """
        inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Move inputs to the same device as the model
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
        return outputs / outputs.norm(p=2, dim=1, keepdim=True)

    def extract_objects_with_embeddings(self, frame):
        """
        Detects objects in a frame and computes embeddings for each detected object.

        Args:
            frame (numpy.ndarray): Input frame in BGR format.

        Returns:
            list: List of dictionaries containing embeddings and bounding boxes.
        """
        results = self.yolo_model(frame)
        objects = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cropped = frame[y1:y2, x1:x2]
            if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                emb = self.get_clip_embeddings(pil_image)
                objects.append({"embedding": emb, "bbox": (x1, y1, x2, y2)})
        return objects

    def process_input_image_or_video(self, input_path):
        """
        Processes the input image or video to extract the reference frame.

        Args:
            input_path (str): Path to the input image or video.

        Returns:
            numpy.ndarray: Reference frame.
        """
        if input_path.endswith((".mp4", ".avi", ".mov")):
            cap = cv2.VideoCapture(input_path)
            success, first_frame = cap.read()
            cap.release()
            if not success:
                raise ValueError(f"Unable to read the first frame from video: {input_path}")
            return first_frame
        else:
            return cv2.imread(input_path)

    def run_pipeline(self, input_path, parse_dir, similarity_threshold=0.7):
        """
        Runs the clustering pipeline.

        Args:
            input_path (str): Path to the initial input image or video.
            parse_dir (str): Directory containing images or videos to parse.
            similarity_threshold (float): Threshold for cosine similarity.

        Returns:
            list: List of similar objects with their bounding boxes and similarity scores.
        """
        query_frame = self.process_input_image_or_video(input_path)
        query_image = Image.fromarray(cv2.cvtColor(query_frame, cv2.COLOR_BGR2RGB))
        query_embedding = self.get_clip_embeddings(query_image)

        similar_objects = []
        for image_file in os.listdir(parse_dir):
            image_path = os.path.join(parse_dir, image_file)
            frame = cv2.imread(image_path)
            objects = self.extract_objects_with_embeddings(frame)
            for obj in objects:
                emb = obj["embedding"]
                bbox = obj["bbox"]
                similarity = cosine_similarity(query_embedding.cpu().numpy(), emb.cpu().numpy())[0][0]
                if similarity > similarity_threshold:
                    similar_objects.append({"image_file": image_file, "bbox": bbox, "similarity": similarity})

        return similar_objects

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MDBC Cluster Pipeline")
    parser.add_argument("--input_path", required=True, help="Path to the initial input image or video")
    parser.add_argument("--parse_dir", required=True, help="Directory containing images or videos to parse")
    parser.add_argument("--similarity_threshold", type=float, default=0.7, help="Similarity threshold for clustering")
    args = parser.parse_args()

    pipeline = MDBCClusterPipeline()
    results = pipeline.run_pipeline(args.input_path, args.parse_dir, args.similarity_threshold)

    for result in results:
        print(f"Match found in {result['image_file']} with similarity {result['similarity']:.2f}")
        print(f"Bounding box: {result['bbox']}")
