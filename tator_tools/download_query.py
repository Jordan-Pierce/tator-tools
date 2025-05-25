import os
import json
import random
import shutil
import argparse
import traceback

from tqdm import tqdm
import concurrent.futures

import tator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import supervision as sv


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class QueryDownloader:
    def __init__(self,
                 api_token: str,
                 project_id: int,
                 search_string: str,
                 frac: float,
                 dataset_name: str,
                 output_dir: str,
                 label_field: str or list,
                 download_width: int = None):
        """

        :param api_token:
        :param project_id:
        :param media_id:
        :param search_string:
        :param frac:
        :param dataset_name:
        :param output_dir:
        :param label_field: Field name(s) to use as the label(s), can be a single field or a list
        :param download_width: Width to resize images to during download (maintains aspect ratio), else full resolution
        """
        self.api = None

        self.token = api_token
        self.project_id = project_id
        self.search_string = search_string
        self.frac = frac
        self.download_width = download_width
        
        # Support both single field and list of fields
        if isinstance(label_field, list):
            self.label_field = label_field
        else:
            self.label_field = [label_field]

        self.dataset_name = dataset_name

        # Convert paths to absolute paths
        self.output_dir = os.path.abspath(output_dir)
        self.dataset_dir = os.path.abspath(f"{self.output_dir}/{self.dataset_name}")
        self.image_dir = os.path.abspath(f"{self.dataset_dir}/images")

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

        self.query = None
        self.data = None

        self.authenticate()
        self.save_query_string()

    def authenticate(self):
        """

        :return:
        """
        try:
            # Authenticate with TATOR
            self.api = tator.get_api(host='https://cloud.tator.io', token=self.token)
            print(f"NOTE: Authentication successful for {self.api.whoami().username}")

        except Exception as e:
            raise Exception(f"ERROR: Could not authenticate with provided API Token\n{e}")

    def save_query_string(self):
        """
        Write the search string to text file

        :return:
        """
        try:
            # Write search string to file for the posterity
            if self.search_string:
                query_string_path = os.path.abspath(f"{self.dataset_dir}/search_string.txt")
                with open(query_string_path, 'w') as file:
                    file.write(self.search_string)
                    
            print(f"NOTE: Search string saved to {query_string_path}")

        except Exception as e:
            print(f"WARNING: An error occurred while writing search string to file:\n{e}")

    def query_tator(self):
        """
        Query TATOR for the desired data

        :return:
        """
        try:
            # Query TATOR for the desired data
            print("NOTE: Querying Tator for labeled data")
            self.query = self.api.get_localization_list(project=self.project_id, 
                                                        encoded_search=self.search_string)
            
            if not self.query:
                print("WARNING: No localization data found for search string; downloading media only")
                self.query = self.api.get_media_list(project=self.project_id, 
                                                     encoded_search=self.search_string)
            if not self.query:
                raise Exception("ERROR: No data found for search string")

        except Exception as e:
            raise Exception(f"ERROR: Could not query TATOR for data\n{e}")
    
    def process_query(self):
        """
        Process the query by restructuring the data into a DataFrame;
        each row contains a query object, and the media it belongs to.
        Then group by image_path and randomly sample.

        :return: None
        """
        print(f"NOTE: Found {len(self.query)} objects in query")
        
        data = []
        
        # Loop through the queries (Class - tator localization)
        for q in tqdm(self.query, desc="Processing query"):
            # Convert query object to dictionary
            q_type = q.__class__.__name__
            q_dict = q.to_dict()
        
            try:        
                # Handle labels differently based on whether we have one field or multiple
                if len(self.label_field) == 1:
                    # Single field - store as string
                    label = q_dict['attributes'].get(self.label_field[0], None)
                else:
                    # Multiple fields - store as nested dictionary
                    label = {}
                    for field in self.label_field:
                        label[field] = q_dict['attributes'].get(field, None)
                        
            except Exception as e:
                label = None

            # Determine if the localization is a bounding box or a polygon
            if q_dict.get("points", None):
                # Get the polygon from the query
                polygon = q_dict['points']
                # Get the bounding box from the polygon
                x, y, xmax, ymax = sv.polygon_to_xyxy(polygon)
                width, height = xmax - x, ymax - y
            elif q_dict.get('x') and q_dict.get('y') and q_dict.get('width') and q_dict.get('height'):
                # No polygon, set to None
                polygon = []
                # Get the bounding box from the query
                x, y, width, height = q_dict['x'], q_dict['y'], q_dict['width'], q_dict['height']
            else:    
                # No polygon or bounding box, set to None
                polygon = []
                x, y, width, height = None, None, None, None
                
            # Determine if q is a media or localization
            if q_type == "Media":
                # Media object
                media = q_dict['id']
                frame = 0
                image_name = q_dict['name']
                image_width = q_dict['width']
                image_height = q_dict['height']
            elif q_type == "Localization":
                # Localization object
                media = q_dict['media']
                frame = q_dict['frame']
                image_name = f"{media}_{frame}.jpg"
                image_width = None
                image_height = None
            else:
                print(f"WARNING: Query object is not a media or localization\nType:{type(q_type)}")
                continue

            # Create a dictionary for each row with absolute paths
            row_dict = {
                'media': media,
                'frame': frame,
                'image_name': image_name,
                'image_path': os.path.abspath(f"{self.image_dir}/{image_name}"),
                'image_width': image_width,
                'image_height': image_height,
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'polygon': polygon
            }
            # Unnest label dict if label_field is a list
            if isinstance(label, dict):
                row_dict.update(label)
            else:
                row_dict['label'] = label
             
            # Add the dictionary to the list
            data.append(row_dict)

        # Create DataFrame from the list of dictionaries
        self.data = pd.DataFrame(data)

        if self.frac < 1 and not self.data.empty:
            # Group by image_path
            image_paths = self.data['image_path'].unique().tolist()
            sampled_images = random.sample(image_paths, int(len(image_paths) * self.frac))
            sampled_data = self.data[self.data['image_path'].isin(sampled_images)]

            # Reset index after sampling
            self.data = sampled_data.reset_index(drop=True)
        
        print(f"NOTE: Found {len(self.data)} query objects after sampling")
        
        # Save the data to disk
        self.to_csv(f"{self.dataset_dir}/data.csv")
        self.to_json(f"{self.dataset_dir}/data.json")
                    
    def as_dict(self):
        """
        Convert the data to a dictionary for saving to disk.

        :return: Dictionary of the data
        """
        # Convert data to dictionary
        return self.data.to_dict('records')
    
    def as_dataframe(self):
        """
        Convert the data to a pandas DataFrame.

        :return: DataFrame of the data
        """
        return self.data
    
    def to_csv(self, output_path="data.csv"):
        """
        Convert the data to a CSV file and save to disk.

        :param output_path: Optional file path to save CSV to disk
        :return: DataFrame of the data
        """       
        # If output path is provided, save to disk
        if output_path:
            output_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.data.to_csv(output_path, index=False)
            print(f"NOTE: Data saved to {output_path}")
    
    def to_json(self, output_path="data.json"):
        """
        Convert the data to a JSON string and optionally save to disk.

        :param output_path: Optional file path to save JSON to disk
        :return: JSON string of the data
        """
        # Convert dictionary to JSON string
        json_data = json.dumps(self.as_dict(), indent=2)
        
        # If output path is provided, save to disk
        if output_path:
            output_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(json_data)
            print(f"NOTE: Data saved to {output_path}")
            
        return json_data

    def download_images(self, max_workers=os.cpu_count() // 3, max_retries=3):
        """
        Multithreading function for downloading multiple images in parallel.
        
        :param max_workers: Maximum number of concurrent downloads
        :param max_retries: Maximum number of retry attempts per download
        """
        if self.data.empty:
            print("ERROR: No data to download images for")
            return
        
        print(f"NOTE: Downloading images to {self.image_dir}")
        
        # Get unique combinations of media ID and frame number
        media_images = self.data[['media', 'frame', 'image_path']].drop_duplicates().to_dict('records')
        
        # Use ThreadPoolExecutor for parallel downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download tasks for each unique frame
            futures = [executor.submit(self.download_image, item, max_retries) for item in media_images]
            
            # Process results as they complete with a progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading images"):
                try:
                    future.result()  # This will raise any exceptions that occurred during execution
                except Exception as e:
                    print(f"Failed download task: {str(e)}")
                    
        print(f"NOTE: Images downloaded to {self.image_dir}")

    def download_image(self, item, max_retries=3):
        """
        Downloads a frame and saves it to disk with retry logic.
        
        :param item: Dictionary with media, frame, and image_path keys
        :param max_retries: Maximum number of retry attempts
        :return: Path to saved image or None if failed
        """
        # Skip download if the file already exists
        if os.path.exists(item['image_path']):
            return item['image_path']
            
        for attempt in range(max_retries):
            try:
                # Get image dimensions from first matching row
                frame_data = self.data[(self.data['media'] == item['media']) & 
                                       (self.data['frame'] == item['frame'])].iloc[0]
                
                # If user wants to resize the image, calculate the new dimensions
                if self.download_width:
                    # If provided with dimensions, use them to calculate the new dimensions
                    if frame_data['image_width'] and frame_data['image_height']:
                        # Use the existing dimensions
                        original_width = frame_data['image_width']
                        original_height = frame_data['image_height']
                        # Calculate height to maintain aspect ratio
                        width = self.download_width
                        height = int(original_height * (self.download_width / original_width))
                    else:
                        # Else, calculate the new dimensions using default aspect ratio (1.33)
                        width = self.download_width
                        height = int(self.download_width / 1.33)
                else:
                    width = None
                    height = None
                    
                if width and height:
                    force_scale = f"{width}x{height}"
                else:
                    force_scale = None
                
                # Get the image from Tator at the specified resolution
                temp = self.api.get_frame(id=item['media'],
                                          force_scale=force_scale,
                                          frames=[int(item['frame'])])

                # Move the image to the correct directory
                shutil.move(temp, item['image_path'])
                return item['image_path']
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Retrying download for media {item['media']}, frame {item['frame']} again...")
                else:
                    print(f"Error downloading {item['image_path']}: {str(e)}")
                    
    def calculate_scale(self, width, height):
        """Calculate the scale factor to maintain aspect ratio."""
        if self.download_width:
            # Calculate height to maintain aspect ratio
            height = int(height * (self.download_width / width))
            return self.download_width, height
        else:
            return width, height
        
    def download_data(self):
        """
        Download and process labels, creating a YOLO-format dataset.
        """
        # Query TATOR given the search string
        self.query_tator()

        # Extract data from query
        self.process_query()

        # Download all the images
        self.download_images()
        
    def display_sample(self, label_column='label'):
        """
        Display a sample of the data with images and labels overlaid.
        """
        # Get a sample of the data
        if self.data.empty:
            print("No data available to display sample")
            return
            
        image_path = self.data.sample(1)['image_path'].values[0]
        sample = self.data[self.data['image_path'] == image_path]
        
        # Load the image
        image_ = plt.imread(image_path)
        image_height, image_width = image_.shape[:2]
        
        # Create a single figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image_)
        
        # Create a color map for unique labels
        labels = sample[label_column].unique().tolist()
        
        if len(labels):
            color_map = plt.cm.get_cmap('hsv')(np.linspace(0, 1, len(labels)))
            label_to_color = {label: color_map[i][:3] for i, label in enumerate(labels)}

        # Loop through the sample
        for i, row in sample.iterrows():
            # Get the bounding box coordinates and polygon
            x, y, w, h = row['x'], row['y'], row['width'], row['height']
            polygon = row['polygon']
            label = row[label_column]
            color = label_to_color[label]
            
            try:
                # Draw polygon if it exists
                if polygon is not None and len(polygon) > 0:
                    # Denormalize polygon coordinates
                    x_coords = [p[0] * image_width for p in polygon]
                    y_coords = [p[1] * image_height for p in polygon]
                    plt.fill(x_coords, y_coords, alpha=0.3, color=color)
                    plt.plot(x_coords, y_coords, color=color, linewidth=2)
            except Exception as e:
                print(f"Error displaying polygon: {str(e)}")
                
            try:
                # Draw bounding box if coordinates exist
                if all(coord is not None for coord in [x, y, w, h]):
                    # Denormalize bounding box coordinates
                    x, y, w, h = x * image_width, y * image_height, w * image_width, h * image_height
                    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    
            except Exception as e:
                print(f"Error displaying bounding box: {str(e)}")
                
        # Display the image with the name from the first row
        if not sample.empty:
            plt.title(f"Image: {os.path.basename(image_path)}, Labels: {labels}")
            
        plt.axis('off')  # Hide axes for cleaner display
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download queried images and labels from TATOR")

    parser.add_argument("--api_token", type=str, required=True,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--project_id", type=int, required=True, default=70,
                        help="Project ID for desired media")

    parser.add_argument("--search_string", type=str, required=True,
                        help="Search string for localizations")

    parser.add_argument("--frac", type=float, default=1.0,
                        help="Sub-sample the amount of images being downloaded")

    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to download data to")

    parser.add_argument("--label_field", type=str, required=True, nargs='+',
                        help="Field name(s) to use as the label; can be single field or a list of fields")
    
    parser.add_argument("--download_width", type=int, default=None,
                        help="Width to resize images to when downloading (maintains aspect ratio)")
    
    args = parser.parse_args()

    try:
        # Convert output_dir to absolute path
        output_dir = os.path.abspath(args.output_dir)
        
        # Handle label_field as either single field or list
        label_field = args.label_field
        if len(label_field) == 1:
            label_field = label_field[0]  # If only one field provided, convert from list to string
        
        # Download the data
        downloader = QueryDownloader(api_token=args.api_token,
                                     project_id=args.project_id,
                                     search_string=args.search_string,
                                     frac=args.frac,
                                     dataset_name=args.dataset_name,
                                     output_dir=output_dir,
                                     label_field=label_field,
                                     download_width=args.download_width)

        # Download the data
        downloader.download_data()

        print("Done.")

    except Exception as e:
        print(f"ERROR: Could not finish process.\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
