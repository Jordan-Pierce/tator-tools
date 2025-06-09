# Model weights
Need to place the model weights in a top folder `Data`

# Rock Algorithm
Run the rock algorithm inference script to create Rock Mask localization polygons for a given image or video frame. It's best to run this script in the docker container.

```
docker build -t rock-masks-tator -f ./tator/docker/glbm_rocks.docker .

docker run --rm -ti -v /var/run/docker.sock:/var/run/docker.sock --gpus device=0 rock-masks-tator

python3 tator_rock_infer.py --host https://cloud.tator.io --token $CLOUD_TATOR_IO_TOKEN --media-id <> --frame <> --version-id <> --algorithm-config ../../Algorithms/Rocks/configs/rock_config_v1.4.0.yaml --tator-config ../../Algorithms/Rocks/configs/tator_config.yaml

python3 tator_rock_infer.py --host https://cloud.tator.io --token $CLOUD_TATOR_IO_TOKEN --media-id 17943896 --frame 2700 --version-id 546 --algorithm-config ../../Algorithms/Rocks/configs/rock_config_v1.4.0.yaml --tator-config ../../Algorithms/Rocks/configs/tator_config.yaml
```

## Model Server
Expected to be run using the model server framework here: `https://gitlab.com/bgwoodward/keras-model-server-fast-api`

### Docker file
```
docker build -t rock-masks-model-server -f ./tator/docker/glbm_rocks_model_server.docker .
```

# Coral Algorithm

```
docker build -t coral-detector-tator -f ./tator/Coral/docker/mdbc_coral.docker .

docker run --rm -ti -v /var/run/docker.sock:/var/run/docker.sock --gpus device=0 coral-detector-tator

python3 tator_corals_infer.py --host https://cloud.tator.io --token $CLOUD_TATOR_IO_TOKEN --media-id 18055191 --frame 204859 --version-id 410 --algorithm-config ../../Algorithms/Coral/configs/coral_config_v1.0.0.yaml --tator-config ../../Algorithms/Coral/configs/tator_config.yaml

python3 tator_corals_infer.py --host https://cloud.tator.io --token $CLOUD_TATOR_IO_TOKEN --media-id 18055191 --frame 204859 --version-id 410 --algorithm-config ../../Algorithms/Coral/configs/coral_config_v2.0.0.yaml --tator-config ../../Algorithms/Coral/configs/tator_config.yaml

```

## Model Server
```
docker build -t mdbc-coral-model-server -f ./tator/Coral/docker/mdbc_coral_model_server.docker .
```