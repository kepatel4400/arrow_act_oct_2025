# Demo 3: Multimodal Similarity Search with Vector Database - NanoDB

NanoDB is a CUDA-optimized multimodal vector database that uses embeddings from the CLIP vision transformer for txt2img and img2img similarity search. It's running in realtime on 275K images from the MS COCO image captioning dataset, and shows a fundamental capability in multimodal applications - operating in a shared embedding space between text/images/etc., and being able to query with a deep contextual understanding.

<p align="center">
  <a href="https://youtu.be/ayqKpQNd1Jw" target="_blank">
    <img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/nanodb_horse.gif" height="70%" width="70%" alt="Demo Video">
  </a>
</p>

<details>
<summary>Resources</summary>

- Source code - https://github.com/dusty-nv/NanoDB
- Container - https://github.com/dusty-nv/jetson-containers/tree/master/packages/vectordb/nanodb

</details>

## Step 1: Data Preparation
    
- Download COCO 2017 dataset
    
    ```bash
    cd ~/jetson-containers
    mkdir -p data/datasets/coco/2017
    cd data/datasets/coco/2017
    
    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/zips/unlabeled2017.zip
    
    unzip train2017.zip
    unzip val2017.zip
    unzip unlabeled2017.zip
    ```

## Step 2: Download NanoDB Index

- Download NanoDB index
    
    ```bash
    cd ~/jetson-containers/data
    wget https://nvidia.box.com/shared/static/icw8qhgioyj4qsk832r4nj2p9olsxoci.gz -O nanodb_coco_2017.tar.gz
    tar -xzvf nanodb_coco_2017.tar.gz
    ```

## Step 3: Launch NanoDB Container

- The below command will download (if not already present) and launch the NanoDB container

    ```bash
    jetson-containers run dustynv/nanodb:r36.4.0
    ```

## Step 4: Setup NanoDB Web Server

- Inside the container terminal, install utility packages
    
    ```bash
    apt update && apt install featherpad -y
    ```
    
- Run NanoDB web server
    
    - Change directory
        
        ```bash
        cd /opt/NanoDB/nanodb
        ```

    - Replace the `server.py` script with the updated version available [HERE](./server.py)
        - To edit the file, run
            ```python
            featherpad server.py
            ```
    
    - Start the server

        ```bash
        cd /
        ```
        
        ```bash
        python3 -m nanodb \
            --path /data/nanodb/coco/2017 \
            --server --port=7860
        ```
    
    - Open a web browser for the GUI and enter the URL
        
        ```bash
        0.0.0.0:7860
        ```
        

- Sample queries
    
    ```bash
    a person riding a motorcycle
    ```
    
    ```bash
    a person riding a bike
    ```
    
    ```bash
    a person riding a horse
    ```
    
    ```bash
    a person riding a horse jumping
    ```
    
    ```bash
    ocean underwater
    ```
    
    ```bash
    ocean surfing
    ```
    
    ```bash
    A lake by a mountain with people swimming
    ```
    
    ```bash
    orange and apples
    ```
    
    ```bash
    apple
    ```
    
    ```bash
    apple fruit
    ```
