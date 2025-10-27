# Demo 1: Zero-Shot Object Detection - NanoOWL

NanoOWL is a project that optimizes OWL-ViT to run real-time on NVIDIA Jetson Orin Platforms with NVIDIA TensorRT. NanoOWL also introduces a new "tree detection" pipeline that combines OWL-ViT and CLIP to enable nested detection and classification of anything, at any level, simply by providing text.

- **Resources**
    - Source code -  https://github.com/NVIDIA-AI-IOT/nanoowl
    - Container - https://github.com/dusty-nv/jetson-containers/tree/master/packages/vit/nanoowl
    - Docker Images - https://hub.docker.com/r/dustynv/nanoowl/tags

### Step 1
- **Download container**
    - Make sure to have jetson-container available
    
    ```bash
    jetson-containers run dustynv/nanoowl:r36.4.0
    ```

