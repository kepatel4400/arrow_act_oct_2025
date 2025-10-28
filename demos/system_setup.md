# System Setup

<details>
<summary><strong>Checkpoints 🏁</strong></summary>

- Run `jtop` and verify the jetson software stack
    - JetPack 6.2.1
        - With Jetson Linux (L4T) R36.4.4
            - Linux Kernel 5.15
            - Ubuntu 22.04-based rootfs
            - NVIDIA drivers
        - Jetson AI Stack
            - CUDA 12.6
            - TensorRT 10.3
            - cuDNN 9.3
- Verify `SUPER` model
    - Run the below command, it should print the board name with `super`
        
        ```csharp
        cat /sys/firmware/devicetree/base/model
        ```
        
    - Check power mode
        
        ```csharp
        sudo nvpmodel -q
        ```
        
        - 1: 25W (7W and 15W were the old power modes)
        - 2: Uncapped MAXN power mode
    - Set power model to MAXN_SUPER
        
        ```bash
        sudo nvpmodel -m 2
        ```
        
    - Check CPU, EMC frequency
        - CPU: 1.5 GHz (old) → 1.7 GHz (super mode)
        - GPU: 625 MHz  (old) → 1020 MHz  (super mode)
        
        ```csharp
        sudo jetson_clocks --show
        ```
        
- Check NVMe SSD
    
    ```csharp
    lspci
    ```
    
    - Check that the root partition is on SSD
        - Verify that root partition is on the NVMe disk
            
            ```csharp
            lsblk
            ```
            
            - Output
            
            ```bash
            ...
            nvme0n1      259:0    0 931.5G  0 disk 
            ├─nvme0n1p1  259:1    0 930.1G  0 part /
            ....
            ....
            ```
</details>            

<details>
<summary><strong>Docker Setup</strong></summary>

- Check whether docker is installed
    
    ```csharp
    sudo docker
    ```
    
    <details>
    <summary>(If not) Install Docker</summary>
    
    - Install Docker
        
        ```csharp
        sudo apt update
        sudo apt install -y nvidia-container curl
        curl https://get.docker.com | sh && sudo systemctl --now enable docker
        sudo nvidia-ctk runtime configure --runtime=docker
        ```
            
- To run docker commands without sudo
    
    Add user to the docker group (to stop getting sudo permission error)
    
    - First check whether docker already exist in the user group
        
        ```bash
        groups
        ```
        
        - Look for `docker` in the output
    1. Create the `docker` group (if it doesn’t already exist):
        
        ```bash
        sudo groupadd docker
        ```
        
    2. Add your user to the `docker` group:
        
        ```bash
        sudo usermod -aG docker $USER
        ```
        
        - usermod = modifying the user account
        - aG = ‘a’ append the user to the group specified by ‘G’, ‘G’ groups that user should be a member of
    3. Apply the new group membership:
        
        You can either log out and log back in, or run:
        
        ```bash
        newgrp docker
        ```
        
    4. Test it:
        
        Try:
        
        ```bash
        docker run hello-world
        ```
        
        - It should work without `sudo`.
- Add default runtime in `/etc/docker/daemon.json`
    - Why
        
        By default, Docker containers don’t know about the GPU. NVIDIA provides the nvidia-container-runtime that plugs into Docker and makes CUDA, TensorRT, cuDNN, etc. visible inside the container.
        
    - Open this file in a text editor
        
        ```css
        sudo vi /etc/docker/daemon.json
        ```
        
    - Insert the `"default-runtime": "nvidia"` line as following:
        
        ```css
        {
            "runtimes": {
                "nvidia": {
                    "path": "nvidia-container-runtime",
                    "runtimeArgs": []
                }
            },
            "default-runtime": "nvidia"
        }
        ```
        
- Restart docker
    
    ```css
    sudo systemctl daemon-reload && sudo systemctl restart docker
    ```
</details> 

<details>
<summary><strong>Download Docker Images</strong></summary>

1. NanoOWL
    
    ```bash
    docker pull dustynv/nanoowl:r36.4.0
    ```
    
2. NanoDB
    
    ```bash
    docker pull dustynv/nanodb:r36.4.0
    ```
    
3. Ultralytics
    
    ```bash
    docker pull ultralytics/ultralytics:latest-jetson-jetpack6
    ```
    
4. Ollama
    
    ```bash
    docker pull dustynv/ollama:r36.4.0
    ```
    
- Or all in one command
    ```bash
    $docker pull dustynv/nanoowl:r36.4.0; docker pull dustynv/nanodb:r36.4.0; docker pull ultralytics/ultralytics:latest-jetson-jetpack6; docker pull dustynv/ollama:r36.4.0
    ```

</details>

<details>
<summary><strong>Download `jetson-containers`</strong></summary>

- Clone jetson-containers repo
    
    ```bash
    git clone https://github.com/dusty-nv/jetson-containers
    ```
    
- Create a virtual environment
    
    ```bash
    apt install python3.10-venv
    ```
    
    ```bash
    python3 -m venv jetson_venv
    ```
    
    ```bash
    source jetson_venv/bin/activate
    ```
    
- Install
    
    ```bash
    bash jetson-containers/install.sh
    ```

</details>

<details>
<summary><strong>Install Chromium Browser</strong></summary>

- If already installed and not working, remove it
    - Remove Chromium
        - From APT
            
            ```bash
            sudo apt remove --purge chromium-browser chromium
            ```
            
        - From snap
            
            ```bash
            sudo snap remove chromium
            ```
            
- Install snap
    
    ```bash
    sudo apt install snapd && \
    snap download snapd --revision=24724 && \
    sudo snap ack snapd_24724.assert && \
    sudo snap install snapd_24724.snap && \
    sudo snap refresh --hold snapd
    ```
    
- Install Chromium
    
    ```bash
    sudo snap install chromium
    ```
    
- (optional) All in one
    
    ```bash
    sudo apt install snapd && \
    snap download snapd --revision=24724 && \
    sudo snap ack snapd_24724.assert && \
    sudo snap install snapd_24724.snap && \
    sudo snap refresh --hold snapd && \
    sudo snap install chromium
    ```

</details>