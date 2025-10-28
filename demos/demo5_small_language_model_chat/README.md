# Demo 5: Small Language Model (SLM) Chat

This demo showcases how to run and interact with Small Language Models (SLMs) on edge devices using Ollama. Lightweight AI models locally via CLI or through a web-based chat interface using Open WebUI. These compact models enable privacy-focused, low-latency inference without requiring cloud connectivity.

<details>
<summary><strong>Ollama CLI</strong></summary>
        
- Run a model in CLI
    
    Download any model from the Ollama registry - [Link ↗️](https://ollama.com/search)
    
    - To get print verbose output with stats
        
        ```bash
        ollama run qwen3:1.7b --verbose
        ```

</details>

<details>
<summary><strong>Open WebUI (GUI)</strong></summary>
    
- Launch Ollama docker container and keep it running in background in detach mode

    ```bash
    docker run -d -it --runtime nvidia --network=host -v ~/ollama:/ollama -e OLLAMA_MODELS=/ollama --name ollama-server dustynv/ollama:r36.4.0
    ```
        
- Download and launch Open WebUI container
    
    ```bash
    docker run -d --network=host     -v ${HOME}/open-webui:/app/backend/data     -e OLLAMA_BASE_URL=http://127.0.0.1:11434     --name open-webui     --restart always     ghcr.io/open-webui/open-webui:main
    ```

- Access Open WebUI in your browser at `0.0.0.0:8080`

<details>
<summary><strong>Example Prompts</strong></summary>
    
```python
Hello?
```

```python
What is jetson orin nano?
```

```python
Tell me a random space fact in under 10 words.
```

```python
List 5 real-world edge AI use cases for Jetson
```

```python
Write a short poem about AI at the edge.
```

```python
Give me three possible use cases for Jetson in agriculture.
```

- Reasoning
    
    ```python
    What is the probability of getting two heads in three coin flips?
    ```
    
    ```python
    Solve this: A GPU processes 480 frames in 20 seconds. What is the FPS?
    ```
    
    ```python
    Find the derivative of x² + 3x + 5.
    ```
    
- Context
    
    ```python
    Summarize Jetson Orin Nano in one line.
    ```
    
    ```python
    Now make it sound like a movie trailer.
    ```
    
    ```python
    Convert that into a tweet under 200 characters.”
    ```
</details>
</details>  
    

<details>
<summary><strong>Troubleshooting</strong></summary>

- Open WebUI login problem
    
    ```python
    rm -rf ~/open-webui/*
    ```
</details>