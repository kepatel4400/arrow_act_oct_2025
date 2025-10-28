# Demo 2: Visual Question Answering (VQA)

This demo shows how to run Visual Question Answering (VQA) using Ollama's VLM models (e.g., qwen2.5vl:3b) on Jetson devices.

<details>
<summary>Resources</summary>

- Ollama Documentation - https://ollama.com/docs
- Ollama Model Library - https://ollama.com/search
- Qwen Model: [https://ollama.com/library/qwen2.5vl](https://ollama.com/library/qwen2.5vl)

</details>

## Step 1: Setup Ollama

- Download Ollama CLI
    
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```
    
    - Check ollama usage
        
        ```bash
        ollama
        ```
        
- Download Ollama python package
        
    ```bash
    pip install ollama pillow
    ```

## Step 2: Download VLM Models

- Download VLM models using Ollama CLI
    - Download models (list of available models: https://ollama.com/search)
        ```bash
        ollama pull qwen2.5vl:3b
        ```
        
        ```bash
        ollama pull gemma3:4b
        ```
    - Check model details
        ```python
        ollama show qwen2.5vl:3b
        ```
        ```python
        ollama show gemma3:4b
        ```

## Step 3: Run Visual Question Answering (VQA)    

- Change directory
    
    ```bash
    cd ~/arrow_act_oct_2025/demos/demo2_visual_question_answering
    ``` 
        
- VQA reasoning (example 1)

    ```bash
    python vqa.py --image ../../assets/images/golf.png --query "which sport is this?"
    ```
    - Where
        - `--image`: path to input image
        - `--query`: question related to the image
        - `--long-side`: (optional) resize the long side of the image to the specified size (default: 512)

- VQA reasoning with smaller image size
    ```bash
    python vqa.py --image ../../assets/images/golf.png --query "which sport is this?" --long-side 224
    ```

- VQA reasoning (example 2)

    ```bash
    python vqa.py --image ../../assets/images/arrow_building.jpg --query "what is the name of the company logo? and its address?"
    ```
    
    ```bash
    python vqa.py --image ../../assets/images/arrow_building.jpg --query "what is the name of the company logo? and its address?" --long-side 224
    ```

- VQA reasoning (example 3)

    ```bash
    python vqa.py --image ../../assets/images/two_robots.png --query "what are these robots doing?"
    ```
    
    ```bash
    python vqa.py --image ../../assets/images/two_robots.png --query "what are these robots doing?" --long-side 224
    ```

- OCR

    ```bash
    python vqa.py --image ../../assets/images/groot_paper_authors.png --query "who is the product lead?"
    ```
    
    ```python
    python vqa.py --image ../../assets/images/groot_paper_authors.png --query "what is the name of this research paper?"
    ```

- VQA counting

    ```bash
    python vqa.py --image ../../assets/images/workers.png --query "how many workers are there?"
    ```

- VQA counting and reasoning

    ```bash
    python vqa.py --image ../../assets/images/workers_2.png --query "how many workers are there? and which color cap they are wearing"
    ```

- Image captioning

    ```bash
    python vqa.py --image ../../assets/images/jetson_orin_nano.png --query "Describe the whole image in 100 words."
    ```