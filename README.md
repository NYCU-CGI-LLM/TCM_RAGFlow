# RAGFlow

## Prerequisites

- Docker Engine 24+ with the Compose plugin, [installation guide](https://docs.docker.com/engine/install/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.4 or newer for Python environment management
- Node.js >= 18.20 (and npm) for running the RAGFlow web UI locally
- NVIDIA GPU drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — only required if you plan to run the `tcm_embedding` container or fine-tune the embedding model on a GPU
- (Linux only) System libraries for the backend: `pkg-config`, `libicu-dev`, `libjemalloc-dev`

## Running the Project

There are two supported workflows. Pick the one that matches how much you want to run on the host machine versus in Docker.

### Option A — Full container stack

1. Review `.env` and adjust ports, credentials, or `COMPOSE_PROFILES` if necessary. The default values work for local development.
2. From the repository root, start the full stack (this includes the optional `tcm_embedding` container, which starts in an idle state; see [TCMEmbeddingModel (optional)](#tcmembeddingmodel-optional) for usage details or the vLLM CLI alternative):

    ```bash
    docker compose up -d --build
    ```

3. Wait for the services to become healthy:

    ```bash
    docker compose ps
    docker compose logs -f ragflow  # optional, shows backend status
    ```

   The `ragflow_eval` and `tcm_embedding` services start in an idle `sleep infinity` state so you can `docker compose exec` into them when needed. If you do not need the embedding container, you can stop it independently with `docker compose stop tcm_embedding`.

4. When you're done, stop everything with `docker compose down`.

### Option B — Local backend/frontend with shared containers

This mode keeps databases and storage in Docker but runs the API server and web UI directly on your machine.

1. Start the shared infrastructure:

    ```bash
    cd ragflow/docker
    docker compose -f docker-compose-base.yml up -d
    docker compose ps
    ```

2. Set up the backend (from `ragflow/`):

    ```bash
    cd ragflow
    # Install once on Debian/Ubuntu; use the equivalent packages on other distros.
    sudo apt update
    sudo apt install -y pkg-config libicu-dev libjemalloc-dev

    uv venv --python 3.11
    uv sync
    uv run download_deps.py
    source .venv/bin/activate
    bash docker/launch_backend_service.sh 9380
    ```

   The backend will listen on `http://127.0.0.1:9380`.

3. (Optional) Run the TCM embedding model locally. This is useful if you prefer not to start the Docker container:

    ```bash
    cd TCMEmbeddingModel
    uv venv --python 3.10
    uv sync
    source .venv/bin/activate
    cd scripts
    bash deploy.sh
    ```

    Refer to [TCMEmbeddingModel (optional)](#tcmembeddingmodel-optional) for more details and configuration options.

4. Run the web UI (in a separate terminal):

    ```bash
    cd ragflow/web
    npm install
    npm run dev
    ```

   By default the UI is available at `http://localhost:9222`.

5. Shut everything down with `Ctrl+C` for the local processes and `docker compose -f ragflow/docker/docker-compose-base.yml down` for the infrastructure.

## TCMEmbeddingModel (optional)

The TCM embedding model service is optional. Run it only if you want to inspect the training scripts, host your own embedding endpoint, or continue fine-tuning.

- **Docker workflow** — dependencies are baked into the image, so no extra `uv sync` is required.

    ```bash
    docker compose up -d tcm_embedding
    docker compose exec tcm_embedding bash -lc "cd /workspace/TCMEmbeddingModel/scripts && bash deploy.sh"
    ```

    Adjust `model_path`, `model_name`, and `serve_port` in the scripts to match your environment. Stop the container with `docker compose stop tcm_embedding` when you are done.

- **vLLM CLI workflow** — you can host the embedding model without any containers. Install vLLM by following the [official quickstart](https://docs.vllm.ai/en/latest/getting_started/installation.html) and serve the published weights directly:

    ```bash
    vllm serve \
      --model NYCU-CGI-LLM/tcm-qwen3-embedding \
      --tensor-parallel-size 1 \
      --port 8000
    ```

    The model weights are published at [https://huggingface.co/collections/NYCU-CGI-LLM/tcm-qwen3-embedding](https://huggingface.co/collections/NYCU-CGI-LLM/tcm-qwen3-embedding).

## Setup RAGFlow

To connect the UI to your vLLM host, create a knowledge base, obtain a RAGFlow API key, and finish the setup flow, follow [docs/GUIDE.md](docs/GUIDE.md).


## RAGFlow-eval

You can run evaluations locally or inside the `ragflow_eval` container.

- **Local execution**

    ```bash
    cd ragflow_eval
    uv venv --python 3.11
    uv sync
    source .venv/bin/activate
    python run_evaluation.py --config config/generation_example.yaml
    ```

- **Docker execution**

    ```bash
    docker compose up -d ragflow_eval  # ensure the service is running
    docker compose exec ragflow_eval bash -lc "cd /workspace/ragflow_eval && uv run python run_evaluation.py --config config/generation_example.yaml"
    ```

- Refer the [README.md](http://README.md) for more
- A Quick sanity test
    1. Edit `ragflow_eval/test_req.sh`
    2. Set 
        - `API_BASE`: RAGFlow backend
        - `API_KEY`: RAGFlow API Key
        - `DATASET_NAME`: The `knowledge base` name (can override with cli agrument)
    3. Execute
        
        ```python
        source .venv/bin/activate
        bash test_req.sh simple_rag_qwen 
        ```
        
    4. Expected output:
        - Resolve dataset name into UUID and then perform retrieval
        
        ![image.png](docs/image%2022.png)
