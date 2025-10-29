Official RAGFlow document: https://ragflow.io/docs/dev/

## RAGFlow Model provider

1. Click on the user avatar (top right corner)
    
    ![image.png](./image%205.png)
    
2. Click `Model providers`
    
    ![image.png](./image%206.png)
    
3. Add OpenAI model for generation, Input the OpenAI API key by clicking the `API-Key`
    
    ![image.png](./image%207.png)
    
4. Add vLLM model for retrieval, find vLLM provider and choose, click `Add Model`
    
    ![image.png](./image%208.png)
    
5. Fill the model information:
    
    ![image.png](./image%209.png)
    
    - `Model type`: embedding
    - `Model name`: match with the `MODEL_NAME` when we host vllm in `TCMEmbeddingModel`
    - `Base url` : match with the host ip address and `SERVE_PORT` when we host vllm in `TCMEmbeddingModel`

![image.png](./image%2010.png)

## RAGFlow API Key (Used for evaluation)

1.  Get the ragflow API Key (use for evaluation later)
    
    ![image.png](./image%2011.png)
    
    ![image.png](./image%2012.png)
    

## Knowledge base and ingestion (for retreival evaluation)

1. Create new knowledge base
    
    ![image.png](./image%2013.png)
    

![image.png](./image%2014.png)

1. Configuration only need to change the `Embedding model` to use the `Qwen 3 finetuned`
    
    ![image.png](./image%2015.png)
    
2. Upload the `cn_syndrome_knowledge_fixed.json`
    
    ![image.png](./image%2016.png)
    
3. Select `Chunking method` and select `JSON`
    
    ![image.png](./image%2017.png)
    
    ![image.png](./image%2018.png)
    
4. Click play to process ingestion
    
    ![image.png](./image%2019.png)
    
5. Finished ingestion, confirm `SUCCESS` status
    
    ![image.png](./image%2020.png)
    

## Chat Assistant (for generation evaluation)

![image.png](./image%2021.png)
- The Assistant name is used for the `chat_name` in generation evaluation`ragflow_eval`
- Make sure select the `Knowledge bases`
![image.png](./image%2023.png)