# RAG - Ops: RAG with Operations

We help you to deploy RAG with changing in datasources.

Note: Everything are under-construction and experimental!

## Start application

1. Clone the project

```bash
git clone https://github.com/hllj/rag-ops
```

2. Go to the directory 

```bash
cd rag-ops
```

3. Install libraries

```bash
pip install -r requirements.txt
```

4. Start services with Docker

Start MLFlow, Milvus, Minio, API endpoint

```bash
docker compose up -d
```

5. Start Data Ingestion

We only support Folder watcher to update documents in pdf files.

```bash
python start_ingestion.py
```

Upload pdf files in documents/ folder

Data will automatically be processed and stored in vector store.

```
documents/
│   *.pdf : new update files
└───processed/: processed files
└───error/: error files
```


## Stack

- MLFlow: Experiments management.
- Minio: Storage.
- Milvus: Vector Store.
- LangChain: Data and RAG Orchestration.
- FastAPI.

Much more to come, stay tune.