# Vietnamese Wikipedia RAG

## Introduction
A Natural Language Processing school project that aims to build a Vietnamese Question-Answering system based on 10,000 articles from Vietnamese Wikipedia using Retrieval Augmented Generation (RAG).

## Preparation
Download the required libraries:
``` bash
cd VietnameseWikiRAG
pip install -r requirements.txt`
```

Unzip the dataset:
``` bash
unzip data/data_raw10k.zip`
```

Chunk the whole corpus and embed it:
``` bash
python src/chunking_embedding.py`
```

## Future direction
* Train a custom LLM model instead of using the pretrained VinAI's PhoGPT model.
* Enhance the retrieval stage using advanced techniques (e.g., query routing).
