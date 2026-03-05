#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time

import boto3
import polars as pl
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm  # Great for Jupyter notebooks


# In[ ]:


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

print("Keys loaded successfully!")  # Just to verify, avoid printing the actual keys


# In[ ]:


# 1. Setup Clients
pc = Pinecone(api_key=PINECONE_API_KEY)


# In[ ]:


# Initialize Bedrock with retry configuration to handle minor throttling
from botocore.config import Config

retry_config = Config(
    region_name="us-east-1", retries={"max_attempts": 5, "mode": "standard"}
)

boto_session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name="us-east-1",
)

bedrock_client = boto_session.client(
    service_name="bedrock-runtime", config=retry_config
)


# In[ ]:


# Note: Titan v1 is good, but you might also consider "amazon.titan-embed-text-v2:0"
# which allows flexible dimensions and is often cheaper/better.
embeddings = BedrockEmbeddings(
    client=bedrock_client, model_id="amazon.titan-embed-text-v1"
)


# In[ ]:


# 2. Initialize Pinecone Index
index_name = "arxiv-cs-methodologies"

if index_name not in pc.list_indexes().names():
    print(f"Creating index {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # 1536 for Titan v1. (Titan v2 defaults to 1024 or 512 depending on setup)
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait a moment for the index to be fully initialized before inserting
    time.sleep(10)


# In[ ]:


index = pc.Index(index_name)


# In[ ]:


# 3. Load Parquet & Prepare Documents
print("Loading Parquet data...")
df = pl.read_parquet("../data/cs_papers_data.parquet")


# In[ ]:


df.head()


# In[ ]:


df.null_count()


# In[ ]:


docs = []

print("Preparing documents...")

for row in tqdm(df.iter_rows(named=True), total=df.height, desc="Creating Docs"):
    date_str = row.get("update_date", "")
    date_int = int(date_str.replace("-", "")) if date_str else 0

    # Clean categories safely
    raw_cats = row.get("categories", "")
    categories = raw_cats.split(" ") if isinstance(raw_cats, str) else []

    doc = Document(
        page_content=row["abstract"],
        metadata={
            "title": row.get("title", "Unknown"),
            "categories": categories,
            "update_date": date_int,
            "date_display": date_str,
        },
    )

    docs.append(doc)


# In[ ]:


len(docs) == df.height


# In[ ]:


docs[:3]


# In[ ]:


# 4. Upsert to Pinecone in Batches
# Instantiate the vector store first
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)


# In[ ]:




