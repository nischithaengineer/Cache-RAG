# Cache-RAG with LangGraph

A Retrieval-Augmented Generation (RAG) system with semantic caching capabilities built using LangGraph. This project demonstrates how to implement intelligent caching for LLM responses using semantic similarity search with FAISS.

## Features

- **Semantic Caching**: Cache LLM responses based on semantic similarity rather than exact query matching
- **RAG Pipeline**: Retrieve relevant context from documents before generating answers
- **LangGraph Workflow**: Stateful, multi-step agent workflows with conditional routing
- **FAISS Vector Store**: Efficient similarity search for both document retrieval and cache lookup
- **Performance Optimization**: Significantly reduce LLM API calls and response times through intelligent caching

## Architecture

The system consists of two main components:

### 1. Simple Cache Model
A basic in-memory cache that stores LLM responses keyed by exact query strings.

### 2. Advanced CAG (Conditional Agent Graph)
A sophisticated RAG pipeline with semantic caching that:
- Normalizes user queries
- Checks semantic cache for similar previous queries
- Retrieves relevant documents if cache miss
- Generates answers using retrieved context
- Stores new Q&A pairs in the semantic cache

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Cache-RAG-with-Langraph
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running the Notebook

Open and run the `cag.ipynb` notebook in Jupyter:

```bash
jupyter notebook cag.ipynb
```

### Key Configuration

The system can be configured through the following parameters in the notebook:

- `EMBED_MODEL`: Embedding model for semantic similarity (default: "sentence-transformers/all-MiniLM-L6-v2")
- `LLM_MODEL`: LLM model for generation (default: "gpt-4o-mini")
- `RETRIEVE_TOP_K`: Number of documents to retrieve (default: 4)
- `CACHE_TOP_K`: Number of cache candidates to check (default: 3)
- `CACHE_DISTANCE_THRESHOLD`: Semantic similarity threshold for cache hits (default: 0.45)
- `CACHE_TTL_SEC`: Time-to-live for cache entries in seconds (default: 0, disabled)

## How It Works

1. **Query Normalization**: User queries are normalized (lowercased, stripped)
2. **Semantic Cache Lookup**: The system searches for semantically similar cached queries using FAISS
3. **Cache Hit**: If a similar query is found within the distance threshold, return cached answer
4. **Cache Miss**: If no similar query is found:
   - Retrieve relevant documents from the RAG store
   - Generate answer using LLM with retrieved context
   - Store the Q&A pair in the semantic cache for future use

## Example

```python
# First query - cache miss, generates answer
q1 = "what is agent memory ?"
out1 = app.invoke({"question": q1, "context_docs": [], "citations": []}, thread_cfg)
# Cache hit: False

# Similar query - cache hit, returns cached answer
q2 = "Explain about agent memory ?"
out2 = app.invoke({"question": q2, "context_docs": [], "citations": []}, thread_cfg)
# Cache hit: True
```

## Project Structure

```
Cache-RAG-with-Langraph/
├── cag.ipynb                    # Main notebook with implementation
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .env                         # Environment variables (create this)
```

## Dependencies

- **langchain**: Core LangChain framework
- **langgraph**: Stateful agent workflows
- **faiss-cpu**: Vector similarity search
- **sentence-transformers**: Embedding models
- **langchain-openai**: OpenAI integration
- **python-dotenv**: Environment variable management

## Performance Benefits

- **Reduced API Calls**: Semantic caching significantly reduces redundant LLM API calls
- **Faster Response Times**: Cache hits return instantly (0.00 seconds vs. 3-20+ seconds)
- **Cost Savings**: Fewer API calls mean lower costs
- **Improved User Experience**: Instant responses for similar queries

## Notes

- The semantic cache uses L2 distance for similarity measurement (lower = more similar)
- Cache entries can optionally have a TTL (time-to-live) for expiration
- The RAG store is populated from web documents in the example, but can be customized
- The system uses in-memory storage by default (not persistent across sessions)

## License

MIT License 

