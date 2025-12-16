# NIST RAG Agent - Embeddings

This directory contains pre-built text embeddings for NIST cybersecurity publications.

## Available Documents

The following NIST publications are included as chunked JSON files:

- **NIST.SP.800-53r5** - Security and Privacy Controls for Information Systems
- **NIST.SP.800-37r2** - Risk Management Framework
- **NIST.SP.800-171r3** - Protecting Controlled Unclassified Information (CUI)
- **NIST.SP.800-60r2** - Guide for Mapping Types of Information
- **NIST.SP.800-63-3** - Digital Identity Guidelines
- **NIST.SP.800-53B** - Control Baselines
- **nist800-30r1** - Guide for Conducting Risk Assessments
- **nist800-137** - Continuous Monitoring for Federal Information Systems
- **nist800-18r1** - Guide for Developing Security Plans
- **nistspecialpublication800-60v1r1** - Volume 1: Guide for Mapping Types

## File Format

Each `.chunks.json` file contains an array of text chunks:

```json
[
  {
    "chunk_id": "0",
    "text": "Access Control: Organizations must...",
    "section": "3.1",
    "control_id": "AC-1",
    "title": "Access Control Policy and Procedures"
  }
]
```

## Usage

The agent automatically loads these files on startup and builds a FAISS vector index for fast similarity search.

## Building Your Own Embeddings

To add additional NIST documents:

1. Place PDF in `docs/` directory
2. Run the embedding builder:

```python
from tools.embedding_builder import build_embeddings

build_embeddings(
    pdf_path="docs/NIST.SP.800-XX.pdf",
    output_dir="embeddings/"
)
```

3. Restart the agent to load new embeddings

## Storage

- **JSON files** (~10-50MB each): Stored in Git
- **NPY files** (numpy embeddings, ~100-500MB each): Not stored in Git
  - Generated automatically on first run
  - Cached in `embeddings/faiss_index/`

## Notes

- Text chunks are approximately 1000 characters each
- Overlapping chunks preserve context across boundaries
- Metadata includes section, control ID, and title for citation
