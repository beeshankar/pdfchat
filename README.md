# PDFChat

PDFChat is a Python tool that extracts content from PDF files, generates markdown and JSON chunked documents, and enables interactive chat-based question answering using [Ollama](https://ollama.com/) locally. It leverages [pymupdf4llm](https://github.com/jerryjliu/pymupdf4llm) for PDF parsing and supports models like [Llama 3](https://ollama.com/library/llama3) and [Gemma](https://ollama.com/library/gemma) for conversational AI.

---

## Features

- **PDF Extraction:** Extract full-content and page-based chunks from PDFs as markdown, JSON, and text files.
- **Image Extraction:** Optionally extract and save images from PDFs.
- **RAG Chunks:** Automatically chunk PDF pages for retrieval-augmented generation (RAG) tasks.
- **Local Chatbot:** Chat with PDF content using Ollama's open-source LLMs.
- **Chunk-Relevant Context:** Questions are answered with context retrieved from the most relevant PDF chunks.
- **Citations:** Answers reference the page number when content is quoted.

---

## Installation

1. **Clone this repository:**

    ```sh
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
    ```

2. **Install [Ollama](https://ollama.com/download):**

    Download and install Ollama for your OS. Start Ollama server:

    ```sh
    ollama serve
    ```

    Pull a model, e.g.,

    ```sh
    ollama pull llama3
    ```

3. **Install dependencies:**

    Ensure you have Python 3.8+.

    ```sh
    pip install pymupdf4llm requests
    ```

---

## Usage

### Extract PDF Content

```sh
python pdfchat.py document.pdf
```

This generates `pdfchat-rag-DOCUMENT/` directory containing:
- Full markdown of the document
- Page-based chunks (JSON, combined markdown, separate chunk files)
- Extracted images (optionally)

### Chat with a PDF

```sh
python pdfchat.py document.pdf --chat
```

After extraction, enter questions interactively. The assistant will respond citing relevant page numbers!

### Other Options

- Disable image extraction: `--no-images`
- Specify Ollama model: `--model MODEL_NAME` (default: `gemma3:1b`)
- Combine options, e.g.:
  
    ```sh
    python pdfchat.py document.pdf --chat --no-images --model llama3
    ```

#### Help

```sh
python pdfchat.py --help
```

---

## Example

```sh
python pdfchat.py lecture_notes.pdf --chat
```

**Chat commands:**
- Enter questions directly.
- `chunks <query>` — Shows relevant chunks for your query.
- `history` — Shows conversation history.
- `clear` — Resets history.
- `exit` — Ends the chat.

---

## Output

Extracted files are saved to:

```
pdfchat-rag-<pdfname>/
├── <pdfname>_full.md
├── <pdfname>_chunks.json
├── <pdfname>_chunks_combined.md
├── chunks/
│   ├── chunk_001.md
│   ├── chunk_002.md
│   └── ...
├── images/
│   ├── page_001.png
│   └── ...
```

---

## Requirements

- Python 3.8+
- [pymupdf4llm](https://github.com/jerryjliu/pymupdf4llm)
- [Ollama](https://ollama.com/) (running locally for chat)
- Supported LLM model (e.g., `llama3`, `gemma`, etc.)

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

- [pymupdf4llm](https://github.com/jerryjliu/pymupdf4llm)
- [Ollama](https://ollama.com/)
