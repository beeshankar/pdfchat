import pymupdf4llm
from pathlib import Path
import argparse
import json
import os
import requests
from typing import List, Dict

def setup_output_dirs(pdf_path):
    """Create output directory structure for extracted content."""
    pdf_file = Path(pdf_path)
    base_dir = pdf_file.parent / f"pdfchat-rag-{pdf_file.stem}"
    base_dir.mkdir(exist_ok=True)
    
    chunks_dir = base_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    
    return base_dir, chunks_dir

def save_markdown(content, filepath):
    """Save markdown content to a file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Saved markdown to: {filepath}")

def save_chunks_as_json(chunks, filepath):
    """Save chunks as JSON for easy parsing."""
    # Convert chunks to JSON-serializable format
    serializable_chunks = []
    for chunk in chunks:
        serializable_chunk = {
            'text': chunk.get('text', ''),
            'metadata': {}
        }
        # Convert metadata to serializable format
        metadata = chunk.get('metadata', {})
        for key, value in metadata.items():
            # Convert non-serializable objects to strings
            if hasattr(value, '__dict__'):
                serializable_chunk['metadata'][key] = str(value)
            else:
                serializable_chunk['metadata'][key] = value
        serializable_chunks.append(serializable_chunk)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved chunks (JSON) to: {filepath}")
    return serializable_chunks

def save_chunks_as_text(chunks, chunks_dir):
    """Save individual chunks as separate text files."""
    for idx, chunk in enumerate(chunks, start=1):
        chunk_file = chunks_dir / f"chunk_{idx:03d}.md"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            # Write metadata
            f.write(f"# Chunk {idx}\n\n")
            f.write(f"**Metadata:**\n")
            f.write(f"- Pages: {chunk.get('metadata', {}).get('page', 'N/A')}\n")
            f.write(f"- Source: {chunk.get('metadata', {}).get('file_path', 'N/A')}\n\n")
            f.write("---\n\n")
            # Write content
            f.write(chunk.get('text', ''))
    print(f"✓ Saved {len(chunks)} individual chunk files to: {chunks_dir}")

def extract_pdf_to_markdown(pdf_path, page_chunks=True, write_images=True, 
                           image_format="png", dpi=150):
    """
    Extract PDF content as markdown with optional chunking.
    
    Args:
        pdf_path: Path to the PDF file
        page_chunks: Whether to create page-based chunks
        write_images: Whether to extract and save images
        image_format: Format for extracted images (png, jpg, etc.)
        dpi: DPI for image extraction
    
    Returns:
        list: List of chunks if page_chunks=True, else None
    """
    try:
        if not Path(pdf_path).exists():
            print(f"❌ Error: File not found: {pdf_path}")
            return None

        print(f"\n{'='*80}")
        print(f"Processing PDF: {Path(pdf_path).name}")
        print(f"{'='*80}\n")

        # Setup output directories
        base_dir, chunks_dir = setup_output_dirs(pdf_path)
        
        # Create images directory if needed
        if write_images:
            images_dir = base_dir / "images"
            images_dir.mkdir(exist_ok=True)
            image_path = str(images_dir)
        else:
            image_path = None

        # Extract as complete markdown document
        print("📄 Extracting full document as markdown...")
        full_markdown = pymupdf4llm.to_markdown(
            pdf_path,
            page_chunks=False,
            write_images=write_images,
            image_path=image_path,
            image_format=image_format,
            dpi=dpi
        )
        
        # Save full markdown
        full_md_file = base_dir / f"{Path(pdf_path).stem}_full.md"
        save_markdown(full_markdown, full_md_file)

        chunks_data = None
        # Extract as page chunks
        if page_chunks:
            print("\n📑 Extracting document with page chunks...")
            chunks = pymupdf4llm.to_markdown(
                pdf_path,
                page_chunks=True,
                write_images=write_images,
                image_path=image_path,
                image_format=image_format,
                dpi=dpi
            )
            
            print(f"\n✓ Created {len(chunks)} page-based chunks")
            
            # Display chunk statistics
            print("\n📊 Chunk Statistics:")
            print(f"{'='*80}")
            total_chars = sum(len(chunk.get('text', '')) for chunk in chunks)
            avg_chars = total_chars / len(chunks) if chunks else 0
            print(f"Total chunks: {len(chunks)}")
            print(f"Total characters: {total_chars:,}")
            print(f"Average characters per chunk: {avg_chars:,.0f}")
            
            # Show sample chunk info
            if chunks:
                print(f"\n📝 First chunk preview:")
                print(f"{'='*80}")
                first_chunk = chunks[0]
                print(f"Metadata: {first_chunk.get('metadata', {})}")
                preview = first_chunk.get('text', '')[:200]
                print(f"Content preview: {preview}...")
                print(f"{'='*80}")
            
            # Save chunks in different formats
            chunks_json_file = base_dir / f"{Path(pdf_path).stem}_chunks.json"
            chunks_data = save_chunks_as_json(chunks, chunks_json_file)
            
            save_chunks_as_text(chunks, chunks_dir)
            
            # Create a combined chunks markdown file
            combined_chunks_md = base_dir / f"{Path(pdf_path).stem}_chunks_combined.md"
            with open(combined_chunks_md, 'w', encoding='utf-8') as f:
                for idx, chunk in enumerate(chunks, start=1):
                    f.write(f"\n\n{'='*80}\n")
                    f.write(f"# CHUNK {idx}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(f"**Page:** {chunk.get('metadata', {}).get('page', 'N/A')}\n\n")
                    f.write(chunk.get('text', ''))
                    f.write("\n")
            print(f"✓ Saved combined chunks markdown to: {combined_chunks_md}")

        # Summary
        print(f"\n{'='*80}")
        print("✅ Extraction Complete!")
        print(f"{'='*80}")
        print(f"\nOutput directory: {base_dir}")
        print(f"\nFiles created:")
        print(f"  - Full markdown: {full_md_file.name}")
        if page_chunks:
            print(f"  - Chunks (JSON): {chunks_json_file.name}")
            print(f"  - Chunks (combined): {combined_chunks_md.name}")
            print(f"  - Individual chunks: {len(chunks)} files in chunks/")
        if write_images:
            print(f"  - Images: Saved to images/")
        print(f"{'='*80}\n")
        
        return chunks_data

    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

class PDFChatBot:
    """Chat with PDF using Ollama and Llama 3.1 8B"""
    
    def __init__(self, chunks: List[Dict], model="gemma3:1b", ollama_url="http://localhost:11434"):
        self.chunks = chunks
        self.model = model
        self.ollama_url = ollama_url
        self.chat_history = []
        
        # Test Ollama connection
        if not self.test_ollama_connection():
            raise ConnectionError("Cannot connect to Ollama. Make sure Ollama is running.")
    
    def test_ollama_connection(self):
        """Test if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def find_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Simple keyword-based chunk retrieval."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score chunks based on keyword overlap
        scored_chunks = []
        for idx, chunk in enumerate(self.chunks):
            text_lower = chunk['text'].lower()
            chunk_words = set(text_lower.split())
            
            # Calculate overlap score
            overlap = len(query_words & chunk_words)
            # Check if query appears as phrase
            phrase_match = query_lower in text_lower
            
            score = overlap + (10 if phrase_match else 0)
            
            if score > 0:
                scored_chunks.append((score, idx, chunk))
        
        # Sort by score and return top_k
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, _, chunk in scored_chunks[:top_k]]
    
    def create_context(self, relevant_chunks: List[Dict]) -> str:
        """Create context from relevant chunks."""
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            page = chunk['metadata'].get('page', 'Unknown')
            text = chunk['text']
            context_parts.append(f"[Chunk {i} - Page {page}]\n{text}")
        
        return "\n\n".join(context_parts)
    
    def chat(self, user_message: str, use_context: bool = True) -> str:
        """Send a message and get response from Ollama."""
        
        # Find relevant chunks
        relevant_chunks = []
        context = ""
        if use_context and self.chunks:
            relevant_chunks = self.find_relevant_chunks(user_message)
            if relevant_chunks:
                context = self.create_context(relevant_chunks)
        
        # Build the prompt
        if context:
            system_prompt = """You are a helpful assistant that answers questions based on the provided PDF document context. 
Use the context below to answer the user's question. If the answer is not in the context, say so clearly.
Always cite the page number when referencing information from the document.

CONTEXT:
""" + context
        else:
            system_prompt = "You are a helpful assistant answering questions about a PDF document."
        
        # Prepare messages for Ollama
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add chat history (last 5 exchanges)
        for msg in self.chat_history[-10:]:
            messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Call Ollama API
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result['message']['content']
                
                # Update chat history
                self.chat_history.append({"role": "user", "content": user_message})
                self.chat_history.append({"role": "assistant", "content": assistant_message})
                
                return assistant_message
            else:
                return f"Error: Ollama returned status code {response.status_code}"
        
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The model might be taking too long to respond."
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"
    
    def show_relevant_chunks(self, query: str, top_k: int = 3):
        """Display relevant chunks for a query."""
        relevant_chunks = self.find_relevant_chunks(query, top_k)
        
        print(f"\n📚 Found {len(relevant_chunks)} relevant chunks:")
        print("="*80)
        for i, chunk in enumerate(relevant_chunks, 1):
            page = chunk['metadata'].get('page', 'Unknown')
            text_preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            print(f"\nChunk {i} (Page {page}):")
            print(f"{text_preview}")
            print("-"*80)

def interactive_chat(chunks: List[Dict], model: str = "gemma3:1b"):
    """Start an interactive chat session with the PDF."""
    
    print(f"\n{'='*80}")
    print("🤖 PDF Chat Bot with Ollama (Llama 3.1 8B)")
    print(f"{'='*80}\n")
    print("Initializing chat bot...")
    
    try:
        chatbot = PDFChatBot(chunks, model=model)
        print("✅ Chat bot ready!\n")
        print("Commands:")
        print("  - Type your question to chat")
        print("  - 'chunks <query>' - Show relevant chunks for a query")
        print("  - 'history' - Show chat history")
        print("  - 'clear' - Clear chat history")
        print("  - 'exit' or 'quit' - Exit chat")
        print(f"\n{'='*80}\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            elif user_input.lower() == 'clear':
                chatbot.chat_history = []
                print("✓ Chat history cleared.\n")
                continue
            
            elif user_input.lower() == 'history':
                print("\n📜 Chat History:")
                print("="*80)
                for msg in chatbot.chat_history:
                    role = msg['role'].capitalize()
                    content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                    print(f"{role}: {content}\n")
                print("="*80 + "\n")
                continue
            
            elif user_input.lower().startswith('chunks '):
                query = user_input[7:].strip()
                chatbot.show_relevant_chunks(query)
                print()
                continue
            
            # Regular chat
            print("\n🤔 Thinking...\n")
            response = chatbot.chat(user_input)
            print(f"Assistant: {response}\n")
            print("-"*80 + "\n")
    
    except ConnectionError as e:
        print(f"\n❌ {e}")
        print("Please make sure Ollama is installed and running.")
        print("Install: https://ollama.com/download")
        print(f"Run: ollama run {model}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description='Extract PDF content and chat with it using Ollama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract and chat
  python script.py document.pdf --chat
  
  # Just extract without chat
  python script.py document.pdf
  
  # Use different Ollama model
  python script.py document.pdf --chat --model gemma3:1b
  
  # Extract without images
  python script.py document.pdf --no-images --chat
        """
    )
    
    parser.add_argument('pdf_file', help='Path to the PDF file to process')
    parser.add_argument('--chat', action='store_true',
                       help='Start interactive chat after extraction')
    parser.add_argument('--model', default='gemma3:1b',
                       help='Ollama model to use (default: gemma3:1b)')
    parser.add_argument('--no-chunks', action='store_true', 
                       help='Disable page-based chunking')
    parser.add_argument('--no-images', action='store_true',
                       help='Do not extract images from PDF')
    parser.add_argument('--image-format', default='png',
                       choices=['png', 'jpg', 'jpeg'],
                       help='Format for extracted images (default: png)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for image extraction (default: 150)')
    
    args = parser.parse_args()
    
    # Process the PDF
    chunks = extract_pdf_to_markdown(
        args.pdf_file,
        page_chunks=not args.no_chunks,
        write_images=not args.no_images,
        image_format=args.image_format,
        dpi=args.dpi
    )
    
    # Start chat if requested
    if args.chat:
        if chunks:
            interactive_chat(chunks, model=args.model)
        else:
            print("\n❌ Cannot start chat: No chunks were created.")
            print("Make sure --no-chunks is not used when starting chat.")

if __name__ == "__main__":
    main()