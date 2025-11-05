from pypdf import PdfReader
from pathlib import Path
import argparse

def extract_pdf_text(pdf_path):
    try:
        if not Path(pdf_path).exists():
            print(f"Error: File not found: {pdf_path}")
            return
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"\nTotal pages in PDF: {total_pages}")
        for page_num in range(total_pages):
            print(f"\n{'='*80}")
            print(f"Page {page_num + 1} Content:")
            print(f"{'='*80}")
            
            page = reader.pages[page_num]
            text = page.extract_text()
            print(text)
            
            print(f"\nPage {page_num + 1} Details:")
            print(f"Media Box: {page.mediabox}")
            print(f"Rotation: {page.rotation}")
            print(f"{'='*80}\n")

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract text from a PDF file')
    parser.add_argument('pdf_file', help='Path to the PDF file to process')
    args = parser.parse_args()
    
    extract_pdf_text(args.pdf_file)