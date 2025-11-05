import pdfplumber
from pathlib import Path
import argparse

def extract_pdf_text(pdf_path):
    try:
        if not Path(pdf_path).exists():
            print(f"Error: File not found: {pdf_path}")
            return
            
        with pdfplumber.open(pdf_path) as pdf:
            print("\nDocument Information:")
            print("="*80)
            print(f"Total Pages: {len(pdf.pages)}")
            if pdf.metadata:
                for key, value in pdf.metadata.items():
                    print(f"{key}: {value}")
            print("="*80)

            for page_num, page in enumerate(pdf.pages):
                print(f"\nPage {page_num + 1} Content:")
                print("="*80)
                
                text = page.extract_text()
                print(text)
                
                print(f"\nPage {page_num + 1} Details:")
                print(f"Dimensions: Width={page.width}, Height={page.height}")
                print(f"Crop Box: {page.cropbox}")
                print(f"Rotation: {page.rotation}")
                
                tables = page.extract_tables()
                if tables:
                    print(f"\nTables found on page {page_num + 1}:")
                    for table_num, table in enumerate(tables, 1):
                        print(f"\nTable {table_num}:")
                        for row in table:
                            print(row)
                
                images = page.images
                if images:
                    print(f"\nNumber of images on page: {len(images)}")
                    for img_num, img in enumerate(images, 1):
                        print(f"Image {img_num}: {img['width']}x{img['height']} pixels")
                
                print("="*80)

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract text from a PDF file')
    parser.add_argument('pdf_file', help='Path to the PDF file to process')
    args = parser.parse_args()
    
    extract_pdf_text(args.pdf_file)