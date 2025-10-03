#reingest-seam.py
import os
import shutil
# Import necessary functions from core/ingest (assuming it is the latest version)
from core.ingest import process_document, SUPPORTED_TYPES 

# Define constants
DATA_DIR = "data"
SEAM_DOC_ID = "seam"
VECTORSTORE_DIR = "vectorstore" 

# -------------------------------------------------------------
print(f"--- Starting Re-Ingestion for Document ID: '{SEAM_DOC_ID}' ---")

# 1. Manually delete the old vectorstore folder for 'seam' (to avoid deleting the source file)
print("\n🧹 Cleaning up old vector store (only the vector embeddings folder)...")
vectorstore_path = os.path.join(VECTORSTORE_DIR, SEAM_DOC_ID)
try:
    if os.path.exists(vectorstore_path):
        shutil.rmtree(vectorstore_path)
        print(f"  - Deleted old vector store folder: {vectorstore_path}")
    else:
        print(f"  - Warning: Vector store path '{vectorstore_path}' not found. Skipping deletion.")

except Exception as e:
    # Handle potential errors during deletion (e.g., permission issues)
    print(f"  ❌ Critical Error during vector store deletion: {e}") 
        
print("🧹 Cleanup complete.")
# -------------------------------------------------------------

# 2. Search for the original 'seam' file
seam_file_found = False
target_file_path = None
target_file_name = None

if not os.path.exists(DATA_DIR):
    print(f"❌ Error: Data directory '{DATA_DIR}' not found. Cannot proceed.")
else:
    print(f"\n🔎 Searching for '{SEAM_DOC_ID}.*' in the '{DATA_DIR}' directory...")
    
    found_files = [] 
    
    # Search for files named 'seam' with any supported extension in DATA_DIR
    for filename in os.listdir(DATA_DIR):
        # Skip hidden files
        if filename.startswith('.'):
            continue
            
        found_files.append(filename)

        name_part, ext = os.path.splitext(filename)
        if name_part.lower() == SEAM_DOC_ID and ext.lower() in SUPPORTED_TYPES:
            target_file_name = filename
            target_file_path = os.path.join(DATA_DIR, filename)
            seam_file_found = True
            break
            
    if not seam_file_found and found_files:
        print(f"  ℹ️ Files found in '{DATA_DIR}' but none matched '{SEAM_DOC_ID}.*': {', '.join(found_files)}")
    
    if seam_file_found:
        print(f"\nProcessing target file: '{target_file_name}'...")
        
        # 3. Process and Re-Ingest the document
        try:
            # IMPORTANT: Explicitly pass collection_id='seam' to override the default folder-based ID ('data')
            process_document(target_file_path, target_file_name, collection_id=SEAM_DOC_ID) 
            
            print(f"  ✅ Successfully re-indexed '{target_file_name}'. The new vector store should be accessible via doc_id: '{SEAM_DOC_ID}'.")
            
        except Exception as e:
            print(f"  ❌ Critical Error processing {target_file_name}: {e}")
    else:
        print(f"\n❌ Error: Could not find the file '{SEAM_DOC_ID}.*' in the '{DATA_DIR}' directory. ")
        print(f"   Please ensure the source document is named exactly 'seam' (e.g., 'seam.pdf') and is located directly inside the '{DATA_DIR}' folder.")

print(f"\n--- Document Re-Ingestion Complete ---")
