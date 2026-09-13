from dotenv import load_dotenv
from anthropic import Anthropic
import voyageai
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)


load_dotenv()


def load_document(file_path):
    """
    Load a document based on its file type.
    Supports: .txt, .pdf, .docx
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"\n📄 Loading file: {file_path.name}")
    print(f"   File type: {file_path.suffix}")
    print(f"   File size: {file_path.stat().st_size / 1024:.2f} KB")

    # Choose loader based on file extension
    if file_path.suffix == ".txt":
        loader = TextLoader(str(file_path))
    elif file_path.suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif file_path.suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    documents = loader.load()
    print(f"   ✅ Loaded {len(documents)} document(s)")

    return documents


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents: List of LangChain Document objects
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of chunked Document objects
    """
    print(f"\n✂️  Chunking documents...")
    print(f"   Chunk size: {chunk_size} characters")
    print(f"   Chunk overlap: {chunk_overlap} characters")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    print(f"   ✅ Created {len(chunks)} chunks")

    # Show sample of first chunk
    if chunks:
        print(f"\n📝 Sample chunk (first 200 chars):")
        print(f"   {chunks[0].page_content[:200]}...")

    return chunks


def main():
    print("=" * 80)
    print("📚 DEMONSTRATION: Document Loader + Text Splitter")
    print("=" * 80)

    # Use the existing FAQ text file
    faq_file = Path("faq_txt_file.txt")

    if not faq_file.exists():
        print(f"❌ Error: {faq_file.name} not found in the project root!")
        print("   Please make sure the file exists before running this demo.")
        return

    print(f"\n✅ Using file: {faq_file.name}")

    # STEP 1: Load the document
    print("\n" + "🔹" * 40)
    print("STEP 1: LOADING THE DOCUMENT")
    print("🔹" * 40)

    documents = load_document(str(faq_file))

    print(f"\n📊 Document Details:")
    print(f"   • Number of documents loaded: {len(documents)}")
    print(f"   • Total characters: {len(documents[0].page_content)}")
    print(f"   • Total words: {len(documents[0].page_content.split())}")

    print(f"\n📖 Full Document Preview (first 300 chars):")
    print("   " + "-" * 76)
    print(f"   {documents[0].page_content[:300]}...")
    print("   " + "-" * 76)

    # STEP 2: Chunk the document with different sizes to show the difference
    print("\n" + "🔹" * 40)
    print("STEP 2: CHUNKING THE DOCUMENT")
    print("🔹" * 40)

    print("\n🔍 Demo A: Small chunks (chunk_size=300, overlap=50)")
    print("-" * 80)
    small_chunks = chunk_documents(documents, chunk_size=300, chunk_overlap=50)

    print("\n   Preview of chunks created:")
    for i, chunk in enumerate(small_chunks[:3]):  # Show first 3 chunks
        print(f"\n   📄 Chunk {i+1} ({len(chunk.page_content)} chars):")
        print(f"   {chunk.page_content[:150]}...")
        if i < len(small_chunks) - 1:
            print("   " + "─" * 76)

    print("\n\n🔍 Demo B: Medium chunks (chunk_size=500, overlap=100)")
    print("-" * 80)
    medium_chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=100)

    print("\n   Preview of chunks created:")
    for i, chunk in enumerate(medium_chunks[:2]):  # Show first 2 chunks
        print(f"\n   📄 Chunk {i+1} ({len(chunk.page_content)} chars):")
        print(f"   {chunk.page_content[:200]}...")
        if i < len(medium_chunks) - 1:
            print("   " + "─" * 76)

    print("\n\n🔍 Demo C: Large chunks (chunk_size=1000, overlap=200)")
    print("-" * 80)
    large_chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)

    print("\n   Preview of chunks created:")
    for i, chunk in enumerate(large_chunks[:2]):  # Show first 2 chunks
        print(f"\n   📄 Chunk {i+1} ({len(chunk.page_content)} chars):")
        print(f"   {chunk.page_content[:250]}...")
        if i < len(large_chunks) - 1:
            print("   " + "─" * 76)

    # SUMMARY
    print("\n" + "=" * 80)
    print("📊 CHUNKING COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Original Document: {len(documents[0].page_content)} characters")
    print(f"\nChunk Size: 300  → Chunks Created: {len(small_chunks)}")
    print(f"Chunk Size: 500  → Chunks Created: {len(medium_chunks)}")
    print(f"Chunk Size: 1000 → Chunks Created: {len(large_chunks)}")

    print("\n💡 Key Takeaway:")
    print("   • Smaller chunks = More chunks, better precision, but less context")
    print("   • Larger chunks = Fewer chunks, more context, but less precision")
    print("   • Overlap ensures important information isn't lost at chunk boundaries")
    print("=" * 80)

    return large_chunks  # Return for further processing if needed

    # Example file paths (update these paths to point to your documents)
    # file_paths = [
    #     "./faq_txt_file.txt",
    # ]

    # for file_path in file_paths:
    #     try:
    #         documents = load_document(file_path)
    #         for i, doc in enumerate(documents, 1):
    #             print(f"\n--- Document {i} Content Start ---\n")
    #             print(doc.page_content[:500])  # Print first 500 characters
    #             print(f"\n--- Document {i} Content End ---\n")
    #     except Exception as e:
    #         print(f"Error loading document {file_path}: {e}")


if __name__ == "__main__":
    main()
