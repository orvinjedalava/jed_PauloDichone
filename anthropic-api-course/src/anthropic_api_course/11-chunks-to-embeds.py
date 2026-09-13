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
import chromadb

load_dotenv()


def save_to_chromadb(
    embedded_chunks, collection_name="faq_embeddings", db_path="./chroma_db"
):
    """
    Save embeddings to ChromaDB vector database.

    Args:
        embedded_chunks: List of dicts with 'text', 'embedding', 'metadata'
        collection_name: Name of the collection to store embeddings
        db_path: Path where ChromaDB will store the database

    Returns:
        ChromaDB collection object
    """
    print(f"\n💾 Saving embeddings to ChromaDB...")
    print(f"   Database path: {db_path}")
    print(f"   Collection name: {collection_name}")

    # Initialize ChromaDB client (creates database if it doesn't exist)
    client = chromadb.PersistentClient(path=db_path)

    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"description": "FAQ document embeddings"}
    )

    # Prepare data for ChromaDB
    ids = [f"chunk_{chunk['chunk_id']}" for chunk in embedded_chunks]
    embeddings = [chunk["embedding"] for chunk in embedded_chunks]
    documents = [chunk["text"] for chunk in embedded_chunks]
    metadatas = [chunk["metadata"] for chunk in embedded_chunks]

    # Add to collection
    collection.add(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )

    print(f"   ✅ Saved {len(embedded_chunks)} embeddings to ChromaDB")
    print(f"   Collection size: {collection.count()} items")

    return collection


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


def search_similar_chunks(collection, query_text, voyage_client, n_results=3):
    """
    Search for similar chunks using a query.

    Args:
        collection: ChromaDB collection
        query_text: Text to search for
        voyage_client: Voyage AI client for generating query embedding
        n_results: Number of results to return

    Returns:
        Search results from ChromaDB
    """
    print(f"\n🔍 Searching for: '{query_text}'")
    print(f"   Returning top {n_results} results...")

    # Generate embedding for the query
    query_embedding = generate_embedding(
        voyage_client, query_text, input_type="query"  # Use 'query' for searching
    )

    # Search in ChromaDB
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    print(f"\n📋 Search Results:")
    print("=" * 70)

    for i, (doc, distance, metadata) in enumerate(
        zip(results["documents"][0], results["distances"][0], results["metadatas"][0])
    ):
        print(f"\n🔹 Result {i+1} (Distance: {distance:.4f})")
        print(f"   {doc[:200]}...")
        print(f"   Metadata: {metadata}")
        print("-" * 70)

    return results


def generate_answer_with_context(query, search_results, anthropic_client):
    """
    Use Claude to generate an answer based on retrieved context.

    Args:
        query: User's question
        search_results: Results from ChromaDB search
        anthropic_client: Anthropic client

    Returns:
        Claude's answer based on the context
    """
    print(f"\n🤖 Generating answer with Claude...")

    # Extract the relevant documents from search results
    retrieved_docs = search_results["documents"][0]

    # Build context from retrieved chunks
    context = "\n\n".join(
        [f"Context {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)]
    )

    # Create the prompt with context
    system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
    
Instructions:
- Use ONLY the information from the provided context to answer the question
- If the context doesn't contain enough information to answer the question, say so
- Be concise but thorough in your answer
- Cite which context section you're using when relevant"""

    user_message = f"""Context from the knowledge base:

{context}

Question: {query}

Please provide a clear and accurate answer based on the context above."""

    print(f"   Sending request to Claude...")

    # Call Claude API
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text

    print(f"   ✅ Answer generated")
    print(f"\n" + "=" * 70)
    print("💬 CLAUDE'S ANSWER:")
    print("=" * 70)
    print(answer)
    print("=" * 70)

    return answer


def search_and_answer(
    collection, query_text, voyage_client, anthropic_client, n_results=3
):
    """
    Search for similar chunks and generate an answer using Claude.

    Args:
        collection: ChromaDB collection
        query_text: Text to search for
        voyage_client: Voyage AI client for generating query embedding
        anthropic_client: Anthropic client for generating answers
        n_results: Number of results to retrieve

    Returns:
        Dict with search results and generated answer
    """
    print(f"\n🔍 Searching for: '{query_text}'")
    print(f"   Retrieving top {n_results} relevant chunks...")

    # Generate embedding for the query
    query_embedding = generate_embedding(voyage_client, query_text, input_type="query")

    # Search in ChromaDB
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    print(f"\n📋 Retrieved Context:")
    print("=" * 70)

    for i, (doc, distance) in enumerate(
        zip(results["documents"][0], results["distances"][0])
    ):
        print(f"\n🔹 Context {i+1} (Similarity Score: {1 - distance:.4f})")
        print(f"   {doc[:200]}...")
        print("-" * 70)

    # Generate answer using Claude
    answer = generate_answer_with_context(query_text, results, anthropic_client)

    return {"query": query_text, "search_results": results, "answer": answer}


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


def generate_embedding(client, text, model="voyage-3-large", input_type="document"):
    """
    Generate embedding for a single text using Voyage AI.

    Args:
        client: Voyage AI client
        text: Text to embed
        model: Voyage AI model to use
        input_type: 'document' for storage, 'query' for search

    Returns:
        List of floats representing the embedding
    """
    result = client.embed([text], model=model, input_type=input_type)
    return result.embeddings[0]


def embed_chunks(voyage_client, chunks):
    """
    Generate embeddings for all chunks.

    Args:
        voyage_client: Voyage AI client
        chunks: List of LangChain Document objects

    Returns:
        List of tuples (chunk_text, embedding, metadata)
    """
    print(f"\n🔢 Generating embeddings for {len(chunks)} chunks...")

    embedded_chunks = []

    for i, chunk in enumerate(chunks):
        print(f"   Processing chunk {i+1}/{len(chunks)}...", end="\r")

        # Generate embedding
        embedding = generate_embedding(
            voyage_client,
            chunk.page_content,
            input_type="document",  # Use 'document' for storing
        )

        # Store chunk text, embedding, and metadata
        embedded_chunks.append(
            {
                "chunk_id": i,
                "text": chunk.page_content,
                "embedding": embedding,
                "metadata": chunk.metadata,
            }
        )

    print(f"\n   ✅ Generated {len(embedded_chunks)} embeddings")
    print(f"   Embedding dimension: {len(embedded_chunks[0]['embedding'])}")

    return embedded_chunks


def main():
    voyage_client = voyageai.Client()
    anthropic_client = Anthropic()

    print("=" * 70)
    print("📚 Document Processing & Embedding Pipeline with RAG")
    print("=" * 70)

    # Check if collection already exists
    db_path = "./chroma_db"
    collection_name = "faq_embeddings"

    client = chromadb.PersistentClient(path=db_path)
    existing_collections = [col.name for col in client.list_collections()]

    if collection_name in existing_collections:
        print(f"\n✅ Found existing collection: '{collection_name}'")
        collection = client.get_collection(collection_name)
        print(f"   Collection contains {collection.count()} embeddings")
        print(f"   Skipping document loading and embedding process...")

        # Skip to search demo
        print("\n🔹 SEARCH & ANSWER DEMO (RAG)")
        print("-" * 70)
        print("\n💡 Using existing embeddings for Retrieval Augmented Generation")

        query = input(
            "\nEnter a question about the document (or press Enter to skip): "
        ).strip()

        if query:
            result = search_and_answer(
                collection, query, voyage_client, anthropic_client, n_results=3
            )
            print(f"\n💾 Full response stored in result variable")
        else:
            print("   Skipping search demo")

        print("\n💡 To re-process documents, delete the ./chroma_db folder")
        return None

    # If no existing collection, proceed with full pipeline
    print("\n🔹 STEP 1: Upload & Load Document")
    print("-" * 70)

    file_path = input("Enter file path (or press Enter for demo text): ").strip()

    if not file_path:
        demo_file = Path("faq_txt_file.txt")
        file_path = str(demo_file)
        print(f"📝 Using demo file: {file_path}")

    try:
        # Load the document
        documents = load_document(file_path)

        # Chunk the documents
        print("\n🔹 STEP 2: Chunk Documents")
        print("-" * 70)
        chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)

        # Generate embeddings
        print("\n🔹 STEP 3: Generate Embeddings")
        print("-" * 70)
        embedded_chunks = embed_chunks(voyage_client, chunks)

        # Show sample embeddings
        print("\n🔹 STEP 4: Sample Embeddings")
        print("-" * 70)
        num_samples = min(3, len(embedded_chunks))

        for i in range(num_samples):
            chunk_data = embedded_chunks[i]
            print(f"\n📊 Chunk {i+1} Embedding:")
            print(f"   Text preview: {chunk_data['text'][:100]}...")
            print(
                f"   Embedding vector (first 10 values): {chunk_data['embedding'][:10]}"
            )
            print(
                f"   Embedding vector (last 10 values): {chunk_data['embedding'][-10:]}"
            )
            print(f"   Total dimensions: {len(chunk_data['embedding'])}")

        # Save to ChromaDB
        print("\n🔹 STEP 5: Save to Vector Database (ChromaDB)")
        print("-" * 70)
        collection = save_to_chromadb(
            embedded_chunks, collection_name=collection_name, db_path=db_path
        )

        # Search & Answer Demo (RAG)
        print("\n🔹 STEP 6: Search & Answer Demo (RAG)")
        print("-" * 70)
        print("\n💡 This demonstrates Retrieval Augmented Generation (RAG):")
        print("   1. Your question is converted to an embedding")
        print("   2. Most relevant document chunks are retrieved")
        print("   3. Claude uses those chunks to generate an accurate answer")

        query = input(
            "\nEnter a question about the document (or press Enter to skip): "
        ).strip()

        if query:
            result = search_and_answer(
                collection, query, voyage_client, anthropic_client, n_results=3
            )
            print(f"\n💾 Full response stored in result variable")
            print(f"   - result['query']: Your question")
            print(f"   - result['search_results']: Retrieved chunks")
            print(f"   - result['answer']: Claude's answer")
        else:
            print("   Skipping RAG demo")

        # Summary
        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE")
        print("=" * 70)
        print(f"📊 Summary:")
        print(f"   • Original documents: {len(documents)}")
        print(f"   • Total chunks: {len(chunks)}")
        print(f"   • Embeddings generated: {len(embedded_chunks)}")
        print(f"   • Embedding dimension: {len(embedded_chunks[0]['embedding'])}")
        print(f"   • Saved to ChromaDB: {collection.count()} items")
        print(f"   • Database location: {db_path}")
        print(f"\n💡 Your RAG system is ready!")
        print(f"   - Documents are embedded and stored")
        print(f"   - Next run will use existing embeddings (faster!)")
        print(f"   - Questions retrieve relevant context automatically")
        print(f"   - Claude answers based on YOUR documents")
        print("=" * 70)

        return embedded_chunks

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None


if __name__ == "__main__":
    main()
