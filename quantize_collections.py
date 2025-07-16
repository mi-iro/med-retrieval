import argparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType
)

def quantize_collection(client: QdrantClient, collection_name: str):
    """
    Applies scalar quantization to the specified collection.

    Args:
        client: An instance of QdrantClient.
        collection_name: The name of the collection to be quantized.
    """
    try:
        # Correctly structure the quantization configuration
        quantization_config = ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,  # Use top 99% of data to build quantization
                always_ram=True # Keep quantized vectors in RAM for faster search
            )
        )

        client.update_collection(
            collection_name=collection_name,
            quantization_config=quantization_config
        )
        print(f"Successfully applied scalar quantization to collection '{collection_name}'.")
    except Exception as e:
        print(f"An error occurred while applying quantization to collection '{collection_name}': {e}")

def main():
    """
    Main function to connect to Qdrant and quantize all specified text collections.
    """
    parser = argparse.ArgumentParser(description="Apply quantization to Qdrant collections for performance optimization.")
    parser.add_argument("--host", type=str, default="localhost", help="The hostname of the Qdrant service.")
    parser.add_argument("--port", type=int, default=6333, help="The port number of the Qdrant service.")
    args = parser.parse_args()

    # Collections to be quantized, identified from text_search.py
    collections_to_quantize = ["guideline_qwen", "research_qwen", "book_qwen", "wiki_qwen"]

    try:
        # Initialize the Qdrant client
        client = QdrantClient(host=args.host, port=args.port)
        print(f"Successfully connected to Qdrant at {args.host}:{args.port}")

        # Iterate through the collections and apply quantization
        for collection in collections_to_quantize:
            quantize_collection(client, collection)

    except Exception as e:
        print(f"Failed to connect to Qdrant or an error occurred during the process: {e}")
        print("Please ensure the Qdrant service is running and the host/port are correctly configured.")

if __name__ == "__main__":
    main()