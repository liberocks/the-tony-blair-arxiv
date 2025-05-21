import os
import time
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
import getpass
from upstash_vector import Index, Vector

# =====================================================
# Global instances
# =====================================================
openai_client = None  # Will be initialized in main()
upstash_index = None  # Will be initialized in main()

# =====================================================
# Set up OpenAI
# =====================================================
MODEL = "text-embedding-3-small"
MAX_TOKENS_PER_REQUEST = 8191


# =====================================================
# Input from user
# =====================================================
def get_openai_api_key():
    # Try to get API key from environment variable first
    api_key = os.getenv("OPENAI_API_KEY")

    # If not found, prompt user for input
    if not api_key:
        print("OpenAI API key not found in environment variables.")
        api_key = getpass.getpass("Please enter your OpenAI API key: ")

        if not api_key:
            raise ValueError("No API key provided. Cannot continue.")

    return api_key


def get_upstash_token():
    # Try to get API key from environment variable first
    token = os.getenv("UPSTASH_TOKEN")

    # If not found, prompt user for input
    if not token:
        print("Upstash token not found in environment variables.")
        token = getpass.getpass("Please enter your Upstash Token key: ")

        if not token:
            raise ValueError("No token provided. Cannot continue.")

    return token


def embed_texts(texts):
    response = openai_client.embeddings.create(model=MODEL, input=texts)
    return [e.embedding for e in response.data]


def main():
    global openai_client
    global upstash_index

    # Initialize OpenAI client with user-provided API key
    openai_api_key = get_openai_api_key()
    openai_client = OpenAI(api_key=openai_api_key)

    # Initialize Upstash index with user-provided token
    upstash_token = get_upstash_token()
    upstash_index = Index(
        url="https://capable-midge-9649-eu1-vector.upstash.io",
        token=upstash_token,
    )

    print("Loading dataset...")
    dataset = load_dataset("ccdv/arxiv-summarization", "section", split="train")

    articles = dataset["article"]
    abstracts = dataset["abstract"]

    successful_upserts = 0
    failed_upserts = 0

    print("Embedding articles and abstracts and upserting to Upstash...")

    for i in tqdm(range(len(dataset))):
        text = f"Abstract: {abstracts[i]}"
        try:
            # Embed each pair as one input
            embedding = embed_texts([text])[0]

            # Upsert to Upstash
            upstash_index.upsert(
                vectors=[
                    Vector(
                        id=f"arxiv_{i}",
                        vector=embedding,
                        metadata={"abstract": abstracts[i]},
                    )
                ]
            )
            successful_upserts += 1
        except Exception as e:
            print(f"Error at index {i}: {e}")
            failed_upserts += 1
            time.sleep(2)  # backoff strategy

        time.sleep(0.15)  # delay to avoid rate limit

    print(f"Done. {successful_upserts} embeddings successfully upserted to Upstash.")
    print(f"Failed upserts: {failed_upserts}")


if __name__ == "__main__":
    main()
