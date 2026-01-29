import os
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import google.generativeai as genai

# VoyageAI is imported conditionally when needed
try:
    import voyageai
    VOYAGEAI_AVAILABLE = True
except ImportError:
    VOYAGEAI_AVAILABLE = False


class FinancialSituationMemory:
    def __init__(self, name, config):
        self.llm_provider = config.get("llm_provider", "openai").lower()
        
        if config["backend_url"] == "http://localhost:11434/v1":
            # Use local Ollama for embeddings
            self.embedding_provider = "ollama"
            self.embedding_model = "nomic-embed-text"
            self.openai_client = OpenAI(base_url=config["backend_url"])
        elif self.llm_provider == "google":
            # Use Google Gemini embeddings when using Gemini models
            # See: https://ai.google.dev/gemini-api/docs/embeddings
            self.embedding_provider = "google"
            self.embedding_model = "gemini-embedding-001"
            # genai uses GOOGLE_API_KEY env var automatically
        elif self.llm_provider == "anthropic" and VOYAGEAI_AVAILABLE and os.environ.get("VOYAGE_API_KEY"):
            # Use VoyageAI embeddings when using Anthropic/Claude models
            # Recommended by Anthropic: https://platform.claude.com/docs/en/build-with-claude/embeddings
            self.embedding_provider = "voyageai"
            self.embedding_model = "voyage-3.5"  # Best general-purpose model
            self.voyageai_client = voyageai.Client()  # Uses VOYAGE_API_KEY env var
        else:
            # Use OpenAI for embeddings (OpenAI, OpenRouter, or Anthropic without VoyageAI)
            # Anthropic doesn't have its own embeddings API
            self.embedding_provider = "openai"
            self.embedding_model = "text-embedding-3-small"
            self.openai_client = OpenAI()  # Uses OPENAI_API_KEY env var
            
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        """Get embedding for a text using the configured provider"""
        
        if self.embedding_provider == "google":
            # Use Google Gemini embeddings
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="SEMANTIC_SIMILARITY"
            )
            return result['embedding']
        elif self.embedding_provider == "voyageai":
            # Use VoyageAI embeddings (recommended for Anthropic/Claude)
            result = self.voyageai_client.embed(
                [text],
                model=self.embedding_model,
                input_type="document"
            )
            return result.embeddings[0]
        else:
            # Use OpenAI-compatible API (OpenAI or Ollama)
            response = self.openai_client.embeddings.create(
                model=self.embedding_model, input=text
            )
            return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using embeddings from the configured provider"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
