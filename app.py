import gradio as gr
import sys
import os
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Gracefully handle spaces GPU decorator if running on Hugging Face ZeroGPU
try:
    import spaces
    gpu_decorator = spaces.GPU
except (ImportError, Exception):
    gpu_decorator = lambda f: f

from src.retriever import SpotifyRAGRetriever, get_default_processed_dir
from src.classifier import IntentClassifier
from src.agent import SpotifySupportAgent

# Global state for agent
_agent = None
_init_error = None

def get_agent():
    """Lazily initialize the agent and return (agent, error_msg)."""
    global _agent, _init_error
    if _agent is not None:
        return _agent, None

    # Check for GEMINI_API_KEY
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        data_dir = get_default_processed_dir()
        clusters_found = (data_dir / "sample_with_clusters.csv").exists()
        _init_error = (
            "GEMINI_API_KEY is not set in the environment.\n\n"
            "👉 Fix: Go to your Space's 'Settings' tab > 'Variables and secrets' > 'New secret'.\n"
            "   - Name: GEMINI_API_KEY\n"
            "   - Value: <your_api_key>\n"
            "   - Important: After saving, click 'Restart Space' at the top of Settings!\n\n"
            f"Diagnostics:\n"
            f"- Data directory found: {data_dir} (sample_with_clusters.csv exists: {clusters_found})\n"
            f"- Current working dir: {os.getcwd()}"
        )
        return None, _init_error

    try:
        print("Loading retriever, classifier, and agent...")
        rag_retriever = SpotifyRAGRetriever()
        intent_clf = IntentClassifier(retriever_model=rag_retriever.model)
        _agent = SpotifySupportAgent(retriever=rag_retriever, classifier=intent_clf)
        _init_error = None
        return _agent, None
    except Exception as e:
        data_dir = get_default_processed_dir()
        _init_error = (
            f"Initialization error ({type(e).__name__}): {str(e)}\n\n"
            f"Diagnostics:\n"
            f"- Data directory: {data_dir}\n"
            f"- sample_with_clusters.csv exists: {(data_dir / 'sample_with_clusters.csv').exists()}\n"
            f"- embeddings_cache.npy exists: {(data_dir / 'embeddings_cache.npy').exists()}\n"
            f"- GEMINI_API_KEY present: {'GEMINI_API_KEY' in os.environ}"
        )
        print(f"Error loading agent: {_init_error}\n{traceback.format_exc()}")
        return None, _init_error

# Pre-warm agent at startup if possible
get_agent()

@gpu_decorator
def chat_logic(message, history):
    agent, err = get_agent()
    if agent is None:
        return f"⚠️ **Configuration Issue:**\n\n{err}"
        
    try:
        res = agent.process(message)
        
        esc_badge = "🚨 **[ESCALATED TO HUMAN]**" if res["is_escalation"] else "✅ **[AUTO-HANDLED]**"
        reply = res["draft_reply"]
        intent = res["predicted_intent"]
        confidence = round(res.get("classifier_confidence", 0) * 100, 2)
        reason = res["escalation_reason"]
        
        response = f"{esc_badge}\n\n**Reply:**\n{reply}\n\n"
        response += "---\n"
        response += f"🔍 **Agent Internal Reasoning:**\n"
        response += f"- **Intent:** `{intent}` ({confidence}% confidence)\n"
        response += f"- **Routing Decision:** {reason}"
        
        return response
    except Exception as e:
        return f"❌ Failed to process query ({type(e).__name__}): {str(e)}"

demo = gr.ChatInterface(
    fn=chat_logic,
    title="🎧 Spotify AI Support Agent",
    description="End-to-End RAG Support Agent (Intent Classification, Historical Resolution Retrieval, and Grounded Reply Generation).",
    examples=[
        "@SpotifyCares I got double charged for my Family plan!",
        "@SpotifyCares my account was hacked and weird songs are playing",
        "The app keeps crashing when I try to open my downloaded playlists.",
        "How do I create a collaborative playlist with my friend?"
    ]
)

if __name__ == "__main__":
    demo.launch()
