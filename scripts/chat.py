"""
scripts/chat.py - Interactive CLI for Spotify AI Support Agent

Usage:
  1. Interactive Chat Mode:
     .venv\\Scripts\\python scripts/chat.py

  2. One-off Message:
     .venv\\Scripts\\python scripts/chat.py --message "@SpotifyCares my account was hacked and songs are playing"

  3. Side-by-Side Comparison with Baselines:
     .venv\\Scripts\\python scripts/chat.py --compare --message "@SpotifyCares I was billed twice for Family Premium"
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import SpotifyRAGRetriever
from src.classifier import IntentClassifier
from src.agent import SpotifySupportAgent
from src.baselines import TrivialBaselineAgent, SimpleBaselineAgent

# Configure UTF-8 for Windows console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def render_banner():
    print("=" * 75)
    print(" 🎵 SPOTIFY AI SUPPORT AGENT — INTERACTIVE TESTING TERMINAL")
    print("=" * 75)
    print(" Type any incoming customer message or tweet to test the agent pipeline.")
    print(" Type 'exit', 'quit', or 'q' to end the session.")
    print("=" * 75 + "\n")


def display_rag_result(res: dict, verbose: bool = False):
    esc_badge = "🚨 [ESCALATE TO HUMAN SPECIALIST]" if res["is_escalation"] else "✅ [AUTO-HANDLED BY AI]"
    
    print("\n" + "─" * 75)
    print(f"📥 CUSTOMER QUERY: {res['customer_tweet']}")
    print("─" * 75)
    print(f"🎯 PREDICTED INTENT : {res['predicted_intent']} (Classifier confidence: {res['classifier_confidence']*100:.1f}%)")
    print(f"⚙️  ROUTING DECISION : {esc_badge}")
    print(f"📝 STATED REASON    : {res['escalation_reason']}")
    print(f"💬 DRAFT REPLY      :\n   \"{res['draft_reply']}\"")
    
    if verbose and "retrieved_docs" in res:
        print("\n📚 GROUNDED HISTORICAL RESOLUTIONS (TOP MATCHES):")
        for doc in res["retrieved_docs"]:
            print(f"   [{doc['rank']}] Sim: {doc['similarity']:.2f} | Intent: {doc['intent_label']}")
            print(f"       User : {doc['customer_text'][:70]}...")
            print(f"       Brand: {doc['spotify_reply'][:80]}...")
    print("─" * 75 + "\n")


def display_comparison(tweet: str, rag_agent, simple_agent, trivial_agent, verbose: bool = False):
    print("\n" + "=" * 75)
    print(f"📥 INCOMING TWEET: {tweet}")
    print("=" * 75)
    
    # 1. Baseline 1
    triv_res = trivial_agent.process(tweet)
    print("\n[1] BASELINE 1 (TRIVIAL - MAJORITY CLASS)")
    print(f"    Intent    : {triv_res['predicted_intent']}")
    print(f"    Escalate? : {triv_res['is_escalation']}")
    print(f"    Reply     : {triv_res['draft_reply']}")
    
    # 2. Baseline 2
    simp_res = simple_agent.process(tweet)
    print("\n[2] BASELINE 2 (SIMPLE - TF-IDF + KEYWORDS)")
    print(f"    Intent    : {simp_res['predicted_intent']} (Similarity: {simp_res['confidence']:.2f})")
    print(f"    Escalate? : {simp_res['is_escalation']} ({simp_res['escalation_reason']})")
    print(f"    Reply     : {simp_res['draft_reply']}")
    
    # 3. Proposed RAG Agent
    rag_res = rag_agent.process(tweet)
    print("\n[3] PROPOSED RAG SUPPORT AGENT (MINILM + RAG + GEMINI)")
    display_rag_result(rag_res, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(description="Test Spotify AI Support Agent from Terminal")
    parser.add_argument("-m", "--message", type=str, default=None, help="One-off customer message to test")
    parser.add_argument("-c", "--compare", action="store_true", help="Compare side-by-side with Baseline 1 & 2")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show top historical retrieved resolutions")
    args = parser.parse_args()

    print("Loading models and historical index (takes ~3-5 seconds)...")
    rag_retriever = SpotifyRAGRetriever()
    intent_clf = IntentClassifier(retriever_model=rag_retriever.model)
    rag_agent = SpotifySupportAgent(retriever=rag_retriever, classifier=intent_clf)

    simple_agent = SimpleBaselineAgent() if args.compare else None
    trivial_agent = TrivialBaselineAgent() if args.compare else None

    # Case 1: One-off message passed via argument
    if args.message:
        if args.compare:
            display_comparison(args.message, rag_agent, simple_agent, trivial_agent, verbose=args.verbose)
        else:
            res = rag_agent.process(args.message)
            display_rag_result(res, verbose=args.verbose)
        return

    # Case 2: Interactive REPL Mode
    render_banner()
    while True:
        try:
            user_input = input("Customer Tweet > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nGoodbye! 👋\n")
                break

            if args.compare:
                display_comparison(user_input, rag_agent, simple_agent, trivial_agent, verbose=args.verbose)
            else:
                res = rag_agent.process(user_input)
                display_rag_result(res, verbose=args.verbose)

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated. Goodbye! 👋\n")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}\n")


if __name__ == "__main__":
    main()
