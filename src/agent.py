"""
src/agent.py - RAG-Grounded Spotify Customer Support Agent

Architecture:
1. Intent Classification: Calibrated classifier predicts customer intent & confidence.
2. Semantic Retrieval: Retrieves top-3 historical brand resolutions from Spotify.
3. Decision & Generation: LLM determines escalation (with stated reason) and drafts
   an empathetic response grounded directly in Spotify's historical resolutions.
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from src.retriever import SpotifyRAGRetriever
from src.classifier import IntentClassifier
from src.utils import call_with_retry

load_dotenv()


class AgentDecision(BaseModel):
    """Structured decision output schema for the Support Agent."""
    reasoning: str = Field(description="Step-by-step reasoning explaining intent and escalation determination.")
    predicted_intent: str = Field(description="One of: subscription_issue, account_access, general_inquiry, off_topic_chatter, app_bug, billing_dispute, feature_request")
    is_escalation: bool = Field(description="True if human intervention is required; False if auto-handled by agent.")
    escalation_reason: str = Field(description="Explicit reason for escalation decision (e.g. 'Billing refund requires private account verification via human support team').")
    draft_reply: str = Field(description="Customer-facing response grounded in Spotify's historical resolutions. Concise (under 60 words, suitable for Twitter).")


AGENT_SYSTEM_PROMPT = """You are an expert AI customer support agent for Spotify (@SpotifyCares).
Your mission is to support Spotify customers on Twitter with high empathy, brand fidelity, and precision.

You have access to:
1. Historical Spotify resolutions from similar past customer tweets.
2. Verified Spotify policies and product tiers (Free, Premium, Family, Student).

YOUR RESPONSIBILITIES:
1. Intent: Classify the customer's intent accurately into one of the 7 classes.
2. Escalation Decision: Carefully determine whether this tweet requires HUMAN ESCALATION or can be AUTO-HANDLED.
   - ESCALATE (is_escalation = true) ONLY IF:
     * Unauthorized charges, double billing, refund requests, or payment transaction failures
     * Account compromised, hacked, suspicious device activity, email/password changed without permission, or sign-in lockouts
     * Explicit requests requiring internal backstage database lookup or private DM account verification
   - AUTO-HANDLE (is_escalation = false) IF:
     * General subscription FAQs, pricing inquiries, or how-to questions (unless customer reports an active billing/payment failure)
     * App troubleshooting (cache clear, reinstall, offline sync advice, audio stutter)
     * Feature requests, feedback, or UI suggestions
     * General inquiries, catalog questions, availability in specific countries
     * Off-topic chatter, banter, or compliments
3. Escalation Reason: Provide an explicit, transparent justification for your decision.
4. Grounded Reply: Draft a reply GROUNDED in how Spotify has historically resolved similar issues.
   - Match Spotify's brand voice: empathetic, friendly, and concise (<60 words).
   - If escalating: politely prompt for a DM with account email so human agents can help backstage.
   - If auto-handling: provide clear, direct actionable guidance.
   - Never generate broken shortlinks (use https://support.spotify.com for help articles).
"""


class SpotifySupportAgent:
    """
    Production-grade RAG Support Agent combining semantic search,
    calibrated intent classification, and structured LLM generation.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        retriever: Optional[SpotifyRAGRetriever] = None,
        classifier: Optional[IntentClassifier] = None,
    ):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("GEMINI_API_KEY is required in environment.")

        self.model_name = model_name or os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash")
        self.client = genai.Client(api_key=self.api_key)

        # Initialize or share components
        print("Initializing RAG Retriever...")
        self.retriever = retriever or SpotifyRAGRetriever()

        print("Initializing Intent Classifier...")
        self.classifier = classifier or IntentClassifier(retriever_model=self.retriever.model)

    def process(self, customer_tweet: str, top_k: int = 3) -> Dict[str, Any]:
        """
        End-to-end processing of an incoming customer message.
        """
        # 1. Classify intent via supervised model
        intent_result = self.classifier.predict(customer_tweet)
        clf_intent = intent_result["intent"]
        clf_conf = intent_result["confidence"]

        # 2. Retrieve historical Spotify resolutions
        retrieved_docs = self.retriever.retrieve(customer_tweet, top_k=top_k)
        context_str = self.retriever.format_context_for_prompt(retrieved_docs)

        # 3. Formulate prompt for Gemini
        prompt = (
            f"INCOMING CUSTOMER TWEET:\n"
            f'"{customer_tweet}"\n\n'
            f"PREDICTED INTENT (Confidence: {clf_conf:.2f}): {clf_intent}\n\n"
            f"{context_str}\n\n"
            f"Based on the customer's issue and Spotify's historical resolutions above, "
            f"provide your structured decision (reasoning, predicted_intent, is_escalation, escalation_reason, draft_reply)."
        )

        def _generate():
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=AGENT_SYSTEM_PROMPT,
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=AgentDecision,
                ),
            )
            data = json.loads(response.text)
            return AgentDecision(**data)

        decision = call_with_retry(_generate)

        return {
            "system": "RAG Support Agent",
            "customer_tweet": customer_tweet,
            "predicted_intent": decision.predicted_intent,
            "classifier_intent": clf_intent,
            "classifier_confidence": clf_conf,
            "is_escalation": decision.is_escalation,
            "escalation_reason": decision.escalation_reason,
            "draft_reply": decision.draft_reply,
            "agent_reasoning": decision.reasoning,
            "retrieved_docs": retrieved_docs,
        }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    agent = SpotifySupportAgent()
    test_tweets = [
        "@SpotifyCares Someone logged into my account from Russia and changed my email! Help!",
        "@SpotifyCares Why is the new lyrics feature not showing on my Android app?",
        "@SpotifyCares I was billed twice for Spotify Family this month. Need refund.",
    ]

    for tweet in test_tweets:
        print("\n" + "=" * 60)
        print(f"Customer: {tweet}")
        res = agent.process(tweet)
        print(f"Intent    : {res['predicted_intent']} (clf: {res['classifier_intent']} @ {res['classifier_confidence']:.2f})")
        print(f"Escalate? : {res['is_escalation']}")
        print(f"Reason    : {res['escalation_reason']}")
        print(f"Draft Reply: {res['draft_reply']}")
