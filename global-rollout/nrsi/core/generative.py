"""NRS Native Generative Adapter.

Wraps the GenerativeEngine with NRSI cognitive primitives so that
knowledge retrieval and answer composition use the native NRS reasoning
pipeline (decompose → intent match → compose) rather than raw template
interpolation alone.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NRSIGenerativeAdapter:
    """Enriches GenerativeEngine input via NRSI cognitive primitives.

    Sits between the NRS pipeline and the GenerativeEngine, adding
    decomposition, intent matching, and compositional synthesis to the
    knowledge_facts before the engine assembles the final response.
    """

    def __init__(self, engine):
        self._engine = engine
        self._decompose = None
        self._intent_match = None
        self._compose = None
        self._semantic_distance = None
        self._load_primitives()

    def _load_primitives(self) -> None:
        try:
            from nrsi.lang.cognitive_primitives import (
                nrsi_decompose,
                nrsi_intent_match,
                nrsi_compose,
                nrsi_semantic_distance,
            )
            self._decompose = nrsi_decompose
            self._intent_match = nrsi_intent_match
            self._compose = nrsi_compose
            self._semantic_distance = nrsi_semantic_distance
        except Exception as e:
            logger.debug("NRSI cognitive primitives unavailable: %s", e)

    def enrich_knowledge(
        self,
        query: str,
        knowledge_facts: List[str],
        domain: str = "",
    ) -> List[str]:
        """Use NRSI primitives to decompose, rank, and compose facts."""
        enriched = list(knowledge_facts)

        if self._decompose:
            try:
                decomp = self._decompose(query)
                if hasattr(decomp, "sub_goals") and decomp.sub_goals:
                    for sg in decomp.sub_goals[:3]:
                        sg_text = getattr(sg, "description", str(sg))
                        if sg_text and len(sg_text) > 10 and not sg_text.startswith("["):
                            enriched.append(sg_text)
            except Exception as e:
                logger.debug("nrsi_decompose skipped: %s", e)

        if self._intent_match:
            try:
                self._intent_match(query)
            except Exception as e:
                logger.debug("nrsi_intent_match skipped: %s", e)

        if self._compose and knowledge_facts:
            try:
                strategy = "synthesis"
                if domain in ("mathematics", "logic"):
                    strategy = "analytical"
                elif domain in ("creative", "philosophy"):
                    strategy = "narrative"
                elif domain in ("science", "medical"):
                    strategy = "hierarchical"

                composed = self._compose(
                    knowledge_facts[:25],
                    strategy=strategy,
                )
                if hasattr(composed, "text") and composed.text and len(composed.text) > 20:
                    enriched.insert(0, composed.text)
            except Exception as e:
                logger.debug("nrsi_compose skipped: %s", e)

        if self._semantic_distance and knowledge_facts:
            try:
                ranked = []
                for fact in knowledge_facts:
                    dist = self._semantic_distance(query, fact)
                    score = getattr(dist, "similarity", 0.5) if dist else 0.5
                    ranked.append((score, fact))
                ranked.sort(key=lambda x: x[0], reverse=True)
                reranked = [f for _, f in ranked]
                base_enrichments = [f for f in enriched if f not in knowledge_facts]
                enriched = reranked + base_enrichments
            except Exception as e:
                logger.debug("nrsi_semantic_distance ranking skipped: %s", e)

        seen: set = set()
        deduped: List[str] = []
        for f in enriched:
            key = f[:100].lower().strip().rstrip(".,;:!")
            if key not in seen:
                seen.add(key)
                deduped.append(f)
        return deduped[:15]

    def chat(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        lobe_facts: Optional[List[str]] = None,
        knowledge_facts: Optional[List[str]] = None,
        tone_directive=None,
    ) -> str:
        """Enrich facts via NRSI primitives, then delegate to engine."""
        kf = list(knowledge_facts or [])
        domain = ""
        if hasattr(self._engine, "synthesizer") and hasattr(self._engine.synthesizer, "_detect_domain"):
            try:
                domain = self._engine.synthesizer._detect_domain(message)
            except Exception:
                pass

        enriched = self.enrich_knowledge(message, kf, domain=domain)
        return self._engine.chat(
            message=message,
            conversation_history=conversation_history,
            lobe_facts=lobe_facts,
            knowledge_facts=enriched if enriched else None,
            tone_directive=tone_directive,
        )

    def generate_followup_options(
        self,
        query: str,
        answer: str,
        domain: str = "",
        **kwargs,
    ) -> List[str]:
        """Delegate follow-up generation to the engine."""
        return self._engine.generate_followup_options(
            query=query, answer=answer, domain=domain, **kwargs,
        )

    def __getattr__(self, name):
        """Forward any other attribute access to the underlying engine."""
        return getattr(self._engine, name)
