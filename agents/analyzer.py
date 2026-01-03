"""
Analyzer Agent - Analyzes research findings with scientific integrity.

CRITICAL: Uses negative constraints to prevent fabricating contradictions.

Responsibilities:
- Extract key findings from FAISS index
- Detect TRUE contradictions (not minor differences)
- Identify complementary findings
- Find consensus points and research gaps
"""

from typing import Dict, List
from loguru import logger

from graph.state import ResearchState, Analysis
from memory.vector_store import FAISSVectorStore
from langchain_ollama import ChatOllama
from config import settings


# CRITICAL: Negative Constraint Prompt
CONTRADICTION_VERIFICATION_PROMPT = """You are analyzing research papers for factual contradictions.

CRITICAL INSTRUCTION - Negative Constraint:
- If you cannot find a DIRECT, CLEAR contradiction between two papers, do NOT invent one
- Do NOT report minor differences in methodology as contradictions
- Do NOT report complementary findings (same topic, different aspects) as contradictions
- Only report TRUE contradictions where papers make mutually exclusive claims about the same phenomenon

Compare these two research findings:

Finding 1: {text1}
Source: {citation1}

Finding 2: {text2}
Source: {citation2}

Answer with EXACTLY one of these categories:
- CONTRADICTION: The findings make mutually exclusive claims (explain why)
- COMPLEMENTARY: The findings address different aspects of the same topic
- UNRELATED: The findings are on different topics
- INSUFFICIENT: Not enough context to determine relationship

Format:
CATEGORY: [your choice]
EXPLANATION: [1-2 sentences]
"""


def extract_explanation(response_text: str) -> str:
    """Extract explanation from LLM response."""
    lines = response_text.strip().split('\n')
    for line in lines:
        if line.startswith('EXPLANATION:'):
            return line.replace('EXPLANATION:', '').strip()
    return response_text[:200]  # Fallback


def analyzer_node(state: ResearchState) -> Dict:
    """
    Analyzer agent node for LangGraph.

    Args:
        state: Current research state

    Returns:
        State updates (analysis)
    """
    job_id = state["job_id"]
    logger.info(f"\n═══ Analyzer Agent Starting ═══")
    logger.info(f"Job ID: {job_id}")

    if not state.get("faiss_index_path"):
        logger.error("❌ No FAISS index available - cannot analyze")
        return {
            "analysis": {
                "key_findings": [],
                "contradictions": [],
                "complementary_findings": [],
                "trends": [],
                "consensus_points": [],
                "gaps": []
            },
            "processing_stage": "synthesizing"
        }

    # Initialize tools
    vector_store = FAISSVectorStore()
    vector_store.load_index(state["faiss_index_path"])

    llm = ChatOllama(
        model=settings.OLLAMA_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        base_url=settings.OLLAMA_BASE_URL
    )

    logger.info(f"Loaded FAISS index with {len(vector_store.metadata)} chunks")

    # Extract key findings
    logger.info(f"\n📊 Extracting key findings...")
    key_findings = _extract_key_findings(vector_store, llm)
    logger.info(f"  ✓ Found {len(key_findings)} key findings")

    # Detect contradictions WITH NEGATIVE CONSTRAINT
    logger.info(f"\n🔍 Detecting contradictions (with negative constraint)...")
    contradictions, complementary_findings = _detect_contradictions(vector_store, llm)
    logger.info(f"  ✓ Contradictions: {len(contradictions)}")
    logger.info(f"  ✓ Complementary findings: {len(complementary_findings)}")

    # Identify trends
    logger.info(f"\n📈 Identifying trends...")
    trends = _identify_trends(vector_store, llm)
    logger.info(f"  ✓ Found {len(trends)} trends")

    # Find consensus points
    logger.info(f"\n🤝 Finding consensus points...")
    consensus_points = _find_consensus(vector_store, llm)
    logger.info(f"  ✓ Found {len(consensus_points)} consensus points")

    # Detect research gaps
    logger.info(f"\n🔬 Detecting research gaps...")
    gaps = _detect_gaps(vector_store, llm)
    logger.info(f"  ✓ Found {len(gaps)} research gaps")

    # Build analysis
    analysis: Analysis = {
        "key_findings": key_findings,
        "contradictions": contradictions,
        "complementary_findings": complementary_findings,
        "trends": trends,
        "consensus_points": consensus_points,
        "gaps": gaps
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzer Agent Summary:")
    logger.info(f"  Key findings: {len(key_findings)}")
    logger.info(f"  Contradictions: {len(contradictions)}")
    logger.info(f"  Complementary: {len(complementary_findings)}")
    logger.info(f"  Trends: {len(trends)}")
    logger.info(f"  Consensus: {len(consensus_points)}")
    logger.info(f"  Gaps: {len(gaps)}")
    logger.info(f"{'='*60}\n")

    return {
        "analysis": analysis,
        "processing_stage": "synthesizing"
    }


def _extract_key_findings(vector_store: FAISSVectorStore, llm: ChatOllama) -> List[Dict]:
    """Extract key findings from research papers."""
    findings = []

    # Get diverse chunks - use similarity search to get SUBSTANTIVE content
    # This avoids headers/metadata chunks and finds actual research content
    # CRITICAL: Include queries for REVIEW PAPERS and CONCEPTUAL FRAMEWORKS
    key_queries = [
        "main findings results conclusions",
        "we demonstrate show report",
        "our results indicate suggest",
        "we observed measured found",
        "experimental results data analysis",
        "discussion implications significant",
        "novel approach method technique",
        # NEW: For review papers and conceptual frameworks (like Ahmad et al.)
        "challenges limitations trade-offs",
        "key insights critical problem",
        "framework approach strategy",
        "potential solutions opportunities"
    ]

    substantive_chunks = []
    papers_seen = set()
    paper_chunk_counts = {}  # Track how many chunks per paper

    for query in key_queries:
        results = vector_store.similarity_search(query, k=50)  # Fetch 50 candidates for diversity

        for text, metadata in results:
            arxiv_id = metadata["arxiv_id"]

            # Skip very short chunks (likely headers)
            if len(text) < 200:
                continue

            # DIVERSITY ENFORCEMENT: Limit chunks per paper to ensure we sample ALL papers
            # This simulates MMR by preventing one paper from dominating results
            current_count = paper_chunk_counts.get(arxiv_id, 0)
            if current_count >= 3:  # Max 3 chunks per paper per query
                continue

            # Skip if text is too similar to already collected chunks (basic deduplication)
            is_duplicate = False
            for existing_chunk in substantive_chunks:
                # Simple similarity check: if first 100 chars match, skip
                if existing_chunk["text"][:100] == text[:100]:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            substantive_chunks.append({
                "text": text,
                "citation": metadata["citation"],
                "arxiv_id": arxiv_id
            })

            # Track this paper
            if arxiv_id not in papers_seen:
                papers_seen.add(arxiv_id)
            paper_chunk_counts[arxiv_id] = current_count + 1

            # Stop once we have enough diverse chunks
            if len(substantive_chunks) >= 20:  # Collect more chunks for better coverage
                break

        if len(substantive_chunks) >= 20:
            break

    # Log diversity metrics
    logger.info(f"Collected {len(substantive_chunks)} substantive chunks from {len(papers_seen)} papers")
    for arxiv_id, count in paper_chunk_counts.items():
        logger.info(f"  - {arxiv_id}: {count} chunks")

    # Extract findings from substantive chunks
    for chunk in substantive_chunks[:10]:  # Increased from 5 to 10 findings
        prompt = f"""Extract the main research finding from this text. Be concise (1-2 sentences).

CRITICAL INSTRUCTIONS:
- If the text contains a specific research result, measurement, or observation, extract it
- If the text is only discussion, limitations, or meta-commentary, respond with: "SKIP - no concrete finding"
- DO NOT respond conversationally (no "I'd be happy to help" or "please provide text")
- DO NOT make up findings if none are present

Text: {chunk['text'][:800]}

Finding:"""

        try:
            response = llm.invoke(prompt)
            finding_text = response.content.strip()

            # Skip non-substantive findings
            if finding_text.startswith("SKIP") or "no concrete finding" in finding_text.lower():
                logger.info(f"Skipping non-substantive chunk from {chunk['arxiv_id']}")
                continue

            # Skip conversational responses
            if any(phrase in finding_text.lower() for phrase in [
                "i'd be happy", "please provide", "it seems", "unfortunately",
                "you forgot", "didn't provide", "no text provided"
            ]):
                logger.info(f"Skipping conversational response from {chunk['arxiv_id']}")
                continue

            logger.info(f"Extracted finding from {chunk['arxiv_id']}: {finding_text[:100]}...")

            findings.append({
                "finding": finding_text,
                "citation": chunk["citation"],
                "arxiv_id": chunk["arxiv_id"]
            })
        except Exception as e:
            logger.warning(f"Failed to extract finding from {chunk['arxiv_id']}: {e}")

    return findings


def _detect_contradictions(
    vector_store: FAISSVectorStore,
    llm: ChatOllama
) -> tuple[List[Dict], List[Dict]]:
    """
    Detect contradictions using NEGATIVE CONSTRAINT.

    Returns:
        (contradictions, complementary_findings)
    """
    contradictions = []
    complementary_findings = []

    # Search for potentially conflicting content
    conflict_queries = [
        "conflicting results",
        "opposing findings",
        "different conclusions"
    ]

    potential_conflicts = []
    for query in conflict_queries:
        results = vector_store.similarity_search(query, k=3)
        potential_conflicts.extend(results)

    # Deduplicate by arxiv_id pairs
    seen_pairs = set()

    for i, (text1, meta1) in enumerate(potential_conflicts):
        for text2, meta2 in potential_conflicts[i+1:]:
            # Skip same paper
            if meta1["arxiv_id"] == meta2["arxiv_id"]:
                continue

            # Skip if we've seen this pair
            pair_key = tuple(sorted([meta1["arxiv_id"], meta2["arxiv_id"]]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Limit comparisons (expensive LLM calls)
            if len(seen_pairs) >= 5:
                break

            # Use NEGATIVE CONSTRAINT prompt
            prompt = CONTRADICTION_VERIFICATION_PROMPT.format(
                text1=text1[:300],
                citation1=meta1["citation"],
                text2=text2[:300],
                citation2=meta2["citation"]
            )

            try:
                response = llm.invoke(prompt)
                response_text = response.content

                # Parse response
                if "CATEGORY: CONTRADICTION" in response_text:
                    contradictions.append({
                        "finding_1": text1[:200],
                        "finding_2": text2[:200],
                        "citation_1": meta1["citation"],
                        "citation_2": meta2["citation"],
                        "explanation": extract_explanation(response_text)
                    })
                    logger.info(f"    Found contradiction: {meta1['arxiv_id']} vs {meta2['arxiv_id']}")

                elif "CATEGORY: COMPLEMENTARY" in response_text:
                    complementary_findings.append({
                        "finding_1": text1[:200],
                        "finding_2": text2[:200],
                        "citation_1": meta1["citation"],
                        "citation_2": meta2["citation"],
                        "relationship": extract_explanation(response_text)
                    })

            except Exception as e:
                logger.warning(f"Failed to verify contradiction: {e}")

    return contradictions, complementary_findings


def _identify_trends(vector_store: FAISSVectorStore, llm: ChatOllama) -> List[str]:
    """Identify emerging trends across papers."""
    # Search for methodology and results patterns
    trend_results = vector_store.similarity_search("emerging trends methodology results", k=5)

    prompt = f"""Based on these research excerpts, identify 2-3 emerging trends in the field.

Excerpts:
{chr(10).join([f'- {text[:150]}...' for text, _ in trend_results[:3]])}

Trends (bullet points):"""

    try:
        response = llm.invoke(prompt)
        trends = [line.strip('- ').strip() for line in response.content.split('\n') if line.strip().startswith('-')]
        return trends[:3]
    except Exception as e:
        logger.warning(f"Failed to identify trends: {e}")
        return []


def _find_consensus(vector_store: FAISSVectorStore, llm: ChatOllama) -> List[str]:
    """Find consensus points across papers."""
    consensus_results = vector_store.similarity_search("widely agreed established consensus", k=5)

    prompt = f"""Based on these research excerpts, identify 2-3 points of scientific consensus.

Excerpts:
{chr(10).join([f'- {text[:150]}...' for text, _ in consensus_results[:3]])}

Consensus points (bullet points):"""

    try:
        response = llm.invoke(prompt)
        points = [line.strip('- ').strip() for line in response.content.split('\n') if line.strip().startswith('-')]
        return points[:3]
    except Exception as e:
        logger.warning(f"Failed to find consensus: {e}")
        return []


def _detect_gaps(vector_store: FAISSVectorStore, llm: ChatOllama) -> List[str]:
    """Detect research gaps."""
    gap_results = vector_store.similarity_search("future work limitations gaps", k=5)

    prompt = f"""Based on these research excerpts, identify 2-3 research gaps or areas needing further investigation.

Excerpts:
{chr(10).join([f'- {text[:150]}...' for text, _ in gap_results[:3]])}

Research gaps (bullet points):"""

    try:
        response = llm.invoke(prompt)
        gaps = [line.strip('- ').strip() for line in response.content.split('\n') if line.strip().startswith('-')]
        return gaps[:3]
    except Exception as e:
        logger.warning(f"Failed to detect gaps: {e}")
        return []
