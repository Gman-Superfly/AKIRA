# Observability: Supporting Literature

## Cheap Probing Techniques

**Wu, S., & Yao, Q. (2025).** *Asking LLMs to Verify First is Almost Free Lunch.* arXiv:2511.21734v1.  
📄 Local: `supporting_literature_and_reference/010_EXP_TICKLING_TECHNIQUES/Wu_Yao_2024_Verification_First.pdf`

**Key Findings:**
- Verification requires minimal additional tokens (sometimes none explicit)
- "Almost free lunch" = significant performance gain with negligible compute cost
- Random/trivial candidate answer sufficient to trigger verification
- Iterative verification-generation (Iter-VF) scales efficiently
- **Error correction through verification more efficient than self-reflection**
- Avoids context overflow and error accumulation common in sequential reflection methods

**Relevance to AKIRA:**
- **Validates "tickling the manifold" methodology**: Cheap probes reveal structure
- **Supports free information assets**: Verification uses existing model knowledge
- **Confirms cost-aware observability**: Maximum information, minimum computation
- **Demonstrates probe effectiveness**: Even trivial inputs trigger useful reasoning

**Connection to Theory:**
- Tickling: Perturb system cheaply, observe response, infer structure
- Free Information: Model already has knowledge, just need cheap access method
- Cost hierarchy: Verification << Generation << Full sampling
- Manifold probing: Reverse path reveals complementary structure

---

## Mechanistic Interpretability

*(Space for additional references)*

---

## Free Information Assets

*(Space for additional references)*

