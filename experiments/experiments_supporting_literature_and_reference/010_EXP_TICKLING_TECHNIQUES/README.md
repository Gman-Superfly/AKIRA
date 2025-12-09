# Experiment 010: Tickling Techniques - Supporting Literature

## Primary References

**Wu, S., & Yao, Q. (2025).** *Asking LLMs to Verify First is Almost Free Lunch.* arXiv:2511.21734v1.  
📄 `Wu_Yao_2024_Verification_First.pdf`

### Why This Paper Supports EXP_010

**Core Finding:** Verification-first prompting achieves significant performance improvements with minimal computational overhead, often fewer tokens than standard reasoning.

**Key Evidence:**
1. **Verification is cognitively easier than generation** (Baker et al., 1975)
   - Aligns with "cheap probes reveal expensive structure"
   - Validates tickling methodology: minimal input → maximum information

2. **"Almost Free Lunch"** = high information gain, low cost
   - Directly supports cost hierarchy in observability
   - Confirms: smart probing beats brute force sampling

3. **Reverse reasoning provides complementary information**
   - Different from forward CoT, not redundant
   - Validates multi-perspective probing strategy

4. **Random/trivial candidate sufficient**
   - Don't need high-quality probe
   - Supports "tickle with anything" principle

5. **Error Correction Through Verification**
   - Verification process naturally detects and corrects errors
   - Avoids context overflow and error accumulation of standard self-correction
   - Markovian refinement: each iteration uses only previous answer, not full history
   - Reduces egocentric bias by critiquing external answers

### Experimental Validation

**From Paper:**
- Tested across: Math (GSM8K, MATH), Coding (HumanEval, MBPP), Agentic (API-Bank), Graduate-level QA (GPQA)
- Models: 1B to GPT-4, including commercial thought-hidden models
- Consistent improvements with minimal token overhead

**Connection to AKIRA Experiment:**
- EXP_010 tests: cheap perturbations (temperature tweaks, noise injection) reveal manifold structure
- Wu & Yao show: cheap verification reveals reasoning structure
- Same principle, different domains

### Theoretical Connections

**To AKIRA Theory:**
1. **POMDP Framework:** Verification as belief update mechanism
2. **The Old Lady:** Learning from observation cheaper than trial-and-error
3. **Free Information Assets:** Model already has knowledge, just need access method
4. **Tickling Manifold:** Perturb cheaply, observe response, infer structure
5. **Error Detection:** Verification reveals inconsistencies in belief state
6. **Iterative Refinement:** Iter-VF avoids error accumulation of standard reflection

### Implementation Implications

**For AKIRA:**
1. Design probes that trigger verification-like processes
2. Use reverse paths (belief → observation) as well as forward (observation → belief)
3. Leverage "almost free" techniques before expensive ones
4. Random perturbations may be sufficient (don't over-engineer)
5. **Error correction through verification cheaper than self-reflection**
6. **Markovian iteration avoids context overflow** (use previous answer, not full history)

### Citation

Wu, S., & Yao, Q. (2025). Asking LLMs to Verify First is Almost Free Lunch. *arXiv preprint arXiv:2511.21734v1.*

---

## Additional References

*(Space for future papers on tickling/cheap probing techniques)*

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*

