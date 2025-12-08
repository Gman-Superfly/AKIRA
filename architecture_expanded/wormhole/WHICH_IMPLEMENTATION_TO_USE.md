## Which Implementation Should I Use?

### Use Coherence-Gated Wormholes If:
- ✓ You want theory-aligned implementation
- ✓ You need adaptive gating based on belief distribution structure
- ✓ You're running experiments to validate BEC-inspired predictions
- ✓ You care about entropy dynamics and collapse detection
- ✓ You're implementing the full Spectral Belief Machine

**Files to use:**
- `architecture_base/attention/spectral_wormhole/SPECTRAL_WORMHOLE_ATTENTION.md`
- `architecture_base/attention/ATTENTION_STACK.md`
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` (updated)

---

### Use Energy-Gated Wormholes If:
- ✓ You want quick prototype/baseline
- ✓ You need simplest possible implementation
- ✓ You're comparing against theory-aligned version
- ✓ Computational speed is critical (no entropy calculation)
- ✓ You're testing if wormhole structure alone helps (without theory)

**Files to use:**
- `architecture_expanded/wormhole/WORMHOLE_HYBRID.md`

**Caveat:** This is not theory-aligned and may miss key dynamics.