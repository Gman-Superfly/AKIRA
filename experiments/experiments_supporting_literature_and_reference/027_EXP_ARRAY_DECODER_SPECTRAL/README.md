# 027 - Array Decoder with Spectral Bands

Validates spectral band separation using array reconstruction on synthetic signals.

## Key Concepts

- **Array decoder**: Continuous signal reconstruction (not token prediction)
- **Per-position temporal memory**: Each sample sees only its own history
- **Fixed attention, learnable decoder**: Separates "what to look at" from "how to reconstruct"
- **Synthetic signals**: Controlled frequencies to validate band separation

## Quick Start

```bash
cd code
python train_decoder.py
```

## Files

- `027_EXP_ARRAY_DECODER_SPECTRAL.md` - Full experiment documentation
- `code/array_decoder_spectral.py` - Main model implementation
- `code/synthetic_signals.py` - Signal generation utilities
- `code/train_decoder.py` - Training script



*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*