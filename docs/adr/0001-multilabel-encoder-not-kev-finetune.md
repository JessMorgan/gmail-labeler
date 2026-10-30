# Multi-label encoder, not a Kev fine-tune

This project grew out of auto-github's Kev classifier, but Gmail labeling is a
multi-label task (one message, several concurrent labels, ~40+ candidate
labels) while Kev/TypeSafe is inherently single-choice (one answer token's
probability distribution). We decided to use a small encoder model
(DeBERTa-v3/ModernBERT class, ~100–300M params) with per-label sigmoid heads
instead, accepting that the training pipeline shares only the *pattern* with
the kev toolchain, not the library. Multi-label via 40 binary Kev queries per
message or generative set-output were both rejected on latency and
calibration grounds; the encoder also survives the long-tail labels (dozens of
examples) far better than a 4B LoRA fine-tune.
