"""train — the fine-tune recipe (ticket #19).

The ONLY code in the project that imports torch/transformers. It is
reached exclusively through lazy imports (the CLI's model factory,
``model.load_model``'s caller-supplied factory, the opt-in smoke/gpu
tests), so the importable core — and the CI matrix — stay torch-free
(torch is the optional ``train`` extra; AC#1).

Everything here is fully local (AC#6): the encoder weights come from a
LOCAL path supplied by the config (``model.encoder_path``) — nothing is
ever downloaded, no Hugging Face Hub call, no network of any kind.
Training touches no network outside the ``GmailClient`` seam (which
this module does not use at all).
"""
