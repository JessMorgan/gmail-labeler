"""HF ``Trainer`` fine-tune over the builder's Training Examples
(ticket #19, ADR-0001: small multi-label encoder, one sigmoid head per
Label Path).

The ONLY module that imports torch/transformers — reached exclusively
through lazy imports, so the importable core and CI stay torch-free
(torch is the optional ``train`` extra). Fully local (AC#6): the
encoder weights come from a LOCAL path (``encoder_path``);
``local_files_only=True`` on every ``from_pretrained`` — nothing is
ever downloaded.

Weighting: the recency weight (GLOSSARY: Recency Weight) is applied as
a WEIGHTED MEAN of per-example BCE losses — per-example BCE comes from
``outputs.logits`` with ``reduction='none'`` (the model's built-in
``outputs.loss`` is a scalar mean, and scalar x weight vector is
mathematically inert: loss x mean(w)).

Device: single-device by contract. CPU runs pass ``use_cpu=True``;
the GPU path REFUSES a multi-GPU host (Trainer would DataParallel
the model and the per-example loss vector would break per replica) —
point ``CUDA_VISIBLE_DEVICES`` at one device.

Memory: tokenization is LAZY — the dataset holds raw texts and the
collator tokenizes per batch, so TOKEN memory is O(batch x max
length), not O(corpus x max length). Raw texts and the multi-hot label
matrix remain O(corpus) by design (the matrix is ~0.18 GiB at 394k
messages x 120 labels — held, not streamed). Serving predict is
BATCHED the same way (``_PREDICT_BATCH_SIZE``): eval is the first bulk
caller (ticket #20), and a single pass over a large test split would
pad every text to the longest one in the split — token memory
O(corpus x max length), the exact hazard the lazy collator exists to
avoid. Rows are concatenated per batch; row i still corresponds to
text i (pinned by the smoke test).

Checkpoint output convention: ``output_dir`` is the plain HF-style
directory (model weights + tokenizer + config); the CLI writes the
JSON Checkpoint Sidecar next to it (``model.SIDECAR_FILENAME``) AFTER
this returns — a directory without a sidecar has no verifiable
identity and the loader refuses it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast

# torch/transformers are the optional `train` extra — absent from the
# CI/local-ci venvs, so the imports are ignore-annotated (the module is
# only ever reached through lazy imports; the smoke/gpu tests that
# exercise it are marker-excluded from those environments)
import torch  # type: ignore[reportMissingImports]
import torch.nn.functional as F  # type: ignore[reportMissingImports]
from torch.utils.data import Dataset  # type: ignore[reportMissingImports]
from transformers import (  # type: ignore[reportMissingImports]
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from gmail_labeler.dataset_build import TrainingExample

_MAX_LENGTH = 512

#: Serving batch size for ``HfModel.predict`` (ticket #20): token
#: memory stays O(batch x max length), not O(corpus x max length), on
#: a bulk caller like eval over the whole test split. 32 is the token
#: budget: ~32 x 512 tokens (the _MAX_LENGTH cap) bounded per batch.
#: Named in one place — the smoke test crosses it at the REAL default
#: (33 texts) and forces a small value to cross batch boundaries.
_PREDICT_BATCH_SIZE = 32


class HfModel:
    """A loaded encoder behind the ``Model`` Protocol (GLOSSARY: Model
    seam): batch predict → per-label sigmoid scores, rows aligned to
    the input texts, columns aligned to ``vocabulary`` (the
    checkpoint's Label-Path vocabulary)."""

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        vocabulary: tuple[str, ...],
        device: str,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.vocabulary = vocabulary
        self._device = device

    def predict(self, texts: Iterable[str]) -> tuple[tuple[float, ...], ...]:
        texts = tuple(texts)  # materialize once (Protocol contract)
        if not texts:
            return ()
        self._model.eval()
        rows: list[tuple[float, ...]] = []
        for start in range(0, len(texts), _PREDICT_BATCH_SIZE):
            batch = texts[start : start + _PREDICT_BATCH_SIZE]
            with torch.no_grad():
                inputs = self._tokenizer(
                    list(batch),
                    truncation=True,
                    max_length=_MAX_LENGTH,
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)
                scores = torch.sigmoid(self._model(**inputs).logits)
            # one host transfer per batch (cpu().tolist()), not one
            # per row: tolist() yields Python floats, so no per-value
            # float() coercion is needed
            rows.extend(tuple(row) for row in scores.cpu().tolist())
        return tuple(rows)


class _WeightedDataset(Dataset):
    """Builder examples as a torch dataset: raw text, one multi-hot
    label row over the vocabulary, and the recency weight. Token
    ids are NOT precomputed — the collator tokenizes per batch, so
    memory stays O(batch), not O(corpus)."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        self._texts = list(texts)
        self._labels = labels
        self._weights = weights

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "text": self._texts[index],
            "labels": self._labels[index],
            "weight": self._weights[index],
        }


class _Collator:
    """Explicit batch collation: per-batch tokenization + padding of
    the texts (the memory fix — never corpus-wide padding) plus the
    label/weight stacking, spelled out on purpose: relying on the
    stock DataCollatorWithPadding to pass through the non-tokenizer
    keys is undocumented behavior, and the only coverage for it is the
    marker-excluded smoke test. Row i of every stacked tensor
    corresponds to feature i (row alignment pinned by test)."""

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self._tokenizer = tokenizer

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        encodings = self._tokenizer(
            [str(feature["text"]) for feature in features],
            truncation=True,
            max_length=_MAX_LENGTH,
            padding=True,
        )
        return {
            "input_ids": torch.tensor(encodings["input_ids"]),
            "attention_mask": torch.tensor(encodings["attention_mask"]),
            "labels": torch.stack(
                [cast(torch.Tensor, feature["labels"]) for feature in features]
            ),
            "weight": torch.stack(
                [cast(torch.Tensor, feature["weight"]) for feature in features]
            ),
        }


class _WeightedTrainer(Trainer):
    """Applies the per-example recency weight to the multi-label loss
    (the weight rides on the Training Example; the split identity does
    not depend on it — dataset_build's doctrine)."""

    # v5: a custom compute_loss returning the micro-batch mean keeps the
    # legacy divide-by-gradient-accumulation behavior (the docs require
    # subclasses to say which scaling they apply).
    loss_is_scaled_for_ga = False

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, object]:
        weight = inputs.pop("weight")
        outputs = model(**inputs)
        # per-example BCE from the LOGITS (reduction='none', mean over
        # the label axis), then the weighted mean (per*w).sum()/w.sum().
        # NOT outputs.loss: that is a scalar (reduction='mean') and
        # scalar x weight vector = loss x mean(w) — the weight would be
        # mathematically inert (and would damp the effective LR by
        # mean(w)).
        per_example = F.binary_cross_entropy_with_logits(
            outputs.logits, inputs["labels"], reduction="none"
        ).mean(dim=-1)
        loss = (per_example * weight).sum() / weight.sum()
        return (loss, outputs) if return_outputs else loss


def train_encoder(
    corpus: Sequence[TrainingExample],
    texts: Sequence[str],
    vocabulary: Sequence[str],
    *,
    encoder_id: str,
    encoder_path: Path,
    output_dir: Path,
    device: str = "cpu",
    max_steps: int = 200,
    batch_size: int = 8,
) -> HfModel:
    """Fine-tune the local encoder on the builder's train examples and
    save the HF-style checkpoint to ``output_dir``.

    - ``corpus``: the TRAIN split's Training Examples (manifest mode —
      the CLI rebuilds through ``build_dataset`` against the Split
      Manifest; test-split ids must not be in here);
    - ``texts``: the train texts ALIGNED to ``corpus`` (record i's text
      at index i — the CLI builds them via
      ``corpus_record.preprocess_text``; train and infer share it);
    - ``vocabulary``: the manifest's label vocabulary (Label Paths) —
      one sigmoid head per entry;
    - ``encoder_path``: LOCAL weights (no download, AC#6);
    - ``encoder_id``: the config's encoder identifier — recorded on
      the saved HF config (the checkpoint is self-describing) and
      stamped into the sidecar by the caller;
    - ``device``: 'cpu' (use_cpu=True) or 'cuda' — a multi-GPU host is
      REFUSED on the GPU path (single-device contract; set
      CUDA_VISIBLE_DEVICES to one device).
    """
    if not corpus:
        raise ValueError("no training examples — nothing to fine-tune")
    if len(texts) != len(corpus):
        raise ValueError(f"texts ({len(texts)}) not aligned to corpus ({len(corpus)})")
    if not encoder_path.is_dir():
        raise ValueError(
            f"encoder path {encoder_path} is not a directory — training "
            "downloads nothing; point model.encoder_path at local weights"
        )
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda" and torch.cuda.device_count() > 1:
        raise ValueError(
            "multi-GPU is not supported — the recipe is single-device "
            "(Trainer would DataParallel and the per-example loss "
            "vector would break per replica); set CUDA_VISIBLE_DEVICES "
            "to one device"
        )
    vocab = tuple(vocabulary)
    if not vocab:
        raise ValueError("empty vocabulary — nothing to fine-tune")

    tokenizer = AutoTokenizer.from_pretrained(str(encoder_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(encoder_path),
        num_labels=len(vocab),
        # v5 literal ("multi_label" is the 4.x spelling the v5 line
        # renamed — the extra is pinned to v5, see pyproject)
        problem_type="multi_label_classification",
        local_files_only=True,
    )
    model.to(device)
    # the checkpoint is self-describing: the encoder identifier rides on
    # the saved HF config (save_pretrained persists it)
    model.config.encoder_id = encoder_id

    # multi-hot over the vocabulary: column j is 1.0 iff vocabulary[j]
    # is one of the example's labels (one tensor, not a list of lists)
    labels = torch.tensor(
        [
            [1.0 if name in example.labels else 0.0 for name in vocab]
            for example in corpus
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([example.weight for example in corpus], dtype=torch.float32)
    dataset = _WeightedDataset(texts, labels, weights)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        report_to=[],
        logging_strategy="no",
        save_strategy="no",
        remove_unused_columns=False,
        # single-device contract: CPU runs stay on CPU (Trainer would
        # otherwise re-place the model on CUDA if it sees one)
        use_cpu=(device == "cpu"),
    )
    trainer = _WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=_Collator(tokenizer),
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return HfModel(model, tokenizer, vocab, device)


def load_hf_model(checkpoint_dir: Path, vocabulary: tuple[str, ...]) -> HfModel:
    """The real-encoder factory for ``model.load_model``: loads the
    HF-style checkpoint directory (local files only — no download) and
    wraps it behind the Model Protocol. The sidecar's refusal contract
    has already run by the time this is called; the head-count check
    here is the sidecar-class drift guard the sidecar cannot see
    (a checkpoint whose num_labels does not match the served
    vocabulary is refused, typed)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir), local_files_only=True
    )
    if model.config.num_labels != len(vocabulary):
        raise ValueError(
            f"checkpoint head count {model.config.num_labels} != "
            f"vocabulary size {len(vocabulary)} — the checkpoint does "
            "not match the vocabulary it is asked to serve"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir), local_files_only=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return HfModel(model, tokenizer, vocabulary, device)
