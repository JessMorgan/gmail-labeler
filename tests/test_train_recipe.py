"""Real-encoder recipe tests — opt-in markers, excluded from CI.

AC#1/AC#2: torch is an optional extra and the CI matrix stays
torch-free. These are the CPU-smoke and GPU tests, behind the opt-in
``smoke``/``gpu`` markers (pyproject addopts excludes both from every
CI/local-ci run). torch/transformers are imported INSIDE the test
functions on purpose: the module must stay COLLECTABLE without torch
(collection imports the module; a top-level import would crash CI's
collect phase), and deselected tests never run.

AC#6: the tiny encoder is built locally in tmp_path (random weights,
no download) — training touches no network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gmail_labeler.dataset_build import TrainingExample

VOCAB = ("Bills", "Shipping")
NOW = datetime(2026, 10, 30, 12, 0, 0, tzinfo=UTC)


def _examples(n: int = 24) -> list[TrainingExample]:
    return [
        TrainingExample(
            id=f"{i:016x}",
            date=NOW,
            labels=(VOCAB[i % 2],),
            weight=1.0,
            date_provenance="internal_date",
        )
        for i in range(n)
    ]


def _texts(examples: list[TrainingExample]) -> list[str]:
    # aligned to `examples` (index i ↔ examples[i])
    return [
        f"message {i} about {example.labels[0].lower()} from a shop"
        for i, example in enumerate(examples)
    ]


def _tiny_encoder(dir: Path, num_labels: int) -> Path:
    """A tiny LOCAL encoder (random weights — nothing is downloaded):
    a 1-layer 32-wide Bert with a 128-token vocab, saved HF-style."""
    from transformers import (  # type: ignore[reportMissingImports]
        AutoModelForSequenceClassification,
        BertConfig,
        BertTokenizerFast,
    )

    dir.mkdir(parents=True)
    (dir / "vocab.txt").write_text("\n".join(f"tok{i}" for i in range(128)))
    tokenizer = BertTokenizerFast(vocab_file=str(dir / "vocab.txt"))
    tokenizer.save_pretrained(dir)
    config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=num_labels,
    )
    AutoModelForSequenceClassification.from_config(config).save_pretrained(dir)
    return dir


def _assert_serving_shape(rows: tuple[tuple[float, ...], ...], n: int) -> None:
    assert len(rows) == n  # one row per input text
    assert all(len(row) == len(VOCAB) for row in rows)  # one score per label
    assert all(0.0 <= v <= 1.0 for row in rows for v in row)  # sigmoid range


@pytest.mark.smoke
def test_cpu_smoke_trains_and_serves(tmp_path: Path) -> None:
    """A few dozen steps on CPU with the tiny local encoder: the
    checkpoint is written HF-style, loads back through the real
    factory, and serves aligned sigmoid scores (AC#2, AC#4's real-
    encoder half)."""
    from train.recipe import load_hf_model, train_encoder

    encoder_dir = _tiny_encoder(tmp_path / "tiny", num_labels=len(VOCAB))
    examples = _examples(24)
    model = train_encoder(
        examples,
        _texts(examples),
        VOCAB,
        encoder_id="tiny-bert",
        encoder_path=encoder_dir,
        output_dir=tmp_path / "ckpt",
        max_steps=24,  # a few dozen steps — smoke, not a real fine-tune
    )
    _assert_serving_shape(model.predict(["a bill arrived", "a package shipped"]), 2)
    # the checkpoint dir is a real HF-style dir: the real factory loads
    # it and the round trip serves the same shape
    loaded = load_hf_model(tmp_path / "ckpt", VOCAB)
    assert loaded.vocabulary == VOCAB
    _assert_serving_shape(loaded.predict(["a bill arrived"]), 1)


@pytest.mark.smoke
def test_weighted_loss_is_order_sensitive(tmp_path: Path) -> None:
    """The recency weight must actually weight: per-example BCE
    weighted mean. A scalar loss x weight vector is inert (loss x
    mean(w)) — swapping the weights MUST change the loss, and the
    value must equal the independently computed weighted mean."""
    import torch  # type: ignore[reportMissingImports]
    import torch.nn.functional as F  # type: ignore[reportMissingImports]
    from train.recipe import _WeightedTrainer
    from transformers import (  # type: ignore[reportMissingImports]
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
    )

    encoder_dir = _tiny_encoder(tmp_path / "tiny", num_labels=len(VOCAB))
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(encoder_dir),
        num_labels=len(VOCAB),
        problem_type="multi_label_classification",
    )
    # use_cpu=True: a Trainer without it re-places the model on CUDA in
    # a CUDA box (the "device decorative" hazard) — this test is about
    # the loss math, not the device
    trainer = _WeightedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "out"), max_steps=1, report_to=[], use_cpu=True
        ),
    )
    enc = tokenizer(
        ["a bill arrived", "a package shipped"],
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
    )
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    inputs_w1 = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "weight": torch.tensor([1.0, 0.1]),
    }
    inputs_w2 = dict(inputs_w1, weight=torch.tensor([0.1, 1.0]))
    loss_w1 = trainer.compute_loss(model, dict(inputs_w1))
    loss_w2 = trainer.compute_loss(model, dict(inputs_w2))
    # order sensitivity: the inert-scalar bug gives loss_w1 == loss_w2
    assert loss_w1 != loss_w2
    # independent worked example: weighted mean of per-example BCE
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        ).logits
        per_example = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        ).mean(dim=-1)
        expected = (per_example * torch.tensor([1.0, 0.1])).sum() / torch.tensor(
            [1.0, 0.1]
        ).sum()
    assert torch.allclose(loss_w1, expected)


@pytest.mark.smoke
def test_collator_aligns_rows_on_mismatched_lengths(tmp_path: Path) -> None:
    """The collator pads per batch and stacks labels/weight with row i
    ↔ feature i — pinned on MISMATCHED text lengths (a row-swap
    regression fails this)."""
    from train.recipe import _Collator
    from transformers import (  # type: ignore[reportMissingImports]
        AutoTokenizer,
    )

    encoder_dir = _tiny_encoder(tmp_path / "tiny", num_labels=len(VOCAB))
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir))
    collator = _Collator(tokenizer)
    import torch  # type: ignore[reportMissingImports]

    features = [
        {
            "text": "short",
            "labels": torch.tensor([1.0, 0.0]),
            "weight": torch.tensor(1.0),
        },
        {
            "text": "a much longer text than the other one, definitely",
            "labels": torch.tensor([0.0, 1.0]),
            "weight": torch.tensor(0.5),
        },
    ]
    batch = collator(features)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    # row alignment: row i carries feature i's label row and weight
    assert batch["labels"].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert batch["weight"].tolist() == [1.0, 0.5]


@pytest.mark.smoke
def test_local_files_only_is_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC#6's tripwire at the wire: every from_pretrained call site
    (2 in train_encoder, 2 in load_hf_model) passes local_files_only —
    a deletion of the flag at ANY site fails this."""
    from train.recipe import load_hf_model, train_encoder
    from transformers import (  # type: ignore[reportMissingImports]
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    seen: list[tuple[str, object]] = []
    orig_model = AutoModelForSequenceClassification.from_pretrained
    orig_tok = AutoTokenizer.from_pretrained

    def spy_model(*args: object, **kwargs: object) -> object:
        seen.append(("model", kwargs.get("local_files_only")))
        return orig_model(*args, **kwargs)  # type: ignore[arg-type]

    def spy_tok(*args: object, **kwargs: object) -> object:
        seen.append(("tokenizer", kwargs.get("local_files_only")))
        return orig_tok(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        AutoModelForSequenceClassification, "from_pretrained", staticmethod(spy_model)
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", staticmethod(spy_tok))

    encoder_dir = _tiny_encoder(tmp_path / "tiny", num_labels=len(VOCAB))
    examples = _examples(4)
    train_encoder(
        examples,
        _texts(examples),
        VOCAB,
        encoder_id="tiny-bert",
        encoder_path=encoder_dir,
        output_dir=tmp_path / "ckpt",
        max_steps=2,
    )
    load_hf_model(tmp_path / "ckpt", VOCAB)
    assert seen == [
        ("tokenizer", True),  # train_encoder
        ("model", True),
        ("model", True),  # load_hf_model
        ("tokenizer", True),
    ]


@pytest.mark.smoke
def test_cpu_smoke_refuses_missing_local_encoder(tmp_path: Path) -> None:
    # AC#6's tripwire: no local weights → loud refusal, no download
    from train.recipe import train_encoder

    examples = _examples(4)
    with pytest.raises(ValueError, match="downloads nothing"):
        train_encoder(
            examples,
            _texts(examples),
            VOCAB,
            encoder_id="tiny-bert",
            encoder_path=tmp_path / "nope",
            output_dir=tmp_path / "ckpt",
        )


@pytest.mark.gpu
def test_gpu_smoke_trains(tmp_path: Path) -> None:
    """The GPU test: same recipe on CUDA (AC#2). Skips on boxes
    without a device — the marker is opt-in on the GPU box."""
    import torch  # type: ignore[reportMissingImports]

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this box")
    from train.recipe import train_encoder

    if torch.cuda.device_count() > 1:
        # pin the single-device refusal before pinning the device: a
        # multi-GPU call must fail loudly, not silently DataParallel.
        # non-empty corpus so the empty-corpus guard doesn't fire first
        probe = TrainingExample(
            id="0000000000000001",
            date=NOW,
            labels=("A",),
            weight=1.0,
            date_provenance="internal_date",
        )
        with pytest.raises(ValueError, match="multi-GPU is not supported"):
            encoder_dir = tmp_path / "encoder"
            encoder_dir.mkdir()  # passes the local-weights guard; the
            # multi-GPU refusal fires before any model load
            train_encoder(
                (probe,),
                ("x",),
                ("A",),
                encoder_id="refusal-probe",
                encoder_path=encoder_dir,
                output_dir=tmp_path / "refuse",
                device="cuda",
            )
        pytest.skip(
            "GPU smoke needs a single visible device "
            "(CUDA_VISIBLE_DEVICES=0); the refusal is pinned above"
        )

    encoder_dir = _tiny_encoder(tmp_path / "tiny", num_labels=len(VOCAB))
    examples = _examples(8)
    model = train_encoder(
        examples,
        _texts(examples),
        VOCAB,
        encoder_id="tiny-bert",
        encoder_path=encoder_dir,
        output_dir=tmp_path / "ckpt",
        device="cuda",
        max_steps=8,
    )
    _assert_serving_shape(model.predict(["a bill arrived"]), 1)
