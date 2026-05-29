"""
Build a curriculum-ordered SFT dataset.

SFTTrainer doesn't natively support curriculum, so we pre-sort the dataset
by difficulty_score ascending before passing it to the trainer.

Epoch 1: easy (score < 0.33)
Epoch 2: mixed (shuffled)
Epoch 3: hard (score > 0.66)

In practice this is done by concatenating the splits in order and setting
num_train_epochs=1 per pass, or by using a custom data_collator that cycles.
The simpler approach: sort once and let the trainer iterate linearly.
"""

from __future__ import annotations

from datasets import Dataset, concatenate_datasets


def build_curriculum(dataset: Dataset) -> Dataset:
    """Return dataset sorted ascending by difficulty_score."""
    if "difficulty_score" not in dataset.column_names:
        return dataset

    sorted_ds = dataset.sort("difficulty_score")
    return sorted_ds


def split_by_difficulty(dataset: Dataset) -> tuple[Dataset, Dataset, Dataset]:
    """Return (easy, medium, hard) splits for explicit curriculum scheduling."""
    easy = dataset.filter(lambda x: x["difficulty_score"] < 0.33)
    medium = dataset.filter(lambda x: 0.33 <= x["difficulty_score"] <= 0.66)
    hard = dataset.filter(lambda x: x["difficulty_score"] > 0.66)
    return easy, medium, hard
