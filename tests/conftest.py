"""Shared fixtures. Every test runs on the committed tiny fixture corpus."""

from __future__ import annotations

import numpy as np
import pytest

from spoofline.config import FIXTURE_DIR, profile
from spoofline.data.dataset import load_corpus, make_splits
from spoofline.data.sources import NpzCorpusSource
from spoofline.data.synth import audio_envelope, make_identity, synthesize_audio, synthesize_video
from spoofline.seeding import rng

SEED = 4242


@pytest.fixture(scope="session")
def config():
    return profile("tiny")


@pytest.fixture(scope="session")
def source():
    return NpzCorpusSource(FIXTURE_DIR)


@pytest.fixture(scope="session")
def corpus(source):
    return load_corpus(source)


@pytest.fixture(scope="session")
def splits(source, config):
    return make_splits(
        source.records,
        config.unseen_families,
        config.seed,
        config.train_fraction,
        config.calib_fraction,
    )


@pytest.fixture(scope="session")
def bona_clip():
    """A bona fide audio and video pair, rendered directly rather than loaded."""
    identity = make_identity(SEED, 0)
    audio = synthesize_audio(identity, SEED, "fixture", 16000, 32000)
    frames = synthesize_video(identity, SEED, "fixture", 16, 64, audio_envelope(audio, 16))
    return audio, frames


@pytest.fixture(scope="session")
def donor_clip():
    identity = make_identity(SEED, 3)
    audio = synthesize_audio(identity, SEED, "donor", 16000, 32000)
    frames = synthesize_video(identity, SEED, "donor", 16, 64, audio_envelope(audio, 16))
    return audio, frames


@pytest.fixture
def attack_rng():
    def factory(label: str) -> np.random.Generator:
        return rng(SEED, label)

    return factory
