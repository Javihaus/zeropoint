"""The 1.0 model files: they load, their hashes are checked, and the order of the options never matters."""
import itertools
import os
import random
import shutil

import numpy as np
import pytest

from zeropoint._runtime import MODEL_DIR, ModelFilesError, Runtime, canonical_order


@pytest.fixture(scope="module")
def rt():
    return Runtime(threads=1)


def decide(rt, text, options):
    canon = [options[i] for i in canonical_order(options)]
    d, _ = rt.depths(text, canon)
    p = rt.distribution(d)
    return {o: p[canon.index(o)] for o in options}, p[-1]


def test_manifest_hashes_are_checked(tmp_path):
    copy = tmp_path / "model"
    shutil.copytree(MODEL_DIR, copy)
    Runtime(str(copy), threads=1)
    with open(copy / "head.json", "r+b") as f:
        data = bytearray(f.read())
        data[-2] = (data[-2] + 1) % 256
        f.seek(0)
        f.write(data)
    with pytest.raises(ModelFilesError):
        Runtime(str(copy), threads=1)


def test_missing_file_is_refused(tmp_path):
    copy = tmp_path / "model"
    shutil.copytree(MODEL_DIR, copy)
    os.remove(copy / "vocab_scale.npy")
    with pytest.raises(ModelFilesError):
        Runtime(str(copy), threads=1)


def test_empty_and_duplicate_options_are_refused():
    with pytest.raises(ValueError):
        canonical_order(["book a flight", " "])
    with pytest.raises(ValueError):
        canonical_order(["café", "café"])          # same text once normalised to NFC


def test_every_order_gives_the_same_bits(rt):
    text = "Can you move my dentist appointment to Friday afternoon?"
    options = ["cancel an appointment", "move an appointment", "book a table", "check the weather"]
    ref_p, ref_none = decide(rt, text, options)
    for perm in itertools.permutations(options):
        p, none = decide(rt, text, list(perm))
        assert none == ref_none
        assert all(p[o] == ref_p[o] for o in options)


def test_long_list_in_blocks_is_order_free(rt):
    rnd = random.Random(7)
    options = [f"option number {i}" for i in range(45)] + ["transfer money to another account"]
    text = "send 50 euros to my sister"
    ref_p, ref_none = decide(rt, text, options)
    for _ in range(5):
        perm = options[:]
        rnd.shuffle(perm)
        p, none = decide(rt, text, perm)
        assert none == ref_none and all(p[o] == ref_p[o] for o in options)


def test_repeat_gives_the_same_bits(rt):
    text = "Mi tarjeta no funciona en el cajero"
    options = ["bloquear una tarjeta", "consultar el saldo", "pedir una tarjeta nueva"]
    a = decide(rt, text, options)
    b = decide(rt, text, options)
    assert a == b


def test_probabilities_sum_to_one(rt):
    canon = sorted(["play music", "set an alarm", "turn the lights on"])
    d, _ = rt.depths("wake me up at seven", canon)
    p = rt.distribution(d)
    assert abs(p.sum() - 1.0) < 1e-12 and np.all(p >= 0)
