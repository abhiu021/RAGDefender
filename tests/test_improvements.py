import numpy as np

from ragdefender.core import defender as defender_mod
from ragdefender.core.defender import ImprovedRAGDefender, hdbscan_group
from ragdefender.filtering.mad_threshold import compute_suspicion_scores, dynamic_threshold
from ragdefender.filtering.nli_filter import nli_contradiction_filter


class DummyEncoder:
    def encode(self, texts, convert_to_tensor=False):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            if 'not' in t.lower() or 'berlin' in t.lower():
                vecs.append([0.0, 1.0])
            elif 'france' in t.lower() or 'paris' in t.lower() or 'capital' in t.lower():
                vecs.append([1.0, 0.0])
            else:
                vecs.append([0.5, 0.5])
        arr = np.array(vecs, dtype=float)
        return arr


class DummyNLI:
    def predict(self, pairs):
        out = []
        for a, b in pairs:
            contradiction = 0.9 if ('paris' in a.lower() and 'berlin' in b.lower()) or ('berlin' in a.lower() and 'paris' in b.lower()) else 0.1
            out.append([contradiction, 0.05, 1 - contradiction - 0.05])
        return np.array(out)


def test_hdbscan_flags_dense_cluster_identical_docs():
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    clean, adv = hdbscan_group(emb, threshold=0.0)
    assert len(adv) >= 2
    assert len(clean) + len(adv) == 4


def test_dynamic_threshold_adapts_between_poison_ratios():
    low_poison = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
    high_poison = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0.9, 0.1]])
    thr_low = dynamic_threshold(compute_suspicion_scores(low_poison))
    thr_high = dynamic_threshold(compute_suspicion_scores(high_poison))
    assert thr_high >= thr_low or np.isclose(thr_high, 0.85)


def test_nli_filter_catches_contradiction_pair():
    docs = ['Paris is the capital of France.', 'Berlin is the capital of France.', 'France is in Europe.']
    out = nli_contradiction_filter('capital of france', docs, DummyEncoder(), DummyNLI(), contradiction_threshold=0.65)
    assert len(out) < len(docs)


def test_full_pipeline_runs_end_to_end_small_example(monkeypatch):
    monkeypatch.setattr(defender_mod, 'SentenceTransformer', lambda *_args, **_kwargs: DummyEncoder())
    monkeypatch.setattr(defender_mod, 'CrossEncoder', lambda *_args, **_kwargs: DummyNLI())
    d = ImprovedRAGDefender(device='cpu')
    docs = ['Paris is the capital of France.', 'Berlin is the capital of France.', 'France is in Europe.']
    out = d.defend('capital of france', docs, mode='singlehop')
    assert isinstance(out, list)


def test_fallback_when_optional_dependencies_missing(monkeypatch):
    monkeypatch.setattr(defender_mod, 'SentenceTransformer', lambda *_args, **_kwargs: DummyEncoder())
    monkeypatch.setattr(defender_mod, 'CrossEncoder', lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('x')))
    d = defender_mod.RAGDefender(device='cpu', use_nli=True)
    assert d.nli_model is None
