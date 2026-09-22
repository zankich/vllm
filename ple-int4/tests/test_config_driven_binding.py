"""Config-driven binding: the pinned-host global follows the PLE dtype.

The incident (2026-09-21, infra FP8 arm): the plugin rebound
Qwen4ExpPLEPinnedHostEmbedding to the int4 subclass unconditionally at
install, so any non-int4 config inherited the int4 lookup and died in
profile_run behind a prefetch assertion naming neither plugin nor dtype.
from_quant_config sees the checkpoint's embedding dtype BEFORE the
construction site resolves the global, so the binding decision rides
there: int4 marker binds the subclass, anything else restores stock.
"""

import pytest


@pytest.fixture
def plugin_installed():
    import vllm.models.qwen4_exp.nvidia.ngram_embedding as ng

    from ple_int4 import install, uninstall

    uninstall()
    install()
    yield ng
    uninstall()


def test_fp8_dtype_leaves_stock_binding(plugin_installed):
    ng = plugin_installed
    ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config(
        None, "prefix", "float8_e4m3fn"
    )
    from ple_int4.method import Qwen4ExpPLEPinnedHostInt4Embedding

    assert ng.Qwen4ExpPLEPinnedHostEmbedding is not Qwen4ExpPLEPinnedHostInt4Embedding


def test_int4_dtype_binds_int4_subclass(plugin_installed):
    ng = plugin_installed
    ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, "prefix", "int4")
    from ple_int4.method import Qwen4ExpPLEPinnedHostInt4Embedding

    assert ng.Qwen4ExpPLEPinnedHostEmbedding is Qwen4ExpPLEPinnedHostInt4Embedding


def test_stock_binding_survives_unknown_dtype(plugin_installed):
    ng = plugin_installed
    ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, "prefix", None)
    from ple_int4.method import Qwen4ExpPLEPinnedHostInt4Embedding

    assert ng.Qwen4ExpPLEPinnedHostEmbedding is not Qwen4ExpPLEPinnedHostInt4Embedding


def test_binding_flips_back_after_fp8_follows_int4(plugin_installed):
    """A single process serving both configs sees the right class each time."""
    ng = plugin_installed
    from ple_int4.method import Qwen4ExpPLEPinnedHostInt4Embedding

    ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, "p", "int4")
    assert ng.Qwen4ExpPLEPinnedHostEmbedding is Qwen4ExpPLEPinnedHostInt4Embedding
    ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, "p", "float8_e4m3fn")
    assert ng.Qwen4ExpPLEPinnedHostEmbedding is not Qwen4ExpPLEPinnedHostInt4Embedding
