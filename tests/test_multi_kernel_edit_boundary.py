"""The MLIR prompts must let the optimizer change a multi-kernel launch sequence.

The hard constraints said "Nothing ELSE inside @main is yours: leave the
allocations ...". On a 5-kernel SwiGLU chain the launches and the intermediate
gpu.alloc buffers live in a host function @payload that @main calls, and the
fusion stage read the rule as forbidding any change to them: "The host-side
payload sequence cannot legally be collapsed under the specified edit boundary"
(2026-10-15). Fusing kernels needs exactly that change.
"""
from xe_forge.agents import optimizer_agent, react_agent


def _mlir_signature_docs():
    docs = []
    for mod in (optimizer_agent, react_agent):
        for name in dir(mod):
            obj = getattr(mod, name)
            doc = getattr(obj, "__doc__", None) or ""
            if isinstance(obj, type) and "gpu.launch_func" in doc and "HARD CONSTRAINTS" in doc:
                docs.append((f"{mod.__name__}.{name}", doc))
    return docs


def test_every_mlir_signature_allows_fusing_launches():
    docs = _mlir_signature_docs()
    assert len(docs) >= 3, [n for n, _ in docs]
    for name, doc in docs:
        assert "MULTI-KERNEL MODULES" in doc, name
        assert "merge kernels into fewer launches" in doc, name
        assert "`@main` itself stays the" in doc, name
