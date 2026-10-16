"""Helico: an AlphaFold3 clone for experimentation."""

def __getattr__(name):
    # Dataset staging and cluster submission do not require PyTorch/CUDA.
    if name in {"Helico", "HelicoConfig"}:
        from helico.model import Helico, HelicoConfig
        return {"Helico": Helico, "HelicoConfig": HelicoConfig}[name]
    raise AttributeError(name)

__all__ = ["Helico", "HelicoConfig"]
