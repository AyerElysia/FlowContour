def make_network(cfg):
    """Lazily import the full image pipeline.

    Lightweight architecture tests can import diffusion components without
    requiring optional data/visualization dependencies such as OpenCV.
    """

    from .make_network import make_network as _make_network

    return _make_network(cfg)

__all__ = ("make_network",)
