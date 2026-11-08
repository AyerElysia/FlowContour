"""Build the single network shipped by the mainline package."""

from lib.networks.mainline import Network


def make_network(cfg):
    """Construct the only network shipped by the package."""
    return Network(down_ratio=int(cfg.down_ratio))
