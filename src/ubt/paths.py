"""Locations of the large artifacts, which live outside the tracked files.

  UBT_WORK      models, checkpoints and evaluation outputs        (default <repo>/work)
  UBT_DATASETS  raw sources, the data-u1 build, external corpora   (default <repo>/datasets)
  UBT_PAPER     LaTeX sources of the paper (table/figure writers)  (default <repo>/paper)

The patched liblouis build is installed in <UBT_WORK>/liblouis, its source tree in <UBT_WORK>/liblouis-src.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _root(env: str, name: str) -> str:
    return os.path.abspath(os.environ.get(env) or os.path.join(REPO, name))


WORK = _root("UBT_WORK", "work")
DATASETS = _root("UBT_DATASETS", "datasets")
PAPER = _root("UBT_PAPER", "paper")
LIBLOUIS = os.path.join(WORK, "liblouis")
LIBLOUIS_SRC = os.path.join(WORK, "liblouis-src")
