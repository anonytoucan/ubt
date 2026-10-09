import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "generation.yaml")


@pytest.fixture(scope="session")
def cfg():
    with open(CFG_PATH) as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def fork_tables(cfg):
    return cfg["table_dirs"]["ko2020"]


@pytest.fixture(scope="session")
def translator(fork_tables):
    from ubt.translate import ForwardTranslator

    return ForwardTranslator(extra_table_dirs=[fork_tables])
