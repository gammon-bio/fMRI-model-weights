import copy

import pytest
import torch

from fmritest.config import load_config

torch.set_num_threads(1)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def data_cfg(cfg):
    return copy.deepcopy(cfg["data"])
