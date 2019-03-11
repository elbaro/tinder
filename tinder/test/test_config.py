import tinder
import os
import sys


def test_config():
    config = {
        "mode": "unit_test",
        "gpu": tinder.config.Placeholder.STR,
        "something": 2.4,
    }
    sys.argv = ["", "gpu=5,6", "something=-3.7"]
    tinder.config.override(config)

    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5,6"
    assert config["mode"] == "unit_test"
    assert config["something"] == -3.7
