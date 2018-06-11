import tinder
import os
import sys

def test_setup():
    sys.argv = ['', 'mode=unit_test', 'gpu=5,6', 'something=-3.7']
    tinder.setup(parse_args=True)

    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '5,6'
    assert os.environ['mode'] == 'unit_test'
    assert os.environ['something'] == '-3.7'
