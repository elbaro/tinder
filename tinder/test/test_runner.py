import tinder
import os

def test_setup():
    tinder.setup(args=['unit_test', '5,6', 'something=-3.7'])

    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '5,6'
    assert os.environ['mode'] == 'unit_test'
    assert os.environ['something'] == '-3.7'
