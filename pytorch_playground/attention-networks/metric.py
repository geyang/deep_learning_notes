from visdom import Visdom
from log import log

vis = Visdom()
vis.env = "sequence-to-sequence-experiment"

def plot_accuracy():
    log('')
