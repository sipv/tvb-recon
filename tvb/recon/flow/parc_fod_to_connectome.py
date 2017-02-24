
import logging
import time
from ..cli import runner, mrtrix
from .core import Flow

class ParcFodToConnectome(Flow):

    def __init__(self):
        pass

    def run(self, runner: runner.Runner):
        log = logging.getLogger('parc_fod_to_connectome')
        tic = time.time()

        # tckgen
        # Until convergence is reached:
        #     tckgen more
        #     tckedit to join
        #     tcksift
        #     tckstats -dump
        #     tck2connectome -out_assignment
        #     get streamlines statistics (region-region)

        log.info('complete in %0.2fs', time.time() - tic)


