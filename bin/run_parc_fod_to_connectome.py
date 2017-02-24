import logging

from tvb.recon.flow.parc_fod_to_connectome import ParcFodToConnectome
from tvb.recon.cli.runner import SimpleRunner


logging.basicConfig(level=logging.INFO)

runner = SimpleRunner()
convtest = ParcFodToConnectome()

convtest.run(runner)