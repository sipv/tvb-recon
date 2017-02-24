import logging
import sys

from tvb.recon.flow.parc_fod_to_connectome import ParcFodToConnectome
from tvb.recon.cli.runner import SimpleRunner


logging.basicConfig(level=logging.INFO)

parc, fod, gmwmi, ftt = sys.argv[1:]
runner = SimpleRunner()
convtest = ParcFodToConnectome(parc, fod, gmwmi, ftt)

convtest.run(runner)