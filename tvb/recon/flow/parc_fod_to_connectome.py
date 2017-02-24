
import os
import logging
import time
from ..cli import runner, mrtrix
from .core import Flow

class ParcFodToConnectome(Flow):

    def __init__(self, parc: os.PathLike, fod: os.PathLike, gmwmi: os.PathLike=None, ftt: os.PathLike=None):
        self.parc = parc
        self.fod = fod
        self.gmwmi = gmwmi
        self.ftt = ftt

    def run(self, runner: runner.Runner):
        log = logging.getLogger('parc_fod_to_connectome')
        tic = time.time()

        # Relabel parcellation
        lut_in = os.path.join(os.environ['FREESURFER_HOME'], 'FreeSurferColorLUT.txt')
        lut_out = os.path.join(os.environ['MRT3'], 'src/connectome/tables/fs_default.txt')
        parc_lbl = runner.tmp_fname('parc_lbl.nii.gz')
        runner.run(mrtrix.run_labelconvert(self.parc, lut_in, lut_out, parc_lbl))

        FRACTION_SIFT = 0.2
        ntracks = 1000

        # tckgen
        tracks_A = runner.tmp_fname('tracks_A_N%s.tck' % ntracks)
        runner.run(mrtrix.run_tckgen(self.fod, tracks_A, ntracks, seed_gmwmi=self.gmwmi, act=self.ftt))

        # Until convergence is reached:
        for _ in range(2):

            # TODO: seeding for tckgen?
            # TODO: catching errors, warnings

            # tckgen more
            tracks_B = runner.tmp_fname('tracks_B_N%s.tck' % ntracks)
            runner.run(mrtrix.run_tckgen(self.fod, tracks_B, ntracks, seed_gmwmi=self.gmwmi, act=self.ftt))

            # Merge the track files
            ntracks *= 2
            tracks_merged = runner.tmp_fname('tracks_A_N%s.tck' % ntracks)
            runner.run(mrtrix.merge_trackfiles(tracks_A, tracks_B, tracks_merged))
            tracks_A = tracks_merged

            # tcksift
            ntracks_sift = int(ntracks*FRACTION_SIFT)
            tracks_sifted = runner.tmp_fname('tracks_sifted_N%s.tck' % ntracks)
            runner.run(mrtrix.run_tcksift(tracks_A, self.fod, tracks_sifted, ntracks_sift))

            # tckstats -dump
            tracks_lengths = runner.tmp_fname('track_lengths_N%s.tck' % ntracks)
            runner.run(mrtrix.dump_streamlines_length(tracks_A, tracks_lengths))

            # tck2connectome -out_assignment
            conn = runner.tmp_fname('con_N%s.csv' % ntracks)
            tracks_assignments = runner.tmp_fname('track_assignments_N%s.tck' % ntracks)
            runner.run(mrtrix.run_tck2connectome(tracks_sifted, parc_lbl, conn,
                                                 assignment=mrtrix.tck2connectome.AssignmentRadialSearch(2.0),
                                                 out_assignments=tracks_assignments
                                                 ))


            # get streamlines statistics (region-region)

        log.info('complete in %0.2fs', time.time() - tic)


