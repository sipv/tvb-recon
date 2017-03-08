import enum
import numpy as np
import os
import os.path
import logging
import time
from ..cli.runner import Runner, File
from ..cli import fs
from .core import Flow

class Hemisphere(enum.Enum):
    rh = 'rh'
    lh = 'lh'


class SurfacesToStructuralDataset(Flow):

    def __init__(self, surf_direc: os.PathLike):
        self.surf_direc = surf_direc

    @staticmethod
    def _pial_to_verts_and_triangs(runner: Runner, pial_surf: File) -> (np.ndarray, np.ndarray):
        pial_asc = runner.tmp_fname(pial_surf.path + ".asc")
        pial_info = runner.tmp_fname(pial_surf.path + ".info.txt")

        runner.run(fs.mris_convert_run(pial_surf, pial_asc))
        runner.run(fs.mris_convert_run(pial_surf, pial_info))

        with open(pial_info, 'r') as f:
            lines = f.readlines()

        c_ras_line = lines[32]
        ista = c_ras_line.index(' (') + 2
        iend = c_ras_line.index(')\n')
        lc_ras = c_ras_line[ista:iend].split(',')
        c_ras = np.array(lc_ras).astype('float')

        with open(pial_asc, 'r') as f:
            f.readline()
            nb_vert = f.readline().split(' ')[0]
            read_data = [[np.double(line.rstrip('\n').split()[0]),
                          np.double(line.rstrip('\n').split()[1]),
                          np.double(line.rstrip('\n').split()[2])] for line in f]

        a = np.array(read_data)
        vertices = a[0:int(nb_vert), 0:3] + c_ras
        triangles = a[int(nb_vert):, 0:3]

        return vertices, triangles

    @staticmethod
    def _reunify_both_regions(verts_l, verts_r, triangs_l, triangs_r, region_mapping_l, region_mapping_r):
        verts = np.vstack([verts_l, verts_r])
        triangs = np.vstack([triangs_l,  triangs_r + verts_l.shape[0]])
        region_mapping = np.hstack([region_mapping_l, region_mapping_r])
        return verts, triangs, region_mapping


    def run(self, runner: Runner):

        log = logging.getLogger('SurfacesToStructuralDataset')
        tic = time.time()

        verts_l, triangs_l = self._pial_to_verts_and_triangs(runner,
                                                             os.path.join(self.surf_direc, Hemisphere.lh + ".pial"))
        verts_r, triangs_r = self._pial_to_verts_and_triangs(runner,
                                                             os.path.join(self.surf_direc, Hemisphere.rh + ".pial"))

        region_mapping_l = self.magic()
        region_mapping_r = self.magic()

        verts, triangs, region_mapping = self._reunify_both_regions(verts_l, verts_r, triangs_l, triangs_r,
                                                                    region_mapping_l, region_mapping_r)



        log.info('complete in %0.2fs', time.time() - tic)


