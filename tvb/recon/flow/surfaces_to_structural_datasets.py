import enum
import numpy as np
import os
import os.path
import logging
import time
from typing import List
from ..cli.runner import Runner, File
from ..cli import fs
from .core import Flow
from ..model.constants import FS_TO_CONN_INDICES_MAPPING_PATH


SUBCORTICAL_REG_INDS = ['008', '010', '011', '012', '013', '016', '017', '018', '026', '047', '049',
                        '050', '051', '052', '053', '054', '058']


class Hemisphere(enum.Enum):
    rh = 'rh'
    lh = 'lh'


class SurfacesToStructuralDataset(Flow):

    def __init__(self, cort_surf_direc: os.PathLike, subcort_surf_direc):
        """

        Parameters
        ----------
        cort_surf_direc: Directory that should contain:
                           rh.pial
                           lh.pial
        subcort_surf_direc: Directory that should contain:
                              aseg_<NUM>.srf
                            for each <NUM> in SUBCORTICAL_REG_INDS
        """
        self.cort_surf_direc = cort_surf_direc
        self.subcort_surf_direc = subcort_surf_direc

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

    @staticmethod
    def _unify_regions(verts_list: List[np.ndarray],
                       triangs_list: List[np.ndarray],
                       regmap_list: List[np.ndarray]) -> (np.ndarray, np.ndarray, np.ndarray):

        offsets = np.cumsum([0] + [vs.shape[0] for vs in verts_list][:-1])
        verts = np.vstack(verts_list)
        triangs = np.vstack([ts + offset for ts, offset in zip(triangs_list, offsets)])
        regmap = np.hstack(regmap_list)

        return verts, triangs, regmap

    def _get_cortical_surfaces(self, runner: Runner):
        verts_l, triangs_l = self._pial_to_verts_and_triangs(
            runner, File(os.path.join(self.cort_surf_direc, Hemisphere.lh + ".pial")))
        verts_r, triangs_r = self._pial_to_verts_and_triangs(
            runner, File(os.path.join(self.cort_surf_direc, Hemisphere.rh + ".pial")))

        region_mapping_l = self.magic()
        region_mapping_r = self.magic()

        verts, triangs, region_mapping = self._unify_regions([verts_l, verts_r], [triangs_l, triangs_r],
                                                             [region_mapping_l, region_mapping_r])
        return verts, triangs, region_mapping

    @staticmethod
    def _read_fs_to_conn_indices_mapping(mapping_path):
        fs_to_conn_indices_mapping = {}
        with open(mapping_path, 'r') as fd:
            for line in fd.readlines():
                key, _, val = line.strip().split()
                fs_to_conn_indices_mapping[int(key)] = int(val)

        return fs_to_conn_indices_mapping

    def _get_subcortical_surfaces(self):
        indices_mapping = self._read_fs_to_conn_indices_mapping(FS_TO_CONN_INDICES_MAPPING_PATH)

        verts_list = []
        triangs_list = []
        region_mapping_list = []

        for fs_idx in SUBCORTICAL_REG_INDS:
            conn_idx = indices_mapping[fs_idx]
            filename = os.path.join(self.subcort_surf_direc, 'aseg_' + fs_idx + '.srf')
            with open(filename, 'r') as f:
                f.readline()
                nverts, ntriangs = f.readline().rstrip().split(' ')

            a = np.loadtxt(filename, skiprows=2, usecols=(0, 1, 2))

            verts_list.append(a[:nverts])
            triangs_list.append(a[nverts:].astype(int))
            region_mapping_list.append(conn_idx * np.ones(nverts, dtype=int))

        verts, triangs, region_mapping = self._unify_regions(verts_list, triangs_list, region_mapping_list)
        return verts, triangs, region_mapping

    def run(self, runner: Runner):

        log = logging.getLogger('SurfacesToStructuralDataset')
        tic = time.time()


        log.info('complete in %0.2fs', time.time() - tic)
