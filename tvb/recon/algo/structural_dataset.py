
import numpy as np
import os
import tempfile
from typing import List
from zipfile import ZipFile


class StructuralDataset():
    def __init__(self, orientations: np.ndarray, areas: np.ndarray, centers: np.ndarray, names: List[str]):
        self.orientations = orientations
        self.areas = areas
        self.centers = centers
        self.names = names

        nregions = len(self.names)
        assert self.orientations.shape == (nregions, 3)
        assert self.areas.shape == (nregions,)
        assert self.centers.shape == (nregions, 3)

    def save_to_txt_zip(self, filename: os.PathLike):

        tmpdir = tempfile.TemporaryDirectory()
        file_areas = os.path.join(tmpdir, 'areas.txt')
        file_orientations = os.path.join(tmpdir, 'average_orientations.txt')
        file_centres = os.path.join(tmpdir, 'centres.txt')

        np.savetxt(file_areas, self.areas, fmt='%.2f')
        np.savetxt(file_orientations, self.orientations, fmt='%.2f %.2f %.2f')

        with open(file_centres, 'w') as f:
            for i, name in enumerate(self.names):
                f.write('%s %.4f %.4f %.4f\n' % (name, self.centers[i, 0], self.centers[i, 1], self.centers[i, 2]))

        with ZipFile(filename, 'w') as zip:
            zip.write(file_areas)
            zip.write(file_orientations)
            zip.write(file_centres)

    def save_to_h5(self, filename):
        pass
