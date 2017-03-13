
import numpy as np
import os
import tempfile
from typing import List
from zipfile import ZipFile


class StructuralDataset():
    def __init__(self, orientations: np.ndarray, areas: np.ndarray, centers: np.ndarray,
                 weights: np.ndarray, tract_lengths: np.ndarray, names: List[str]):

        nregions = len(self.names)
        assert orientations.shape == (nregions, 3)
        assert areas.shape == (nregions,)
        assert centers.shape == (nregions, 3)
        assert weights.shape == (nregions, nregions)
        assert tract_lengths.shape == (nregions, nregions)

        # Upper triangular -> symmetric matrices
        assert np.sum(np.tril(weights, 1)) == 0
        assert np.sum(np.tril(tract_lengths, 1)) == 0
        weights = weights + weights.transpose() - np.diag(np.diag(weights))
        tract_lengths = tract_lengths + tract_lengths.transpose() - np.diag(np.diag(tract_lengths))

        self.orientations = orientations
        self.areas = areas
        self.centers = centers
        self.weights = weights
        self.tract_lengths = tract_lengths
        self.names = names

    def save_to_txt_zip(self, filename: os.PathLike):

        tmpdir = tempfile.TemporaryDirectory()
        file_areas = os.path.join(tmpdir, 'areas.txt')
        file_orientations = os.path.join(tmpdir, 'average_orientations.txt')
        file_centres = os.path.join(tmpdir, 'centres.txt')
        file_weights = os.path.join(tmpdir, 'weights.txt')
        file_tract_lengths = os.path.join(tmpdir, 'tract_lengths.txt')

        np.savetxt(file_areas, self.areas, fmt='%.2f')
        np.savetxt(file_orientations, self.orientations, fmt='%.2f %.2f %.2f')
        np.savetxt(file_weights, self.weights, fmt='%d')
        np.savetxt(file_tract_lengths, self.tract_lengths, fmt='%.3f')

        with open(file_centres, 'w') as f:
            for i, name in enumerate(self.names):
                f.write('%s %.4f %.4f %.4f\n' % (name, self.centers[i, 0], self.centers[i, 1], self.centers[i, 2]))

        with ZipFile(filename, 'w') as zip:
            zip.write(file_areas)
            zip.write(file_orientations)
            zip.write(file_centres)

    def save_to_h5(self, filename):
        pass
