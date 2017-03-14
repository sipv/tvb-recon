# -*- coding: utf-8 -*-

import os
import numpy as np
from tvb.recon.tests.base import BaseTest, data_path

import tvb.recon.flow.surfaces_to_structural_datasets as stsd
from tvb.recon.flow.surfaces_to_structural_datasets import Surface, RegionIndexMapping
from tvb.recon.cli.runner import SimpleRunner

FLOAT_TOL = 1e-16


class MinimalSurfaceTest(BaseTest):

    def setUp(self):
        super().setUp()

        self.surf1 = Surface(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
                             np.array([[0, 1, 2], [1, 3, 2]]),
                             np.array([1, 1, 1, 1]))
        self.surf2 = Surface(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]),
                             np.array([[0, 1, 2], [1, 3, 2]]),
                             np.array([1, 1, 1, 1]))

        self.nverts1 = self.surf1.vertices.shape[0]
        self.nverts2 = self.surf2.vertices.shape[0]
        self.ntri1 = self.surf1.triangles.shape[0]
        self.ntri2 = self.surf2.triangles.shape[0]

    def test_merge_surfaces(self):
        surf_merged = stsd.merge_surfaces([self.surf1, self.surf2])

        self.assertEqual(surf_merged.vertices.shape, (self.nverts1 + self.nverts2, 3))
        self.assertEqual(surf_merged.triangles.shape, (self.ntri1 + self.ntri2, 3))
        self.assertEqual(surf_merged.region_mapping.shape, (self.nverts1 + self.nverts2,))

    def test_compute_triangle_normals(self):
        normals = stsd.compute_triangle_normals(self.surf1.triangles, self.surf1.vertices)
        self.assertTrue(np.allclose(normals,
                                    np.array([[0., 0., 1.], [0., 0., 1.]]),
                                    atol=FLOAT_TOL))

    def test_compute_vertex_normals(self):
        vert_triangles = stsd.compute_vertex_triangles(self.nverts1, self.ntri1, self.surf1.triangles)
        tri_angles = stsd.compute_triangle_angles(self.surf1.vertices, self.ntri1, self.surf1.triangles)
        tri_normals = stsd.compute_triangle_normals(self.surf1.triangles, self.surf1.vertices)
        vert_normals = stsd.compute_vertex_normals(self.nverts1, vert_triangles, self.surf1.triangles,
                                                   tri_angles, tri_normals, self.surf1.vertices)
        self.assertTrue(np.allclose(vert_normals,
                                    np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]]),
                                    atol=FLOAT_TOL))

    def test_compute_triangle_areas(self):
        areas = stsd.compute_triangle_areas(self.surf1.vertices, self.surf1.triangles)
        self.assertTrue(np.allclose(areas,
                                    np.array([0.5, 0.5]),
                                    atol=FLOAT_TOL))

    def test_compute_region_params(self):
        surf = stsd.merge_surfaces([self.surf1, self.surf2])
        regions, areas, orientations, centers = stsd.compute_region_params(surf, False)

        self.assertTrue(np.equal(regions, [1]))
        self.assertTrue(np.allclose(areas, [2.0], atol=FLOAT_TOL))
        self.assertTrue(np.allclose(orientations, np.array([1, 0, 1])*np.sqrt(2)/2, atol=FLOAT_TOL))
        self.assertTrue(np.allclose(centers, [0.25, 0.5, 0.25], atol=FLOAT_TOL))


class CorticalSurfaceTest(BaseTest):

    TEST_FS_SUBJECT = "freesurfer_fsaverage"
    TEST_SURF_DIR = "surf"
    TEST_LABEL_DIR = "label"
    FS_TO_CONN_INDICES_MAPPING_PATH = "mapping_FS_88.txt"
    NREGIONS = 88

    def setUp(self):
        super().setUp()
        self.runner = SimpleRunner()

    def test_get_cortical_surfaces(self):
        cort_surf_direc = os.path.join(data_path, self.TEST_FS_SUBJECT, self.TEST_SURF_DIR)
        label_direc = os.path.join(data_path, self.TEST_FS_SUBJECT, self.TEST_LABEL_DIR)
        region_index_mapping = RegionIndexMapping(os.path.join(data_path, self.FS_TO_CONN_INDICES_MAPPING_PATH))

        surf = stsd.get_cortical_surfaces(self.runner, cort_surf_direc, label_direc, region_index_mapping)
        nverts = surf.vertices.shape[0]

        self.assertTrue(((surf.triangles >= 0) & (surf.triangles <= nverts - 1)).all())
        self.assertTrue(((surf.region_mapping >= 0) & (surf.region_mapping <= self.NREGIONS - 1)).all())
