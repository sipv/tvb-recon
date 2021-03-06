# -*- coding: utf-8 -*-

import numpy
from tvb.recon.model.constants import *
from trimesh import Trimesh, intersections
#from tvb.recon.algo.service.surface import  SurfaceService


class Surface(object):
    """
    Hold a surface mesh (vertices and triangles).

    Has also few methods to read from this mesh (e.g. a contour cut).
    """

    def __init__(self, vertices, triangles, area_mask=None, center_ras=[], vertices_coord_system=None,
                 generic_metadata=None, vertices_metadata=None, triangles_metadata=None):

        if len(vertices) == 0:
            self.vertices = numpy.empty((0, 3))
        else:
            # numpy array of n_vertices x 3 [x,y,z] vertices' coordinates
            self.vertices = numpy.array(vertices)
        if len(triangles) == 0:
            self.triangles = numpy.empty((0, 3), dtype='i')
        else:
            # numpy array of n_triangles x 3 [v1, v2, v3] indices in vertices'
            # array
            self.triangles = numpy.array(triangles)

        self.center_ras = center_ras  # [x, y, z]

        self.n_vertices = self.vertices.shape[0]
        self.n_triangles = self.triangles.shape[0]

        self.generic_metadata = generic_metadata
        self.vertices_metadata = vertices_metadata
        self.triangles_metadata = triangles_metadata

        self.vertices_coord_system = vertices_coord_system

        if area_mask is None:
            self.area_mask = numpy.ones((self.n_vertices,), dtype='bool')
        else:
            self.area_mask = area_mask

    def get_main_metadata(self):
        if self.vertices_metadata is not None:
            return self.vertices_metadata
        return self.generic_metadata

    def set_main_metadata(self, new_metadata):
        if self.vertices_metadata is not None:
            self.vertices_metadata = new_metadata
        else:
            self.generic_metadata = new_metadata

    # TODO: it will fail if one tries to add empty inputs
    def add_vertices_and_triangles(
            self, new_vertices, new_triangles, new_area_mask=[]):
        self.triangles = numpy.r_[self.triangles,
                                  new_triangles + self.n_vertices]
        self.vertices = numpy.r_[self.vertices, new_vertices]
        n_new_vertices = new_vertices.shape[0]
        self.n_vertices += n_new_vertices
        if len(new_area_mask) == 0:
            new_area_mask = numpy.ones((n_new_vertices,), dtype='bool')
        numpy.r_[self.area_mask, new_area_mask]
        # self.stack_vertices_and_triangles()

    # def stack_vertices_and_triangles(self):
    #     self.vertices = numpy.vstack(self.vertices)
    #     self.triangles = numpy.vstack(self.triangles)
    #     self.n_vertices = len(self.vertices)
    #     self.n_triangles = len(self.triangles)

    def _get_plane_origin(self, ras):
        plane_origin = numpy.subtract(ras, self.center_ras)
        return list(plane_origin)

    def cut_by_plane(self, projection=SAGITTAL, ras=ORIGIN):
        """
        :param projection:
        :param ras:
        :return: Y_array, X_array
        """
        mesh = Trimesh(self.vertices, self.triangles)
        contours = intersections.mesh_plane(
            mesh, PLANE_NORMALS[projection], self._get_plane_origin(ras))
        x_array = [0] * len(contours)
        y_array = [0] * len(contours)

        for s in range(len(contours)):
            x_array[s] = contours[s][:, X_Y_INDEX[projection][0]]
            y_array[s] = contours[s][:, X_Y_INDEX[projection][1]]

        return x_array, y_array

    # def compute_area(self):
    #     if numpy.all(self.area_mask):
    #         vertices=self.vertices
    #         triangles=self.triangles
    #     else:
    #         surface_service = SurfaceService()
    #         (vertices,triangles) = surface_service.extract_subsurf(self,self.area_mask,output="verts_triangls")[:2]
    #     return numpy.sum(surface_service.tri_area(vertices[triangles]))

    def compute_normals(self):
        """
        :return: array of triangle normal vectors
        """
        normals = [[0, 0, 0] for _ in range(len(self.triangles))]

        for i, tri in enumerate(self.triangles):
            u = self.vertices[tri[1]] - self.vertices[tri[0]]
            v = self.vertices[tri[2]] - self.vertices[tri[0]]
            normals[i][0] = u[1] * v[2] - u[2] * v[1]
            normals[i][1] = u[2] * v[0] - u[0] * v[2]
            normals[i][2] = u[0] * v[1] - u[1] * v[0]

        return normals

    def vertex_normals(self):
        # TODO test by generating points on unit sphere: vtx pos should equal
        # normal

        vf = self.vertices[self.triangles]
        fn = numpy.cross(vf[:, 1] - vf[:, 0], vf[:, 2] - vf[:, 0])
        vf = [set() for _ in range(len(self.vertices))]
        for i, fi in enumerate(self.triangles):
            for j in fi:
                vf[j].add(i)
        vn = numpy.zeros_like(self.vertices)
        for i, fi in enumerate(vf):
            fni = fn[list(fi)]
            norm = fni.sum(axis=0)
            norm /= numpy.sqrt((norm ** 2).sum())
            vn[i] = norm
        return vn
