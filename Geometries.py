from mpi4py import MPI
from dolfinx import *
from dolfinx import fem, default_scalar_type
import numpy as np
from slepc4py import SLEPc
from mshr import *
import ufl
import os
import basix
import gmsh
from dolfinx.io import gmshio
import dolfinx.fem.petsc as fem_petsc


class Geometries:
    def __init__(
            self,
            hmin,
            hmax

    ):
        self.hmin = hmin
        self.hmax = hmax

    def create_circle(self):
        gmsh.initialize()
        gmsh.model.add("circle")
        gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(2, [1], tag=1)
        gmsh.model.setPhysicalName(2, 1, "circle")

        gmsh.option.setNumber("Mesh.MeshSizeMax", self.hmax)
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.hmin)

        gmsh.model.mesh.generate(2)

        domain, _, _ = io.gmshio.model_to_mesh(
            gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()

        return domain

    def create_L_mesh(self):
        gmsh.initialize()
        gmsh.model.add("L_mesh")

        # Geometría: cuadrado grande - cuadrado chico
        big = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
        small = gmsh.model.occ.addRectangle(0.5, 0.5, 0, 0.5, 0.5)
        cut = gmsh.model.occ.cut(
            [(2, big)], [(2, small)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(2, [cut[0][0][1]], tag=1)
        gmsh.model.setPhysicalName(2, 1, "L_domain")

        # Aquí definimos el tamaño máximo
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.hmax)

        gmsh.model.mesh.generate(2)
        domain, _, _ = io.gmshio.model_to_mesh(
            gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()
        return domain

    def nuclear_core(self, r_fuel=0.3, pitch=1.0, r_hex=2.0):
        '''  
        r_fuel: radio de cada disco de combustible
        pitch: 
        r_hex: radio de la circunferencia circunscrita al hexágono
        '''
        n_rings = 1

        if gmsh.isInitialized():
            gmsh.finalize()
        gmsh.initialize()
        gmsh.model.add("hex_lattice")

        # Generar el hexágono
        pts = [gmsh.model.occ.addPoint(r_hex*np.cos(2*np.pi*k/6), r_hex*np.sin(
            2*np.pi*k/6), 0) for k in range(6)]  # Genera los vértiices
        # Une los vértices con lineas
        lines = [gmsh.model.occ.addLine(
            pts[i], pts[(i+1) % 6]) for i in range(6)]
        # Une las lineas creando una sola frontera
        loop = gmsh.model.occ.addCurveLoop(lines)
        hex_surf = gmsh.model.occ.addPlaneSurface([loop])  # Crea la superficie

        # Centros de los cilindros de combustible
        fuel_centers = [(0.0, 0.0)]
        if n_rings >= 1:
            for i in range(6):
                ang = 2*np.pi*i/6
                fuel_centers.append((pitch*np.cos(ang), pitch*np.sin(ang)))

        # Lista de discos
        fuel_disks = [(2, gmsh.model.occ.addDisk(cx, cy, 0, r_fuel, r_fuel))
                      for (cx, cy) in fuel_centers]

        gmsh.model.occ.fragment([(2, hex_surf)], fuel_disks)
        gmsh.model.occ.synchronize()

        surfs = [sid for (d, sid) in gmsh.model.getEntities(2)]
        fuel_surf_tags, taken = [], set()
        for (cx, cy) in fuel_centers:
            best_sid, best_d2 = None, 1e99
            for sid in surfs:
                if sid in taken:
                    continue
                xg, yg, _ = gmsh.model.occ.getCenterOfMass(2, sid)
                d2 = (xg-cx)**2 + (yg-cy)**2
                if d2 < best_d2:
                    best_sid, best_d2 = sid, d2
            fuel_surf_tags.append(best_sid)
            taken.add(best_sid)
        moderator_surf_tags = [sid for sid in surfs if sid not in taken]

        outer_curves, interface_curves = [], []
        for _, cid in gmsh.model.getEntities(1):
            _, higher = gmsh.model.getAdjacencies(1, cid)
            if len(higher) == 1:
                outer_curves.append(cid)
            else:
                interface_curves.append(cid)

        gmsh.model.addPhysicalGroup(2, moderator_surf_tags, 1)
        gmsh.model.setPhysicalName(2, 1, "moderator")
        gmsh.model.addPhysicalGroup(2, fuel_surf_tags, 2)
        gmsh.model.setPhysicalName(2, 2, "fuel")
        if outer_curves:
            gmsh.model.addPhysicalGroup(1, outer_curves, 11)
            gmsh.model.setPhysicalName(1, 11, "Gamma_hex")
        if interface_curves:
            gmsh.model.addPhysicalGroup(1, interface_curves, 12)
            gmsh.model.setPhysicalName(1, 12, "Gamma_fuel_interface")

        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        for _, sid in gmsh.model.getEntities(2):
            gmsh.model.mesh.setRecombine(2, sid, False)
            gmsh.model.mesh.setAlgorithm(2, sid, 6)  # Frontal-Delaunay

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.hmin)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.hmax)
        gmsh.model.mesh.setOrder(1)
        gmsh.model.mesh.generate(2)

        domain, cell_tags, facet_tags = gmshio.model_to_mesh(
            gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()

        return domain, cell_tags, facet_tags
