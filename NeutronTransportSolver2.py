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


class NeutronTransportSolver2:
    def __init__(
            self,
            domain,
            D1M=1.0, D2M=0.5,
            Sa1M=0.2, Sa2M=0.1,
            nusigf1M=0.3, nusigf2M=0.1,
            S12M=0.1,
            D1F=1.0, D2F=0.5,
            Sa1F=0.2, Sa2F=0.1,
            nusigf1F=0.3, nusigf2F=0.1,
            S12F=0.1,
            N_eig=4,
            k=1,
            bord_cond='dir',
            cell_tags=None,
            facet_tags=None,
            ids=None
    ):
        self.domain = domain
        self.cell_tags = cell_tags
        self.D1M, self.D2M = D1M, D2M
        self.Sa1M, self.Sa2M = Sa1M, Sa2M
        self.nusigf1M, self.nusigf2M = nusigf1M, nusigf2M
        self.S12M = S12M
        self.D1F, self.D2F = D1F, D2F
        self.Sa1F, self.Sa2F = Sa1F, Sa2F
        self.nusigf1F, self.nusigf2F = nusigf1F, nusigf2F
        self.S12F = S12F
        self.N_eig = N_eig
        self.k = k
        self.bord_cond = bord_cond
        self.facet_tags = facet_tags
        self.V = self._function_space()
        self.eigvals = None
        self.vr = None
        self.vi = None
        self.phi1 = None
        self.phi2 = None
        self.phi1_list = None
        self.phi2_list = None
        if ids is None:
            self.MOD, self.FUEL, self.G_HEX, self.G_IFACE = 1, 2, 11, 12
        else:
            self.MOD, self.FUEL, self.G_HEX, self.G_IFACE = ids

    def _function_space(self):
        H = basix.ufl.element("Lagrange", self.domain.basix_cell(), self.k)
        Vm = basix.ufl.mixed_element([H, H])
        V = fem.functionspace(self.domain, Vm)
        return V

    def solve(self):
        phi1, phi2 = ufl.TrialFunctions(self.V)
        v1, v2 = ufl.TestFunctions(self.V)

        # dx = ufl.dx
        dx = ufl.Measure("dx", domain=self.domain,
                         subdomain_data=self.cell_tags)
        ds = ufl.Measure("ds", domain=self.domain,
                         subdomain_data=self.facet_tags)

        A = self.D1M*ufl.inner(ufl.grad(phi1), ufl.grad(v1))*dx(self.MOD)
        A += (self.Sa1M + self.S12M)*phi1*v1*dx(self.MOD)

        A += self.D2M*ufl.inner(ufl.grad(phi2), ufl.grad(v2))*dx(self.MOD)
        A += self.Sa2M*phi2*v2*dx(self.MOD)

        A += self.D1F*ufl.inner(ufl.grad(phi1), ufl.grad(v1))*dx(self.FUEL)
        A += (self.Sa1F + self.S12F)*phi1*v1*dx(self.FUEL)

        A += self.D2F*ufl.inner(ufl.grad(phi2), ufl.grad(v2))*dx(self.FUEL)
        A += self.Sa2F*phi2*v2*dx(self.FUEL)

        A -= self.S12M*phi1*v2*dx(self.MOD) + self.S12F*phi1*v2*dx(self.FUEL)

        F = (self.nusigf1M*phi1*v1 + self.nusigf2M*phi2*v1)*dx(self.MOD)
        F += (self.nusigf1F*phi1*v1 + self.nusigf2F*phi2*v1)*dx(self.FUEL)

        def boundary_all(x):
            return np.full(x.shape[1], True, dtype=bool)

        if self.bord_cond == 'dir':
            if self.facet_tags is None:
                boundary_facets = mesh.locate_entities_boundary(
                    self.domain, self.domain.topology.dim-1,
                    lambda x: np.full(x.shape[1], True, dtype=bool))
            else:
                boundary_facets = self.facet_tags.find(self.G_HEX)

            dofs1 = fem.locate_dofs_topological(self.V.sub(
                0), self.domain.topology.dim-1, boundary_facets)
            dofs2 = fem.locate_dofs_topological(self.V.sub(
                1), self.domain.topology.dim-1, boundary_facets)

            zero = default_scalar_type(0)
            bc1 = fem.dirichletbc(zero, dofs1, self.V.sub(0))
            bc2 = fem.dirichletbc(zero, dofs2, self.V.sub(1))
            bcs = [bc1, bc2]

        elif self.bord_cond == 'neu':
            bcs = []
        elif self.bord_cond == 'rob':
            ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags) \
                if self.facet_tags is not None else ufl.ds
            cst_rob = 0.4692
            alfa1 = cst_rob/self.D1M
            alfa2 = cst_rob/self.D2M
            A += alfa1*phi1*v1*ds(self.G_HEX) + alfa2*phi2*v2*ds(self.G_HEX)
            bcs = []
        elif self.bord_cond == 'mixed':
            cst_rob = 0.4692
            alfa1 = -cst_rob/self.D1
            alfa2 = -cst_rob/self.D2
            ds = ufl.Measure("ds", domain=self.domain,
                             subdomain_data=self.facet_tags)
            A += alfa1 * phi1 * v1 * ds(1) + alfa2 * phi2 * v2 * ds(1)
            A += alfa1 * phi1 * v1 * ds(3) + alfa2 * phi2 * v2 * ds(3)
            bcs = []
        else:
            raise ValueError(
                "Condición de borde inexistente: debe ser 'dir', 'neu' o 'rob' ")

        # ensamble del sistema
        a = fem_petsc.assemble_matrix(fem.form(A), bcs=bcs, diagonal=1e2)
        a.assemble()
        f = fem_petsc.assemble_matrix(fem.form(F), bcs=bcs, diagonal=1e-2)
        f.assemble()

        # cálculo de vvalores y vectores propios
        eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
        eigensolver.setDimensions(self.N_eig)
        eigensolver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

        st = SLEPc.ST().create(MPI.COMM_WORLD)
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(1.0)
        st.setFromOptions()
        eigensolver.setST(st)
        eigensolver.setOperators(a, f)
        eigensolver.setFromOptions()

        eigensolver.solve()

        self.vr, self.vi = a.getVecs()
        # self.eigvals = [eigensolver.getEigenpair(i, self.vr, self.vi) for i in range(self.N_eig)]
        self.eigvals = []
        self.phi1_list = []
        self.phi2_list = []
        for i in range(self.N_eig):
            lam = eigensolver.getEigenpair(i, self.vr, self.vi)
            self.eigvals.append(lam)

            phi = fem.Function(self.V)
            phi.x.array[:] = self.vr.array

            phi1, phi2 = phi.split()
            V0 = fem.functionspace(self.domain, ("CG", 1))

            phi1_proj = fem.Function(V0)
            phi1_proj.interpolate(fem.Expression(
                phi1, V0.element.interpolation_points()))
            self.phi1_list.append(phi1_proj)

            phi2_proj = fem.Function(V0)
            phi2_proj.interpolate(fem.Expression(
                phi2, V0.element.interpolation_points()))
            self.phi2_list.append(phi2_proj)

    def phi_norms(self, num=0):
        phi2_norm = np.sqrt(fem.assemble_scalar(
            fem.form(ufl.inner(self.phi2_list[num], self.phi2_list[num]) * ufl.dx)))
        phi1_norm = np.sqrt(fem.assemble_scalar(
            fem.form(ufl.inner(self.phi1_list[num], self.phi1_list[num]) * ufl.dx)))
        return phi1_norm, phi2_norm

    def export(self, modo=1, name='result'):
        if not (0 <= modo <= len(self.phi1_list)-1):
            raise ValueError(
                f"Índice de modo inválido: {modo}. Debe estar entre 1 y {len(self.phi1_list)}")

        phi1 = self.phi1_list[modo]
        phi2 = self.phi2_list[modo]
        V0 = fem.functionspace(self.domain, ("CG", 1))
        phi1_proj = fem.Function(V0)
        phi2_proj = fem.Function(V0)

        phi1_proj.interpolate(fem.Expression(
            phi1, V0.element.interpolation_points()))
        phi2_proj.interpolate(fem.Expression(
            phi2, V0.element.interpolation_points()))

        path = f"outputs/{name}"
        if MPI.COMM_WORLD.rank == 0 and not os.path.exists(path):
            os.makedirs(path)

        with io.VTKFile(MPI.COMM_WORLD, f"{path}/phi1_proj.pvd", "w") as vtk:
            vtk.write_function(phi1_proj)

        with io.VTKFile(MPI.COMM_WORLD, f"{path}/phi2_proj.pvd", "w") as vtk:
            vtk.write_function(phi2_proj)

        MPI.COMM_WORLD.Barrier()
        if MPI.COMM_WORLD.rank == 0:
            print("Archivos guardados en: ", path)
