from mpi4py import MPI
from dolfinx import *
from dolfinx import fem, default_scalar_type
import numpy as np
from slepc4py import SLEPc
from mshr import *
import basix
import ufl
import os
'test'


class NeutronTransportSolver:
    def __init__(
            self,
            domain,
            D1=1.0, D2=0.5,
            Sa1=0.2, Sa2=0.1,
            nusigf1=0.3, nusigf2=0.1,
            S12=0.1,
            N_eig=4,
            k=1,
            bord_cond='dir'
    ):
        self.domain = domain
        self.D1, self.D2 = D1, D2
        self.Sa1, self.Sa2 = Sa1, Sa2
        self.nusigf1, self.nusigf2 = nusigf1, nusigf2
        self.S12 = S12
        self.N_eig = N_eig
        self.k = k
        self.bord_cond = bord_cond

        self.V = self._function_space()
        self.eigvals = None
        self.vr = None
        self.vi = None
        self.phi1 = None
        self.phi2 = None
        self.phi1_list = None
        self.phi2_list = None

    def _function_space(self):
        H = basix.ufl.element("Lagrange", self.domain.basix_cell(), self.k)
        Vm = basix.ufl.mixed_element([H, H])
        V = fem.functionspace(self.domain, Vm)
        return V

    def solve(self):
        phi1, phi2 = ufl.TrialFunctions(self.V)
        v1, v2 = ufl.TestFunctions(self.V)

        dx = ufl.dx

        # Formas bilineales A y F
        A = self.D1 * ufl.inner(ufl.grad(phi1), ufl.grad(v1)) * dx
        A += (self.Sa1 + self.S12) * phi1 * v1 * dx
        A += self.D2 * ufl.inner(ufl.grad(phi2), ufl.grad(v2)) * dx
        A += self.Sa2 * phi2 * v2*dx
        A -= self.S12 * phi1 * v2 * dx

        F = (self.nusigf1 * phi1 * v1 + self.nusigf2 * phi2 * v1) * dx

        def boundary_all(x):
            return np.full(x.shape[1], True, dtype=bool)

        if self.bord_cond == 'dir':
            boundary_facets = mesh.locate_entities_boundary(self.domain,
                                                            self.domain.topology.dim - 1, boundary_all)
            boundary_dofs_x = fem.locate_dofs_topological(self.V.sub(0),
                                                          self.domain.topology.dim - 1, boundary_facets)
            boundary_dofs_x2 = fem.locate_dofs_topological(self.V.sub(1),
                                                           self.domain.topology.dim - 1, boundary_facets)

            bcx = fem.dirichletbc(default_scalar_type(
                0), boundary_dofs_x, self.V.sub(0))
            bc1x = fem.dirichletbc(default_scalar_type(
                0), boundary_dofs_x2, self.V.sub(1))
            bcs = [bc1x, bcx]
        elif self.bord_cond == 'neu':
            bcs = []
        elif self.bord_cond == 'rob':
            cst_rob = 0.4692
            alfa1 = -cst_rob/self.D1
            alfa2 = -cst_rob/self.D2
            ds = ufl.ds
            A += - alfa1 * phi1 * v1 * ds - alfa2 * phi2 * v2 * ds
            bcs = []
        else:
            raise ValueError(
                "Condición de borde inexistente: debe ser 'dir', 'neu' o 'rob' ")

        # ensamble del sistema
        a = assemble_matrix(fem.form(A), bcs=bcs, diagonal=1e2)
        a.assemble()
        f = assemble_matrix(fem.form(F), bcs=bcs, diagonal=1e-2)
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
