import pyvista as pv
from containers import Halo, Event
import os
import numpy as np
from typing import Tuple, List
class Graphic:

    def __init__(self):
        self.name = ''
        self.styles = {}
        return
    
    def setName(self, name):
        self.name = name
        return
    
    def getName(self) -> str:
        return self.name
    
    def writeVTP(self, out_dir, start, stop) -> Tuple[List, List]:
        # write out 
        pass
    
    def getStyle(self):

        return self.styles
    
class Sphere(Graphic):

    def __init__(self, halo : Halo, name = None):
        super().__init__()
        if name is None:
            self.setName(f'sphere_{halo.hid}')
        else:
            self.setName(name)

        self.halo = halo
    
    def writeVTP(self, out_dir, start, stop)-> Tuple[List, List]:
        radius = self.halo.radius
        position = self.halo.pos
        alv = self.halo.getAlive()
        fnames = []
        tstep = []
        for isnap in range(start, stop):
            if not alv[isnap]:
                continue
            # Create sphere mesh in pyvista

            if radius[isnap] < 0:
                raise ValueError(f"halo {self.halo.hid} has invalid radius at snap {isnap}")
            sphere = pv.Sphere(radius=radius[isnap], center=position[isnap], theta_resolution=32, phi_resolution=32)
            # Save to VTP
            fname = os.path.join(out_dir, f"{self.getName()}_{isnap}.vtp")
            sphere.save(fname)
            fnames.append(fname)
            tstep.append(isnap)
        return fnames, tstep

class Marker(Graphic):

    def __init__(self, halo, event):
        self.halo = halo
        self.event = event
        super().__init__()


class Line(Graphic):

    def __init__(self, halo : Halo, name = None):
        super().__init__()
        if name is None:
            self.setName(f'line_{halo.hid}')
        else:
            self.setName(name)

        self.halo = halo


    def writeVTP(self, out_dir, start, stop)-> Tuple[List, List]:
        """
        For each snapshot in [start, stop), write a VTP file showing
        the trajectory up to and including that snapshot.
        """
        pos = self.halo.pos
        alv = self.halo.getAlive()
        fnames = []
        tstep = []
        for isnap in range(start, stop):
            if not alv[isnap]:
                continue
            # Alive mask up to and including isnap
            mask = alv[:isnap + 1]
            traj_points = pos[:isnap + 1][mask]
            if traj_points.shape[0] < 2:
                continue  # Need at least 2 points for a line
            # PyVista expects a lines array like [n_points, 0, 1, ..., n-1]
            pdata = pv.PolyData(traj_points)
            pdata.lines = np.hstack([[traj_points.shape[0]], np.arange(traj_points.shape[0])])
            fname = os.path.join(out_dir, f"{self.getName()}_{isnap}.vtp")
            pdata.save(fname)
            fnames.append(fname)
            tstep.append(isnap)
        return fnames, tstep

class SegmentedLine(Graphic):

    def __init__(self, halo):
        self.halo = halo
        super().__init__()

class Arrow(Graphic):

    def __init__(self, halo_from, halo_to):
        self.halo_from = halo_from
        self.halo_to = halo_to
        super().__init__()



