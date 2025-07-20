import pyvista as pv
from containers import Halo, Event
import os
import numpy as np
from typing import Tuple, List, Union
import global_names as gn
import copy

def _check_color(value):
    """Check if value is a valid RGB or RGBA color."""
    if isinstance(value, (tuple, list, np.ndarray)) and (len(value) == 3 or len(value) == 4):
        if all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in value):
            return True
    raise ValueError(f"Color must be a tuple/list of 3 or 4 floats in [0,1], got {value}")

def _check_positive_float(val, name):
    if not isinstance(val, (int, float)) or val <= 0:
        raise ValueError(f"{name} must be a positive float, got {val}")

class Graphic:
    """Base class for graphics objects in the animation."""

    def __init__(self):
        self.name = ''
        self.start = -1
        self.stop = -1
        self.styles = {}

    def setName(self, name: str):
        if name in gn.INT_GRAPHIC_NAMES:
            raise ValueError(f"{name} cannot be used as graphic name, used internally in phalanx")
        self.name = str(name)

    def getName(self) -> str:
        return self.name

    def getStyle(self):
        styles = copy.deepcopy(self.styles)
        if self.start != -1:
            styles['start'] = str(self.start)
        if self.stop != -1:
            styles['stop'] = str(self.stop)
        return styles

    def setLabel(self, label: str):
        """Set the label for the graphic."""
        if not isinstance(label, str):
            raise TypeError("Label must be a string.")
        self.styles[gn.LABEL] = label

    def setColor(self, color: Union[tuple, list, np.ndarray]):
        """Set the color for the graphic."""
        _check_color(color)
        self.styles[gn.COLOR] = color

    def setSnaps(self, start, stop):
        self.start = start
        self.stop = stop
        return
    
    def getVTPName(self, snap):
        return f"{self.getName()}_{snap:04d}.vtp"
    
    def setOpacity(self, opacity: float):
        """Set opacity between 0 and 1."""
        if not isinstance(opacity, (float, int)) or not (0 <= opacity <= 1):
            raise ValueError(f"Opacity must be a float between 0 and 1, got {opacity}")
        self.styles[gn.OPACITY] = float(opacity)

    def writeVTP(self, out_dir, start, stop) -> Tuple[List, List]:
        
        pass


class Sphere(Graphic):

    def __init__(self, halo: Halo, name=None):
        super().__init__()
        self.halo = halo
        self.setName(f'sphere_{halo.hid}' if name is None else name)

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
            fname = os.path.join(out_dir, f"{self.getName()}_{isnap:04d}.vtp")
            sphere.save(fname)
            fnames.append(fname)
            tstep.append(isnap)
        return fnames, tstep

class Marker(Graphic):

    def __init__(self, halo: Halo, event: Event, name=None):
        super().__init__()
        self.halo = halo
        self.event = event
        self.setName(f'marker_{halo.hid}_{event.name}' if name is None else name)


    def setSize(self, size: float):
        _check_positive_float(size, "Size")
        self.styles[gn.SIZE] = float(size)

    def setShape(self, shape: str):
        """Set marker shape. Allowed: sphere, cube, cone, arrow, cylinder, point."""
        if shape not in gn.ALLOWED_MARKER_SHAPES:
            raise ValueError(f"Shape '{shape}' not recognized. Allowed: {gn.ALLOWED_MARKER_SHAPES}")
        self.styles[gn.SHAPE] = shape

    def writeVTP(self, out_dir, start, stop):
        event_snap = self.event.getSnap()  # snapshot when event occurs
        self.setSnaps(event_snap, stop)
        pos = self.halo.pos
        fnames, tsteps = [], []
        for isnap in range(start, stop):
            if isnap < event_snap:
                continue  # no marker before the event
            # Marker stays at the event location after it occurs:
            center = pos[event_snap]
            shape = self.styles.get(gn.SHAPE, 'sphere')
            size = self.styles.get(gn.SIZE, 1)
            # Create geometry based on shape
            if shape == 'sphere':
                mesh = pv.Sphere(center=center, radius=size)
            elif shape == 'cube':
                mesh = pv.Cube(center=center, x_length=size, y_length=size, z_length=size)
            elif shape == 'cone':
                mesh = pv.Cone(center=center, direction=(0,0,1), height=2*size, radius=size)
            elif shape == 'arrow':
                mesh = pv.Arrow(start=center, direction=(0,0,1), scale=size)
            elif shape == 'cylinder':
                mesh = pv.Cylinder(center=center, direction=(0,0,1), height=2*size, radius=size)
            elif shape == 'point':
                # Represent as a single point (no cells)
                mesh = pv.PolyData(np.array([center]))
            # Save the mesh for this snapshot
            fname = os.path.join(out_dir, self.getVTPName(isnap))
            mesh.save(fname)
            fnames.append(fname); tsteps.append(isnap)
        return fnames, tsteps


class Line(Graphic):

    def __init__(self, halo: Halo, name=None):
        super().__init__()
        self.halo = halo
        self.setName(f'line_{halo.hid}' if name is None else name)
        nsnaps = len(halo.z)
        snaps = np.arange(nsnaps)
        alv = self.halo.getAlive()
        self.setSnaps(snaps[alv][0], snaps[alv][-1])


    def setLinestyle(self, linestyle: str):
        """Allowed: solid, dashed, dotted."""
        if not isinstance(linestyle, str):
            raise TypeError("Linestyle must be a string.")
        if linestyle not in gn.ALLOWED_LINESTYLES:
            raise ValueError(f"Linestyle '{linestyle}' not recognized. Allowed: {gn.ALLOWED_LINESTYLES}")
        self.styles[gn.LSTYLE] = linestyle

    def setLinewidth(self, linewidth: float):
        _check_positive_float(linewidth, "Linewidth")
        self.styles[gn.LWIDTH] = float(linewidth)

    def setShowPoints(self, show: bool):
        if not isinstance(show, bool):
            raise TypeError("show_points must be a boolean.")
        self.styles[gn.SHOW_POINTS] = show
    
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
            
            # if snapshot is less than start, don't write vtp
            if (isnap < self.start) and (self.start != -1):
                continue
            # if snapshot is greater than stop, don't write vtp
            if (isnap > self.stop) and (self.stop != -1):
                continue

            # Alive mask up to and including isnap
            mask = alv[:isnap + 1]
            traj_points = pos[:isnap + 1][mask]
            if traj_points.shape[0] < 2:
                continue  # Need at least 2 points for a line
            # PyVista expects a lines array like [n_points, 0, 1, ..., n-1]
            pdata = pv.PolyData(traj_points)
            pdata.lines = np.hstack([[traj_points.shape[0]], np.arange(traj_points.shape[0])])
            fname = os.path.join(out_dir, self.getVTPName(isnap))
            pdata.save(fname)
            fnames.append(fname)
            tstep.append(isnap)
        return fnames, tstep

class Arrow(Graphic):

    def __init__(self, halo_from: Halo, halo_to: Halo, name=None):
        super().__init__()
        self.halo_from = halo_from
        self.halo_to = halo_to
        self.setName(f'arrow_{halo_from.hid}_{halo_to.hid}' if name is None else name)

    def setLinewidth(self, linewidth: float):
        _check_positive_float(linewidth, "Linewidth")
        self.styles[gn.LWIDTH] = float(linewidth)

    def setTipsize(self, tipsize: float):
        _check_positive_float(tipsize, "Tipsize")
        self.styles[gn.TIPSIZE] = float(tipsize)

    def writeVTP(self, out_dir, start, stop):
        posA = self.halo_from.pos; posB = self.halo_to.pos
        aliveA = self.halo_from.getAlive(); aliveB = self.halo_to.getAlive()
        fnames, tsteps = [], []
        # Determine tip and shaft sizes from styles or defaults
        tip_frac = self.styles.get(gn.TIPSIZE, 0.25)  # fraction of length for arrow tip
        base_tip_radius = 0.1; base_shaft_radius = 0.05  # PyVista defaults
        if gn.LWIDTH in self.styles:
            lw = self.styles[gn.LWIDTH]
            base_tip_radius *= lw
            base_shaft_radius *= lw
        for isnap in range(start, stop):
            if not (aliveA[isnap] and aliveB[isnap]):
                continue  # only draw if both halos exist at this snap
            start_pt = posA[isnap]; end_pt = posB[isnap]
            direction = end_pt - start_pt
            if np.linalg.norm(direction) == 0:
                continue  # skip if positions coincide (degenerate arrow)
            # Create arrow mesh from start_pt to end_pt
            arrow_mesh = pv.Arrow(start=start_pt, direction=direction, 
                                tip_length=tip_frac, 
                                tip_radius=base_tip_radius, shaft_radius=base_shaft_radius)
            fname = os.path.join(out_dir, self.getVTPName(isnap))
            arrow_mesh.save(fname)
            fnames.append(fname); tsteps.append(isnap)
        return fnames, tsteps

    
class Plot2D:
    def __init__(self, halos, yprop):
        self.halos = halos
        self.yprop = yprop
        return
    
    # TODO implement later
        

class LineSegment(Graphic):

    def __init__(self):
        super().__init__()
    
    #TODO implement later, basically contains a series of lines that change properties