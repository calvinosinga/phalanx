import pyvista as pv
from containers import Halo, Event
import os
import numpy as np
from typing import Tuple, List, Union
from scenes import SCENE_KEY
COLOR = 'color'
OPACITY = 'opacity'
LSTYLE = 'linestyle'
LWIDTH = 'linewidth'
SIZE = 'size'
SHAPE = 'shape'
LABEL = 'label'
TIPSIZE = 'tipsize'
SHOW_POINTS = 'show_points'

ALLOWED_LINESTYLES = {'solid', 'dashed', 'dotted'}
ALLOWED_MARKER_SHAPES = {'sphere', 'cube', 'cone', 'arrow', 'cylinder', 'point'}

INT_GRAPHIC_NAMES = {SCENE_KEY}
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
        self.styles = {}

    def setName(self, name: str):
        if name in INT_GRAPHIC_NAMES:
            raise ValueError(f"{name} cannot be used as graphic name, used internally in phalanx")
        self.name = str(name)

    def getName(self) -> str:
        return self.name

    def getStyle(self):
        return self.styles

    def setLabel(self, label: str):
        """Set the label for the graphic."""
        if not isinstance(label, str):
            raise TypeError("Label must be a string.")
        self.styles[LABEL] = label

    def setColor(self, color: Union[tuple, list, np.ndarray]):
        """Set the color for the graphic."""
        _check_color(color)
        self.styles[COLOR] = color

    def writeVTP(self, out_dir, start, stop) -> Tuple[List, List]:
        
        pass


class Sphere(Graphic):

    def __init__(self, halo: Halo, name=None):
        super().__init__()
        self.halo = halo
        self.setName(f'sphere_{halo.hid}' if name is None else name)

    def setOpacity(self, opacity: float):
        """Set sphere opacity between 0 and 1."""
        if not isinstance(opacity, (float, int)) or not (0 <= opacity <= 1):
            raise ValueError(f"Opacity must be a float between 0 and 1, got {opacity}")
        self.styles[OPACITY] = float(opacity)

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

    def __init__(self, halo: Halo, event: Event, name=None):
        super().__init__()
        self.halo = halo
        self.event = event
        self.setName(f'marker_{halo.hid}_{event.name}' if name is None else name)


    def setSize(self, size: float):
        _check_positive_float(size, "Size")
        self.styles[SIZE] = float(size)

    def setShape(self, shape: str):
        """Set marker shape. Allowed: sphere, cube, cone, arrow, cylinder, point."""
        if shape not in ALLOWED_MARKER_SHAPES:
            raise ValueError(f"Shape '{shape}' not recognized. Allowed: {ALLOWED_MARKER_SHAPES}")
        self.styles[SHAPE] = shape


class Line(Graphic):

    def __init__(self, halo: Halo, name=None):
        super().__init__()
        self.halo = halo
        self.setName(f'line_{halo.hid}' if name is None else name)


    def setLinestyle(self, linestyle: str):
        """Allowed: solid, dashed, dotted."""
        if not isinstance(linestyle, str):
            raise TypeError("Linestyle must be a string.")
        if linestyle not in ALLOWED_LINESTYLES:
            raise ValueError(f"Linestyle '{linestyle}' not recognized. Allowed: {ALLOWED_LINESTYLES}")
        self.styles[LSTYLE] = linestyle

    def setLinewidth(self, linewidth: float):
        _check_positive_float(linewidth, "Linewidth")
        self.styles[LWIDTH] = float(linewidth)

    def setShowPoints(self, show: bool):
        if not isinstance(show, bool):
            raise TypeError("show_points must be a boolean.")
        self.styles[SHOW_POINTS] = show
    
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

class Arrow(Graphic):

    def __init__(self, halo_from: Halo, halo_to: Halo, name=None):
        super().__init__()
        self.halo_from = halo_from
        self.halo_to = halo_to
        self.setName(f'arrow_{halo_from.hid}_{halo_to.hid}' if name is None else name)

    def setLinestyle(self, linestyle: str):
        if not isinstance(linestyle, str):
            raise TypeError("Linestyle must be a string.")
        if linestyle not in ALLOWED_LINESTYLES:
            raise ValueError(f"Linestyle '{linestyle}' not recognized. Allowed: {ALLOWED_LINESTYLES}")
        self.styles[LSTYLE] = linestyle

    def setLinewidth(self, linewidth: float):
        _check_positive_float(linewidth, "Linewidth")
        self.styles[LWIDTH] = float(linewidth)

    def setTipsize(self, tipsize: float):
        _check_positive_float(tipsize, "Tipsize")
        self.styles[TIPSIZE] = float(tipsize)


class Plot2D:
    def __init__(self, halos, yprop):
        self.halos = halos
        self.yprop = yprop
        return
    
    # TODO implement later
        

