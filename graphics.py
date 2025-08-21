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
        # snapshots that determine when to show the graphic
        self.disp_start = None
        self.disp_stop = None
        self.styles = {}
        self.units = gn._DEF_UNITS

    def setName(self, name: str):
        if name in gn.INT_GRAPHIC_NAMES:
            raise ValueError(f"{name} cannot be used as graphic name, used internally in phalanx")
        self.name = str(name)

    def getName(self) -> str:
        return self.name

    def getStyle(self):
        styles = copy.deepcopy(self.styles)
        if self.disp_start is not None:
            styles['start'] = str(self.disp_start)
        if self.disp_stop is not None:
            styles['stop'] = str(self.disp_stop)
        return styles

    def setStyle(self, sdict: dict) -> None:
        """Apply *sdict* entries to a Graphic object in a unified way."""
        if gn.COLOR in sdict:
            self.setColor(sdict[gn.COLOR])
        if gn.OPACITY in sdict:
            self.setOpacity(sdict[gn.OPACITY])
        if gn.LWIDTH in sdict:
            self.setLineWidth(sdict[gn.LWIDTH])
        if gn.LSTYLE in sdict:
            self.setLineStyle(sdict[gn.LSTYLE])
        if gn.LABEL in sdict:
            self.setLabel(sdict[gn.LABEL])
        if gn.SHAPE in sdict:
            self.setShape(sdict[gn.SHAPE])
        if gn.TIPSIZE in sdict:
            self.setTipsize(sdict[gn.TIPSIZE])
        if gn.LIGHTING in sdict:
            self.setLighting(sdict[gn.LIGHTING])
        
    def _doDisplay(self, snap)-> bool:
        if self.disp_start is None or self.disp_stop is None:
            raise ValueError(f"display timesteps not set for graphic {self.name}")
        doDisplay = (snap >= self.disp_start) and (snap <= self.disp_stop)

        return doDisplay
    
    def _writeEmptyVTP(self, outdir, snap):
        out = pv.PolyData()
        name = self.getVTPName(snap)
        fname = os.path.join(outdir, name)
        out.save(fname)
        return fname
        
    def setLabel(self, label: str):
        """Set the label for the graphic."""
        if not isinstance(label, str):
            raise TypeError("Label must be a string.")
        self.styles[gn.LABEL] = label

    def addFieldToLabel(self,
                        fmt: str,
                        fields: List[str],
                        start: int,
                        stop: int) -> None:
        """
        Default implementation for halo-backed graphics.
        Appends formatted field values to label; emits a list if time-varying.
        """
        halo = getattr(self, "halo", None)
        if halo is None:
            raise NotImplementedError(f"{self.__class__.__name__} does not provide a halo to read fields from.")

        arrays = []
        lengths = []
        for fname in fields:
            if hasattr(halo, "getField"):
                arr = halo.getField(fname)
            else:
                arr = halo.fields[fname]
            arr = np.asarray(arr)[start:stop]
            arrays.append(arr)
            lengths.append(arr.shape[0])

        nframes = stop - start
        # If any array has >1 distinct value across frames, treat as dynamic
        is_dynamic = any(len(np.unique(a)) > 1 for a in arrays)

        base = self.getLabel()
        if base is None:
            # default base label
            hid = getattr(halo, "id", None)
            base = f"Halo {hid}" if hid is not None else self.name

        def _fmt(vals):
            # Support both printf-style and str.format
            try:
                return fmt % tuple(vals) if len(vals) > 1 else (fmt % vals[0])
            except TypeError:
                return fmt.format(*vals)

        if is_dynamic:
            out = []
            for i in range(nframes):
                vals_i = [a[i] if a.ndim == 1 else a[i] for a in arrays]
                out.append(f"{base}, {_fmt(vals_i)}")
            self.setLabel(out)
        else:
            vals0 = [a[0] if a.ndim == 1 else a[0] for a in arrays]
            self.setLabel(f"{base}, {_fmt(vals0)}")

    def setColor(self, color: Union[tuple, list, np.ndarray]):
        """Set the color for the graphic."""
        _check_color(color)
        self.styles[gn.COLOR] = color

    def setSize(self, size: float):
        raise NotImplementedError(f"size not compatible for graphic {self.__class__}")

    def setShape(self, shape: str):
        raise NotImplementedError(f"shape not compatible for graphic {self.__class__}")
    
    def setLineStyle(self, linestyle: str):
        raise NotImplementedError(f"linestyle not compatible for graphic {self.__class__}")


    def setLineWidth(self, linewidth: float):
        raise NotImplementedError(f"linewidth not compatible for graphic {self.__class__}")


    def setShowPoints(self, show: bool):
        raise NotImplementedError(f"show_points not compatible for graphic {self.__class__}")

    def setTipsize(self, tipsize: float):
        raise NotImplementedError(f"tip size not compatible for graphic {self.__class__}")

    def setLighting(self, light: bool):
        raise NotImplementedError(f"lighting not compatible with graphic {self.__class__}")
    
    def setDisplaySnaps(self, start, stop):
        self.disp_start = start
        self.disp_stop = stop
        return
    
    def _showOnlyInView(self, mins, maxs):
        if not hasattr(self, 'halo'):
            raise NotImplementedError(f"showOnlyInView by default assumes halo attr, DNE in {self.__class__}")
        # Ensure NumPy arrays with shape (3,)
        mins = np.asarray(mins, dtype=float)
        maxs = np.asarray(maxs, dtype=float)

        # Boolean mask: True where the position is inside the box for *all* axes
        pos = self.halo.getPos()
        if self.units == 'phy':
            pos /= (1 + self.halo.z[:, np.newaxis])
        in_box = np.all((pos >= mins) & (pos <= maxs), axis=1)

        # Find the first True; np.flatnonzero avoids the 0-when-all-False pitfall
        hits = np.flatnonzero(in_box)
        if hits.size:
            self.disp_start = int(hits[0])
        else:
            self.disp_start = np.inf # never display
    

    def getVTPName(self, snap):
        return f"{self.getName()}_{snap:04d}.vtp"
    
    def setOpacity(self, opacity: float):
        """Set opacity between 0 and 1."""
        if not isinstance(opacity, (float, int)) or not (0 <= opacity <= 1):
            raise ValueError(f"Opacity must be a float between 0 and 1, got {opacity}")
        self.styles[gn.OPACITY] = float(opacity)
    
    def setUnits(self, utype):
        allowed_units = ['phy', 'com']
        if utype in allowed_units:
            self.units = utype
        else:
            raise ValueError(f"{utype} is not allowed unit type ({allowed_units}).")
        return
    
    """
    All the different start/stops can be a little confusing. Each graphic contains
    its own display start/stop that determines what timesteps we display that
    particular graphic. For example, markers that indicate when/where events happen
    should appear at the timestep they occur and then be maintained afterward. The
    start/stop input into writeVTP delineate the start/stop of the animation. This
    way we only create the VTP files the user desires. Some graphics, like the Line,
    contain their own start/stop that indicate WHAT to display. 
    For example, we sometimes want to create two different line segments from
    the same trajectory that have different stylings -- in this case, we only want
    to display a subset of the full trajectory.
    """
    def writeVTP(self, out_dir, start, stop) -> Tuple[List, List]:
        """_summary_

        Args:
            out_dir (_type_): _description_
            start (_type_): _description_
            stop (_type_): _description_

        Returns:
            Tuple[List, List]: filenames and timesteps that were written
        """
        
        pass

class Sphere(Graphic):

    def __init__(self, halo: Halo, name=None):
        super().__init__()
        self.halo = halo
        self.setName(f'sphere_{halo.hid}' if name is None else name)
        # by default, we display the sphere when the halo is alive
        self.setDisplaySnaps(halo._first, halo._last)
        self.setLighting(True)

    def setLighting(self, light: bool):
        self.styles[gn.LIGHTING] = light
        return
    
    def writeVTP(self, out_dir, start, stop)-> Tuple[List, List]:
        if self.units == 'phy':
            radius = self.halo.radius
            position = self.halo.getPos() / (1 + self.halo.z[:, np.newaxis])
        else: # comoving
            radius = self.halo.radius * (1 + self.halo.z)
            position = self.halo.getPos()
        
        fnames = []
        tstep = []
        for isnap in range(start, stop):
            if not self._doDisplay(isnap):
                fname = self._writeEmptyVTP(out_dir, isnap)
                fnames.append(fname)
                tstep.append(isnap)
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
        # by default, we display from when event occurs to halo's death
        self.setDisplaySnaps(self.event.getSnap(), self.halo._last)

    def setSize(self, size: float):
        _check_positive_float(size, "Size")
        self.styles[gn.SIZE] = float(size)

    def setShape(self, shape: str):
        """Set marker shape. Allowed: sphere, cube, cone, arrow, cylinder, point."""
        if shape not in gn.ALLOWED_MARKER_SHAPES:
            raise ValueError(f"Shape '{shape}' not recognized. Allowed: {gn.ALLOWED_MARKER_SHAPES}")
        self.styles[gn.SHAPE] = shape

    def writeVTP(self, out_dir, start, stop):

        pos = self.halo.getPos()
        if self.units == 'phy':
            pos /= (1 + self.halo.z[:, np.newaxis])
        fnames, tsteps = [], []
        center = pos[self.event.getSnap()]
        shape = self.styles.get(gn.SHAPE, 'sphere')
        size = self.styles.get(gn.SIZE, 1)
        for isnap in range(start, stop):
            if not self._doDisplay(isnap):
                fname = self._writeEmptyVTP(out_dir, isnap)
                fnames.append(fname)
                tsteps.append(isnap)            # Marker stays at the event location after it occurs
                continue

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
        self.setDisplaySnaps(self.halo._first, self.halo._last)
        # by default we display full tjy
        self.tjy_start = self.halo._first
        self.tjy_stop = self.halo._last

    def setTjySnaps(self, start, stop):
        self.halo._checkSnap(start)
        self.halo._checkSnap(stop)
        self.tjy_start = start
        self.tjy_stop = stop

    def _doDisplay(self, snap):
        # if the snapshot is before tjy_start, theres nothing to show.
        return super()._doDisplay(snap) and (snap >= self.tjy_start)
    
    def setLineStyle(self, linestyle: str):
        """Allowed: solid, dashed, dotted."""
        if not isinstance(linestyle, str):
            raise TypeError("Linestyle must be a string.")
        if linestyle not in gn.ALLOWED_LINESTYLES:
            raise ValueError(f"Linestyle '{linestyle}' not recognized. Allowed: {gn.ALLOWED_LINESTYLES}")
        self.styles[gn.LSTYLE] = linestyle

    def setLineWidth(self, linewidth: float):
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
        pos = self.halo.getPos()
        if self.units == 'phy':
            pos /= (1 + self.halo.z[:, np.newaxis])
        fnames = []
        tstep = []
        for isnap in range(start, stop):
            if not self._doDisplay(isnap):
                fname = self._writeEmptyVTP(out_dir, isnap)
                fnames.append(fname)
                tstep.append(isnap)
                continue

            # should be able to assume that slice is within halos livetime
            end = min(isnap + 1, self.tjy_stop)
            traj_points = pos[self.tjy_start:end]
            if traj_points.shape[0] < 2:
                fname = self._writeEmptyVTP(out_dir, isnap)
                fnames.append(fname)
                tstep.append(isnap)
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
        self.mask = None
        self.norm = None

    def setNorm(self, norm):
        self.norm = norm
        return
    
    def setDisplayMask(self, mask):
        if len(mask) != len(self.halo_from.z):
            raise ValueError("mask is incorrect size")
        self.mask = mask
        return
    
    def _doDisplay(self, snap):
        if self.mask is None:
            return super()._doDisplay(snap)
        
        return self.mask[snap]
    
    def setLineWidth(self, linewidth: float):
        _check_positive_float(linewidth, "Linewidth")
        self.styles[gn.LWIDTH] = float(linewidth)

    def setTipsize(self, tipsize: float):
        _check_positive_float(tipsize, "Tipsize")
        self.styles[gn.TIPSIZE] = float(tipsize)

    def _showOnlyInView(self, mins, maxs):

        # Ensure NumPy arrays with shape (3,)
        mins = np.asarray(mins, dtype=float)
        maxs = np.asarray(maxs, dtype=float)

        # Boolean mask: True where the position is inside the box for *all* axes
        to_pos = self.halo_to.getPos()
        if self.units == 'phy':
            to_pos /= (1 + self.halo_to.z[:, np.newaxis])
        from_pos = self.halo_from.getPos()
        if self.units == 'phy':
            from_pos /= (1 + self.halo_from.z[:, np.newaxis])
        to_in_box = np.all((to_pos >= mins) & (to_pos <= maxs), axis=1)
        from_in_box = np.all((from_pos >= mins) & (from_pos <= maxs), axis=1)
        in_box = to_in_box & from_in_box
        # Find the first True; np.flatnonzero avoids the 0-when-all-False pitfall
        hits = np.flatnonzero(in_box)
        if hits.size:
            self.disp_start = int(hits[0])
        else:
            self.disp_start = np.inf # never display
    
    def addFieldToLabel(self, *args, **kwargs):
        raise NotImplementedError("Arrow does not expose a halo; cannot add field-driven labels.")
    
    def writeVTP(self, out_dir, start, stop):
        posA = self.halo_from.getPos(); posB = self.halo_to.getPos()
        if self.units == 'phy':
            posA /= (1 + self.halo_from.z[:, np.newaxis])
            posB /= (1 + self.halo_to.z[:, np.newaxis])
        fnames, tsteps = [], []
        # Determine tip and shaft sizes from styles or defaults
        tip_frac = self.styles.get(gn.TIPSIZE, 0.05)  # fraction of length for arrow tip
        base_tip_radius = 0.1; base_shaft_radius = 0.05  # PyVista defaults
        if gn.LWIDTH in self.styles:
            lw = self.styles[gn.LWIDTH]
            base_tip_radius *= lw
            base_shaft_radius *= lw
        for isnap in range(start, stop):
            if not self._doDisplay(isnap): # throws error if disp snaps not defined
                fname = self._writeEmptyVTP(out_dir, isnap)
                fnames.append(fname)
                tsteps.append(isnap)
                continue
            start_pt = posA[isnap]; end_pt = posB[isnap]
            direction = end_pt - start_pt
            dist = np.linalg.norm(direction)
            if self.norm is None:
                self.norm = dist

            if np.linalg.norm(direction) == 0:
                fname = self._writeEmptyVTP(out_dir, isnap)
                fnames.append(fname)
                tsteps.append(isnap)
            # Create arrow mesh from start_pt to end_pt
            arrow_mesh = pv.Arrow(start=start_pt, direction=direction, 
                                tip_length=tip_frac, scale=dist,
                                tip_radius=base_tip_radius * (self.norm/dist),
                                shaft_radius=base_shaft_radius * (self.norm/dist))
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
        