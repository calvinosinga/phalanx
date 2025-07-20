from graphics import Graphic, Sphere, Line, Marker, Arrow
from containers import System, Event
from typing import List
import numpy as np
import global_names as gn
import copy

class Scene:

    def __init__(self, system : System) -> None:
        if len(system.halos) < 1:
            raise ValueError("system has no halos, nothing to show")
        self.sys = system
        self.graphics : List[Graphic] = []
        self.start = 0
        self.stop = len(self.sys.halos[0].pos)

        self.render_props = {}
        self.setBackgroundColor((0, 0, 0)) # want black background for almost all scenes
        self.setCameraParallelProjection(True) # almost always don't want perspective projection
        return
    
    def getGraphics(self) -> List[Graphic]:
        return self.graphics
    
    def addGraphic(self, graphic) -> None:
        self.graphics.append(graphic)
        return
    
    def rmGraphic(self, name) -> None:
        # TODO find the graphic that matches the name, print warning if no graphic with that name is found
        return
    
    def rmGraphicHalo(self, halo_id):
        # TODO remove all graphics associated with a particular halo
        return
    
    def addEvents(self):
        for h in self.sys.halos:
            events = h.getEvents()
            for ev in events:
                m = Marker(h, ev)
                ds = ev.def_style
                m.setColor(ds[gn.COLOR])
                m.setLabel(ds[gn.LABEL])
                m.setShape(ds[gn.SHAPE])
                if gn.SIZE not in ds:
                    m.setSize(self.sys._getDefaultPointSize())
                else:
                    m.setSize(ds[gn.SIZE])
                m.setSnaps(ev.getSnap(), self.stop)
                self.graphics.append(m)

        return
    
    def setSnap(self, start, stop):
        """
        the snapshots the animation will show
        Args:
            start (_type_): _description_
            stop (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.start = start
        self.stop = stop
        return
    
    # --- Scene Property Setters ---
    def getSceneProps(self):
        rprops = copy.deepcopy(self.render_props)
        rprops['start'] = str(self.start)
        rprops['stop'] = str(self.stop)
        rprops['nframes'] = str(self.stop - self.start + 1)
        return rprops

    def setProp(self, name, value):
        self.render_props[name] = value

    def setBackgroundColor(self, color):
        """
        Set the scene background color.
        Expects a tuple or list of 3 floats (RGB, 0–1).
        """
        if (isinstance(color, (tuple, list)) and len(color) == 3 
            and all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in color)):
            self.render_props[gn.BACKGROUND_COLOR] = tuple(float(c) for c in color)
        else:
            raise ValueError("Background color must be a tuple/list of 3 floats in [0,1].")
        return

    def setCameraPosition(self, pos):
        """
        Set camera position (x, y, z).
        """
        if (isinstance(pos, (tuple, list)) and len(pos) == 3 
            and all(isinstance(x, (int, float)) for x in pos)):
            self.render_props[gn.CAM_POS] = tuple(float(x) for x in pos)
        else:
            raise ValueError("Camera position must be a tuple/list of 3 numbers.")
        return

    def setCameraFocusPoint(self, foc):
        """
        Set camera focal point (where the camera looks).
        """
        if (isinstance(foc, (tuple, list)) and len(foc) == 3 
            and all(isinstance(x, (int, float)) for x in foc)):
            self.render_props[gn.CAM_FOC] = tuple(float(x) for x in foc)
        else:
            raise ValueError("Camera focus point must be a tuple/list of 3 numbers.")
        return

    def setCameraViewUp(self, up):
        """
        Set camera view up direction (usually [0,0,1] or [0,1,0]).
        """
        if (isinstance(up, (tuple, list)) and len(up) == 3 
            and all(isinstance(x, (int, float)) for x in up)):
            self.render_props[gn.CAM_UP] = tuple(float(x) for x in up)
        else:
            raise ValueError("Camera view up must be a tuple/list of 3 numbers.")
        return

    def setCameraParallelScale(self, scale):
        """
        Set camera zoom (parallel scale, must be positive float).
        """
        if isinstance(scale, (int, float)) and scale > 0:
            self.render_props[gn.CAM_ZOOM] = float(scale)
        else:
            raise ValueError("Camera parallel scale must be a positive number.")
        return

    def setCameraParallelProjection(self, flag):
        """
        Set camera projection mode: True (parallel) or False (perspective).
        """
        if isinstance(flag, bool):
            self.render_props[gn.CAM_PROJ] = flag
        else:
            raise ValueError("Camera parallel projection must be a boolean.")
        return

    def setViewSize(self, size):
        if (isinstance(size, (tuple, list)) and len(size) == 2
                and all(isinstance(x, (int, float)) for x in size)):
            self.render_props[gn.VIEW_SIZE] = size
        else:
            raise ValueError("View size must be tuple/list of 2 numbers.")



    def autoCam(self, padding_frac: float = 0.05) -> None:
        """
        Automatically set a static parallel‐projection camera so that all
        halos in the system just fit in view with a bit of padding.

        Args:
        padding_frac (float): fraction of the max span to pad on each side.
        """
        # get bounding box of all halos
        mins, maxs = self.sys.getViewBox()
        center = (mins + maxs) / 2.0
        spans = maxs - mins            # [dx, dy, dz]
        # rank axes by span: [smallest, middle, largest]
        small_ax, mid_ax, large_ax = np.argsort(spans)

        # view direction ≡ +unit on the smallest‐span axis
        view_normal = np.zeros(3)
        view_normal[small_ax] = 1.0
        # up vector ≡ +unit on the middle‐span axis
        view_up = np.zeros(3)
        view_up[mid_ax] = 1.0

        # padding in world‐units
        pad = padding_frac * spans.max()
        width  = spans[large_ax] + 2 * pad
        height = spans[mid_ax]   + 2 * pad

        # parallel scale is half the vertical extent
        parallel_scale = height / 2.0
        # camera position: just outside the box along view_normal
        distance = spans[small_ax] / 2.0 + pad
        cam_pos = center + view_normal * distance

        # apply to render_props
        self.setCameraFocusPoint(tuple(center.tolist()))
        self.setCameraPosition(tuple(cam_pos.tolist()))
        self.setCameraViewUp(tuple(view_up.tolist()))
        self.setCameraParallelScale(parallel_scale)
        self.setViewSize((width, height))

    
    ### GENERAL GRAPHIC CREATION METHODS USED IN ALL SCENES ###
    
    def makeTjy(self, halo_id) -> Line:
        halo = self.sys.getHalo(halo_id)
        traj = Line(halo)
        return traj
    
    def makeSphere(self, halo_id) -> Sphere:
        halo = self.sys.getHalo(halo_id)
        sphere = Sphere(halo)
        return sphere
    
    def makeArrow(self, halo_from_id, halo_to_id, start = -1, stop = -1) -> Arrow:
        halo_from = self.sys.getHalo(halo_from_id)
        halo_to = self.sys.getHalo(halo_to_id)
        arrow = Arrow(halo_from, halo_to)
        if start >= 0 and stop >= 0:
            arrow.setSnaps(start, stop)

        return arrow

    def makeSegment(self, halo_id, start, stop):
        halo = self.sys.getHalo(halo_id)
        seg = Line(halo)
        seg.setSnaps(start, stop)
        seg.setName(f"{seg.getName()}_{start}_{stop}")
        return seg
    
    
class HaloView(Scene):
    """
    For scenes that involve following a single halo. Special methods are provided to give
    special treatment to the pov halo. Most common usage will involve following the host
    of a particular system and seeing how subhalos interact with it.
    """
    def __init__(self, system : System, pov_id : int): # also allow pov to be halo object
        self.pov = system.getHalo(pov_id)
        super().__init__(system)
        self.sys.setHaloOrigin(self.pov.hid)
        nsnaps = len(self.sys.halos[0].pos)
        snaps = np.arange(nsnaps)
        alv = self.pov.getAlive()
        self.setSnap(snaps[alv][0], snaps[alv][-1])
        return

    def POVBoundary(self):
        sphere = self.makeSphere(self.pov.hid)
        sphere.setOpacity(0.15)
        self.addGraphic(sphere)
        return
    
    def showOrigin(self, org_size : float = 0.2):
        """_summary_

        Args:
            org_size (float, optional): size of origin marker, expressed as fraction of
             default calculated by the system. Defaults to 0.2.
        """
        org = Event(self.start, "origin")
        org.def_style = {
            gn.COLOR: (1, 1, 1), # WHITE
            gn.SHAPE: 'sphere',
            gn.LABEL: str(self.pov.hid),
            gn.SIZE: self.sys._getDefaultPointSize() * org_size
        }
        self.pov.addEvent(org)
        return

    def showTjys(self):
        # show tjys of all halos not pov
        for id_i in self.sys.hids:
            if id_i == self.pov.hid:
                continue
            tjy = self.makeTjy(id_i)
            self.addGraphic(tjy)
        return
        
    def startAtFirstInfall(self):
        """
        delay the start of the animation until another halo appears near the pov.
        """
        pov
        return
    
    def showTjyByStatus(self, host_color=None, sub_color=None,
                        alt_sub_color=None, do_ghost=False):
        """
        Draws trajectories of all halos except pov, coloring by host/sub/alt_sub status,
        and optionally applies ghost style overlay (dashed) if status >= 30.
        Single-snapshot segments get a Marker instead of a line (except ghost-only).
        """
        if host_color is None:
            host_color = (0.2, 0.4, 1.0)      # medium blue
        if sub_color is None:
            sub_color = (1.0, 0.2, 0.2)       # red
        if alt_sub_color is None:
            alt_sub_color = (1.0, 0.55, 0.41) # salmon

        ghost_style = {gn.LSTYLE: 'dashed'}

        pov_id = self.pov.hid

        for hid in self.sys.hids:
            if hid == pov_id:
                continue

            halo = self.sys.getHalo(hid)
            alive_mask = halo.getAlive()
            snaps = np.arange(self.start, self.stop)
            alive_range = alive_mask[self.start:self.stop]
            relevant_snaps = snaps[alive_range]
            if relevant_snaps.size == 0:
                continue

            pid = halo.pid[self.start:self.stop][alive_range]
            status = halo.status[self.start:self.stop][alive_range]

            # Determine the main category and ghost overlay for each frame
            def get_category(pid_val):
                if pid_val == -1:
                    return 'host'
                elif pid_val == pov_id:
                    return 'sub'
                elif pid_val > 0:
                    return 'alt_sub'
                else:
                    return 'unknown'

            N = len(relevant_snaps)
            if N == 0:
                continue

            categories = [get_category(pid[i]) for i in range(N)]
            ghosts = [do_ghost and status[i] >= 30 for i in range(N)]

            # Segment the trajectory by contiguous blocks of the same category and ghost flag
            seg_starts = [0]
            seg_cats = [(categories[0], ghosts[0])]
            for i in range(1, N):
                if (categories[i], ghosts[i]) != (categories[i-1], ghosts[i-1]):
                    seg_starts.append(i)
                    seg_cats.append((categories[i], ghosts[i]))
            seg_starts.append(N)  # end

            for i in range(len(seg_starts)-1):
                start_idx = seg_starts[i]
                stop_idx = seg_starts[i+1]
                seg_len = stop_idx - start_idx
                seg_snap = relevant_snaps[start_idx]
                seg_cat, seg_ghost = seg_cats[i]

                # Pick color
                if seg_cat == 'host':
                    color = host_color
                elif seg_cat == 'sub':
                    color = sub_color
                elif seg_cat == 'alt_sub':
                    color = alt_sub_color
                else:
                    color = (1,1,1)

                # Style dict
                style = {}
                style[gn.COLOR] = color
                if seg_ghost:
                    style.update(ghost_style)

                if seg_len == 1:
                    # For ghost-only, skip marker if only occurs for one frame
                    if seg_ghost:
                        continue
                    # Add a Marker instead of a Line
                    m = Marker(halo, snap=seg_snap)
                    m.setColor(color)
                    m.setSize(self.sys._getDefaultPointSize())
                    m.setLabel(seg_cat)
                    m.setShape('sphere')
                    m.setName(f"marker_{hid}_{seg_cat}_{seg_snap}")
                    self.addGraphic(m)
                else:
                    seg_start_snap = relevant_snaps[start_idx]
                    seg_stop_snap = relevant_snaps[stop_idx-1]+1
                    segment = self.makeSegment(hid, seg_start_snap, seg_stop_snap)
                    segment.setColor(color)
                    if seg_ghost:
                        segment.setStyle(ghost_style)
                    self.addGraphic(segment)
        return

        
    
class TjyComp(Scene):

    def __init__(self):
        super().__init__()

class HostSubComp(Scene):

    def __init__(self):
        super().__init__()

class MultiView(Scene):
    """
    For scenes that involve following multiple halos. This is usually intended to 
    assess mergers of multi-level host-sub systems.
    """
    def __init__(self):
        super().__init__()