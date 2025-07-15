from graphics import Graphic, Sphere, Line, Marker
from containers import System
from typing import List
import numpy as np
import global_names as gn


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
        return self.render_props

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
    


    ### GENERAL GRAPHIC CREATION METHODS USED IN ALL SCENES ###
    
    def showTjy(self, halo_id) -> Line:
        halo = self.sys.getHalo(halo_id)
        traj = Line(halo)
        return traj
    
    def showBoundary(self, halo_id) -> Sphere:
        halo = self.sys.getHalo(halo_id)
        sphere = Sphere(halo)
        return sphere
    
    def autoCam(self):
        """
        finds the axis that has the largest position variation, sets that axis to go from left
        to right, with the axis of the second most position variation going from top to bottom.
        Does this via setting the focal point, up view, and position of the camera.
        Automatically sets the zoom to just fit all objects within the view.
        """
        return

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
        nsnaps = self.stop
        snaps = np.arange(nsnaps)
        alv = self.pov.getAlive()
        self.setSnap(snaps[alv][0], snaps[alv][-1])
        return

    def POVBoundary(self):
        sphere = self.showBoundary(self.pov.hid)
        self.addGraphic(sphere)
        return
    
    def showTjys(self):
        # show tjys of all halos not pov
        for id_i in self.sys.hids:
            if id_i == self.pov.hid:
                continue
            tjy = self.showTjy(id_i)
            self.addGraphic(tjy)
        return
        
    def onlyInteractions(self):
        """
        ignore all (if any) snapshots at the start where the halo is isolated, limit to 3
        snapshots before another halo enters its R200m 
        """
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