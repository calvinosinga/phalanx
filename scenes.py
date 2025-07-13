from graphics import Graphic, Sphere, Line
from containers import System
from typing import List
import numpy as np

class Scene:

    def __init__(self, system : System) -> None:
        if len(system.halos) < 1:
            raise ValueError("system has no halos, nothing to show")
        self.sys = system
        self.graphics : List[Graphic] = []
        self.start = 0
        self.stop = len(self.sys.halos[0].pos)
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
    
    ### GENERAL GRAPHIC CREATION METHODS USED IN ALL SCENES ###
    
    def showTjy(self, halo_id) -> Line:
        halo = self.sys.getHalo(halo_id)
        traj = Line(halo)
        return traj
    
    def showBoundary(self, halo_id) -> Sphere:
        halo = self.sys.getHalo(halo_id)
        sphere = Sphere(halo)
        return sphere
    

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