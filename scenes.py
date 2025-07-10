from graphics import Graphic, Sphere, Line
from containers import System
from typing import List

class Scene:

    def __init__(self, system : System) -> None:
        self.sys = system
        self.graphics : List[Graphic] = []
        return
    
    def getGraphics(self) -> List[Graphic]:
        return self.graphics
    
    def addGraphic(self, graphic) -> None:
        self.graphics.append(graphic)
        return
    
    def rmGraphic(self, name) -> None:
        # TODO find the graphic that matches the name, print warning if no graphic with that name is found
        return
    ### GENERAL GRAPHIC CREATION METHODS USED IN ALL SCENES ###
    
    
    def makeTjy(self, halo_id) -> Line:
        halo = self.sys.getHalo(halo_id)
        traj = Line(halo)
        return traj
    
    def makeSphere(self, halo_id) -> Sphere:
        halo = self.sys.getHalo(halo_id)
        sphere = Sphere(halo)
        return sphere
    

class CenSat(Scene):
    """
    Create 
    """
    def __init__(self, system : System, cen_id : float):
        self.cen = system.getHalo(cen_id)
        super().__init__(system)
        self.sys.setHostOrigin(self.cen.id)
        return

    def hostBoundary(self, **style):
        sphere = self.makeSphere(self.cen.id)
        self.addGraphic(sphere)
        return
class TjyComp(Scene):

    def __init__(self):
        super().__init__()

class HostSubComp(Scene):

    def __init__(self):
        super().__init__()

class MergeSys(Scene):

    def __init__(self):
        super().__init__()