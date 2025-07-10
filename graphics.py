

class Graphic:

    def __init__(self):
        self.name = ''
        self.styles = {}
        return
    
    def setName(self, name):
        self.name = name
        return
    
    def getName(self):
        return self.name
    
    def writeVTP(self, out_dir):

        return
    
    def getStyle(self):

        return self.styles
    
class Sphere(Graphic):

    def __init__(self, halo, name = None):
        super().__init__()
        if name is None:
            self.setName(f'sphere_{halo.id}')
        else:
            self.setName(name)

        self.halo = halo
    

class Marker(Graphic):

    def __init__(self, halo, event):
        self.halo = halo
        self.event = event
        super().__init__()


class Line(Graphic):

    def __init__(self, halo):
        super().__init__()
        self.halo = halo
        # TODO repeat the naming scheme from sphere

class SegmentedLine(Graphic):

    def __init__(self, halo):
        self.halo = halo
        super().__init__()

class Arrow(Graphic):

    def __init__(self, halo_from, halo_to):
        self.halo_from = halo_from
        self.halo_to = halo_to
        super().__init__()



