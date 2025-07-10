import numpy as np

##### CONSTANT FIELD NAMES ##########
RADIUS = 'radius'
MASS = 'mass'
VR = 'rad_vel'
VT = 'tan_vel'
VEL = 'velocity'  # optionally update to ['vx', 'vy', 'vz']
PID = 'pid'
UPID = 'upid'

class Halo:
    """
    Class that stores data relevant to an individual halo.
    Core attributes (id, pos, redshift) are required.
    Optional fields (e.g., radius, mass, velocity) are stored in self.fields.
    """
    def __init__(self, id, pos, redshift, **opt_fields):
        self.id = id
        self.pos = np.asarray(pos)
        self.z = redshift
        self.fields = {}
        for k, v in opt_fields.items():
            self.addField(k, v)
        self.events = []


    def addField(self, name, arr):
        arr = np.asarray(arr)
        nz = len(self.z)

        if arr.ndim == 0:
            # Scalar input → broadcast to shape (nz,)
            arr = np.full((nz,), arr)
        elif arr.shape[0] != nz:
            raise ValueError(f"Field '{name}' must have first dimension of length {nz}, but got shape {arr.shape}")
        
        self.fields[name] = arr


    def hasField(self, name):
        return name in self.fields

    def getField(self, name):
        if not self.hasField(name):
            raise ValueError(f"{name} not defined.")
        return self.fields.get(name, None)

    def addEvent(self, event):
        self.events.append(event)

    # --- Property Accessors for Common Fields --- #
    @property
    def radius(self):
        return self.getField(RADIUS)

    @property
    def mass(self):
        return self.getField(MASS)

    @property
    def radVel(self):
        return self.getField(VR)

    @property
    def tanVel(self):
        return self.getField(VT)

    @property
    def velocity(self):
        return self.getField(VEL)

    @property
    def pid(self):
        return self.getField(PID)

    @property
    def upid(self):
        return self.getField(UPID)

    
    
class System:
    """
    A class that handles how halos interface within a particular system
    """
    def __init__(self, halo_list):
        self.halos = halo_list
        self.ids = np.zeros(len(halo_list))
        for i,h in enumerate(self.halos):
            self.ids[i] = h.id

        return
    
    def setCoMOrigin(self):
        return
    
    def setCustomOrigin(self, orig_pos):
        return
    
    def setHostOrigin(self, host_id):
        """
        Adjust the positions of all the subhalos to be with respect to 
        """
        return

    def getPidIdx(self):

        return

    def getUpidIdx(self):
        return
    
    def getHalo(self, id):
        return np.where(id == self.ids)[0]
    
    def getId(self, idx):
        return self.halos[idx].id
    

class Event:
    """
    stores info about some event that happens to halo
    """
    def __init__(self):
        pass

class Peri(Event):

    def __init__(self):
        super().__init__()

class Apo(Event):

    def __init__(self):
        super().__init__()

class Death(Event):

    def __init__(self):
        super().__init__()

class Inf(Event):

    def __init__(self):
        super().__init__()

class SwitchHost(Event):

    def __init__(self):
        super().__init__()