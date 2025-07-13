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
    def __init__(self, hid, pos, redshift, **opt_fields):
        self.hid = hid
        self.pos = np.asarray(pos)
        if len(self.pos.shape) != 2 or self.pos.shape[1] != 3:
            raise ValueError(f"position needs to have shape (n, 3), has {self.pos.shape}")
        self.z = redshift
        self.alive = np.all(self.pos > 0, axis = 1)
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

    def getAlive(self):
        return self.alive

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
    def __init__(self, halo_list, boxsize):
        self.halos = halo_list
        self.hids = np.zeros(len(halo_list), dtype = int)
        for i,h in enumerate(self.halos):
            self.hids[i] = h.hid
        self.boxsize = boxsize
        return
    
    def getHalo(self, hid) -> Halo:
        if not hid in self.hids:
            raise ValueError(f"{hid} not found in system.")
        idx = np.where(self.hids == hid)[0][0]
        return self.halos[idx]
    
    def getID(self, idx) -> int:
        return self.halos[idx].hid
    
    def setCoMOrigin(self):
        """
        Shift all positions so the center of mass at each snapshot is at the origin,
        using only alive halos for each snapshot.
        """
        nsnaps = self.halos[0].pos.shape[0]
        nhalos = len(self.halos)
        boxsize = self.boxsize

        # Collect positions and masses
        all_pos = np.array([h.pos for h in self.halos])  # shape (nhalos, nsnaps, 3)
        all_mass = np.array([h.mass for h in self.halos])  # shape (nhalos, nsnaps)

        # Mask: halos alive per snapshot (nhalos, nsnaps)
        all_alive = np.array([h.getAlive() for h in self.halos])  # (nhalos, nsnaps)

        for s in range(nsnaps):
            # Only use halos alive at this snapshot
            alive_mask = all_alive[:, s]
            if not np.any(alive_mask):
                continue  # skip if no halos alive

            pos_s = all_pos[alive_mask, s, :]      # (Nalive, 3)
            mass_s = all_mass[alive_mask, s]       # (Nalive,)

            # Use periodic wrapping for CoM calculation
            # Shift all to origin, then compute mean position
            ref = pos_s[0]  # anchor to first alive halo
            dpos = self._wrapPeriodic(pos_s - ref)
            com_offset = (np.sum(mass_s[:, None] * dpos, axis=0) / np.sum(mass_s))
            com = self._wrapPeriodic(ref + com_offset)

            # Now subtract com from every halo's position
            for i, h in enumerate(self.halos):
                # Only shift if alive
                if all_alive[i, s]:
                    h.pos[s] = self._wrapPeriodic(h.pos[s] - com)

    def setCustomOrigin(self, orig_pos):
        """
        Shift all positions so orig_pos is at the origin.
        orig_pos can be shape (nsnaps, 3) or (3,) for static origin.
        """
        nsnaps = self.halos[0].pos.shape[0]

        orig_pos = np.asarray(orig_pos)
        if orig_pos.shape == (3,):
            orig_pos = np.broadcast_to(orig_pos, (nsnaps, 3))
        elif orig_pos.shape != (nsnaps, 3):
            raise ValueError(f"orig_pos must have shape (nsnaps, 3) or (3,), got {orig_pos.shape}")

        for h in self.halos:
            alive = h.getAlive()
            h.pos[alive, :] = self._wrapPeriodic(h.pos[alive, :] - orig_pos[alive, :])

    def setHaloOrigin(self, halo_id):
        """
        Shift all positions so the halo with given ID is at the origin (for each snapshot).
        """
        idxs = np.where(self.hids == halo_id)[0]
        if len(idxs) == 0:
            raise ValueError(f"Host halo ID {halo_id} not found")
        host = self.halos[idxs[0]]

        nsnaps = host.pos.shape[0]
        host_pos = host.pos.copy()
        host_alive = host.getAlive()

        for h in self.halos:
            alive = h.getAlive() & host_alive
            h.pos[alive, :] = self._wrapPeriodic(h.pos[alive, :] - host_pos[alive, :])


    def _wrapPeriodic(self, delta):
        """
        Adjust position differences to respect periodic boundaries.
        """
        bs = self.boxsize
        return (delta + 0.5 * bs) % bs - 0.5 * bs
    
    def getPidIdx(self):

        return

    def getUpidIdx(self):
        return

import graphics as g
class Event:
    """
    stores info about some event that happens to halo. Often particular kinds
    of events require shared plot properties so they appear the
    same in each instance. Each subclass stores the default style for that particular
    kind of event, implemented via the function _setDefaultStyle
    """
    def __init__(self, snaps, name=None):
        # Use provided name, or fall back to class name
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.snaps = snaps
        self._setDefaultStyle()

    def getName(self):
        return self.name
    
    def getSnaps(self):
        return self.snaps
    
    def _setDefaultStyle(self):
        self.def_style = {}
        return
    
class Peri(Event):
    
    def _setDefaultStyle(self):
        self.def_style = {
            g.COLOR : (255, 255, 0), # yellow
            g.LABEL : 'pericenter',
            g.SHAPE : 'sphere'
        }

class Apo(Event):

    def _setDefaultStyle(self):
        self.def_style = {
            g.COLOR : (50, 205, 50), # lime green
            g.LABEL : 'pericenter',
            g.SHAPE : 'sphere'
        }

# class Death(Event):
#     def _setDefaultStyle(self):
#         self.def_style = {
#             g.COLOR : (255, 255, 0), # yellow
#             g.LABEL : 'pericenter',
#             g.SHAPE : 'sphere'
#         }

class Inf(Event):
    def _setDefaultStyle(self):
        self.def_style = {
            g.COLOR : (255, 165, 0), # orange
            g.LABEL : 'infall',
            g.SHAPE : 'sphere'
        }

class SwitchHost(Event):

    def _setDefaultStyle(self):
        self.def_style = {
            g.COLOR : (54, 117, 136), # teal blue
            g.LABEL : 'host switch',
            g.SHAPE : 'cone'
        }