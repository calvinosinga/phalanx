import numpy as np
from numpy.linalg import eigh, norm
import global_names as gn
from typing import List, Tuple, Union
import warnings
class Event:
    """
    stores info about some event that happens to halo. Often particular kinds
    of events require shared plot properties so they appear the
    same in each instance. Each subclass stores the default style for that particular
    kind of event, implemented via the function _setDefaultStyle
    """
    def __init__(self, snap, name=None):
        # Use provided name, or fall back to class name
        self.snap = snap
        if name is None:
            name = f"{self.__class__.__name__}_{self.snap}"
        self.name = name
        self._setDefaultStyle()

    def getName(self):
        return self.name
    
    def getSnap(self):
        return self.snap
    
    def _setDefaultStyle(self):
        self.def_style = {}
        return
    
class Peri(Event):

    
    def _setDefaultStyle(self):
        self.def_style = {
            gn.COLOR : (1, 1, 0), # yellow
            gn.LABEL : 'pericenter',
            gn.SHAPE : 'sphere'
        }

    def setHost(self, host_id):
        self.host_id = host_id
        self.name = f"{self.name}_{host_id}"
        return
    
class Apo(Event):

    def _setDefaultStyle(self):
        self.def_style = {
            gn.COLOR : (50/255, 205/255, 50/255), # lime green
            gn.LABEL : 'pericenter',
            gn.SHAPE : 'sphere'
        }

    def setHost(self, host_id):
        self.host_id = host_id
        self.name = f"{self.name}_{host_id}"
        return


class Ifl(Event):
    def _setDefaultStyle(self):
        self.def_style = {
            gn.COLOR : (1, 165/255, 0), # orange
            gn.LABEL : 'infall',
            gn.SHAPE : 'sphere'
        }

    def setHost(self, host_id):
        self.host_id = host_id
        self.name = f"{self.name}_{host_id}"
        return

class SwitchStatus(Event):

    def _setDefaultStyle(self):
        self.def_style = {
            gn.COLOR : (54/255, 117/255, 136/255), # teal blue
            gn.LABEL : 'host switch',
            gn.SHAPE : 'cone'
        }
    
class Death(Event):

    def _setDefaultStyle(self):
        self.def_style = {
            gn.COLOR : (1, 0, 0), # red
            gn.LABEL : 'death',
            gn.SHAPE : 'cube'
        }
        return

class Halo:
    """
    Class that stores data relevant to an individual halo.
    Core attributes (id, pos, redshift) are required.
    Optional fields (e.g., radius, mass, velocity) are stored in self.fields.
    """
    def __init__(self, hid, pos, redshift, **opt_fields):
        self.hid = hid
        self.pos = np.asarray(pos) # positions are expected to be comoving
        if len(self.pos.shape) != 2 or self.pos.shape[1] != 3:
            raise ValueError(f"position needs to have shape (n, 3), has {self.pos.shape}")
        self.z = redshift
        self.alive = np.all(self.pos > 0, axis = 1)
        nsnaps = len(self.z)
        snapshots = np.arange(nsnaps)
        self._first = snapshots[self.alive][0]
        self._last = snapshots[self.alive][-1] + 1 # add one to make it exclusive
        self.fields = {}
        for k, v in opt_fields.items():
            self.addField(k, v)
        self.events = []

    def _checkSnap(self, snap, msg = None):
        if msg is None:
            msg = f"{snap} outside halo lifetime, cannot set."
        if (snap < self._first) or (snap > self._last):
            raise ValueError(msg)
        return 
    
    def addField(self, name, arr):
        arr = np.asarray(arr)
        nz = len(self.z)

        if arr.ndim == 0:
            # Scalar input → broadcast to shape (nz,)
            arr = np.full((nz,), arr)
        elif arr.shape[0] != nz:
            raise ValueError(f"Field '{name}' must have first dimension of length {nz}, but got shape {arr.shape}")
        arr = np.squeeze(arr)
        self.fields[name] = arr


    def hasField(self, name):
        return name in self.fields

    def getField(self, name):
        if not self.hasField(name):
            raise ValueError(f"{name} not defined.")
        return self.fields.get(name, None)

    def getFieldNameMatch(self, key):
        names = self.fields.keys()
        matches = []
        for n in names:
            if key in n:
                matches.append(n)
        return matches
    
    def getAlive(self):
        return self.alive

    def addEvent(self, event : Event):
        esnap = event.getSnap()
        msg = f"Event {event.getName()} occurs outside of halos lifetime {self._first}, {self._last}"
        self._checkSnap(esnap, msg)
        self.events.append(event)
    
    def getEvents(self) -> List[Event]:
        return self.events
    
    def _getRange(self, snap_slc = None) -> Tuple[np.ndarray, np.ndarray]:
        alv = self.getAlive()
        if snap_slc is not None:
            snap_mask = np.zeros_like(alv, dtype = bool)
            snap_mask[snap_slc] = True
            mask = snap_mask & alv
        else:
            mask = alv
        pos = self.getPos()
        mins = pos[mask, :].min(axis = 0)
        maxs = pos[mask, :].max(axis = 0)
        
        if self.hasField(gn.RADIUS):
            max_r = np.max(self.radius[snap_slc])
            return mins - max_r, maxs + max_r
        return mins, maxs
    # --- Property Accessors for Common Fields --- #

    def getPos(self):
        return self.pos
    
    @property
    def radius(self):
        return self.getField(gn.RADIUS)

    @property
    def mass(self):
        return self.getField(gn.MASS)

    @property
    def radVel(self):
        return self.getField(gn.VR)

    @property
    def tanVel(self):
        return self.getField(gn.VT)

    @property
    def velocity(self):
        return self.getField(gn.VEL)
    
    @property
    def pid(self):
        return self.getField(gn.PID)
    
    @property
    def upid(self):
        return self.getField(gn.UPID)
    
    @property
    def status(self):
        return self.getField(gn.STATUS)



class System:
    """
    A class that handles how halos interface within a particular system
    """
    def __init__(self, halo_list : List[Halo], boxsize : float):
        self.halos = halo_list
        self.hids = np.array([h.hid for h in self.halos], dtype=int)
        self.boxsize = boxsize  # comoving
        self._rmPBC()           # unwrap + align once at construction
        return
    
    # ---------- GETTERS ----------

    
    def getHalo(self, hid) -> Halo:
        if hid not in self.hids:
            raise ValueError(f"{hid} not found in system.")
        idx = int(np.where(self.hids == hid)[0][0])
        return self.halos[idx]
    
    def getID(self, idx) -> int:
        return self.hids[idx]
    
    def addHalos(self, halos : Union[Halo, List[Halo]]):
        if isinstance(halos, Halo):
            halos = [halos]
        newids = np.array([h.hid for h in halos], dtype = int)

        self.halos.extend(halos)
        self.hids = np.hstack((self.hids, newids))
        if len(self.halos) != len(self.hids):
            raise Exception(f"length of halo {len(self.halos)} and hids {len(self.hids)} do not match")
        
        for h in range(len(self.halos)):
            if not self.halos[h].hid == self.hids[h]:
                raise Exception(f"id mismatch at idx {h}")

        self._rmPBC()
        
    
    # ---------- INTERNAL CONVENIENCE METHODS FOR SCENE CLASS  ----------

    def _checkKey(self, key_name):
        for h in self.sys.halos:
            if not h.hasField(key_name):
                raise KeyError(f"halo {h.hid} does not have field {key_name}.") 
    
    def _createDeathEvents(self, last_snap = -1):
        for h in self.halos:
            if h._last != last_snap:
                evt = Death(h._last - 1, f'death_{h.hid}') # _last is exclusive
                h.addEvent(evt)
        return
    
    def _getRange(self, start = 0, stop = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the overall axis‐aligned bounding box ([mins], [maxs]) in x,y,z
        that encloses all halos (including their radii, if present).
        """
        if not self.halos:
            raise Exception("Cannot calculate view box without halos.")
        mins = np.full(3, np.inf)
        maxs = np.full(3, -np.inf)
        if stop == -1:
            snap_slc = slice
        snap_slc = slice(start, stop)
        for h in self.halos:
            hmins, hmaxs = h._getRange(snap_slc)
            mins = np.minimum(mins, hmins)
            maxs = np.maximum(maxs, hmaxs)
        return mins, maxs

    def _findOptAxis(self):
        """
        Find the data-driven orthonormal axes that capture the most positional 
        variance among *all* halos.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (viewDir, viewUp)  – 3-vectors, already unit-normalised
            * viewDir : axis of **least** variance (3rd PC).  Point the camera
              along -viewDir so the screen plane shows maximal spread.
            * viewUp  : axis of **greatest** variance (1st PC).  Aligns the
              screen’s Y-direction with the direction of biggest motion.
        """
        # 1. Collect every valid position from every halo
        pos_chunks = []
        for halo in self.halos:
            pos_chunks.append(halo.getPos()[halo.getAlive()])

        if not pos_chunks:
            raise RuntimeError("System has no positions to analyze")

        pts = np.concatenate(pos_chunks, axis=0)
        pts -= pts.mean(axis=0)            # centre on the mean

        # 2. PCA via covariance eigen-decomposition
        cov = np.cov(pts, rowvar=False)
        w, v = eigh(cov)                   # v columns are eigenvectors
        order = w.argsort()[::-1]          # descending variance
        v = v[:, order]

        # 3. Return the principal axes, normalised
        axis1 = v[:, 0] / norm(v[:, 0])    # most variance
        axis2 = v[:, 1] / norm(v[:, 1])    # second most
        axis3 = v[:, 2] / norm(v[:, 2])    # least variance

        return axis3, axis1               # (viewDir, viewUp)
    
    
    # ---------- INTERNAL METHODS FOR HANDLING PERIODIC BOUNDARIES ----------
    
    def _Lvec(self) -> np.ndarray:
        L = np.asarray(self.boxsize, dtype=float)
        return np.array([L, L, L], dtype=float) if L.ndim == 0 else L

    def _alive_median(self, pos: np.ndarray, alive: np.ndarray) -> np.ndarray:
        """Median of positions over alive frames (3-vector)."""
        if not np.any(alive):
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        return np.median(pos[alive, :], axis=0)

    def _unwrap_positions(self, pos: np.ndarray, alive: np.ndarray) -> np.ndarray:
        """
        Unwrap (T,3) trajectory into a continuous path using minimal-image increments
        across alive frames. Non-alive frames are left as NaN.
        Assumes per-frame displacement < ~boxsize/2 along each axis.
        """
        pos = np.asarray(pos, dtype=float)
        L = self._Lvec()

        out = np.full_like(pos, np.nan)
        prev = None
        for t in range(pos.shape[0]):
            if not alive[t]:
                prev = None
                continue
            if prev is None:
                out[t] = pos[t]
            else:
                d = pos[t] - pos[prev]
                # minimal-image increment (handles multi-box via nearest integer)
                d -= np.round(d / L) * L
                out[t] = out[prev] + d
            prev = t
        return out

    def _rmPBC(self):
        """
        1) Unwrap every halo in time so trajectories are continuous.
        2) Apply a single integer box shift (per halo) so the *system* is in one image,
           i.e., halos are not split across opposite sides (if feasible).
        3) If the resulting system span exceeds L/2 on any axis, warn and continue.
        """
        if not self.halos:
            return

        L = self._Lvec()

        # 1) Unwrap each halo's trajectory
        for h in self.halos:
            alive = h.getAlive()
            h.pos = self._unwrap_positions(h.pos, alive)

        # 2) Single-image alignment across halos
        # Use the first halo with any alive frames as reference
        ref_idx = next((i for i, h in enumerate(self.halos) if np.any(h.getAlive())), None)
        if ref_idx is None:
            return  # nothing alive
        ref = self.halos[ref_idx]
        ref_ctr = self._alive_median(ref.pos, ref.getAlive())

        # For every halo, choose an integer shift so its median is closest to ref median
        for h in self.halos:
            alive = h.getAlive()
            if not np.any(alive):
                continue
            ctr = self._alive_median(h.pos, alive)
            # integer image index (3-vector)
            n = np.round((ctr - ref_ctr) / L)
            # shift entire (unwrapped) trajectory by -n*L
            h.pos = h.pos - n * L

        # 3) Check span; warn if any axis > L/2
        # Span computed over all halos & alive frames
        mins = np.array([ np.inf,  np.inf,  np.inf], dtype=float)
        maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        for h in self.halos:
            alive = h.getAlive()
            if np.any(alive):
                p = h.pos[alive]
                mins = np.minimum(mins, np.min(p, axis=0))
                maxs = np.maximum(maxs, np.max(p, axis=0))
        span = maxs - mins
        if np.any(span > 0.5 * L + 1e-9):
            warnings.warn(
                "System spans more than half the box along at least one axis "
                f"(span={span}, box={L}). Keeping unwrapped coordinates; "
                "views may show a large cosmological volume with PBC context."
            )


    # ---------- Origin setters ----------

    def setCoMOrigin(self):
        """
        Shift all positions so the center of mass at each snapshot is at the origin,
        using only alive halos for each snapshot. No periodic wrapping expected.
        """
        nsnaps = self.halos[0].pos.shape[0]
        all_alive = np.array([h.getAlive() for h in self.halos])  # (H, T)

        for s in range(nsnaps):
            com = self.getCoM(s)

            # subtract COM from every alive halo at s
            k = 0
            for h in self.halos:
                if all_alive[k, s]:
                    h.pos[s] = h.pos[s] - com
                k += 1

    def getCoM(self, s):
        all_pos  = np.array([h.pos  for h in self.halos])   # (H, T, 3)
        all_mass = np.array([h.mass for h in self.halos])   # (H, T)
        all_alive = np.array([h.getAlive() for h in self.halos])  # (H, T)
        alive_mask = all_alive[:, s]
        if not np.any(alive_mask):
            return np.array([np.nan]*3)
        pos_s  = all_pos[alive_mask, s, :]            # (N, 3)
        mass_s = all_mass[alive_mask, s][:, None]     # (N, 1)
        com = (mass_s * pos_s).sum(axis=0) / mass_s.sum()

        return com
    
    def setCustomOrigin(self, orig_pos):
        """
        Shift all positions so orig_pos is at the origin.
        orig_pos can be shape (nsnaps, 3) or (3,) for static origin.
        No wrapping; PBC presumed handled already.
        """
        nsnaps = self.halos[0].pos.shape[0]
        orig_pos = np.asarray(orig_pos, dtype=float)
        if orig_pos.shape == (3,):
            orig_pos = np.broadcast_to(orig_pos, (nsnaps, 3))
        elif orig_pos.shape != (nsnaps, 3):
            raise ValueError(f"orig_pos must have shape (nsnaps, 3) or (3,), got {orig_pos.shape}")

        for h in self.halos:
            alive = h.getAlive()
            h.pos[alive, :] = h.pos[alive, :] - orig_pos[alive, :]

    def setHaloOrigin(self, halo_id):
        """
        Shift all positions so the halo with given ID sits at the origin (per snapshot).
        Assumes all trajectories are already continuous (PBCs removed).
        """
        idxs = np.where(self.hids == halo_id)[0]
        if len(idxs) == 0:
            raise ValueError(f"Host halo ID {halo_id} not found")
        host = self.halos[idxs[0]]

        host_pos   = host.pos.copy()
        host_alive = host.getAlive()

        for h in self.halos:
            both = h.getAlive() & host_alive
            h.pos[both, :] = h.pos[both, :] - host_pos[both, :]
