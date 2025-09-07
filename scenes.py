from graphics import Graphic, Sphere, Line, Marker, Arrow
from containers import Halo, Peri, System, Event
from typing import List, Union, Optional, Dict, Tuple
import numpy as np
import global_names as gn
import copy

def _toIterable(var):
    if not isinstance(var, List):
        return [var]
    return var

class Scene:
    """
    Class that handles first how to view the data (camera settings, background, lighting, animation snapshots)
    and then what data/graphics get shown. 
    """
    def __init__(self, system : System) -> None:
        if len(system.halos) < 1:
            raise ValueError("system has no halos, nothing to show")
        self.sys = copy.deepcopy(system)
        self.graphics : List[Graphic] = []
        self.start = 0
        self.stop = len(self.sys.halos[0].pos)  # exclusive

        # settings for how to display graphics
        self.units = gn._DEF_UNITS
        # leaving all graphics up permanently can cause the scene to look busy, allow for deletion after
        # halo death
        self.rm_graphics = False  

        # camera properties and defaults
        self.render_props = {}
        self.setBackgroundColor((0, 0, 0)) # want black background for almost all scenes
        self.setCameraParallelProjection(True) # almost always don't want perspective projection

        # text/legend display properties
        self.texts = []
        return


    # --- Scene Property Methods ---
    def setAnimSnap(self, start, stop):
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
    
    # ---------------------------
    # Serialization to style.json
    # ---------------------------
    def getSceneProps(self) -> Dict:
        """
        Return dict to be serialized under _scene in style.json,
        including the scene-wide text specs under key 'text'.
        """
        props = {
            "start": int(self.start),
            "stop": int(self.stop),
        }
        rprops = copy.deepcopy(self.render_props)
        props.update(rprops)
        
        # turn scene properties into lists
        for k,v in props.items():
            if isinstance(v, np.ndarray):
                props[k] = v.tolist() # arrays are not serializable
        
        if self.texts:
            props["text"] = self.texts
        return props

    # ---------------------------
    # Overlays / legend API
    # ---------------------------
    def showRedshift(self,
                     fmt: str = 'z = %.2f',
                     color: tuple = (1.0, 1.0, 1.0),
                     font: int = 12,
                     position: List[float] = [0.05, 0.95]) -> None:
        """
        Create a dynamic scene-wide annotation for redshift.
        Produces a string per animation timestep [start:stop].
        """
        z = np.asarray(self.sys.getZ())
        z_slice = z[self.start:self.stop]
        # Support printf-style or str.format
        texts = []
        for val in z_slice:
            try:
                s = fmt % val
            except TypeError:
                s = fmt.format(val)
            texts.append(s)

        spec = {
            "text": texts,
            "color": list(color),
            "font": int(font),
            "position": list(position),
        }
        self.texts.append(spec)

    def addText(self,
                text: Union[str, List[str]],
                color: tuple = (1.0, 1.0, 1.0),
                font: int = 12,
                position: List[float] = [0.90, 0.95]) -> None:
        """
        Add an arbitrary scene-wide annotation (static string or list per frame).
        """
        if isinstance(text, list):
            # ensure it matches animation length if dynamic
            if len(text) != (self.stop - self.start):
                raise ValueError("Length of dynamic text must equal number of frames (stop-start).")
        self.texts.append({
            "text": text,
            "color": list(color),
            "font": int(font),
            "position": list(position),
        })

    # ---------------------------
    # Per-graphic labels from halo fields
    # ---------------------------
    def addFieldsToLabels(self,
                          fmt: str,
                          fields: Union[str, List[str]],
                          halo_ids: Optional[List[int]] = None) -> None:
        """
        Append formatted field values to each matching graphic's label.
        If any field is time-varying, a per-frame label list is created for that graphic.
        Positions will be auto-stacked in render step.
        """
        if isinstance(fields, str):
            field_list = [fields]
        else:
            field_list = list(fields)
        if not field_list:
            raise ValueError("fields must be a non-empty string or list of strings.")
        target_hids = self.sys.hids.tolist() if halo_ids is None else set(halo_ids)
        for g in self.graphics:
            halo = getattr(g, "halo", None)
            if halo is None:
                # graphics that don’t wrap a halo (e.g., Arrow) will reject via NotImplemented in graphic method
                continue
            if (halo.hid not in target_hids):
                continue
            if hasattr(g, "addFieldToLabel") and g.hasLabel():
                g.addFieldToLabel(fmt, field_list, start=self.start, stop=self.stop)
            else:
                # If a subclass opts out
                pass

    # --- Basic camera controls ---

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
    
    def setCameraPosition(self, pos: np.ndarray) -> None:
        """
        Accepts:
          - static: np.ndarray of shape (3,), or list/tuple of length 3
          - dynamic: np.ndarray of shape (N, 4) [[t, x, y, z], ...],
                     or list of lists with 4 numbers each
        Stores a NumPy array in render_props[gn.CAM_POS].
        """
        key = gn.CAM_POS

        if isinstance(pos, np.ndarray):
            if pos.ndim == 1 and pos.shape == (3,):
                arr = pos.astype(float, copy=False)
            elif pos.ndim == 2 and pos.shape[1] == 4:
                arr = pos.astype(float, copy=False)
            else:
                raise ValueError("Camera position ndarray must be shape (3,) or (N,4) for dynamic.")
        elif isinstance(pos, (list, tuple)):
            # list/tuple → prefer converting to ndarray
            if len(pos) == 3 and all(isinstance(v, (int, float)) for v in pos):
                arr = np.asarray(pos, dtype=float)
            elif len(pos) > 0 and isinstance(pos[0], (list, tuple)):
                if not all(len(e) == 4 and all(isinstance(v, (int, float)) for v in e) for e in pos):
                    raise ValueError("Dynamic camera position must be [[t, x, y, z], ...].")
                arr = np.asarray(pos, dtype=float)
            else:
                raise ValueError("Camera position must be (3,) or [[t, x, y, z], ...].")
        else:
            raise TypeError("Unsupported type for camera position.")

        self.render_props[key] = arr

    def setCameraFocusPoint(self, foc: np.ndarray) -> None:
        """
        static: (3,)
        dynamic: (N,4) [[t, fx, fy, fz], ...]
        """
        key = gn.CAM_FOC

        if isinstance(foc, np.ndarray):
            if foc.ndim == 1 and foc.shape == (3,):
                arr = foc.astype(float, copy=False)
            elif foc.ndim == 2 and foc.shape[1] == 4:
                arr = foc.astype(float, copy=False)
            else:
                raise ValueError("Camera focus ndarray must be shape (3,) or (N,4).")
        elif isinstance(foc, (list, tuple)):
            if len(foc) == 3 and all(isinstance(v, (int, float)) for v in foc):
                arr = np.asarray(foc, dtype=float)
            elif len(foc) > 0 and isinstance(foc[0], (list, tuple)):
                if not all(len(e) == 4 and all(isinstance(v, (int, float)) for v in e) for e in foc):
                    raise ValueError("Dynamic camera focus must be [[t, fx, fy, fz], ...].")
                arr = np.asarray(foc, dtype=float)
            else:
                raise ValueError("Camera focus must be (3,) or [[t, fx, fy, fz], ...].")
        else:
            raise TypeError("Unsupported type for camera focus.")

        self.render_props[key] = arr

    def setCameraViewUp(self, up : np.ndarray) -> None:
        """
        static: (3,)
        dynamic: (N,4) [[t, ux, uy, uz], ...]
        """
        key = gn.CAM_UP

        if isinstance(up, np.ndarray):
            if up.ndim == 1 and up.shape == (3,):
                arr = up.astype(float, copy=False)
            elif up.ndim == 2 and up.shape[1] == 4:
                arr = up.astype(float, copy=False)
            else:
                raise ValueError("Camera view-up ndarray must be shape (3,) or (N,4).")
        elif isinstance(up, (list, tuple)):
            if len(up) == 3 and all(isinstance(v, (int, float)) for v in up):
                arr = np.asarray(up, dtype=float)
            elif len(up) > 0 and isinstance(up[0], (list, tuple)):
                if not all(len(e) == 4 and all(isinstance(v, (int, float)) for v in e) for e in up):
                    raise ValueError("Dynamic camera view-up must be [[t, ux, uy, uz], ...].")
                arr = np.asarray(up, dtype=float)
            else:
                raise ValueError("Camera view-up must be (3,) or [[t, ux, uy, uz], ...].")
        else:
            raise TypeError("Unsupported type for camera view-up.")

        self.render_props[key] = arr

    def setCameraParallelScale(self, scale: np.ndarray) -> None:
        """
        static: scalar or ndarray shape (1,)
        dynamic: (N,2) [[t, scale], ...]
        """
        key = gn.CAM_ZOOM

        if isinstance(scale, np.ndarray):
            if scale.ndim == 0 or (scale.ndim == 1 and scale.size == 1):
                arr = np.asarray(float(scale), dtype=float)
            elif scale.ndim == 2 and scale.shape[1] == 2:
                arr = scale.astype(float, copy=False)
            else:
                raise ValueError("Parallel scale ndarray must be scalar-like or shape (N,2).")
        elif isinstance(scale, (int, float)):
            arr = np.asarray(float(scale), dtype=float)
        elif isinstance(scale, (list, tuple)):
            if len(scale) > 0 and isinstance(scale[0], (list, tuple)):
                if not all(len(e) == 2 and all(isinstance(v, (int, float)) for v in e) for e in scale):
                    raise ValueError("Dynamic parallel scale must be [[t, scale], ...].")
                arr = np.asarray(scale, dtype=float)
            elif len(scale) == 1 and isinstance(scale[0], (int, float)):
                arr = np.asarray(float(scale[0]), dtype=float)
            else:
                raise ValueError("Parallel scale must be scalar-like or [[t, scale], ...].")
        else:
            raise TypeError("Unsupported type for parallel scale.")

        self.render_props[key] = arr

    def setCameraViewAngle(self, angle: np.ndarray) -> None:
        """
        Note that this will almost always not be used - for parallel perspective, use the zoom feature
        static: scalar or ndarray shape (1,)
        dynamic: (N,2) [[t, view_angle_degrees], ...]
        """
        key = gn.CAM_ANGLE  # gn.CAM_ANGLE or "camera_view_angle"

        if isinstance(angle, np.ndarray):
            if angle.ndim == 0 or (angle.ndim == 1 and angle.size == 1):
                arr = np.asarray(float(angle), dtype=float)
            elif angle.ndim == 2 and angle.shape[1] == 2:
                arr = angle.astype(float, copy=False)
            else:
                raise ValueError("View angle ndarray must be scalar-like or shape (N,2).")
        elif isinstance(angle, (int, float)):
            arr = np.asarray(float(angle), dtype=float)
        elif isinstance(angle, (list, tuple)):
            if len(angle) > 0 and isinstance(angle[0], (list, tuple)):
                if not all(len(e) == 2 and all(isinstance(v, (int, float)) for v in e) for e in angle):
                    raise ValueError("Dynamic view angle must be [[t, degrees], ...].")
                arr = np.asarray(angle, dtype=float)
            elif len(angle) == 1 and isinstance(angle[0], (int, float)):
                arr = np.asarray(float(angle[0]), dtype=float)
            else:
                raise ValueError("View angle must be scalar-like or [[t, degrees], ...].")
        else:
            raise TypeError("Unsupported type for view angle.")

        self.render_props[key] = arr

    def setCameraParallelProjection(self, flag):
        """
        Set camera projection mode: True (parallel) or False (perspective).
        """
        if isinstance(flag, bool):
            self.render_props[gn.CAM_PROJ] = flag
        else:
            raise ValueError("Camera parallel projection must be a boolean.")
        return
    
    # def setInterpolation(self, interp_type: str, interp_step : int = 2) -> None:
    #     """
    #     Set interpolation mode used by camera keyframes in ParaView.
    #     e.g., "Linear" (default), "Spline", "Step", etc.
    #     """
    #     if interp_type not in gn.ALLOWED_INTERP_TYPES:
    #         raise ValueError(f"{interp_type} not in compatible interpolation types {gn.ALLOWED_INTERP_TYPES}")
    #     self.render_props[gn.INTERP_TYPE] = interp_type
    #     self.render_props[gn.INTERP_STEP] = str(interp_step)

    def setViewSize(self, size):
        if (isinstance(size, (tuple, list)) and len(size) == 2
                and all(isinstance(x, (int, float)) for x in size)):
            self.render_props[gn.VIEW_SIZE] = size
        else:
            raise ValueError("View size must be tuple/list of 2 numbers.")

    def getViewBox(self): # we default to using system's view box
        return self.sys._getRange(self.start, self.stop)

    # --- Complex Camera Controls ---
    def autoCam(self, padding_frac: float = 0.0, overwrite = False) -> None:
        """
        Pick a camera whose screen plane is spanned by the two PCA axes of
        greatest variance and that tightly encloses every halo, with padding.
        If the user has already set a camera property, we do not overwrite
        their settings.
        """
        # 1) Data-driven axes ---------------------------------------------------
        viewDir, viewUp = self.sys._findOptAxis()        # unit vectors
        right   = np.cross(viewUp, viewDir)              # second in-plane axis

        # 2) Gather the eight AABB corners ------------------------------------
        mins, maxs = self.getViewBox()                   # world-axis AABB
        corners = np.array([[x, y, z]                    # 8 combinations
                            for x in (mins[0], maxs[0])
                            for y in (mins[1], maxs[1])
                            for z in (mins[2], maxs[2])])

        center = (mins + maxs) / 2.0
        pad    = padding_frac * (maxs - mins).max()

        # 3) Project the corners into the new camera basis --------------------
        # screen plane axes: right (X), viewUp (Y)
        proj_2d = (corners - center) @ np.vstack([right, viewUp]).T  # (8×2)
        width, height = proj_2d.ptp(axis=0)               # extents in plane
        width  += 2 * pad
        height += 2 * pad
        parallel_scale = height / 2.0                     # ParaView definition

        # depth along viewDir to avoid near-plane clipping
        depth_extent = np.dot(corners - center, viewDir).ptp()
        distance     = depth_extent / 2.0 + pad
        cam_pos      = center + viewDir * distance        # absolute 3-vector

        # 4) Commit camera & view size ----------------------------------------
        if gn.CAM_FOC not in self.render_props or overwrite:
            self.setCameraFocusPoint(tuple(center))
        if gn.CAM_POS not in self.render_props or overwrite:
            self.setCameraPosition(tuple(cam_pos))
        if gn.CAM_UP not in self.render_props or overwrite:
            self.setCameraViewUp(tuple(viewUp))
        if gn.CAM_ZOOM not in self.render_props or overwrite:
            self.setCameraParallelScale(parallel_scale)
        if gn.VIEW_SIZE not in self.render_props or overwrite:
            self.setViewSize((width, height))



    # --- Phase 2 (Graphic creation/manipulation) ---
    
    def getGraphics(self) -> List[Graphic]:
        return self.graphics
    
    def addGraphic(self, graphic) -> None:
        graphic.setUnits(self.units)
        if self.rm_graphics and isinstance(graphic, (Marker, Sphere, Line)):
            death_snap = graphic.halo._last
            # only adjust if the graphic stays after death or appears before death
            if (graphic.disp_stop > death_snap) and (graphic.disp_start < death_snap):
                graphic.setDisplaySnaps(graphic.disp_start, death_snap)
        self.graphics.append(graphic)
        return
    
    def addEvents(self):
        for h in self.sys.halos:
            events = h.getEvents()
            for ev in events:
                m = Marker(h, ev)
                ds = ev.def_style
                m.setStyle(ev.def_style)
                if gn.SIZE not in ds:
                    m.setSize(self._getDefaultPointSize())
                else:
                    m.setSize(ds[gn.SIZE])
                m.setDisplaySnaps(ev.getSnap(), self.stop)
                self.addGraphic(m)

        return

    
    def addTjy(self, halo_id) -> Line:
        halo = self.sys.getHalo(halo_id)
        traj = Line(halo)
        # default: only show tjy when halo exists, but keep path until anim ends
        traj.setDisplaySnaps(halo._first, self.stop) 
        self.addGraphic(traj)
        return traj
    
    def addSphere(self, halo_id) -> Sphere:
        halo = self.sys.getHalo(halo_id)
        sphere = Sphere(halo)
        self.addGraphic(sphere)
        return sphere
    

    def addSegment(self, halo_id, start, stop):
        halo = self.sys.getHalo(halo_id)
        seg = Line(halo)
        seg.setTjySnaps(start, stop)
        seg.setDisplaySnaps(start, self.stop)
        seg.setName(f"{seg.getName()}_{start}_{stop}")
        self.addGraphic(seg)
        return seg
    
    def addParentArrows(self, halo_id, pid_key = gn.UPID) -> List[Arrow]:
        halo = self.sys.getHalo(halo_id)
        pids = halo.getField(pid_key)
        has_parent = pids > 0
        unq_pids = np.unique(pids[has_parent])
        arr_list = []
        for up in unq_pids:
            phalo = self.sys.getHalo(up)
            arrow = Arrow(halo, phalo)
            arrow.setDisplayMask(pids == up)
            default_name = arrow.getName()
            arrow.setName(default_name + '_' + pid_key)
            arrow.setNorm(self._getDefaultArrowSize())
            arr_list.append(arrow)
            self.addGraphic(arrow)

        return arr_list
    
    def createDeathEvents(self):
        self.sys._createDeathEvents(self.stop)
    
    def _getDefaultPointSize(self):
        mins, maxs = self.getViewBox()
        spans = maxs - mins            # [dx, dy, dz]
        # rank axes by span: [smallest, middle, largest]
        large_ax = np.max(spans)

        return 0.005 * large_ax
    
    def _getDefaultArrowSize(self):
        mins, maxs = self.getViewBox()
        spans = maxs - mins            # [dx, dy, dz]
        # rank axes by span: [smallest, middle, largest]
        large_ax = np.max(spans)

        return 0.02 * large_ax
    
    def deleteGraphicsAfterDeath(self):

        
        self.rm_graphics = True
        for gph in self.graphics:
            # by default, arrows are not drawn when halo dies
            # only need to rm lines, markers and spheres
            if isinstance(gph, (Marker, Sphere, Line)):
                death_snap = gph.halo._last
                gph.setDisplaySnaps(gph.disp_start, death_snap)

        return

    def setUnits(self, utype):
        # save setting for future graphics
        allowed_units = ['phy', 'com']
        if utype in allowed_units:
            self.units = utype
        else:
            raise ValueError(f"{utype} is not allowed unit type ({allowed_units}).")
        # set the units of all previously made graphics
        for gph in self.graphics:
            gph.setUnits(utype)
        return


class HostView(Scene):
    """
    For scenes that involve following a single host/satellites system. Methods are provided to give
    special treatment to the host.
    """
    def __init__(self, system : System, pov_id : int): # also allow pov to be halo object
        self.pov = system.getHalo(pov_id)
        super().__init__(system)
        self.setAnimSnap(self.pov._first, self.pov._last)
        self.sys.setHaloOrigin(self.pov.hid)
        return

    def showHostBoundary(self):
        sphere = self.addSphere(self.pov.hid)
        sphere.setOpacity(0.3)
        return
    
    def showOrigin(self, org_size : float = 0.2):
        """_summary_

        Args:
            org_size (float, optional): size of origin marker, expressed as fraction of
             default calculated by the system. Defaults to 0.2.
        """
        org = Event(self.pov._first, "origin")
        org.def_style = {
            gn.COLOR: (1, 1, 1), # WHITE
            gn.SHAPE: 'sphere',
            gn.SIZE: self._getDefaultPointSize() * org_size
        }
        self.pov.addEvent(org)
        return

    def showTjys(self):
        # show tjys of all halos not pov
        for id_i in self.sys.hids:
            if id_i == self.pov.hid:
                continue
            self.addTjy(id_i)
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
            alt_sub_color = (1.0, 0.75, 0.79) # pink

        ghost_style = {gn.LSTYLE: 'dashed'}

        pov_id = self.pov.hid

        for hid in self.sys.hids:
            # don't draw tjy for pov halo
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
            categories = [get_category(pid[i]) for i in range(N)]
            ghosts = [do_ghost and status[i] >= 30 for i in range(N)]

            # Segment the trajectory by contiguous blocks of the same category and ghost flag

            seg_starts = [max(halo._first, self.start)]
            seg_cats = [(categories[0], ghosts[0])]
            for i in range(1, N):
                if (categories[i], ghosts[i]) != (categories[i-1], ghosts[i-1]):
                    seg_starts.append(i + seg_starts[0])
                    seg_cats.append((categories[i], ghosts[i]))
            seg_starts.append(min(halo._last, self.stop))  # end
            for i in range(len(seg_starts)-1):
                if i == 0:
                    start_snap = seg_starts[i]
                else:
                    start_snap = seg_starts[i] - 1
                stop_snap = seg_starts[i+1]
                seg_len = stop_snap - start_snap
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

                
                
                # if it's the first segment *and* genuinely only one point, do a Marker
                if seg_len == 1 and i == 0:

                    # Add a Marker instead of a Line
                    evt = Event(start_snap)
                    m = Marker(halo, evt)
                    m.setColor(color)
                    m.setSize(self._getDefaultPointSize())
                    m.setLabel(seg_cat)
                    m.setShape('sphere')
                    m.setName(f"marker_{hid}_{seg_cat}_{start_snap}")
                    self.addGraphic(m)
                else:
                    segment = self.addSegment(hid, start_snap, stop_snap)

                    segment.setColor(color)
                    if seg_ghost:
                        segment.setStyle(ghost_style)
        return
   


class TrackHost(Scene):

    def __init__(self, system, pov_id):
        super().__init__(system, pov_id)
        self.autoCam(overwrite = True)
        alv = self.pov.getAlive()
        pos = self.pov.getPos()
        snaps = np.arange(len(alv))
        times = snaps[alv]
        pos = pos[alv]
        los_ax = self.render_props[gn.CAM_UP]
        offset_ax = np.array(los_ax) * self.sys.boxsize/2
        offset = np.repeat(offset_ax.reshape(1, 3), times.size, axis = 0)
        if gn._DEF_UNITS == 'com':
            cam_foc = np.hstack((times.reshape(times.size, 1), pos))
            cam_pos = cam_foc.copy()
            cam_pos[:, 1:] += offset
            self.setCameraPosition(cam_pos)
            self.setCameraFocusPoint(cam_foc)
        else:
            pos /= (1 + self.pov.z[:, np.newaxis])
            cam_foc = np.hstack((times.reshape(times.size, 1), pos))
            cam_pos = cam_foc.copy()
            cam_pos[:, 1:] += offset
            self.setCameraPosition(cam_pos)
            self.setCameraFocusPoint(cam_foc)
    
    
    def setUnits(self, utype):
        super().setUnits(utype)
        # camera positions depend on unit type, so we have to adjust that too
        if self.method == 'cam':
            alv = self.pov.getAlive()
            pos = self.pov.getPos()
            snaps = np.arange(len(alv))
            times = snaps[alv]
            pos = pos[alv]
            los_ax = self.render_props[gn.CAM_UP]
            offset_ax = np.array(los_ax) * self.sys.boxsize/2
            offset = np.repeat(offset_ax.reshape(1, 3), times.size, axis = 0)
            if utype == 'com':
                cam_foc = np.hstack((times.reshape(times.size, 1), pos))
                cam_pos = cam_foc.copy()
                cam_pos[:, 1:] += offset
                self.setCameraPosition(cam_pos)
                self.setCameraFocusPoint(cam_foc)
            else:
                pos /= (1 + self.pov.z[:, np.newaxis])
                cam_foc = np.hstack((times.reshape(times.size, 1), pos))
                cam_pos = cam_foc.copy()
                cam_pos[:, 1:] += offset
                self.setCameraPosition(cam_pos)
                self.setCameraFocusPoint(cam_foc)
        return
    
    def setCameraViewUp(self, up : np.ndarray) -> None:
        """
        static: (3,)
        dynamic: (N,4) [[t, ux, uy, uz], ...]
        """
        key = gn.CAM_UP

        if isinstance(up, np.ndarray):
            if up.ndim == 1 and up.shape == (3,):
                arr = up.astype(float, copy=False)
            elif up.ndim == 2 and up.shape[1] == 4:
                arr = up.astype(float, copy=False)
            else:
                raise ValueError("Camera view-up ndarray must be shape (3,) or (N,4).")
        elif isinstance(up, (list, tuple)):
            if len(up) == 3 and all(isinstance(v, (int, float)) for v in up):
                arr = np.asarray(up, dtype=float)
            elif len(up) > 0 and isinstance(up[0], (list, tuple)):
                if not all(len(e) == 4 and all(isinstance(v, (int, float)) for v in e) for e in up):
                    raise ValueError("Dynamic camera view-up must be [[t, ux, uy, uz], ...].")
                arr = np.asarray(up, dtype=float)
            else:
                raise ValueError("Camera view-up must be (3,) or [[t, ux, uy, uz], ...].")
        else:
            raise TypeError("Unsupported type for camera view-up.")
        self.render_props[key] = arr

        # TODO: changing the cameraViewUp requires additional logic when in method 'cam', since
        # the offsets we give to the camera position need to be along the line-of-sight.

    
class TjyComp(HostView):
    """
    For scenes where we want to compare the trajectories of two instances of the
    same subhalo. We expect that these will be from the PoV of the host, so the
    class inherits from HaloView.
    """
    def __init__(self, system : System, pov_id : int, pos_alt_key : str):
        super().__init__(system, pov_id)
        # we need to create duplicate halo containers for the alt positions
        alt_halos = []
        for h in self.sys.hids:
            halo = self.sys.getHalo(h)
            alt_pos = halo.getField(pos_alt_key)
            alt_halos.append(Halo(h, alt_pos, halo.z, **halo.fields))
        self.alt_sys = System(alt_halos, self.sys.boxsize)
        self.alt_sys.setHaloOrigin(pov_id)
        self.fid_style = {
            gn.COLOR: (1, 0, 0)        }
        self.alt_style = {
            gn.COLOR: (0, 0, 1)
        }
        return
    
    # ------------------------------------------------------------------
    # Style setters
    # ------------------------------------------------------------------
    def setFidStyle(
        self,
        color = None,
        opacity = None,
        lwidth = None,
        lstyle = None,
    ) -> None:
        """Update the style used for **fiducial** trajectories."""
        self._update_style_dict(self.fid_style, color, opacity, lwidth, lstyle)

    def setAltStyle(
        self,
        color = None,
        opacity = None,
        lwidth = None,
        lstyle = None,
    ) -> None:
        """Update the style used for **alternative** trajectories."""
        self._update_style_dict(self.alt_style, color, opacity, lwidth, lstyle)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _update_style_dict(
        sdict,
        color,
        opacity,
        lwidth,
        lstyle,
    ) -> None:
        """Validate and write the supplied kwargs into *sdict* in‐place."""
        if color is not None:
            if not (isinstance(color, (tuple, list)) and len(color) == 3):
                raise ValueError("color must be a 3-tuple of floats in [0, 1]")
            sdict[gn.COLOR] = tuple(float(c) for c in color)

        if opacity is not None:
            if not (0.0 <= opacity <= 1.0):
                raise ValueError("opacity must be in the range [0, 1]")
            sdict[gn.OPACITY] = float(opacity)

        if lwidth is not None:
            if lwidth <= 0:
                raise ValueError("linewidth must be positive")
            sdict[gn.LWIDTH] = float(lwidth)

        if lstyle is not None:
            if lstyle not in gn.ALLOWED_LINESTYLES:
                raise ValueError(
                    f"linestyle must be one of {gn.ALLOWED_LINESTYLES}"
                )
            sdict[gn.LSTYLE] = lstyle


            
    def showTjys(self):


        for id_i in self.sys.hids:
            if id_i == self.pov.hid:
                continue
            halo = self.sys.getHalo(id_i)
            alt_halo = self.alt_sys.getHalo(id_i)
            # create line
            traj = Line(halo)
            traj.setDisplaySnaps(halo._first, self.stop) 
            traj.setStyle(self.fid_style)
            self.addGraphic(traj)
            
            traj = Line(alt_halo)
            traj.setDisplaySnaps(alt_halo._first, self.stop)
            traj.setStyle(self.alt_style)
            self.addGraphic(traj)
        return

    
    def showTjyByStatus(self, host_color=None, sub_color=None, alt_sub_color=None, do_ghost=False):
        raise NotImplementedError("function not defined in TjyComp")
        


    
class MultiView(Scene):
    """
    For scenes that involve following multiple halos. This is usually intended to 
    assess mergers or interactions between multi-level host-sub systems.
    """
    def __init__(self, system: System):
        super().__init__(system)
        self.setUnits('com') # assume this will be in comoving units
        # set origin to be around last CoM (remember _last is exclusive)
        last_snap = 0
        for h in self.sys.halos:
            if h._last - 1 > last_snap:
                last_snap = h._last - 1
            
        com = self.sys.getCoM(last_snap)
        self.sys.setCustomOrigin(com)

        
    def showBoundaries(self, halo_ids : Union[int, List[int]]):
        halo_ids = _toIterable(halo_ids)
        for hi in halo_ids:
            sphere = self.addSphere(hi)
            sphere.setOpacity(0.1)

        return
    
    def showTjys(self):
        for h in self.sys.halos:
            self.addTjy(h.hid)
        return
    
    
class SubhaloView(MultiView):
    """
    For following relationships/interactions of a single subhalo
    """
    def __init__(self, system: System, pov_id:int):
        super().__init__(system)
        self.pov = self.sys.getHalo(pov_id)
        self.astyles = []
        self.setAnimSnap(self.pov._first, self.pov._last)
        return
    
    def setArrowTypeStyle(self, pid_keys, astyle = {}, **kwargs):
        if isinstance(pid_keys, str):
            pid_keys = [pid_keys]
        style = copy.deepcopy(astyle)
        style.update(kwargs)
        self.astyles.append((pid_keys, style))
        return
    
    def addParentArrows(self, pid_key):
        style = {}
        for ar in self.astyles:
            if pid_key in ar[0]:
                style.update(ar[1])
        # TODO catch any repeats/conflicts in style (if they appear more than once in astyles)
        arrows = super().addParentArrows(self.pov.hid, pid_key)
        for arr in arrows:
            arr.setStyle(style)

    # --- Methods for applying styles to Traj ---
    def showTjys(self, show_sub = True, id_label = False):
        for h in self.sys.halos:
            if not show_sub and h.hid == self.pov.hid:
                continue
            tjy = self.addTjy(h.hid)
            if id_label:
                tjy.setLabel(f"{h.hid}")
        return
    

    def colorByHalo(self):
        # TODO save color settings, when adding grapchic apply colors. Move to super class
        halo_to_color = {self.pov.hid:(1, 1, 1)}
        cidx = 0
        for gph in self.graphics:
            # color lines by halo
            if isinstance(gph, Line) or isinstance(gph, Sphere):
                if gph.halo.hid in halo_to_color:
                    gph.setColor(halo_to_color[gph.halo.hid])
                else:
                    col = gn.COLOR_CYCLE[cidx]
                    gph.setColor(col)
                    halo_to_color[gph.halo.hid] = col
                    cidx += 1
            elif isinstance(gph, Marker) and hasattr(gph.event, 'host_id'):
                if gph.event.host_id in halo_to_color:
                    gph.setColor(halo_to_color[gph.event.host_id])
                else:
                    col = gn.COLOR_CYCLE[cidx]
                    gph.setColor(col)
                    halo_to_color[gph.event.host_id] = col
                    cidx += 1
        return
    
    def showParentDif(self,
                fid_key,
                alt_key,
                agree_col = (1, 1, 1),
                agree_lab = 'Agree',
                host_to_sub_col = (0, 0, 1),
                host_to_sub_lab = 'Host -> Subhalo',
                sub_to_host_col = (1, 0, 0),
                sub_to_host_lab = 'Subhalo -> Host',
                dif_parent_col = (0, 1, 0),
                dif_parent_lab = 'Parent Dif'
            ):

        h = self.pov
        anim_slc = slice(self.start, self.stop)
        alv = h.getAlive()[anim_slc]
        fid_pids = h.getField(fid_key)[anim_slc][alv]
        alt_pids = h.getField(alt_key)[anim_slc][alv]
        
        snaps = np.arange(self.start, self.stop)
        relevant_snaps = snaps[alv]
        if relevant_snaps.size == 0:
            return
        
        # create masks for each category
        N = len(relevant_snaps)
        fid_sub = fid_pids > 0
        alt_sub = alt_pids > 0
        same = fid_pids == alt_pids
        hts_mask = (~fid_sub & alt_sub)
        sth_mask = (fid_sub & ~alt_sub)
        dp_mask = (fid_sub & alt_sub & ~same)
        categories = np.zeros(N, dtype = int)
        categories[hts_mask] = 1
        categories[sth_mask] = 2
        categories[dp_mask] = 3

        # prepare segment info
        seg_starts = [max(h._first, self.start)]
        seg_cats = [categories[0]]
        for i in range(1, N):
            if categories[i] != categories[i-1]:
                seg_starts.append(i + seg_starts[0])
                seg_cats.append(categories[i])
        seg_starts.append(min(h._last, self.stop))
        # create segments
        colors = [agree_col, host_to_sub_col, sub_to_host_col, dif_parent_col]
        labels = [agree_lab, host_to_sub_lab, sub_to_host_lab, dif_parent_lab]
        for i in range(len(seg_starts) - 1):
            if i == 0:
                start_snap = seg_starts[i]
            else:
                start_snap = seg_starts[i] - 1
            stop_snap = seg_starts[i+1]
            seg_len = stop_snap - start_snap
            seg_cat = seg_cats[i]

            style = {}
            style[gn.COLOR] = colors[seg_cat]
            style[gn.LABEL] = labels[seg_cat]


            # if it's the first segment *and* genuinely only one point, do a Marker
            if seg_len == 1 and i == 0:

                # Add a Marker instead of a Line
                evt = Event(start_snap)
                m = Marker(h, evt)
                m.setStyle(style)
                m.setSize(self._getDefaultPointSize())
                m.setShape('sphere')
                m.setName(f"marker_{h.hid}_{start_snap}")
                self.addGraphic(m)
            else:
                segment = self.addSegment(h.hid, start_snap, stop_snap)
                segment.setStyle(style)
        return


class HierarchyView(Scene):
    # for viewing multi-level systems
    def __init__(self, system):
        super().__init__(system)

    
    

    

