from graphics import Graphic, Sphere, Line, Marker, Arrow
from containers import Halo, System, Event
from typing import Dict, List
import numpy as np
import global_names as gn
import copy

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

        self.render_props = {}
        self.setBackgroundColor((0, 0, 0)) # want black background for almost all scenes
        self.setCameraParallelProjection(True) # almost always don't want perspective projection
        return

    # --- Phase 1: Scene Property Methods ---
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
    

    def getSceneProps(self):
        rprops = copy.deepcopy(self.render_props)
        # can I delete these?
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

    def getViewBox(self): # we default to using system's view box
        return self.sys._getRange(self.start, self.stop)

    def autoCam(self, padding_frac: float = 0.05) -> None:
        """
        Pick a camera whose screen plane is spanned by the two PCA axes of
        greatest variance and that tightly encloses every halo, with padding.
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
        self.setCameraFocusPoint(tuple(center))
        self.setCameraPosition(tuple(cam_pos))
        self.setCameraViewUp(tuple(viewUp))
        self.setCameraParallelScale(parallel_scale)
        self.setViewSize((width, height))

    

    # --- Phase 2 (Graphic creation/manipulation) ---
    
    def getGraphics(self) -> List[Graphic]:
        return self.graphics
    
    def addGraphic(self, graphic) -> None:
        self.graphics.append(graphic)
        return
    
    def addEvents(self):
        for h in self.sys.halos:
            events = h.getEvents()
            for ev in events:
                m = Marker(h, ev)
                ds = ev.def_style
                m.setStyle(ev.def_style)
                # m.setColor(ds[gn.COLOR])
                # m.setLabel(ds[gn.LABEL])
                # m.setShape(ds[gn.SHAPE])
                if gn.SIZE not in ds:
                    m.setSize(self._getDefaultPointSize())
                else:
                    m.setSize(ds[gn.SIZE])
                m.setDisplaySnaps(ev.getSnap(), self.stop)
                self.graphics.append(m)

        return

    
    def makeTjy(self, halo_id) -> Line:
        halo = self.sys.getHalo(halo_id)
        traj = Line(halo)
        # default: only show tjy when halo exists, but keep path until anim ends
        traj.setDisplaySnaps(halo._first, self.stop) 
        # default: label is ID
        traj.setLabel(str(halo.hid))
        return traj
    
    def makeSphere(self, halo_id) -> Sphere:
        halo = self.sys.getHalo(halo_id)
        sphere = Sphere(halo)
        sphere.setLabel(str(halo.hid))
        return sphere
    

    def makeSegment(self, halo_id, start, stop):
        halo = self.sys.getHalo(halo_id)
        seg = Line(halo)
        seg.setTjySnaps(start, stop)
        seg.setDisplaySnaps(start, self.stop)
        seg.setName(f"{seg.getName()}_{start}_{stop}")
        return seg
    
    def makeParentArrows(self, halo_id, pid_key = gn.UPID) -> List[Arrow]:
        halo = self.sys.getHalo(halo_id)
        pids = halo.getField(pid_key)
        alv = halo.getAlive()
        unq_pids = np.unique(pids[alv])
        arr_list = []
        for up in unq_pids:
            phalo = self.sys.getHalo(up)
            arrow = Arrow(halo, phalo)
            arrow.setDisplayMask(pids == up)
            arr_list.append(arrow)
   
        return arr_list
    
    def createDeathEvents(self):
        self.sys._createDeathEvents(self.stop)
    
    def _getDefaultPointSize(self):
        mins, maxs = self.getViewBox()
        spans = maxs - mins            # [dx, dy, dz]
        # rank axes by span: [smallest, middle, largest]
        large_ax = np.max(spans)

        return 0.005 * large_ax

    # phase 3: methods for adjusting graphics
    def deleteGraphicsAfterDeath(self):
        for gph in self.graphics:
            # by default, arrows are not drawn when halo dies
            # only need to rm lines, markers and spheres
            if isinstance(gph, (Marker, Sphere, Line)):
                death_snap = gph.halo._last
                gph.setDisplaySnaps(gph.disp_start, death_snap)

        return
    # doesn't work...
    # def rmLinePoints(self):
    #     for gph in self.graphics:
    #         if isinstance(gph, Line):
    #             gph.setShowPoints(False)

class HostView(Scene):
    """
    For scenes that involve following a single host/satellites system. Methods are provided to give
    special treatment to the host.
    """
    def __init__(self, system : System, pov_id : int): # also allow pov to be halo object
        self.pov = system.getHalo(pov_id)
        super().__init__(system)
        self.sys.setHaloOrigin(self.pov.hid)
        self.setAnimSnap(self.pov._first, self.pov._last)

        return

    def hostBoundary(self):
        sphere = self.makeSphere(self.pov.hid)
        sphere.setOpacity(0.3)
        self.addGraphic(sphere)
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
            gn.LABEL: str(self.pov.hid),
            gn.SIZE: self._getDefaultPointSize() * org_size
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

            seg_starts = [halo._first]
            seg_cats = [(categories[0], ghosts[0])]
            for i in range(1, N):
                if (categories[i], ghosts[i]) != (categories[i-1], ghosts[i-1]):
                    seg_starts.append(i + halo._first)
                    seg_cats.append((categories[i], ghosts[i]))
            seg_starts.append(halo._last)  # end
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
                    segment = self.makeSegment(hid, start_snap, stop_snap)

                    segment.setColor(color)
                    if seg_ghost:
                        segment.setStyle(ghost_style)
                    self.addGraphic(segment)
        return
    
class HostViewZoom(HostView):

    def __init__(self, system, pov_id):
        super().__init__(system, pov_id)
        if self.pov.hasField(gn.RADIUS):
            self.zoom_box_length = 1.05 * np.max(self.pov.radius)
        else:
            print("WARNING: default zoom box length requires host radius, user must set...")
            self.zoom_box_length = None

    def setViewCube(self, box_length):
        self.zoom_box_length = box_length
        return
    
    def setViewMaxRad(self, max_rad_frac):
        
        self.zoom_box_length = max_rad_frac * np.max(self.pov.radius)
    
    def setViewRadSnap(self, rad_frac, snap):
        self.zoom_box_length = rad_frac * self.pov.radius[snap]
    

    def onlyInBox(self): 
        # only show graphics that occur within the box. without this on, graphics that
        # occur along the los will still be shown even without being close by.

        # if we are not zoomed in, then by default all graphics are included

        
        for gph in self.graphics:
            mins, maxs = self.getViewBox()
            gph.showOnlyInView(mins, maxs)
        return
                
    def getViewBox(self):
        if self.zoom_box_length is None:
            raise ValueError("zoom box length not set")
        mins = np.array([-self.zoom_box_length]*3)
        maxs = np.array([self.zoom_box_length]*3)
        return mins, maxs

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
            gn.COLOR: (1, 0, 0),
            gn.LABEL: 'catalog'
        }
        self.alt_style = {
            gn.COLOR: (0, 0, 1),
            gn.LABEL: 'sparta'
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
        

class SubhaloView(HostView):

    def __init__(self, system, pov_id):
        super().__init__(system, pov_id)

    def hostBoundary(self):
        # find the halos that are the pov hosts, create spheres
        return
    
    def showOrigin(self, org_size = 0.2):
        # marker color changes with
        return super().showOrigin(org_size)

    def showHostArrows(self):
        return
    
class MultiView(Scene):
    """
    For scenes that involve following multiple halos. This is usually intended to 
    assess mergers of multi-level host-sub systems.
    """
    def __init__(self, system: System):
        super().__init__(system) 

    


class HostSub(Scene):

    def __init__(self, system: System, alt_upid, alt_pid):
        super().__init__(system)
        self.alt_upid = alt_upid
        self.alt_pid = alt_pid
        self.fid_upid = gn.UPID
        self.fid_pid = gn.PID
        # arrow stylings
        self.fid_style = {
            gn.COLOR: (0, 0, 1),
            gn.LABEL: self.fid_pid
        }
        self.alt_style = {
            gn.COLOR: (1, 0, 0),
            gn.LABEL: self.alt_upid
        }
        self.upid_style = {
            gn.LWIDTH: 2
        }
        self.pid_style = {
            gn.LWIDTH: 1
        }
        
        # we can't display both upid and pid difs, so keep track of which is shown
        self.showing_status = False
        return
    
    
    def setFidKey(self, upid_key, pid_key):
        # check that all halos have these keys
        change_label = (self.fid_pid == self.fid_style[gn.LABEL])
        self.sys._checkKey(upid_key)
        self.sys._checkKey(pid_key)
        self.fid_pid = pid_key
        self.fid_upid = upid_key
        
        # if the label had default, then change. If user set it, don't change
        if change_label:
            self.fid_style[gn.LABEL] = pid_key
        return
    
    def setAltKey(self, upid_key, pid_key):
        # check that all halos have these keys
        change_label = (self.alt_pid == self.alt_style[gn.LABEL])

        self.sys._checkKey(upid_key)
        self.sys._checkKey(pid_key)
        self.alt_pid = pid_key
        self.alt_upid = upid_key

        # if the label had default, then change. If user set it, don't change
        if change_label:
            self.alt_style[gn.LABEL] = pid_key
        return
    
    # --- Parent Arrow Creation Methods ---
    def showParentArrows(self, parent_type = 'pid'):
        if parent_type == 'both':
            self.showParentArrows('upid')
            self.showParentArrows('pid')
            return
        elif parent_type == 'pid':
            pid_style = copy.deepcopy(self.pid_style)
            fid_key = self.fid_pid
            alt_key = self.alt_pid
        elif parent_type == 'upid':
            pid_style = copy.deepcopy(self.upid_style)
            fid_key = self.fid_upid
            alt_key = self.alt_upid
        else:
            raise ValueError(f"undefined parent type {parent_type}")
        
        alt_style = copy.deepcopy(self.alt_style)
        fid_style = copy.deepcopy(self.fid_style)
        alt_style.update(pid_style)
        fid_style.update(pid_style)
        for h in self.sys.hids:
            fid_arrows = self.makeParentArrows(h, fid_key)
            alt_arrows = self.makeParentArrows(h, alt_key)
            
            for fa in fid_arrows:
                fa.setStyle(fid_style)
            for aa in alt_arrows:
                aa.setStyle(alt_style)
            self.graphics.extend(fid_arrows)
            self.graphics.extend(alt_arrows)
        return
    
    
    # --- Methods for applying styles to Traj ---

    def showParentDif(self,
                parent_type = 'pid',
                agree_col = (1, 1, 1),
                agree_lab = 'Agree',
                host_to_sub_col = (0, 0, 1),
                host_to_sub_lab = 'Host -> Subhalo',
                sub_to_host_col = (1, 0, 0),
                sub_to_host_lab = 'Subhalo -> Host',
                dif_parent_col = (0, 1, 0),
                dif_parent_lab = 'Parent Dif'
            ):
        
        if self.showing_status:
            raise Exception("already showing dif, cannot show another")
        
        self.showing_status = True

        if parent_type == 'pid':
            fid_key = self.fid_pid
            alt_key = self.alt_pid
        elif parent_type == 'upid':
            fid_key = self.fid_upid
            alt_key = self.alt_upid
        elif parent_type == 'both':
            raise ValueError(f"parent type {parent_type} not compatible with showParentDif")
        else:
            raise ValueError(f"undefined parent type {parent_type}")
        
        for h in self.sys.halos:
            anim_slc = slice(self.start, self.stop)
            alv = h.getAlive()[anim_slc]
            fid_pids = h.getField(fid_key)[anim_slc]
            alt_pids = h.getField(alt_key)[anim_slc]
            
            snaps = np.arange(self.start, self.stop)
            relevant_snaps = snaps[alv]
            if relevant_snaps.size == 0:
                continue
            
            # create masks for each category
            N = len(relevant_snaps)
            fid_sub = fid_pids > 0
            alt_sub = alt_pids > 0
            same = fid_pids == alt_pids
            hts_mask = (~fid_sub & alt_sub)
            sth_mask = (fid_sub & ~alt_sub)
            dp_mask = (fid_sub & alt_sub & ~same)
            categories = np.zeros(N)
            categories[hts_mask] = 1
            categories[sth_mask] = 2
            categories[dp_mask] = 3

            # prepare segment info
            seg_starts = [0]
            seg_cats = [categories[0]]
            for i in range(1, N):
                if categories[i] != categories[i-1]:
                    seg_starts.append(i)
                    seg_cats.append(categories[i])
            seg_starts.append(N)
            
            # create segments
            colors = [agree_col, host_to_sub_col, sub_to_host_col, dif_parent_col]
            labels = [agree_lab, host_to_sub_lab, sub_to_host_lab, dif_parent_lab]
            for i in range(len(seg_starts) - 1):
                start_idx = seg_starts[i]
                stop_idx = seg_starts[i+1]
                seg_len = stop_idx - start_idx
                seg_snap = relevant_snaps[start_idx]
                seg_cat = seg_cats[i]

                style = {}
                style[gn.COLOR] = colors[seg_cat]
                style[gn.LABEL] = labels[seg_cat]

                # figure out where to start the line:
                if i == 0:
                    # first iter: start at its own first point
                    seg_start_snap = relevant_snaps[start_idx]
                else:
                    # back up one so we include the previous point
                    seg_start_snap = relevant_snaps[start_idx - 1]
                
                # always end at the last point of this run (+1 so makeSegment includes it)
                seg_stop_snap = relevant_snaps[stop_idx - 1] + 1

                # if it's the first segment *and* genuinely only one point, do a Marker
                if seg_len == 1 and i == 0:

                    # Add a Marker instead of a Line
                    m = Marker(h, snap=seg_snap)
                    m.setColor(style[gn.COLOR])
                    m.setSize(self._getDefaultPointSize())
                    m.setShape('sphere')
                    m.setName(f"marker_{h.hid}_{seg_snap}")
                    self.addGraphic(m)
                else:
                    segment = self.makeSegment(h.hid, seg_start_snap, seg_stop_snap)
                    segment.setStyle(style)
                    self.addGraphic(segment)
        return
    

