import os
import global_names as gn
import glob
import numpy as np
def main(scene_dir, verbose = 1):
    import paraview.simple as pvs
    import xml.etree.ElementTree as ET
    import json

    ##### HELPER FUNCTIONS ##########################
    def _applyStyles(rep, styles):
        """
        Reads style dict, applies to rep
        """
        # TODO: in future, reads if the style is given as array, and creates/applies animation track.
        # 1) Color
        if gn.COLOR in styles:
            rep.DiffuseColor = styles[gn.COLOR]

        # 2) Opacity
        if gn.OPACITY in styles:
            rep.Opacity = styles[gn.OPACITY]
        
        # 3) Point size (used for point glyphs or 'Points' representation)
        if gn.SIZE in styles:
            rep.PointSize = styles[gn.SIZE]
        
        # 4) Line width & style (for line/trajectory objects)
        if gn.LWIDTH in styles:
            rep.LineWidth = styles[gn.LWIDTH]

        if gn.LSTYLE in styles:
            # ParaView expects e.g. "Solid", "Dashed", "Dotted"
            try:
                rep.LineStyle = styles[gn.LSTYLE].capitalize()
            except Exception:
                pass

        # 5) Turn on/off rendering points as little spheres
        if gn.SHOW_POINTS in styles:
            rep.RenderPointsAsSpheres = bool(styles[gn.SHOW_POINTS])
        
        # 6) lighting
        if gn.LIGHTING in styles and bool(styles[gn.LIGHTING]):
            rep.Diffuse = 0.8
            rep.Specular = 0.4
            rep.SpecularPower = 25
            rep.Ambient = 0.2

        return
    
    def _is_dynamic(v, expected_width=None):
        """True if v looks like [[t, ...], ...]. Optionally enforce row length."""
        if not isinstance(v, list) or not v:
            return False
        first = v[0]
        if not isinstance(first, (list, tuple)):
            return False
        return expected_width is None or len(first) == expected_width

    def _sorted_unique_rows(rows):
        """Sort rows by time and drop duplicate times (keep last)."""
        by_t = {}
        for r in rows:
            t = float(r[0])
            by_t[t] = r  # last one wins
        return [by_t[t] for t in sorted(by_t.keys())]
    
    def _norm_keytime(snap_t, start_idx, stop_idx):
        """Map snapshot index to [0,1] normalized KeyTime for tracks."""
        s0 = float(start_idx); s1 = float(stop_idx)
        if s1 <= s0:
            return 0.0
        return (float(snap_t) - s0) / (s1 - s0)
    
    def make_animation_track(proxy, prop_name, key_rows, start_idx, stop_idx, interp_mode=None):
        """
        Create an animation track for a property on a proxy.
        key_rows: [[snap, val(s)], ...] where val(s) can be scalar or vector.
        """
        track = pvs.GetAnimationTrack(prop_name, proxy=proxy)
        keyframes = []
        for row in _sorted_unique_rows(key_rows):
            kf = pvs.CompositeKeyFrame()
            kf.KeyTime = _norm_keytime(row[0], start_idx, stop_idx)
            vals = row[1:]
            kf.KeyValues = [float(v) for v in vals]
            if interp_mode:
                kf.Interpolation = interp_mode
            keyframes.append(kf)
        track.KeyFrames = keyframes
        return track

    def _stack_bottom_right(n, x=0.95, y0=0.05, dy=0.04):
        """
        Produce n positions stacked from bottom-right upward (normalized coords).
        We nudge x left slightly so long labels don't get clipped.
        """
        x_adj = max(0.0, x - 0.18)
        return [[x_adj, y0 + i*dy] for i in range(n)]

    def _make_scene_text_actors(view, scene_text_specs):
        """
        Returns a list of dicts with {'actor': textSource, 'series': list_or_none}
        """
        actors = []
        for spec in scene_text_specs:
            txt = spec.get("text")
            color = spec.get("color", [1,1,1])
            font = int(spec.get("font", 12))
            pos = spec.get("position", [0.05, 0.95])

            t = pvs.Text(Text=(txt if isinstance(txt, str) else ""))  # empty if dynamic
            disp = pvs.Show(t, view)
            disp.WindowLocation = "Any Location"
            disp.Position = list(pos)
            disp.Color = color
            disp.FontSize = font

            actors.append({
                "src": t,
                "series": txt if isinstance(txt, list) else None
            })
        return actors

    def _make_graphic_label_actors(view, styles, color_fallback=[1,1,1]):
        """
        Creates stacked labels (bottom-right) for any graphic that has gn.LABEL in style.
        Returns list of dicts {'src': textSource, 'series': list_or_none}.
        """
        # Collect candidates
        items = []
        for name, st in styles.items():
            if name == gn.SCENE_KEY:
                continue
            if gn.LABEL in st:
                label = st[gn.LABEL]
                color = st.get(gn.COLOR, color_fallback)
                items.append((name, label, color))

        # Positions
        pos_list = _stack_bottom_right(len(items))
        actors = []
        for (name, label, color), pos in zip(items, pos_list):
            t = pvs.Text(Text=(label if isinstance(label, str) else ""))
            disp = pvs.Show(t, view)
            disp.WindowLocation = "Any Location"
            disp.Position = pos
            disp.Color = color
            disp.FontSize = 14

            actors.append({
                "src": t,
                "series": label if isinstance(label, list) else None
            })
        return actors

    ##### MAIN FUNCTION BEGINS ##########

    vtp_dir = os.path.join(scene_dir, 'vtp')
    with open(os.path.join(vtp_dir, "style.json")) as f:
        styles = json.load(f)

    
    view = pvs.GetActiveViewOrCreate('RenderView')
    pvs._DisableFirstRenderCameraReset()

    if verbose:
        print("loading graphics...")

    # load all pvd files in vtp_dir (use glob to find them)
    pvd_paths = sorted(glob.glob(os.path.join(vtp_dir, "*.pvd")))

    if not pvd_paths:
        raise FileNotFoundError(f"No PVD files found in {vtp_dir}")
    
    sources = {}
    for ppath in pvd_paths:
        # load pvd file
        tree = ET.parse(ppath)
        root = tree.getroot()
        collection = root.find('Collection')
        # there should only be one dataset in each pvd file (does this work?)
        ds = collection.findall('DataSet')
        # the groupname should match a key in the styles json
        grp = ds[0].get('group')

        if grp not in styles:
            msg = f'{grp} not found in styles json, {list(styles.keys())}'
            raise ValueError(msg)
        
        if not grp:
            raise ValueError(f"no group found for {ppath}, cannot locate styles")
        # create proxy
        reader = pvs.PVDReader(FileName = ppath)
        pvs.RenameSource(grp, reader)
        pvs.UpdatePipeline(proxy = reader)
        sources[grp] = reader


    if verbose:
        print("creating representations...")
    
    do_data_interp = False # for now, never interpolate the data since it's not currently set up for it.
    # do_interp = gn.INTERP_TYPE in scene_style

    reps = {}
    for name, src in sources.items():
        # TODO: wrap src in TimeInterplator. Requires change to graphic class to make consistent topology
        if do_data_interp:
            rep_src = pvs.TemporalInterpolator(Input = src)
        else:
            rep_src = src
        rep = pvs.Show(rep_src, view)
        
        # _applyStyles will handle both static and dynamic styles in the future
        _applyStyles(rep, styles[name])
        reps[name] = rep


    if verbose:
        print("adjusting camera and scene...")

    scene_style = styles.get(gn.SCENE_KEY, {})
    # annotations = scene_style.get(gn.TEXT, [])

    # text_actors = _make_scene_text_actors(view, annotations)
    # grph_label_actors = _make_graphic_label_actors(view, styles)
    start_idx = int(scene_style[gn.START])
    stop_idx = int(scene_style[gn.STOP])

    do_cam_tracks  = True   # allow dynamic camera now

    # camera interpolation type (validate only if present)
    interp_type = scene_style.get(gn.INTERP_TYPE)
    VALID_INTERP = {"Linear", "Spline", "Boolean"}
    if interp_type is not None and interp_type not in VALID_INTERP:
        raise ValueError(f"Invalid interpolation '{interp_type}'. Allowed: {sorted(VALID_INTERP)}")

    # oversampling factor (used only when do_data_interp is True)
    interp_num = int(scene_style.get(gn.INTERP_STEP, 4))
    # always static settings
    if gn.BACKGROUND_COLOR in scene_style:
        view.UseColorPaletteForBackground = 0
        bg = list(scene_style[gn.BACKGROUND_COLOR])
        if max(bg) > 1.0:  # allow 0-255 input
            bg = [c / 255.0 for c in bg]
        view.Background = bg
    
    if gn.CAM_PROJ in scene_style:
        view.CameraParallelProjection = int(bool(scene_style[gn.CAM_PROJ]))

    if gn.VIEW_SIZE in scene_style:
        size = list(scene_style[gn.VIEW_SIZE])
        smax = max(size[0], size[1])
        size[0] /= smax; size[1] /= smax
        # make sure the width and height are even numbers for ffmpeg
        width = round(size[0] * 1024)
        height = round(size[1] * 1024)
        width = width if (width % 2 == 0) else width + 1
        height = height if (height % 2 == 0) else height + 1
        view.ViewSize = [width, height]

    view.OrientationAxesVisibility = 0

    # potentially dynamic settings
    pos_spec = scene_style.get(gn.CAM_POS)    # either [x,y,z] or [[snap,x,y,z],...]
    foc_spec = scene_style.get(gn.CAM_FOC)
    up_spec  = scene_style.get(gn.CAM_UP)
    z_spec   = scene_style.get(gn.CAM_ZOOM)   # scalar or [[snap,scale],...]
    ang_spec = scene_style.get(gn.CAM_ANGLE)  # scalar or [[snap,deg],...]
    interp_type = scene_style.get(gn.INTERP_TYPE)
    # Dynamic → per-property tracks; static → set once on the view.
    if _is_dynamic(pos_spec, 4) and do_cam_tracks:
        make_animation_track(view, "CameraPosition",     pos_spec, start_idx, stop_idx, interp_type)
    elif isinstance(pos_spec, (list, tuple)) and len(pos_spec) == 3:
        view.CameraPosition = [float(pos_spec[0]), float(pos_spec[1]), float(pos_spec[2])]

    if _is_dynamic(foc_spec, 4) and do_cam_tracks:
        make_animation_track(view, "CameraFocalPoint",   foc_spec, start_idx, stop_idx, interp_type)
    elif isinstance(foc_spec, (list, tuple)) and len(foc_spec) == 3:
        view.CameraFocalPoint = [float(foc_spec[0]), float(foc_spec[1]), float(foc_spec[2])]

    if _is_dynamic(up_spec, 4) and do_cam_tracks:
        make_animation_track(view, "CameraViewUp",       up_spec,  start_idx, stop_idx, interp_type)
    elif isinstance(up_spec, (list, tuple)) and len(up_spec) == 3:
        view.CameraViewUp = [float(up_spec[0]), float(up_spec[1]), float(up_spec[2])]

    if _is_dynamic(z_spec, 2) and do_cam_tracks:
        make_animation_track(view, "CameraParallelScale", z_spec,  start_idx, stop_idx, interp_type)
    elif isinstance(z_spec, (int, float)):
        view.CameraParallelScale = float(z_spec)

    if _is_dynamic(ang_spec, 2) and do_cam_tracks:
        make_animation_track(view, "CameraViewAngle",     ang_spec, start_idx, stop_idx, interp_type)
    elif isinstance(ang_spec, (int, float)):
        view.CameraViewAngle = float(ang_spec)

    anim = pvs.GetAnimationScene()
    tk   = pvs.GetTimeKeeper()

    # Get dataset time range once
    if hasattr(tk, "TimeRange"):
        t0, t1 = float(tk.TimeRange[0]), float(tk.TimeRange[1])
    else:
        t0, t1 = 0.0, 1.0

    if do_data_interp:
        # continuous timeline with oversampling
        num_snaps  = max(1, int(round(stop_idx - start_idx)))
        num_frames = max(2, num_snaps * interp_num + 1)

        anim.PlayMode = 'Sequence'
        anim.StartTime = t0
        anim.EndTime = t1
        anim.NumberOfFrames = num_frames

        frame_times = [t0 + (t1 - t0) * (i / (num_frames - 1)) for i in range(num_frames)]
    else:
        # discrete data steps
        anim.UpdateAnimationUsingDataTimeSteps()
        times = list(getattr(tk, "TimestepValues", []))
        if not times:
            times = [t0]  # fallback if empty
        frame_times = times
        num_frames = len(frame_times)

    if verbose:
        print(f"rendering {num_frames} frames...")



    # Prepare frames dir
    frames_dir = os.path.join(scene_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    for i, t in enumerate(frame_times):
        tk.Time = float(t)
        anim.AnimationTime = float(t)
        # # Update scene-wide dynamic texts
        # for a in text_actors:
        #     if a["series"] is not None:
        #         if i < len(a["series"]):
        #             a["src"].Text = a["series"][i]

        # # Update per-graphic dynamic labels
        # for a in grph_label_actors:
        #     if a["series"] is not None:
        #         if i < len(a["series"]):
        #             a["src"].Text = a["series"][i]
        pvs.Render(view=view)
        pvs.SaveScreenshot(os.path.join(frames_dir, f"frame_{i:04d}.png"),
                        viewOrLayout=view,
                        ImageResolution=view.ViewSize)
        
    if verbose:
        print("saving state to pvsm")
    
    pvsm_path = os.path.join(scene_dir, "scene.pvsm")

    # save the state to scene_dir
    pvs.SaveState(pvsm_path)

    return




def render(scene_dir, fps, cleanup_frames = True, verbose = 1):
    import subprocess
    frames_dir = os.path.join(scene_dir, 'frames')

    this_script = os.path.abspath(__file__)
    
    # pvpython command
    pvpython_cmd = [
        "pvpython",
        this_script,
        scene_dir
    ]
    try:
        subprocess.run(pvpython_cmd, check=True)
    except FileNotFoundError:
        print("Error: pvpython not found in PATH. Please add ParaView's bin directory to your PATH.")
        return
    except subprocess.CalledProcessError as e:
        print(f"pvpython failed: {e}")
        return
    
    # 2. Assemble PNGs into a movie with ffmpeg
    png_pattern = os.path.join(frames_dir, "frame_%04d.png")
    movie_out = os.path.join(scene_dir, "scene.mp4")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", png_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        movie_out
    ]
    if verbose:
        print("Calling ffmpeg to assemble movie...")
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Movie successfully created at: {movie_out}")
    except FileNotFoundError:
        print("Error: ffmpeg not found in PATH. Please install ffmpeg.")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed: {e}")

    # 3. Optionally clean up PNG frames
    if cleanup_frames:
        import shutil
        try:
            shutil.rmtree(frames_dir)
            print("Frame images cleaned up.")
        except Exception as e:
            print(f"Could not clean up frames: {e}")
    
    return

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: pvpython render.py /path/to/exported_scene")
    else:
        main(sys.argv[1])