import os

import global_names as gn


def main(scene_dir, verbose = 1):
    import paraview.simple as pvs
    import xml.etree.ElementTree as ET
    import json

    def _applyStyles(rep, styles):
        """
        Reads style dict, applies to rep
        """
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
            rep.LineStyle = styles[gn.LSTYLE].capitalize()

        # 5) Turn on/off rendering points as little spheres
        if gn.SHOW_POINTS in styles:
            rep.RenderPointsAsSpheres = bool(styles[gn.SHOW_POINTS])
        
        return
    
    vtp_dir = scene_dir + '/vtp/'
    with open(os.path.join(vtp_dir, "style.json")) as f:
        styles = json.load(f)

    pvs._DisableFirstRenderCameraReset()
    view = pvs.GetActiveViewOrCreate('RenderView')
    scene_style = styles.get(gn.SCENE_KEY, {})
    if gn.BACKGROUND_COLOR in scene_style:
        bg = scene_style[gn.BACKGROUND_COLOR]
        if max(bg) > 1.0:
            bg = [c / 255.0 for c in bg]
        view.Background = bg
        view.Background2 = bg
    if gn.CAM_POS in scene_style:
        view.CameraPosition = scene_style[gn.CAM_POS]
    if gn.CAM_FOC in scene_style:
        view.CameraFocalPoint = scene_style[gn.CAM_FOC]
    if gn.CAM_UP in scene_style:
        view.CameraViewUp = scene_style[gn.CAM_UP]
    if gn.CAM_ZOOM in scene_style:
        view.CameraParallelScale = scene_style[gn.CAM_ZOOM]
    if gn.CAM_PROJ in scene_style:
        view.CameraParallelProjection = int(bool(scene_style[gn.CAM_PROJ]))
    if gn.VIEW_SIZE in scene_style:
        size = list(scene_style[gn.VIEW_SIZE])
        smax = max(size[0], size[1])
        size[0] /= smax; size[1] /= smax
        view.ViewSize = [int(size[0] * 1024), int(size[1] * 1024)]
    if verbose:
        print("loading pvd...")
    tree = ET.parse(vtp_dir + 'scene.pvd')
    root = tree.getroot()
    collection = root.find('Collection')

    if verbose:
        print("creating groups...")
    groups = {}
    for ds in collection.findall('DataSet'):
        grp = ds.get('group')
        if grp not in styles:
            msg = f'{grp} not found in styles json, {list(styles.keys())}'
            raise ValueError(msg)
        bname = ds.get('file')
        fullpath = os.path.join(vtp_dir, bname)
        groups.setdefault(grp, []).append(fullpath)


    # anim_scene = pvs.GetAnimationScene()
    # anim_scene.UpdateAnimationUsingDataTimeSteps()
    # if verbose:
    #     print("rendering...")
    # pvs.Render()
    # if verbose:
    #     print("saving state...")
    # pvs.SaveState(os.path.join(scene_dir, "scene.pvsm"))
    frames_dir = os.path.join(scene_dir, 'frames/')
    os.makedirs(frames_dir, exist_ok=True)
    if verbose:
        print(f"frames saved in {frames_dir}")
    # # before your manual loop, print out what the reader *knows* about its timesteps:
    # for name, reader in readers.items():
    #     print(f"{name!r} reader.TimestepValues = {reader.TimestepValues}")
    tstart = int(styles[gn.SCENE_KEY][gn.START])
    tstop = int(styles[gn.SCENE_KEY][gn.STOP])
    for t in range(tstart, tstop):
        if verbose:
            print(f"\tcreating snapshot {t}...")
        created = []
        for name in groups:
            # build the exact filename for this time (e.g. line_1061_0005.vtp)
            fname = os.path.join(vtp_dir, f"{name}_{int(t):04d}.vtp")
            if not os.path.isfile(fname):
                continue

            # create a fresh reader for this one file
            reader = pvs.XMLPolyDataReader(FileName=[fname])
            reader.UpdatePipeline()

            # show it and style it
            rep = pvs.Show(reader, view)
            rep.Visibility = 1
            _applyStyles(rep, styles[name])

            # keep track for cleanup
            created.extend([reader, rep])
        # view.ResetCamera()
        pvs.Render()
        shot_path = os.path.join(frames_dir, f'frame_{int(t):04d}.png')
        pvs.SaveScreenshot(shot_path, view)

        for proxy in created:
            pvs.Delete(proxy)
        

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