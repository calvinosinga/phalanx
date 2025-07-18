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
        
        # # 6) Legend label (if you want entries in the color legend)
        # if gn.LABEL in styles:
        #     rep.LegendLabel = styles[gn.LABEL]
        return
    vtp_dir = scene_dir + '/vtp/'
    
    with open(os.path.join(vtp_dir, "style.json")) as f:
        styles = json.load(f)

    view = pvs.GetActiveViewOrCreate('RenderView')
    scene_style = styles.get(gn.SCENE_KEY, {})
    if gn.BACKGROUND_COLOR in scene_style:
        bg = scene_style[gn.BACKGROUND_COLOR]
        if max(bg) > 1.0:
            bg = [c / 255.0 for c in bg]
        view.Background = bg
    if gn.CAM_POS in scene_style:
        view.CameraPosition = scene_style[gn.CAM_POS]
    if gn.CAM_FOC in scene_style:
        view.CameraFocalPoint = scene_style[gn.CAM_FOC]
    if gn.CAM_UP in scene_style:
        view.CameraViewUp = scene_style[gn.CAM_UP]
    if gn.CAM_ZOOM in scene_style:
        view.CameraParallelScale = scene_style[gn.CAM_ZOOM]
    if gn.CAM_PROJ in scene_style:
        view.CameraParallelProjection = int(bool(scene_style[gn.CAMERA_PAR_PROJ]))

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

    if verbose:
        print("applying custom styles...")
    for name, file_list in groups.items():
        reader = pvs.XMLPolyDataReader(FileName = file_list)
        reader.UpdatePipelineInformation()
        reader.UpdatePipeline()

        rep = pvs.Show(reader, view)
        rep.Visibility = 1

        _applyStyles(rep, styles[name])

    view.ResetCamera()
    anim_scene = pvs.GetAnimationScene()
    anim_scene.UpdateAnimationUsingDataTimeSteps()
    if verbose:
        print("rendering...")
    pvs.Render()
    if verbose:
        print("saving state...")
    pvs.SaveState(os.path.join(scene_dir, "scene.pvsm"))
    os.makedirs(scene_dir + '/frames/', exist_ok=True)
    movie_path = os.path.join(scene_dir + '/frames/', "frame.png")
    if verbose:
        print("creating movie...")
    pvs.SaveAnimation(movie_path, view)
    print(f"Movie saved to {movie_path}")
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
    png_pattern = os.path.join(frames_dir, "frame.%04d.png")
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