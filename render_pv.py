import os
import global_names as gn
import glob
import numpy as np
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
    
    vtp_dir = scene_dir + '/vtp/'
    with open(os.path.join(vtp_dir, "style.json")) as f:
        styles = json.load(f)

    if verbose:
        print("adjusting camera...")
    
    view = pvs.GetActiveViewOrCreate('RenderView')
    pvs._DisableFirstRenderCameraReset()
    scene_style = styles.get(gn.SCENE_KEY, {})
    if gn.BACKGROUND_COLOR in scene_style:
        view.BackgroundColorMode = 'Single Color'
        view.UseColorPaletteForBackground = 0
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


    if verbose:
        print('loading pvds...')

    # load all pvd files in vtp_dir (use glob to find them)
    pvd_paths = sorted(glob.glob(os.path.join(vtp_dir, "*.pvd")))

    if not pvd_paths:
        raise FileNotFoundError(f"No PVD files found in {vtp_dir}")
    
    if verbose:
        print("creating graphics...")
    
    readers = []
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
        # create rep
        reader = pvs.PVDReader(FileName = ppath)
        pvs.RenameSource(grp, reader)
        disp = pvs.Show(reader, view)
        pvs.UpdatePipeline(proxy = reader)

        _applyStyles(disp, styles[grp])

        readers.append(reader)

    # # Decide timesteps to render
    # tsteps = sorted(all_times)
    
    # start = styles[gn.SCENE_KEY][gn.START]
    # stop  = styles[gn.SCENE_KEY][gn.STOP]

    # tsteps = np.arange(start, stop)

    anim_scene = pvs.GetAnimationScene()
    anim_scene.UpdateAnimationUsingDataTimeSteps()
    tk = pvs.GetTimeKeeper()
    tsteps = list(getattr(tk, "TimestepValues", []))

    if verbose:
        print(f"Rendering {len(tsteps)} frames...")

    # Prepare frames dir
    frames_dir = os.path.join(scene_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    for i, t in enumerate(tsteps):
        tk.Time = float(t)

        pvs.Render(view = view)
        out_png = os.path.join(frames_dir, f"frame_{i:04d}.png")
        pvs.SaveScreenshot(out_png, viewOrLayout = view, ImageResolution=view.ViewSize)

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