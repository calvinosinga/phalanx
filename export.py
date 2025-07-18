import json
import os
from collections import Counter
from xml.etree.ElementTree import Element, SubElement, ElementTree
from scenes import Scene
import global_names as gn
import shutil
def write_pvd(pvd_path, group_names, vtp_filelist, timestep_list):
    """
    Write a Paraview PVD file listing all VTPs at given timesteps.
    """
    root = Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = SubElement(root, "Collection")
    for grpname, fname, ts in zip(group_names, vtp_filelist, timestep_list):
        SubElement(collection, "DataSet", timestep=str(ts), file=os.path.basename(fname), group=grpname, part="0")
    tree = ElementTree(root)
    tree.write(pvd_path, encoding="utf-8", xml_declaration=True)

def export(scene : Scene, out_dir, delete_prev = True):
    vtp_dir = out_dir+'/vtp'
    # 1. Create output directory if it doesn't exist. If it does, delete any previously exported vtp files
    if os.path.exists(vtp_dir) and delete_prev:
        shutil.rmtree(vtp_dir)
    os.makedirs(vtp_dir, exist_ok=True)
    

    graphics = scene.getGraphics()

    # 2. Check for repeated graphic names
    names = [gph.getName() for gph in graphics]
    name_counts = Counter(names)
    repeats = [n for n, count in name_counts.items() if count > 1]
    if repeats:
        raise ValueError(f"Duplicate graphic names found: {repeats}")

    # 3. Write VTP files and collect outputs
    style = {}
    vtp_outputs = []  # List of tuples: (fname, timestep)
    for gph in graphics:
        fnames, tsteps = gph.writeVTP(vtp_dir, scene.start, scene.stop)
        name = gph.getName()
        for fn, ts in zip(fnames, tsteps):
            vtp_outputs.append((name, fn, ts))
        style[gph.getName()] = gph.getStyle()
    
    # also add scene settings to style json
    style[gn.SCENE_KEY] = scene.getSceneProps()
    
    # 4. Save style dictionary as JSON
    style_path = os.path.join(vtp_dir, 'style.json')
    with open(style_path, 'w') as f:
        json.dump(style, f, indent=2)

    # 5. Create PVD file (scene.pvd)
    vtp_groups, vtp_files, timesteps = zip(*vtp_outputs) if vtp_outputs else ([], [], [])
    pvd_path = os.path.join(vtp_dir, 'scene.pvd')
    write_pvd(pvd_path, vtp_groups, vtp_files, timesteps)

    return

