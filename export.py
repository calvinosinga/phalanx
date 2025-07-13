import json
import os
from collections import Counter
from xml.etree.ElementTree import Element, SubElement, ElementTree

def write_pvd(pvd_path, vtp_filelist, timestep_list):
    """
    Write a Paraview PVD file listing all VTPs at given timesteps.
    """
    root = Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = SubElement(root, "Collection")
    for fname, ts in zip(vtp_filelist, timestep_list):
        SubElement(collection, "DataSet", timestep=str(ts), file=os.path.basename(fname), group="", part="0")
    tree = ElementTree(root)
    tree.write(pvd_path, encoding="utf-8", xml_declaration=True)

def export(scene, out_dir):
    # 1. Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

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
        fnames, tsteps = gph.writeVTP(out_dir, scene.start, scene.stop)
        vtp_outputs.extend(zip(fnames, tsteps))
        style[gph.getName()] = gph.getStyle()
    # 4. Save style dictionary as JSON
    style_path = os.path.join(out_dir, 'style.json')
    with open(style_path, 'w') as f:
        json.dump(style, f, indent=2)

    # 5. Create PVD file (scene.pvd)
    vtp_files, timesteps = zip(*vtp_outputs) if vtp_outputs else ([], [])
    pvd_path = os.path.join(out_dir, 'scene.pvd')
    write_pvd(pvd_path, vtp_files, timesteps)

    return

