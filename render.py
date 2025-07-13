from paraview.simple import (OpenDataFile, Show, Render, SaveState, GetActiveViewOrCreate, SetActiveSource, GetDisplayProperties)
import json
import os
import sys

def render(out_dir):
    

    # Load style
    style_path = os.path.join(out_dir, 'style.json')
    if not os.path.exists(style_path):
        raise FileNotFoundError(f"No style.json found in {out_dir}")
    with open(style_path, 'r') as f:
        styles = json.load(f)

    # Load PVD
    pvd_path = os.path.join(out_dir, 'scene.pvd')
    if not os.path.exists(pvd_path):
        raise FileNotFoundError(f"No scene.pvd found in {out_dir}")
    pvd_reader = OpenDataFile(pvd_path)

    # Apply style per object (basic example for color/point size)
    # Note: This assumes each dataset in the PVD corresponds to a source in ParaView,
    # and the order matches; otherwise, matching by name is needed.
    rep = Show(pvd_reader)
    # (Advanced: you can iterate over sources, set visibility, color, etc, by matching names)
    for gname, st in styles.items():
        # Example: apply color if specified
        if 'color' in st:
            rep.DiffuseColor = st['color']
        if 'point_size' in st:
            rep.PointSize = st['point_size']
        # Add more mappings as needed (e.g., representation, opacity)

    # Render view (create if needed)
    view = GetActiveViewOrCreate('RenderView')
    Render()

    # Save the ParaView state file to the output directory
    pvsm_path = os.path.join(out_dir, "scene.pvsm")
    SaveState(pvsm_path)
    print(f"ParaView state saved to {pvsm_path}")
    return pvsm_path

if __name__ == '__main__':
    render(sys.argv[1])

    # TODO if next argument is delete, delete output directory