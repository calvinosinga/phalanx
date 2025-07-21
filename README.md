# Phalanx

Pipeline for Halo Animation & Lineage Assessment in N-body eXperiments (phalanx) is a tool for visualizing halo systems and interactions, assessing differences in subhalo definitions, and comparing methods of calculating halo trajectories. Generally, the pipeline is broken down into four steps:
1. Create container objects that store the halo data
2. Select a way to display the data and style it
3. Export the cached objects into VTK (visualization toolkit) files
4. Render and save the animation using the VTK files

Currently, phalanx only uses paraview to create animations but will become compatible other plotting softwares.

# Step 1: What to Show
## Loading Functions
An assortment of functions that can quickly create data containers compatible with phalanx from halo catalogs are provided in the load_*.py files. Each catalog is organized differently, so the functions are designed to operate on a particular catalog that is specified in the latter half of the file name. Currently, loading functions are only provided for sparta/moria catalogs.

Example usage:
```python
from sparta_tools import sparta, moria, utils
    
mor = moria.load('moria.hdf5')
spa = sparta.load('sparta.hdf5', res_match = ['ifl', 'tjy', 'oct'])
smi = utils.SMInterface(spa, mor)

import phalanx.load_sparta as ls

host_halo = ls.makeHalo(smi, host_id)
sub_halo = ls.makeHalo(smi, halo_id)
ls.addCat(smi, [host_halo, sub_halo]) # adds all catalog fields for each halo
ls.addPeri(smi, [sub_halo]) # adds pericenter events (see below) to each halo
```
## Containers 
There are three kinds of data containers: Event, Halo, and System. An Event (has several subclasses) indicates when something happens to a halo and contains specifications for how/when that event should be marked. A Halo object contains the data relevant to a single halo. A System contains methods that require interfacing between halos.
```python
from phalanx.containers import System
sys = System([host_halo, sub_halo], spa['simulation']['box_size'])
# adjust halo positions to be wrt the center of mass of the system
sys.setCOMOrigin() 
```

# Step 2: How to Display it

## Scene
An assortment of scenes are provided for common use cases. Use the functions each scene provides to determine what aspects of the data should be shown. 

```python
from phalanx.scenes import HaloView
hv = HaloView(sys, host_halo.hid) # scene that centers on a specific halo
hv.setAnimSnap(150, 192) # set what snapshots we want to animate
hv.showTjys() # show tjys of other halos
hv.POVBoundary() # show boundary of pov halo
hv.showOrigin() # show center of pov halo
hv.addEvents() # show events that are contained in each halo
hv.autoCam() # automatically find the optimal camera settings to display info
```

## Graphics
The Scene class hosts a list of Graphic objects, which contain instructions and stylings for the different visual elements shown within the animation.

# Step 3: Export

Paraview expects the data to be input as visual toolkit files (VTK), so we export the data into VTKs and style instructions into JSONs. The stylings are applied during the render stage.

```python
from phalanx.export import export
export(hv, f'/Users/cosinga/code/orb_algo/{host_id}_{halo_id}_hv')
```

# Step 4: Render
Create the animation
```python
from phalanx.render import render
render('/Users/cosinga/code/orb_algo/2328507_1061_hv', 10, True, True)
```