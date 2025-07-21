# Phalanx

Pipeline for Halo Animation & Lineage Assessment in N-body eXperiments (phalanx) is a pipeline for visualizing halo systems and interactions, assessing differences in subhalo definitions, and comparing methods of calculating halo trajectories. Generally, the pipeline is broken down into four steps:
1. Create container objects that store the halo data
2. Select a way to display the data and style it
3. Export the cached objects into VTK (visualization toolkit) files
4. Render and save the animation using the VTK files


## Step 1: load_*.py
Assortment of functions that can quickly create data containers from halo catalogs. The type of catalog is specified in the latter half of the file name.

Usage:
```python
import load_sparta as ls

host_halo = ls.makeHalo(smi, host_id)
sub_halo = ls.makeHalo(smi, halo_id)
ls.addCat(smi, [host_halo, sub_halo])
ls.addPeri(smi, [sub_halo])
```
## Containers 
We restructure the data in the array into containers that conveniently combine the data to be plotted, associated style elements, and instructions.

## Export
Given a list of containers, we export the data to VTK files and style hints and instructions to JSON files.

## Render
Create the animation