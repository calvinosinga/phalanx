# Phalanx

Pipeline for Halo Animation & Lineage Assessment in N-body eXperiments (phalanx) is a pipeline for visualizing halo position data and dynamical systems.
1. Identify targets in the halo and sparta catalogs
2. Create container objects that store the data, plot stylings, and instructions
3. Export the cached objects into VTK files
4. Render and save the animation using the VTK files


## Identifying Targets
Assortment of unrelated functions that can quickly create lists of containers for common plot types, like
- subhalo trajectories in and around host halos (ROCKSTAR vs SPARTA, for example)
- comparisons of different host/sub algos in system-system interactions
- locations of pericenters 

## Containers 
We restructure the data in the array into containers that conveniently combine the data to be plotted, associated style elements, and instructions.

## Export
Given a list of containers, we export the data to VTK files and style hints and instructions to JSON files.

## Render
Create the animation