# functions for loading sparta data and quickly creating containers
from containers import Halo
import numpy as np

def makeHalo(sminterface, halo_id) -> Halo:
    ogid = sminterface.getOrigID(halo_id)
    # get moria data for this halo
    halo_data = sminterface.getCat(ogid)
    pos = halo_data['x_spa'][:,:]
    pos = np.squeeze(pos)
    halo = Halo(ogid, pos, sminterface.s['simulation']['snap_z'])
    
    return halo


