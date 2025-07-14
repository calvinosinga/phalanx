# functions for loading sparta data and quickly creating containers
from containers import Halo, Peri, Ifl
import containers as c
import numpy as np
from typing import List

def makeHalo(sminterface, halo_id, default_position = 'x_spa') -> Halo:
    ogid = sminterface.getOrigID(halo_id)
    # get moria data for this halo
    halo_data = sminterface.getCat(ogid)
    pos = halo_data[default_position][:,:]
    pos = np.squeeze(pos)
    halo = Halo(ogid, pos, sminterface.s['simulation']['snap_z'])

    return halo

def _makePeriEvents(out) -> List[Peri]:
    peris = []
    if 'snap_hist' in out.dtype.names:
        for snap in out['snap_hist']:
            if snap >= 0:
                peri = Peri(snap)
                peri.setHost(out['host_orig_id'])
                peris.append(peri)
    else:
        if out['last_pericenter_snap'] >= 0:
            peri = Peri(out['last_pericenter_snap'])
            peri.setHost(out['host_orig_id'])
            peris.append(peri)
    return peris

def addPeri(sminterface, halos, host_id = None):
    for h in halos:
        out = sminterface.getRes(host_id, h.hid, result_type = 'oct', return_host_ids = True)
        ntcrs = out.shape[0] # number of instances of this halo as a tracer
        for i in range(ntcrs):
            peris = _makePeriEvents(out[i])
            for p in peris:
                h.addEvent(p)
    return

def addIfl(sminterface, halos, host_id = -1):
    for h in halos:
        out = sminterface.getRes(host_id, h.hid, result_type = 'tjy', return_host_ids = True)
        ntcrs = out.shape[0] # number of instances of this halo as a tracer
        for i in range(ntcrs):
            ifl = Ifl(out['first_snap'][i])
            ifl.setHost(out['host_orig_id'][i])
            h.addEvent(ifl)
    return

def addApo(sminterface, halos, host_id = -1):
    return

def addTjy(sminterface, halos, host_id):
    for h in halos:
        out = sminterface.getRes(host_id, h.hid, result_type = 'tjy', return_host_ids = True)
        # should only have size of 1
        h.addField(c.VR, out['vr'])
        h.addField(c.VT, out['vt'])
        h.addField('dist', out['r'])
        
    return

def addCat(
    sminterface,
    halos : List[Halo],
    default_radius = 'R200m_bnd_cat',
    default_mass = 'M200m_bnd_cat',
    default_pid = 'parent_id_cat',
    default_upid = 'upid_R200m_bnd_cat',
    default_vel = 'v_spa',
    **other_fields
):
    for h in halos:

        # get moria data for this halo
        halo_data = sminterface.getCat(h.hid)
        h.addField(c.RADIUS, halo_data[default_radius])
        h.addField(c.MASS, halo_data[default_mass])
        h.addField(c.PID, halo_data[default_pid])
        h.addField(c.UPID, halo_data[default_upid])
        h.addField(c.VEL, halo_data[default_vel])
        for k,v in other_fields.items():
            h.addField(k, halo_data[v])

    return

