# functions for loading sparta data and quickly creating containers
from containers import Halo, Peri, Ifl
import global_names as gn
import numpy as np
from typing import List

def makeHalo(sminterface, halo_id, default_position = 'x_spa') -> Halo:
    ogid = sminterface.getOrigID(halo_id)
    # get moria data for this halo
    halo_data = sminterface.getCat(ogid)
    pos = halo_data[default_position][:,:]
    pos = np.squeeze(pos)
    # by default, pos is in cMpc/h. convert to comoving kpc / h
    pos = pos * 1e3
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
        out = sminterface.getRes(host_id, h.hid, result_type = 'res_oct', return_host_ids = True)
        ntcrs = out.shape[0] # number of instances of this halo as a tracer
        for i in range(ntcrs):
            peris = _makePeriEvents(out[i])
            for p in peris:
                h.addEvent(p)
    return

def addIfl(sminterface, halos, host_id = -1):
    for h in halos:
        out = sminterface.getRes(host_id, h.hid, result_type = 'res_tjy', return_host_ids = True)
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
        out = sminterface.getRes(host_id, h.hid, result_type = 'res_tjy', return_host_ids = True)
        # should have shape (ntcr = 1, nsnaps)
        h.addField(gn.VR, out['vr'][0, :])
        h.addField(gn.VT, out['vt'][0, :])
        h.addField('dist', out['r'][0, :])
        
    return

def pidToOrig(sminterface, halos : List[Halo], pid_keys = List[str]):
    for h in halos:
        halo_data = sminterface.getCat(h.hid)
        for key in pid_keys:
            pids = halo_data[key].copy(); pid_mask = pids > 0
            og_pids = sminterface.getOrigID(pids[pid_mask])
            pids[pid_mask] = og_pids
            h.addField(key, pids)
    return

def addCat(
    sminterface,
    halos : List[Halo],
    default_radius = 'R200m_bnd_cat',
    default_mass = 'M200m_bnd_cat',
    default_pid = 'parent_id_cat',
    default_upid = 'upid_R200m_bnd_cat',
    default_vel = 'v_spa',
    default_status = 'status_sparta',
    **other_fields
):
    for h in halos:

        # get moria data for this halo
        halo_data = sminterface.getCat(h.hid)
        h.addField(gn.RADIUS, halo_data[default_radius])
        h.addField(gn.MASS, halo_data[default_mass])
        # all parent ids need to be their original ID
        pids = halo_data[default_pid].copy(); pid_mask = pids > 0
        og_pids = sminterface.getOrigID(pids[pid_mask])
        pids[pid_mask] = og_pids
        h.addField(gn.PID, pids)

        upids = halo_data[default_upid].copy(); upid_mask = upids > 0
        og_upids = sminterface.getOrigID(upids[upid_mask])
        upids[upid_mask] = og_upids
        h.addField(gn.UPID, upids)
        h.addField(gn.STATUS, halo_data[default_status])
        h.addField(gn.VEL, halo_data[default_vel])
        for k,v in other_fields.items():
            h.addField(k, halo_data[v])

    return

# status fields
HALO_STATUS_NOT_FOUND = -2
HALO_STATUS_NONE = -1
HALO_STATUS_HOST = 10
HALO_STATUS_SUB = 20
HALO_STATUS_BECOMING_SUB = 21
HALO_STATUS_BECOMING_HOST = 22
HALO_STATUS_BECOMING_SUB_HOST = 23
HALO_STATUS_SWITCHED_HOST = 24
HALO_STATUS_GHOST_HOST = 30
HALO_STATUS_GHOST_SUB = 31
HALO_STATUS_GHOST_SWITCHING = 32

def makeStatus(sminterface, halos : List[Halo], sub_def : str, do_phantom = False):

    for h in halos:
        halo_data = sminterface.getCat(h.hid)
        new_status = np.zeros(halo_data.shape[0])
        cat_status = halo_data['sparta_status']

        # get parent IDs, but set them all to ogID instead of ID at each snapshot
        pid = halo_data[f'pid_{sub_def}']
        pid_mask = pid > 0
        og_pids = sminterface.getOrigID(pid[pid_mask])
        pid[pid_mask] = og_pids

        # when halo DNE, set to negative one
        dne = cat_status == -1
        new_status[dne] = -1

        # when halo exists but has pid == -1, set to host
        host = ~dne & (pid == -1)
        new_status[host] = HALO_STATUS_HOST

        # when pid is positive and halo exists, set to subhalo
        sub = ~dne &  (pid > 0)
        new_status[sub] = HALO_STATUS_SUB

        # ghost and host halo
        
        ghost = cat_status >= 30
        if do_phantom:
            phantom = halo_data['phantom'] == 1
            ghost = ghost | phantom
        
        new_status[ghost & host] = HALO_STATUS_GHOST_HOST

        # ghost and sub
        new_status[ghost & sub] = HALO_STATUS_GHOST_SUB

        # TODO implement the switching host statuses (not needed at moment)

        h.addField(f"{sub_def}_status", new_status)

        
    return