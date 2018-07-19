import numpy as np
import h5py

def save_mesh_animation(filename, verts, tris, **kwargs):
    verts = np.asarray(verts)
    tris = np.asarray(tris)
    assert verts.ndim == 3
    assert tris.ndim == 2
    with h5py.File(filename, 'w') as f:
        f.create_dataset('verts', data=verts, compression='gzip')
        f['tris'] = tris
        for k, v in kwargs.iteritems():
            if k == 'attributes':
                for an, av in v.iteritems():
                    f.attrs[an] = av
            else:
                f[k] = v
    print "saved mesh animation %s" % filename

def load_mesh_animation(filename, *additional_datasets):
    r = []
    with h5py.File(filename, 'r') as f:
        for name in ['verts', 'tris'] + list(additional_datasets):
            if name in f:
                r.append(f[name].value)
            else:
                print "[warn] non-existent dataset %s requested, returning None" % name
                r.append(None)
    return r

def load_first_frame(filename, *additional_datasets):
    r = []    
    with h5py.File(filename, 'r') as f:
        r.append(f['verts'][0].value)
        for name in ['tris'] + list(additional_datasets):
            r.append(f[name].value)
    return r

def save_blendshapes(filename, shapes, tris, blendshape_names=None):
    with h5py.File(filename, 'w') as f:
        f['tris'] = tris
        for i, s in enumerate(shapes):
            if i == 0:
                name = 'default'
            else:
                name = "%03d_%s" % (i, blendshape_names[i-1]) \
                        if blendshape_names is not None and len(blendshape_names) >= i and blendshape_names[i-1] is not None \
                        else '%03d' % i
            f[name] = s
    print "saved blendshapes %s" % filename

def save_components_as_blendshapes(filename, verts0, tris, components, names=None):
    blendshapes = components + verts0[np.newaxis]
    save_blendshapes(filename, [verts0] + list(blendshapes), tris, blendshape_names=names)

def load_components_from_blendshapes(filename):
    with h5py.File(filename, 'r') as f:
        tris = f['tris'].value
        Xmean = f['default'].value
        names = sorted(list(set(f.keys()) - set(['tris', 'default'])))
        names_fixed = []
        for n in names:
            if len(n) > 3:
                n = n[4:]
            names_fixed.append(n)
        components = np.array([
            f[name].value - Xmean 
            for name in names])
    return components, tris, Xmean, names_fixed

