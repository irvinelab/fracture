#!/usr/bin/env python

import numpy as np
try: from . import vector
except: import vector #Allows it to be run from package dir.
try: from . import simple_cl
except: pass #import simple_cl #Allows it to be run from package dir.
import re
import os
from StringIO import StringIO
import warnings

'''In order to use this code with FEniCS, references to pyopencl have been removed -- I have left a few, but they will not work.
The reason is that six.moves does not contain intern in FEniCS's version of six. See /--local_fenics_path--/six.py for more information.
Note: to get the local fenics path to six, import six, then print six.__file__
Unfortunately, this seems to be a deeply rooted issue.'''

class MeshWarning(Warning):
    pass

class ShapeError(Exception):
    pass

class ToDoError(Exception):
    pass
    
def shape_str(shape):
    return '(' + ', '.join(map(lambda d: '-any-' if d < 0 else str(d), shape)) + ')'

def enforce_shape(arr, shape):
    arr = np.asarray(arr)
    
    if arr.shape == shape: #Shortcut
        return arr

    if arr.ndim > len(shape):
        raise ShapeError('input has two many dimensions (%s) to cast into shape %s' % (arr.ndim, shape_str(shape)))
    elif arr.ndim < len(shape):
        arr.shape = (1, ) * (len(shape) - arr.ndim) + arr.shape
        
    new_shape = tuple(so if st<=0 else st for so, st in zip(arr.shape, shape))
        
    if arr.shape == new_shape: #Shortcut
        return arr
    
    else:
        tile = []
        for so, st in zip(arr.shape, new_shape):
            if so == st: tile.append(1)
            elif so == 1: tile.append(st)
            else: raise ShapeError('input not castable into array of shape %s' % shape_str(shape))
        
        return np.tile(arr, tile)
    
STL_DTYPE = [('n', '3f'), ('c1', '3f'), ('c2', '3f'), ('c3', '3f'), ('byte_count', 'u2')]
    
def colors_astype(c, dtype):
    new_dtype = np.dtype(dtype)
    if c.dtype == np.dtype('u1') and new_dtype in ('f', 'd'):
        return c.astype(new_dtype) / 255.
    elif c.dtype in ('f', 'd') and new_dtype == np.dtype('u1'):
        return (np.clip(c, 0, 1) * 255).astype(new_dtype)
    else:
        return c.astype(new_dtype)
        

# CL = simple_cl.CLSession(device='cpu', use_doubles=False)
# CODE = os.path.join(os.path.split(__file__)[0], 'cl_src', 'mesh.cl')

# def _check_code(cl=CL):
#     if 'mesh_dummy_kernel' not in getattr(cl, 'kernels', {}):
#         cl.compile_file(CODE)
#         

# def set_opencl_params(use_doubles=None, device=None, context=None):
#     '''Set the parameters of the OpenCL Session used for computations.
#     
#     See ``simple_cl`` documentation for details.
#     
#     *Note:* this function must be called *before* any functions that use it,
#     or an error will be raised.
#     
#     Parameters
#     ----------
#     use_doubles : bool, optional (default: False)
#     device : string or list, optional (default: 'cpu')
#     context : pyopencl Context or CLSession object, optional
#         If specified, causes CLSession to share a context.
#     '''
#     if use_doubles is not None: CL.use_doubles = use_doubles
#     if device is not None: CL.device = device
#     if context is not None: CL.initialize(context=context)

        
class Mesh(object):
    '''3D triangular mesh object.
    
    Parameters
    ----------
    points : (num points, 3) array
    triangles : (num triangles, 3) array
    normals : (num points, 3) array (optional)
        Vertex normal vectors.
    colors : (num points, 3 or 4) array (optional)
        Vertex colors, as RGB or RGBA.  Alpha channel is optional.
        Type should either be ``"float"`` (range: [0.0-1.0]) or
        ``"u1"`` (range: [0-255]).
    attached : dict of (num points,) arrays (optional)
        Arbitrary attributes attached to vertices.  Shape is checked on
        initialization, and an error will be thrown if it is not a 1D array
        with correct length.  *(Note: vector or other multivalued arrays must
        be split into components; this is necessary to get consistent results
        when saved to a PLY mesh.)*
    '''


    def __init__(self, points=None, triangles=None, normals=None, colors=None, attached={}):
        if points is None: points = np.zeros((0, 3), 'f')
        if triangles is None: triangles = np.zeros((0, 3), 'i')
        
        self.points = enforce_shape(points, (-1, 3))
        self.triangles = enforce_shape(triangles, (-1, 3))
        
        if normals is not None:
            self.normals = enforce_shape(normals, (len(self.points), 3))
        
        if colors is not None:
            self.colors = enforce_shape(colors, (len(self.points), -1))
        
        self.attached = {}
        for k, v in attached.iteritems():
            self.attached[k] = enforce_shape(v, (len(self.points), ))
            
            
    def attach(self, key, val):
        '''Attach a named vertex parameter.
        
        Equivalent to setting an item in the ``attached`` attribute, but also
        does shape checking.
        
        Parameters
        ----------
        key : string
        val : numpy array shaped (num_points, )
        '''
        
        self.attached[key] = enforce_shape(val, (len(self.points), ))

    
    def __getattr__(self, key):
        #Allow access to attached attributes directly.
        if key in self.attached:
            return self.attached[key]
        else:
            raise AttributeError("mesh does not have property '%s'" % key)

        
    def save(self, fn, file_type=None):
        '''Save mesh as a file.
        
        Supported file types:
            * ``ply``: The most useful format; will also save attached parameters.
            * ``stl``: Used primarily by 3D printers.  Only save geometry, not
                            colors, normals, etc.
            * ``pov``: Used to render object in POVRay.  If normals exists,
                            smooth triangles will be saved, but colors, etc.
                            are not retained.  See :meth:`save_pov` for more
                            options and details, including sample scene
                            generation.
        
        
        Parameters
        ----------
        fn : string
            File name
        file_type : string, optional
            File type, should be one of the supported formats.
            If not specified, determined from the file name.
        '''
        
        if file_type is None: file_type = os.path.splitext(fn)[1].strip('.')
        
        if file_type.lower() == 'ply':
            with open(fn, 'wb') as f:
                f.write(encode_ply(self.points, self.triangles,
                                   normals=getattr(self, "normals", None),
                                   colors=getattr(self, "colors", None),
                                   extra_vertex_attributes=self.attached))
                
        elif file_type.lower() == 'stl':
            #if extra_vertex_attributes is not None:
            #    raise ValueError('extra_vertex_attributes can only be saved in PLY files')
                
            stl_dat = np.empty(len(self.triangles), dtype=STL_DTYPE)
            stl_dat['byte_count'] = 0 #Just how STL works.
            
            for i, a in enumerate(['c1', 'c2', 'c3']):
                stl_dat[a] = self._tps(i)
            
            stl_dat['n'] = self.face_normals()
    
            with open(fn, 'wb') as f:
                header = '\x00\x00This is an STL file. (http://en.wikipedia.org/wiki/STL_(file_format))'
                f.write(header + ' ' * (80 - len(header)))
                f.write(np.array(len(self.triangles), 'u4').tostring())
                f.write(stl_dat.tostring())
                
        elif file_type.lower() == 'pov':
            self.save_pov(fn)
                
        else:
            raise ValueError('file type should be "ply", "stl", or "pov"')
        
    
    def rotate(self, rv):
        '''Rotate the mesh (in place).
        
        Parameters
        ----------
        rv : vector
            Rotation vector, will rotate counter-clockwise around axis by an amount
            given be the length of the vector (in radians).
        '''
        self.points = vector.rot(rv, self.points)
        if hasattr(self, "normals"):
            self.normals = vector.rot(rv, self.normals)
    
    
    def rotated(self, rv):
        '''Return a rotated copy of a mesh'''
        m = self.copy()
        m.rotate(rv)
        return m
    
    
    def scale(self, s):
        '''Scale the Mesh in place.
        
        Parameters
        ----------
        scale : float or vector
        '''
        self.points *= s
    
    def scaled(self, s):
        '''Return a scaled copy of a mesh.'''
        m = self.copy()
        m.scale(s)
        return m
    
    
    def invert(self):
        '''Flip the mesh normals, including triangle winding order and the
        ``normals`` field (if present).'''
        self.triangles = self.triangles[:, ::-1]
        if hasattr(self, "normals"):
            self.normals *= -1
    
    def inverted(self):
        '''Return an inverted copy of a mesh.'''
        m = self.copy()
        m.inverted()
        return m
    
    
    def translate(self, delta):
        '''Rotate the mesh (in place).
        
        Parameters
        ----------
        delta : vector
        '''
        self.points += delta
        
    def translated(self, delta):
        '''Return an translated copy of a mesh.'''
        m = self.copy()
        m.translate(delta)
        return m
    
        
    def volume(self):
        '''Return the volume of the mesh.
        
        The mesh is assumed to be closed and non-interecting.
        '''
        px, py, pz = self.tps(0).T
        qx, qy, qz = self.tps(1).T
        rx, ry, rz = self.tps(2).T
        
        return (px*qy*rz + py*qz*rx + pz*qx*ry - px*qz*ry - py*qx*rz - pz*qy*rx).sum() / 6.
        
        
    def is_closed(self, tol=None):
        '''Check if the mesh is closed by calculating the volume of a
        transposed copy.
        
        This routine is not fool-proof, but catches most open meshes.
        
        Parameters
        ----------
        tol : float (optional)
            The relative tolerance of the volume comparison.  Defaults to 1E-12
            if the vertex data type is double precision, otherwise 1E-6.
        '''
        if tol is None: tol = 1E-6 if self.points.dtype == np.dtype('f') else 1E-12
        
        x, y, z = self.points.T
        
        m2 = self.copy()
        m2.points += 2 * np.array((max(x) - min(x), max(y) - min(y), max(z) - min(z)))
        v1 = self.volume()
        v2 = m2.volume()
        
        return abs((v1 - v2) / v1) < tol
        
        
    def colorize(self, c, cmap='jet', clim=None):
        '''Color a mesh using matplotlib specifications.
        
        Parameters
        ----------
        c : string or array
        
            * If type is string, it is interpreted like a normal matplotlib color
              (e.g., ``"r"`` for red, or ``"0.5"`` for 50% gray, or ``"#FF8888"`` for pink).
            
            * If type is array, it should have the same length as the number of points,
              and it will be converted to a colormap.
            
        cmap : string or matplotlib colormap (default: 'jet')
            The color map to use.  Only relevant if ``c`` is an array.
        clim : tuple (default: None)
            The color limits.  If None, the max/min of c are used.
        '''
        
        
        if isinstance(c, basestring):
            import matplotlib.colors
            self.colors = np.tile(matplotlib.colors.colorConverter.to_rgb(c), (len(self.points), 1))
        else:
            import matplotlib.cm
            c = enforce_shape(c, (len(self.points),))
            if clim is None: clim = (c.min(), c.max())
            self.colors = matplotlib.cm.get_cmap(cmap)((c - clim[0]) / (clim[1] - clim[0]))
            
        
    def copy(self):
        '''Return a copy of the mesh, with sub-arrays copied as well.'''
        return Mesh(
            points=self.points.copy(),
            triangles=self.triangles.copy(),
            normals=self.normals.copy() if hasattr(self, 'normals') else None,
            colors=self.colors.copy() if hasattr(self, 'colors') else None            
        )
        
    def _tps(self, n):
        '''Returns the vortices corrsesponding to one corner of each triangle.
        
        Parameters
        ----------
        n : 0, 1, 2 (int)
            Triangle corner number
            
        Returns
        -------
        X : [num points, 3] array
            The vertices corresponding to the specified corner number
        '''
        return self.points[self.triangles[:, n]]
    
    def face_normals(self, right_hand=True, normalize=True):
        '''Compute triangle face normals.
        
        Parameters
        ----------
        right_hand : bool (default: True)
            Right-hand or left hand normals?  Normally, right-hand normals are
            used, meaning triangles are wound counter-clockwise when viewed
            from "outside" the mesh.
        normalize : bool (default: True)
            If True, normals are normalized after being computed.
            
        Returns
        -------
        n : [num triangles, 3] array
            Normals corresponding to each face, generated as a cross product.
        '''
        
        n = vector.cross(self._tps(2) - self._tps(0), self._tps(1) - self._tps(0))
        if normalize:
            n = vector.norm(n)
        if not right_hand:
            n *= -1
        return n
    
    
         
    def convert_colors(self, dtype):
        '''Convert type of colors field, rescaling if required.
        
        **Note: usually the type of the color field is not important, e.g. when
        saving to ``"ply"`` format the color is automatically converted on saving.**
        
        Parameters
        ----------
        dtype : valid numpy data type (usually ``"f"`` or ``"u1"``)
            Data type to convert colors to.  If converting from ``"u1"`` to a float
            type, rescales from [0-255] to [0.0-1.0], and vice-versa.  If converting
            from float to ``"u1"``, colors will first be clipped from (0-1).
        '''
        self.colors = colors_astype(self.colors, dtype)
            
           
    def __iadd__(self, other):
        if hasattr(other, 'points') and hasattr(other, 'triangles'):
            NO = len(self.points)
            self.points = np.vstack([self.points, other.points])
            self.triangles = np.vstack([self.triangles, other.triangles + NO])
            
            shc = hasattr(self, 'colors')
            ohc = hasattr(other, 'colors')
            
            missing_props = []
            
            if shc or ohc:
                if not ohc:
                    #warnings.warn('trying to add two meshes when only one has colors defined; making uncolored mesh white', MeshWarning, stacklevel=2)
                    missing_props.append('colors')
                    oc = np.ones((len(other.points), 3), 'f')
                else:
                    oc = other.colors
            
                if not shc:
                    #warnings.warn('trying to add two meshes when only one has colors defined; making uncolored mesh white', MeshWarning, stacklevel=2)
                    missing_props.append('colors')
                    mc = np.ones((NO, 3), dtype=oc.dtype)
                    if mc.dtype == np.dtype('u1'): mc *= 255
                else:
                    mc = self.colors
                    
                oc = colors_astype(oc, mc.dtype)
                self.colors = np.ones((len(self.points), max(oc.shape[1], mc.shape[1])), mc.dtype)
                if self.colors.shape[1] == 4 and self.colors.dtype == np.dtype('u1'):
                    self.colors[:, 3] = 255
                self.colors[:mc.shape[0], :mc.shape[1]] = mc
                self.colors[mc.shape[0]:, :oc.shape[1]] = oc
                
            
            for prop in set(['normals'] + self.attached.keys() + other.attached.keys()):
                p1 = getattr(self, prop, None)
                p2 = getattr(other, prop, None)
                
                if p1 is not None:
                    if p2 is not None:
                        #self.normals = np.vstack([self.normals, other.normals])
                        p_new = np.concatenate([p1, p2], axis=0)
                        if prop == 'normals': self.normals = p_new
                        else: self.attached[prop] = p_new
                    else:
                        #warnings.warn('trying to add two meshes when only one has normals defined; dropping normals from result', MeshWarning, stacklevel=2)
                        missing_props.append(prop)
                        if prop == 'normals': del self.normals
                        else: del self.attached[prop]
                    
                elif p2 is not None:
                    #warnings.warn('trying to add two meshes when only one has normals defined; dropping normals from result', MeshWarning, stacklevel=2)
                    missing_props.append(prop)
            
            if missing_props:
                warnings.warn('tried to add two meshes with different attached properties (conflicted values: %s)\nmissing properties dropped from result' % missing_props)
            
            
            return self
        
        else: raise TypeError('can only add a Mesh to another Mesh')

    
    def __add__(self, other):
        m = self.copy()
        m += other
        return m
        
    
    def euler_char(self):
        '''Return the *total* Euler Characteristic of a mesh: :math:`\chi = V - E + F`.
        
        Note that orphaned points (not connected to triangles) are automatically
        excluded.
        
        If the mesh is composed of multiple parts, they can first be separated
        with :meth:`Mesh.separate`.
        '''
        
        #http://en.wikipedia.org/wiki/Euler_characteristic
        
        #Create a list of edges with index p0 + NP*p1  [where p1<p0]
        #Because we presorted the triangles, we gaurantee the lowest index point
        #  is p0, thus this is a unique edge index.
        tt = self.triangles.copy()
        tt.sort(axis=-1)
        N = len(self.points)
        num_edges = len(np.unique(np.array([
                tt[:, 0] + N*tt[:, 1],
                tt[:, 0] + N*tt[:, 2],
                tt[:, 1] + N*tt[:, 2],
            ])))

        #Unreferenced points not included this way!
        num_points = len(np.unique(self.triangles)) 
        
        return  num_points - num_edges + len(self.triangles)
         
    def separate(self, return_point_indices=False):
        '''Break the mesh into disconnected parts.
        
        Parts are considered connected if any of their triangles share a corner.
        
        *Note:* This algorithm is unoptimized and may run slower than you expect.
        
        Parameters
        ----------
        return_point_indices : bool, optional (default: False)
            If true, returns the indices of the points in the original mesh
            which are in each sub-mesh.
        
        Returns
        -------
        meshes : list of meshes
        point_indices : list of int arrays (optional)
            Only returned if ``return_point_indices==True``.
        '''

        #----------------------------------------------------------------------
        #Old, slower method
        #----------------------------------------------------------------------
        #
        #t = self.triangles
        #marked = np.zeros(len(self.points), bool)
        #groups = []
        #
        #while len(t):
        #    g = [t[-1:]]
        #    t = t[:-1]
        #    
        #    while True:
        #        ptt = np.unique(g[-1])
        #        #print ptt, t.shape
        #        ptt = ptt[np.where(~marked[ptt])]
        #        
        #        if not len(ptt): break
        #
        #        connected = (ptt.reshape(-1, 1, 1) == t).any(0).any(-1)
        #
        #        if connected.any():
        #            g.append(t[np.where(connected)])
        #            t = t[np.where(~connected)]
        #            marked[ptt] = True
        #        else: break
        #        
        #    groups.append(np.vstack(g))
        #                        
        #parts = []
        #point_i = []
        #for t in groups:
        #    ip, it = np.unique(t, return_inverse=True)
        #    
        #    extras = {}
        #    for k in ['normals', 'colors']:
        #        if hasattr(self, k):
        #            extras[k] = getattr(self, k)[ip]
        #    
        #    parts.append(Mesh(points=self.points[ip], triangles=it.reshape(-1, 3), **extras))
        #    point_i.append(ip)

        
        #----------------------------------------------------------------------
        #New, 10x faster method (~10k points in a few 100 ms on 2012 hardware)
        #----------------------------------------------------------------------
        #
        #Find all the triangles connected to a point
        tn = np.argsort(self.triangles.flat)
        ip = self.triangles.flat[tn]
        tn //= 3
        
        breaks = np.where(ip[:-1] != ip[1:])[0]
        start = np.zeros(len(self.points), 'i8')
        end = np.zeros(len(self.points), 'i8')
        
        #triangles connected to point i => tn[start[i]:end[i]]
        end[ip[breaks]] = breaks+1
        start[ip[breaks+1]] = breaks+1
        end[ip[breaks[-1]+1]] = len(ip)
        
        tris_at_point = {}
        for i in range(len(self.points)):
            tris_at_point[i] = tn[start[i]:end[i]]
            
        
        
        t_free = np.ones(len(self.triangles), bool)
        ffunc = lambda x: t_free[x]
        nt = 0
        
        groups = []
        dummy = np.empty(0, dtype='i8') #Used to prevent type conversion on concatenates
        
        while nt < len(self.triangles):
            new_tris = np.array([np.argmax(t_free)]) #Pick a free one.
            group = []
            
            while True:
                t_free[new_tris] = False
                nt += len(new_tris)
                group.append(new_tris)
                
                connected_tris = np.unique(np.concatenate(
                    [tris_at_point.pop(i, dummy) for i in self.triangles[group[-1]].flat]))
        
                new_tris = connected_tris[np.where(t_free[connected_tris])]
                
                if not len(new_tris):
                    groups.append(np.concatenate(group))
                    break

        parts = []
        point_i = []
        for t in groups:
            ip, it = np.unique(self.triangles[t], return_inverse=True)
            
            extras = {}
            for k in ['normals', 'colors']:
                if hasattr(self, k):
                    extras[k] = getattr(self, k)[ip]
            
            attached = {}
            for k in self.attached:
                attached[k] = self.attached[k][ip]
            
            extras['attached'] = attached
            
            parts.append(Mesh(points=self.points[ip], triangles=it.reshape(-1, 3), **extras))
            point_i.append(ip)

        if return_point_indices: return parts, point_i
        else: return parts
    
        
    def periodic_separate(self, top=None):
        '''Separate into sections and then reconnect across periodic boundaries.
        
        Similar to :meth:`separate`, but checks for periodic reconnections.
        
        Ideally this mase generated by :meth:`ilpm.geometry_extractor.periodic_isosurface`, but in principle
        this is not necessary.  The algorithm looks for sections that touch the
        bottom edge (x/y/z=0) and tries to move them to the top and join them.
    
        Parameters
        ----------
        m : ilpm.mesh.Mesh
        top : list of floats, optional
            The period of each axis.  If not specified it will be the maximum value
            of any points along each axis.
            
            *Note: this is generally ok even
            if the maximum value isn't the period; in this case there should
            also be no points touching the bottom boundary, so this axis will
            have no effect.  However, if you have a mesh where the points touch
            the "bottom" (x/y/z=0) and not the "top", then you must manually
            specify the "top" to get correct results.*
            
        Returns
        -------
        meshes : list of Mesh objects
            The reconnected sections.  Any mesh touching a bottom boundary will be
            pushed to the top and merged with coindident sections.
        '''
        
        if top is None: top = self.points.max(0)
    
        parts = self.separate()
    
        for axis in range(3):
            bulk = []
            edge = []
            
            for m in parts:
                on_top = m.points[:, axis] == top[axis]
                if on_top.any():
                    edge.append((m, on_top))
                else:
                    on_bottom = m.points[:, axis] == 0.0
                    if on_bottom.any():
                        m.points[:, axis] += top[axis]
                        edge.append((m, on_bottom))
                    else:
                        bulk.append(m)
                        
            while len(edge) > 1:
                for i, (m1, mask1) in enumerate(edge):
                    break_now = False
                    
                    for j, (m2, mask2) in enumerate(edge[:i]):
                        merged = _attempt_merge(m1, m2, mask1, mask2)
                        if merged is not False:
                            edge.pop(i)
                            edge.pop(j)
                            
                            mask = merged.points[:, axis] == top[axis]
                            
                            if mask.any():
                                edge.append((merged, mask))
                            else:
                                bulk.append(merged)
        
                            #If we changed the edge array, we need to restart the merging loop.                        
                            break_now = True
                            break
                
                    if break_now: break
                else: break #No parts separated
            
            parts = bulk + map(lambda x: x[0], edge)
            
        return parts

    
    def clip(self, a, level=0, both_sides=False):
        '''Clip a mesh by some parameter defined for each point.
        
        Parameters
        ----------
        a : [N] array
            An array with the same size as the number of points.
        level : float, optional (default: 0)
            The level at which to clip the mesh, keeps all sections with
            (a-level) >=0, clipping triangles that span the boundary.
        both_sides : bool, optional (default: False)
            If True, returns two meshes, one for each side of the split.
            
        Returns
        -------
        clipped_mesh : Mesh
            A clipped version of the mesh.  All attached attributes are clipped
            and retained as well.
        other_side: Mesh, optional
            If both_sides == True, returns the other side of the clip.
        '''
        
        a = enforce_shape(a, (len(self.points), ))
        if level: a -= level
        ab = a >= 0
        
        #Inside and outside point index.
        ipi = np.where(ab)[0]
        ni = len(ipi)
        opi = np.where(~ab)[0]
        no = len(opi)
        
        #Point map for inside and outside points; we can share a map for both,
        #  since there is no overlap.  This map is used to go from old point
        #  index to new point index.
        point_map = np.zeros(len(a), dtype='i8')
        point_map[ipi] = np.arange(ni)
        point_map[opi] = np.arange(no)
        
        
        #Triangle corner in
        tci = ab[self.triangles]
        
        new_tri = []
        if both_sides:
            new_tri_out = []
        
        clipped_edges = {}
        
        ak = filter(lambda k: hasattr(self, k), ['points', 'normals', 'colors']) + self.attached.keys()
        new_points = {}
        for k in ak: new_points[k] = []
        
        def get_edge(i1, i2):
            if i1 > i2: i1, i2 = i2, i1
            ii = (i1, i2)
            
            if ii not in clipped_edges: 
                clipped_edges[ii] = len(new_points['points'])

                z = -a[i1] / (a[i2] - a[i1])
                for k in ak:
                    attr = getattr(self, k)
                    v = attr[i1] * (1-z) + attr[i2] * z
                    #print k, v
                    if k == 'normals': v = vector.norm(v)
                    new_points[k].append(v)
        
            return clipped_edges[ii]

        
        #Partial in triangles
        #if False:
        for i in np.where(tci.all(1) ^ tci.any(1))[0]:
            c = self.triangles[i]
            aab = ab[c]
            
            #Roll the corner indices until index 0 is in and 2 is out.
            #A while loop is avoided for safety, although it should be ok.
            for i in range(2):
                if aab[0] and not aab[2]:
                    break
                else:
                    c = np.roll(c, 1)
                    aab = np.roll(aab, 1)
                    
            cm = point_map[c]
                    
            #Now there are two options: index 1 is in or out, i.e. 1 or 2
            #  corners are inside the clip.  We'll handle each case
            #  individually.
            if aab[1]: #Corners 0, 1 inside.
                #Create clipped points
                e1 = get_edge(c[0], c[2])
                e2 = get_edge(c[1], c[2])
                
                #Add triangles, maintaining winding order!
                new_tri += [(cm[0], e2+ni, e1+ni), (cm[0], cm[1], e2+ni)]
                if both_sides:
                    new_tri_out += [(e1+no, e2+no, cm[2])]

            else: #Same as above, but now only corner 0 is in.
                e1 = get_edge(c[0], c[1])
                e2 = get_edge(c[0], c[2])
                new_tri += [(cm[0], e1+ni, e2+ni)]
                if both_sides:
                    new_tri_out += [(e1+no, cm[1], cm[2]), (e1+no, cm[2], e2+no)]
            
        ai_ti = np.where(tci.all(1))[0]
        kw = dict(
            triangles=np.concatenate([point_map[self.triangles[ai_ti]], np.asarray(new_tri, dtype='i8').reshape(-1, 3)]),
            attached = {})
    
        if both_sides:
            ao_ti = np.where(~tci.any(1))[0]
            kw_out = dict(
                triangles=np.concatenate([point_map[self.triangles[ao_ti]], np.asarray(new_tri_out, dtype='i8').reshape(-1, 3)]),
                attached = {})
            
        for k in ak:
            v = getattr(self, k)
            vn = np.array(new_points[k], dtype=v.dtype).reshape((-1, ) + v.shape[1:])
            
            vi = np.concatenate([v[ipi], vn])
            if both_sides:
                vo = np.concatenate([v[opi], vn])
            
            if k in ('points', 'normals', 'colors'):
                kw[k] = vi
                if both_sides: kw_out[k] = vo
            else:
                kw['attached'][k] = vi
                if both_sides: kw_out['attached'][k] = vo
            
        if both_sides:
            return Mesh(**kw), Mesh(**kw_out)
        else:
            return Mesh(**kw)
    
    
    def color_by_proximity(self, p, c=None, w=None, rgb_only=True, power=-6, period=None):
        '''Color a mesh by proximity to another path, tangle, or set of colored points.
        
        Parameters
        ----------
        p : Path, Tangle or [N, 3] array
        c : [N, 3 or 4] array
            The color of each point; required if p is an array, and ignored if
            it is a Path or Tangle (in this case the ``color`` attribute is
            used).
        w : [N] array, optional
            Weight array.
        rgb_only : bool, optional (default: True)
            If True, the alpha component is dropped
        power : int, optional (default: -6)
            The power of the radial weighting function.
            Recommended values are <= -4.
        period : None, float or (3,) array_like (optional)
            If not None, the period to use for wrapping the proximity
            calculation (data assumed to be go from 0-period on each axis).
            If one number is specified, the period is assumed to be the same
            on all three axes.
        '''
        
        if hasattr(p, '_color_by_proximity_points'):
            p, c, w = p._color_by_proximity_points()
        
        p = enforce_shape(p, (-1, 3))
        
        if c is None: raise ValueError('if specified as points, a color array must also be specified')
        c = enforce_shape(c, (len(p), -1))
        
        if c.shape[-1] == 3: c = np.concatenate([c, np.zeros((len(c), 1))], axis=-1)  
        elif c.shape[-1] != 4: raise ValueError('last dimension of color array must be 3 or 4')
        
        if w is None: w = np.ones((len(c), 1))
        else: w = enforce_shape(w, (len(c), 1) )

        _check_code()
        
        pw = CL.to_device(np.concatenate([p, w], axis=-1).astype(CL.real))
        c = CL.to_device(c.astype(CL.real))
        
        #self._color_by_proximity_inner(pw, c, rgb_only, power)
        
        x = CL.to_device(self.points)
        color = CL.empty((len(self.points), 4))
        
        #__kernel void path_color_by_proximity(
        #    global REAL* X, int nx,
        #    global REAL4* pw, global REAL4* c, int nc,
        #    global REAL4* color, int power
        #    )        

        if period is None:
            CL.mesh_color_by_proximity(
                x, len(x),
                pw, c, len(pw),
                color, power)
        else:
            p = np.ones(3, dtype=CL.real)
            p *= period
            px, py, pz = p

            CL.mesh_color_by_proximity_periodic(
                x, len(x),
                pw, c, len(pw),
                color, power,
                px, py, pz)
        
        color = color.get()
        
        if rgb_only: self.colors = color[..., :3]
        else: self.colors = color
        
        
    def _pov_vec(self, V, prec=5):
        return '<%s>' % (','.join(['%%.%df' % prec % x for x in V]))
        
        
    def _pov_point(self, i, prec=5):
        s = self._pov_vec(self.points[i], prec)
        
        if hasattr(self, 'normals'):
            s += ',' + self._pov_vec(self.normals[i], prec)
            
        return s
        
    def save_pov(self, fn, include_extras=False, mesh_name=None, precision=5):
        '''Save a mesh into a POV file; optionally create a sample scene.
        
        The resulting POV file contains only a definition of the mesh, which
        can be imported into other files.
        
        The POV file will not contain colors or other attached attributes,
        but corner normals will cause it to render smooth triangles.
        
        Parameters
        ----------
        fn : string
            Filename to save to; ``.pov`` extension will be forced.
        include_extras : bool, optional (default: False)
            If True, creates ``<mesh_name>_scene.pov`` and
            ``<mesh_name>_scene.sh``, which are a sample scene and script for
            rendering the mesh.
        mesh_name : string, optional (default: ``<mesh_name>_mesh``)
            The name of the mesh in the declare statement.  (i.e. this is the
            name of the object created in the POV file.)
        precision : int, optional (default: 5)
            The precision of the the points
        '''
        
        fnb, ext = os.path.splitext(fn)
        if not ext: ext = '.pov'
        if mesh_name is None: mesh_name = os.path.split(fnb)[-1] + "_mesh"
        
        main_name = fnb + ext
        output = open(main_name, 'wt')
        
        output.write('#declare %s = mesh\n{\n' % mesh_name)
        
        if hasattr(self, 'normals'):
            for t in self.triangles:
                output.write('    smooth_triangle {%s}\n' % ','.join(map(lambda x: self._pov_point(x, precision), t)))
        else:
            for t in self.triangles:
                output.write('    triangle {%s}\n' % ','.join(map(lambda x: self._pov_point(x, precision), t)))
                    
        output.write('}\n')
        output.close()
        
        if include_extras:
            center = self.points.mean(0)
            p = self.points - center
            max_size = abs(p).max()
            bottom = min(p[..., 1]) / max_size
            
            scene_name = fnb + "_scene" + ext
            
            output = open(scene_name, 'wt')
            output.write('''#include "%s"

camera {
    location <2.5, 2.5, 0>
    look_at 0
}

light_source {
    <10, 0, 0>, color rgb 1.3
    area_light 2*y, 2*z, 10, 10
    adaptive 1
    circular
    
    rotate y * 20
    rotate z * 60
}


object {
    %s
    translate -%s
    scale 1 / %0.5f
    texture {
    pigment {color rgb <1, 0.2, 0.2>}
        finish {phong 0.5}
    }
}

plane {
    <0, 1, 0>, %f
    texture {
        pigment {color rgb 1.5}
        finish {ambient 0.2}
    }
}''' % (main_name, mesh_name, self._pov_vec(center), max_size, bottom)
    )
            output.close()
            
            script_file = fnb + "_scene.sh"
            output = open(script_file, 'wt')
            output.write('#!/bin/bash\npovray  +I%s +O%s.png +A0.3 +W1024 +H768\n' % (scene_name, fnb))
            output.close()
            os.chmod(script_file, 0777)

    def remove_degenerate_triangles(self):
        '''Remove triangles with repeated edges.'''
        
        self.triangles = self.triangles[
            np.where((self.triangles != np.roll(self.triangles, 1, axis=-1)).all(-1))]
        

#class Polygon(object):
#    '''2D Polygon Object.
#    
#    Used primarily by :func:`make_tube` for describing boundaries.
#    
#    Polygons should be defined counter-clockwise for positive area.
#    
#    Parameters
#    ----------
#    points : [N, 2] array
#        The corners of the polygon in order.  Assumed to be closed (the last
#        point should *not* be a repeat of the first point.)
#    normals : [N, 2] array, optional
#        The normal of each corner.  If not specified, automatically generated.
#        (The normal is used when producing a mesh outline from a given shape)
#    '''
#    
#    def __init__(self, points, normals=None):
#        self.points = enforce_shape(points, (-1, 2))
#
#        plus = np.roll(self.points, -1, axis=0)
#        t = vector.norm(plus - self.points)
#        self.seg_n = np.vstack([-t[:, 1:2], t[:, 0:1]])
#
#        if normals is None:
#            self.normals = vector.norm(self.seg_n + np.roll(self.seg_n, +1, axis=0))
#        else: 
#            self.normals = vector.norm(enforce_shape(normals, self.points.shape))
#        
#    def area(self):
#        '''Return the signed polygon area.
#        
#        Counterclockwise winding produces positive area.
#        '''
#        plus = np.roll(self.points, -1, axis=0)
#        return 0.5 * (self.points[:, 0]*plus[:, 1] - plus[:, 0]-self.points[:, 1])
#    
#    def flip(self):
#        '''Flip the polygon winding and normals.'''
#        self.points = self.points[::-1]
#        self.normals *= -1
#        self.seg_n *= -1
#        
#    def check_orientation(self):
#        '''Check if the polygon is wound counterclockwise, flip if it isn't.'''
#        if self.area() < 0: self.flip()
#        
#    def extrude(self, X, A, B, enforce_normality=True, closed=True):
#        '''Extrude the polygon as the boundary of a 3D path, producing a mesh.
#        
#        Parameters
#        ----------
#        X : [N, 3] array
#            The centerline of the extrusion
#        A, B : [N, 3] arrays
#            The vectors which will correspond to the x and y coordinate of the
#            polygon.
#        enforce_normality : bool, optional (default: True)
#            If True, A and B are forced to be unit length and perpendicular.
#            (A is normalized, and B is reorientated and normalized with
#            Graham Schmidt.)
#        closed : bool, optional (default: True)
#            If True, the path is taken to be closed.
#        '''
#        
#        X = enforce_shape(X, (-1, 2))
#        A = enforce_shape(A, X.shape)
#        B = enforce_shape(B, X.shape)
#        
#        if enforce_normality:
#            A = vector.norm(A)
#            B = vector.norm(B - A*vector.dot1(A, B))
#        
        
        
        
        
        
    
def surface_of_revolution(r, z, N=100):
    '''Create a mesh from a surface of revolution.
    
    Surface is revolved around the z axis.
    
    Parameters
    ----------
    r, z : array_like, 1D
        The radius and z coordinate of the surface of revolution.
    N : int, optional (default: 100)
        The number of points in the revolution
        
    Returns
    -------
    surface : Mesh
        The revolved surface.
    '''
    r = enforce_shape(r, (-1,))
    z = enforce_shape(z, r.shape)
    
    phi = np.linspace(0, 2*np.pi, N, endpoint=False).reshape(-1, 1)
    X = np.empty((N, len(r), 3))
    X[..., 0] = r * np.cos(phi)
    X[..., 1] = r * np.sin(phi)
    X[..., 2] = z
            
    p00 = np.arange(np.prod(X.shape[:2])).reshape(-1, len(r))
    p10 = np.roll(p00, -1, axis=0)
    p01 = np.roll(p00, -1, axis=1)
    p11 = np.roll(p10, -1, axis=1)
        
    triangles = np.swapaxes(np.array([[p00, p01, p10], [p01, p11, p10]]), 1, -1)
    #triangles = np.array([[p00, p01, p10], [p01, p11, p10]])
            
    return Mesh(points=X.reshape(-1, 3), triangles=triangles.reshape(-1, 3))
            
            
            
def extrude_shape(X, A, B, outline=10, outline_normals=None, scale=1, closed=True,
              enforce_normality=True, enforce_direction=True, outline_closed=True):
    '''Create a tubular mesh by extruding a shape along a path.
    
    Parameters
    ----------
    X : [N, 3] array
        The centerline of the extrusion
    A, B : [N, 3] arrays
        The vectors which will correspond to the x and y coordinate of the
        polygon.  Nomimally, dX, A, B form an orthonormal basis.
    outline : int or array
        If int, a circle is automatically generated with that number of points.
        If array, shape should be [M, 2] -or- [N, M, 2], where M is the number
        of points in the outline (for an outline which varies along the tube).
    outline_normals: array, optional
        The 2D normals of each point in the outline.  Shape should match
        outline. 
    scale : float or [N] array
        The scale of the outline.
        *Note:* If scale is negative, the mesh will be inside out.
    closed : bool, optional (default: True)
        If True, the centerline is taken to be closed.
    enforce_normality : bool, optional (default: True)
        If True, A and B are forced to be unit length and perpendicular.
        (A is normalized, and B is reorientated and normalized with
        Graham Schmidt.)
    enforce_direction : bool, optional (default: True)
        If True, orients the path counter-clockwise automatically, accounting
        for the fact that the basis vector may be right or left-handed.
        *Note:* if variable outline is specified, it only checks the *first*
        outline, and flips all outlines.
    outline_closed : bool, optional (default: True)
        If True, the outline is assumed to be closed.
    '''
    X = enforce_shape(X, (-1, 3))
    A = enforce_shape(A, X.shape)
    B = enforce_shape(B, X.shape)
    
    #Create tangent vector and ds
    T = vector.norm(np.roll(X, -1, axis=0) - np.roll(X, +1, axis=0))
    ds = vector.mag1(np.roll(X, -1, axis=0) - X)
    if not closed: #End points don't wrap, so fix them!
        T[0] = vector.norm(X[1] - X[0])
        T[-1] = vector.norm(X[-1] - X[-2])
        ds = ds[:-1] #There is one less segment than the number of points for open paths
        
    if enforce_normality:
        A = vector.norm(A)
        B = vector.norm(B - A*vector.dot1(A, B))

    #If there is no outline, make a circle
    if isinstance(outline, int):
        phi = np.arange(outline) * 2*np.pi / outline
        outline = np.array([np.cos(phi), np.sin(phi)]).T
        
    outline = enforce_shape(outline, (-1, -1, 2))
    handedness = np.sign(vector.dot(X[1] - X[0], vector.cross(A[0], B[0])))

    
    #If direction enforcement is on...
    if enforce_direction and outline_closed:
        o1 = outline[0]
        p1 = np.roll(o1, -1, axis=0)
        area = 0.5 * (o1[:, 0]*p1[:, 1] - p1[:, 0]*o1[:, 1]).sum()
    
        #print vector.dot(vector.norm(X[1] - X[0]), vector.cross(A[0], B[0]))
        #print handedness
                             
        if area * handedness < 0:
            outline = outline[:, ::-1]
    
    #print outline.shape,  np.array(scale).reshape(-1, 1, 1).shape
    outline = outline * np.array(scale).reshape(-1, 1, 1)
    
    #Generate outline normals if not specified
    if outline_normals is None:
        if outline_closed:
            t = vector.norm(np.roll(outline, -1, axis=1) - outline)
            seg_n = t[..., ::-1] * (-1, 1) * -handedness
            outline_normals = vector.norm(seg_n + np.roll(seg_n, +1, axis=1))
        else:
            t = vector.norm(outline[:, 1:] - outline[:, :-1])
            seg_n = t[..., ::-1] * (-1, 1) * -handedness     
            outline_normals = np.zeros_like(outline)           
            outline_normals[:, :-1] += seg_n
            outline_normals[:, 1:] += seg_n
            outline_normals = vector.norm(outline_normals)
        
    outline_normals = enforce_shape(outline_normals, outline.shape)
    
    #Make a basis vector
    AB = np.concatenate([A.reshape(-1, 1, 3, 1), B.reshape(-1, 1, 3, 1)], axis=-1)
    #Reshape everything as [centerline index, outline index, xyz, ab index]
    X = X.reshape(-1, 1, 3)
    T = T.reshape(-1, 1, 3)
    ds = ds.reshape(-1, 1, 1)
    
    #sum over ab index of basis vectors to convert outline ab to xyz coordinates
    offset = (outline[:, :, np.newaxis, :] * AB).sum(-1)
    points = X + offset
    #ditto for normal vectors
    normals = (outline_normals[:, :, np.newaxis, :] * AB).sum(-1)
    
    #Correct normal vectors for changes in the radius of the tube.
    #To do this, first compute a m=dr/ds slope for each point
    R = offset - T*vector.dot1(T, offset)
    r = vector.mag1(R) #Radius of point
    dr = np.roll(r, -1, axis=0) - r
    
    if closed:
        ms = dr/ds #Slope on segment
        m = 0.5 * (ms + np.roll(ms, +1, axis=0)) #Slope on point
    else:
        dr = dr[:-1] #One less segment for open paths
        ms = dr/ds
        m = np.zeros_like(r)
        m[0] = ms[0]
        m[-1] = ms[-1]
        m[1:-1] = 0.5 * (ms[:-1] + ms[1:])
    
    #From the slope, we can generate a vector which points along the tube surface
    S = vector.norm(T + m*R/r)
    
    #Remove this vector from the normals
    normals = vector.norm(normals - S*vector.dot1(S, normals)) 
    
    Ntot = np.prod(points.shape[:-1])
    p00 = np.arange(Ntot).reshape(points.shape[:-1])
    
    
    if closed:
        p10 = np.roll(p00, -1, axis=0)
    else:
        p10 = p00[1:]
        p00 = p00[:-1]
    
    if outline_closed:        
        p01 = np.roll(p00, -1, axis=1)
        p11 = np.roll(p10, -1, axis=1)
    else:
        p01 = p00[:, 1:]
        p00 = p00[:, :-1]
        p11 = p10[:, 1:]
        p10 = p10[:, :-1]
        
    triangles = np.swapaxes(np.array([[p00, p01, p10], [p01, p11, p10]]), 1, -1)
    
    return Mesh(points=points.reshape(-1, 3), normals=normals.reshape(-1, 3), triangles=triangles.reshape(-1, 3))
    #return Mesh(points=points.reshape(-1, 3), normals=S.reshape(-1, 3), triangles=triangles.reshape(-1, 3))
    
    

#Used by periodic_separate
def _attempt_merge(m1, m2, mask1=None, mask2=None, r_min=0.0):
    if mask1 is None: mask1 = ones(len(m1.points), bool)
    if mask2 is None: mask2 = ones(len(m2.points), bool)
    
    i1 = np.where(mask1)[0].reshape(-1, 1)
    i2 = np.where(mask2)[0].reshape(1, -1)
    
    r = vector.mag(m1.points[i1] - m2.points[i2])
    mr = r.min(0) #Closest distance for a point in m1 for each point in m2
    
    #No merging possible, return None
    
    if not (mr <= r_min).any(): return False 
    
    new_i2 = []
    keep_i2 = []
    ii = len(m1.points)
    i2i = 0
    
    for j2 in range(len(m2.points)):
        if mask2[j2] and mr[i2i] <= r_min:
            new_i2.append(i1[np.argmin(r[:, i2i])])
        else:
            new_i2.append(ii)
            keep_i2.append(j2)
            ii += 1
        if mask2[j2]: i2i += 1
        
    new_i2 = np.array(new_i2)
    
    new_t = np.vstack([m1.triangles, new_i2[m2.triangles]])
    new_p = np.vstack([m1.points, m2.points[keep_i2]])
    
    extras = {'attached':{}}
    
    baseprop = ['normals', 'colors']
    for prop in baseprop + m1.attached.keys():
        p1 = getattr(m1, prop, None)
        p2 = getattr(m2, prop, None)
        if p1 is not None and p2 is not None:
            p_new = np.concatenate([p1, p2[keep_i2]], axis=0)
            if prop in baseprop: extras[prop] = p_new
            else: extras['attached'][prop] = p_new
           
    return Mesh(new_p, new_t, **extras)
        
            
def all_in(k, d):
    for kk in k:
        if kk not in d: return False
    else:
        return True
    
        
def load_mesh(fn, file_type=None):
    '''Load a triangular mesh from a file, stored either as a "PLY" or "STL" file.
    
    Parameters
    ----------
    fn : string
    file_type : string, optional
        File type, either 'ply' or 'stl'.  If not specified, determined
        from the file name.    
        
    Returns
    -------
    mesh : Mesh object
        Mesh object with the data in the file.  If type is "PLY", extra vertex
        attributes will appear in the ``attached`` attribute.
    '''
    
    if file_type is None: file_type = os.path.splitext(fn)[1].strip('.')
        
    if file_type.lower() == 'ply':
        data = decode_ply(fn)
        
        try: points = np.array([data['vertex'][k] for k in ('x', 'y', 'z')]).T
        except: raise PLYError('ply file %s has missing or invalid vertex position data' % fn)
        
        try: triangles = data['face']['vertex_indices']
        except: raise PLYError('ply file %s has missing or invalid triangle data' % fn)
        
        for pl in (['red', 'green', 'blue', 'alpha'], ['red', 'green', 'blue']):
            if all_in(pl, data['vertex']):
                colors = np.array([data['vertex'][k] for k in pl]).T
                break
        else:
            colors = None
            
        pl = ('nx', 'ny', 'nz')
        if all_in(pl, data['vertex']):
            normals = np.array([data['vertex'][k] for k in pl]).T
        else:
            normals = None

        extras = {}
        for k, v in data['vertex'].iteritems():
            if k not in ('x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue', 'alpha'):
                extras[k] = v
            
        return Mesh(points=points, triangles=triangles, colors=colors, normals=normals, attached=extras)
            
        return m
            
        
    elif file_type.lower() == 'stl':
        with open(fn, 'rb') as f:
            if f.read(5).lower() == 'solid':
                raise ValueError("ASCII STL reading not implemented!")
            
            f.seek(80)
            
            num_triangles = int(np.fromfile(f, 'u4', 1)[0])

            stl_dat = np.fromfile(f, STL_DTYPE, num_triangles)
            points = np.empty((num_triangles*3, 3), 'f')
            for i, a in enumerate(['c1', 'c2', 'c3']):
                points[i::3] = stl_dat[a]
            
            triangles = np.arange(num_triangles*3, dtype='i').reshape(num_triangles, 3)

            return Mesh(points=points, triangles=triangles)
        
    else:
        raise ValueError('file type should be "ply" or "stl"')    
    
    
class PLYError(Exception): pass
    
PLY_FORMAT = re.compile(r'format\s+(\w+)\s+([0-9.]+)')
PLY_FORMAT_TYPES = {'ascii':'',
                    'binary_big_endian':'>',
                    'binary_little_endian':'<' }

PLY_ELEMENT = re.compile(r'element\s+(\w+)\s+([0-9]+)')
PLY_ELEMENT_TYPES = ('vertex', 'face', 'edge')

PLY_PROPERTY_LIST = re.compile(r'property\s+list\s+(\w+)\s+(\w+)\s+(\w+)')

PLY_PROPERTY = re.compile(r'property\s+(\w+)\s+(\w+)')
PLY_PROPERTY_TYPES = {
    'char':'i1', 'uchar':'u1',
    'short':'i2', 'ushort':'u2',
    'int':'i4', 'uint':'u4',
    'float':'f', 'double':'d'
}

PY_TYPE = lambda x: float if x in ('f', 'd') else int


NP_PLY = {}
for k, v in PLY_PROPERTY_TYPES.iteritems(): NP_PLY[np.dtype(v)] = k


def decode_ply(f, require_triangles=True):
    '''Decode a ply file, converting all saved attributes to dictionary entries.
    
    **Note:** this function usually does not need to be used directly; for
    converting ply's to meshes, use :func:`open_mesh`.  Alternatively, this
    function can be used if you require access to fields beyond the basic mesh
    geometry, colors and point normals.
    
    For more info on the PLY format, see:
        * http://www.mathworks.com/matlabcentral/fx_files/5459/1/content/ply.htm
        * http://paulbourke.net/dataformats/ply/
    
    Parameters
    ----------
    f : string or file
        A valid PLY file
    require_triangles : bool (default: True)
        If true, converts ``element_data["face"]["vertex_indices"]`` to an array
        with shape ``([number of triangles], 3)``.  Raises a PLYError if there
        are not triangular faces.
        **Note: if ``require_triangles=False``, ``element_data["face"]["vertex_indices"]``
        will be a numpy array of data type ``"object"`` which contains arrays
        of possibly variable length.**
        
    Returns
    -------
    element_data : dict
        A dictionary of all the properties in the original file.
        The main dictinoary should contain two sub-dictionaries with names
        ``"vertex"`` and ``"face"``.
        These dictionaries contain all of the named properties in the PLY file.
        Minimal entries in ``"vertex"`` are ``"x"``, ``"y"``, and ``"z"``.
        ``"face"`` should at least contain ``"vertex_indices"``.
    '''
    if type(f) is str:
        f = open(f, 'rb')
    
    line = f.readline().strip()
    
    if line != 'ply':
        raise PLYError('First line of file %s is not "ply", this file is not a PLY mesh!\n[%s]' % (repr(f), line))
    
    format_type = None
    format_char = ''
    format_version = None
    line = ''
    
    elements = []
    element_info = {}
    current_element = None
    
    while line != 'end_header':
        line = f.readline()
        
        if not line.endswith('\n'):
            raise PLYError('reached end of file (%s) before end of header; invalid file.' % repr(f))
        
        line = line.strip()
        if not line or line.startswith('comment'): continue
        
        m = PLY_FORMAT.match(line)
        if m:
            if format_type is not None:
                raise PLYError('format type in file %s specified more than once\n[%s]' % (repr(f), line))
            format_type = m.group(1)
            if format_type not in PLY_FORMAT_TYPES:
                raise PLYError('format type (%s) in file %s is not known\n[%s]' % (format_type, repr(f), line))
            format_char = PLY_FORMAT_TYPES[format_type]

            format_version = m.group(2)
            continue
        
        m = PLY_ELEMENT.match(line)
        if m:
            t = m.group(1)
            if t not in PLY_ELEMENT_TYPES:
                raise PLYError('element type (%s) in file %s is not known\n[%s]' % (t, repr(f), line))
            elif t in elements:
                raise PLYError('element type (%s) appears multiple times in file %s\n[%s]' % (t, repr(f), line))
            
            elements.append(t)
            element_info[t] = [int(m.group(2))]
            current_element = element_info[t]
            continue
        
        m = PLY_PROPERTY_LIST.match(line)
        if m:
            if current_element is None:
                raise PLYError('property without element in file %s\n[%s]' % (repr(f), line))
            
            tn = m.group(1)
            tl = m.group(2)
            
            if tn not in PLY_PROPERTY_TYPES:
                raise PLYError('property type (%s) in file %s is not known\n[%s]' % (tn, repr(f), line))
            if tl not in PLY_PROPERTY_TYPES:
                raise PLYError('property type (%s) in file %s is not known\n[%s]' % (tl, repr(f), line))
        
            current_element.append((m.group(3), (format_char + PLY_PROPERTY_TYPES[tn], format_char + PLY_PROPERTY_TYPES[tl])))
            continue
        
        m = PLY_PROPERTY.match(line)
        if m:
            if current_element is None:
                raise PLYError('property without element in file %s\n[%s]' % (repr(f), line))
            
            t = m.group(1)
            
            if t not in PLY_PROPERTY_TYPES:
                raise PLYError('property type (%s) in file %s is not known\n[%s]' % (t, repr(f), line))
            
            current_element.append((m.group(2), format_char + PLY_PROPERTY_TYPES[t]))
            continue    
        
        if line.startswith('end_header'):
            break
        else:
            raise PLYerror('invalid line in header for %s\n[%s]')
            
    element_data = {}
    fast_triangles = False
          
    for e in elements:
        et = element_info[e]
        
        ne = et.pop(0)
        #print e, et
        
        fast_decode = True
        
        dtype_list = []
        py_types = []
        
        
        if e == 'face' and len(et) == 1 and et[0][0] == 'vertex_indices' and require_triangles:
            #This mesh only has vertex indices AND we have require_triangles = True
            #Lets load the data assuming it's triangles, and check later
            dtype_list = [('nv', et[0][1][0]), ('v', '3' + et[0][1][1])]
            fast_triangles = True
            
        else:
            for name, t in et:
                if type(t) is tuple: #This element is a list.
                    fast_decode = False
                    dtype_list.append((name, 'O'))
                    py_types.append((name, map(PY_TYPE, t)))
                else:
                    dtype_list.append((name, t))
                    py_types.append((name, PY_TYPE(t)))

        dtype = np.dtype(dtype_list)

        if fast_decode:
            if format_type == 'ascii':
                s = ''.join(f.readline() for n in range(ne))
                dat = np.genfromtxt(StringIO(s), dtype=dtype)
            else:
                dat = np.fromfile(f, dtype=dtype, count=ne)
        
        else:
            
            def conv(t, x):
                try: return t(x)
                except: raise PLYError('expected %s, found "%s" (in file %s)' % (t, x, repr(f)))
            
            def get(t, count=1):
                t = np.dtype(t)
                n = t.itemsize*count
                s = f.read(n)
                if len(s) != n:
                    raise PLYEror('reached end of file %s before all data was read' % repr(f))
                return np.fromstring(s, t, count=count)
            
            dat = np.zeros(ne, dtype=dtype)
            
            for i in range(ne):
                if format_type == 'ascii':
                    parts = f.readline().split()
                    
                    for name, t in py_types:
                        if not parts:
                            raise PLYError('when decoding %s #%s, not enough items in line (in file %s)' % (e, i, repr(f)))
                        if type(t) in (list, tuple):
                            nl = conv(t[0], parts.pop(0))
                            if len(parts) >= nl:
                                dat[name][i] = map(lambda x: conv(t[1], x), parts[:nl])
                                parts = parts[nl:]
                            else:
                                raise PLYError('when decoding %s #%s, not enough items for list (in file %s)' % (e, i, repr(f)))
                        else:
                            dat[name][i] = conv(t, parts.pop(0))
                        
                else:
                    for name, t in et:
                        if type(t) is tuple:
                            nl = get(t[0])
                            dat[name][i] = get(t[1], nl)
                        else:
                            dat[name][i] = get(t)
                        
        
        if fast_triangles and e == 'face':
            if (dat['nv'] != 3).any():
                raise PLYError("require_triangles=True, but file contains non-triangular faces")
            element_data[e] = {'vertex_indices':dat['v']}
        else:
            element_data[e] = dict((name, dat[name]) for name, t in et)


    if require_triangles:
        if not fast_triangles:
            v = list(element_data["face"]["vertex_indices"])
    
            if not (np.array(map(len, v)) == 3).all():
                raise PLYError("require_triangles=True, but file contains non-triangular faces")
            
            element_data["face"]["vertex_indices"] = np.array(v)

    return element_data
         
         
#def load_ply(f):
#    dat = decode_ply(f)
#    
#    if 'face' not in dat: raise PLYError('"%s" does not include face data' % f)
#    if 'vertex' not in dat: raise PLYError('"%s" does not include vertex data' % f)
#                
#    v = dat['vertex']
#
#    for p in ('x', 'y', 'z'):
#        if p not in v: raise PLYError('vertices in "%s" missing "%s"' % (f, p))
    

def encode_ply(points, tris, normals=None, colors=None, enforce_types=True, extra_vertex_attributes=None):
    '''Convert triangular mesh data into a PLY file data.
    
    **Note:** if you wish to convert a :class:`Mesh` object to a "ply" file, use
    :meth:`Mesh.save`.
    
    Parameters
    ----------
    points : (num points, 3) array
    tris : (num triangles, 3) array
    normals : (num points, 3) array, optional
        Vertex normals.
    colors : (num points, 3 or 4) array, optional
        Vertex colors, as ``RGBA``.  Alpha channel is optional, and will not
        be saved if not included.  Type should either be ``float`` (ranging from
        0-1) or ``uchar`` (ranging from 0-255).  If type is ``float``, will
        be clipped, scaled, and converted to ``uchar`` if ``enforce_types=True``.
    enforce_types : bool, optional (default: True)
        If True, the data types of all the inputs will be converted to standard
        values (``float`` for points, tris, and normals and ``uchar`` for colors).
        Although saving data with non-standard types is supported by the PLY
        format, files may not open in standard programs (e.g. Meshlab).
    extra_vertex_attributes : dict, optional
        A dictionary of extra attributes to be added to each vertex.
        Each entry should consist of: ``[string name]:[numpy array]``
        Type will not be altered, regardless of the value for ``enforce_types``.
        If vectors need to be saved, the need to be broken out into subarrays:
        all arrays in the dictionary should have shape ``(num. vert., )``.
        
    Returns
    -------
    ply_data : string
        The PLY file data as a string.
    '''
    
    if extra_vertex_attributes is None: extra_vertex_attributes = {}
    
    #http://www.mathworks.com/matlabcentral/fx_files/5459/1/content/ply.htm
    if enforce_types:
        points = np.asarray(points, 'f')
        tris = np.asarray(tris, 'i')
        if normals is not None: normals = np.asarray(normals, 'f')
        if colors is not None:
            colors = np.asarray(colors)
            if colors.dtype in ('f', 'd'):
                colors = np.clip(colors, 0, 1) * 255
            colors = np.asarray(colors, 'u1')
            
    else:
        points = np.asarray(points)
        tris = np.asarray(tris)
    
    #8 byte point indices not supported in PLYs
    if tris.dtype == 'i8': tris = tris.astype('i4')
    if tris.dtype == 'u8': tris = tris.astype('u4')
    
    x_dt = points.dtype
    p_dt = [('x', x_dt), ('y', x_dt), ('z', x_dt)]
    N = len(points)
    
    header = [
        'ply',
        'format binary_little_endian 1.0',
        'element vertex %d' % N
    ]
    
    if normals is not None:
        normals = np.asarray(normals)
        if normals.shape != (N, 3):
            raise ValueError('shape of normals array should be ([num points], 3), found %s' % (normals.shape,))

        n_dt = normals.dtype
        p_dt += [('nx', n_dt), ('ny', n_dt), ('nz', n_dt)]
        
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape not in [(N, 3), (N, 4)]:
            raise ValueError('shape of colors array should be ([num points], [3, 4]), found %s' % normals.shape)

        c_dt = colors.dtype
        p_dt += [('red', c_dt), ('green', c_dt), ('blue', c_dt), ('alpha', c_dt)][:colors.shape[1]]
    
    eva = {}
    for k, v in extra_vertex_attributes.iteritems():
        v = np.asarray(v)
        eva[k] = v
        if v.shape != (N, ):
            raise PLYError('extra vertex attributes must have same length as number of vertices\n(attribute %s has shape %s, not %s)' % (k, v.shape, (N, )))

        p_dt.append((k, eva[k].dtype))

    #print p_dt
        
    header += ['property %s %s' % (NP_PLY[dt], prop) for prop, dt in p_dt]
    
    header += [
        'element face %s' % len(tris),
        'property list uchar %s vertex_indices' % NP_PLY[tris.dtype],
        'end_header'
    ]
    
    point_data = np.empty(N, p_dt)
    point_data['x'] = points[:, 0]
    point_data['y'] = points[:, 1]
    point_data['z'] = points[:, 2]
    
    if normals is not None:
        point_data['nx'] = normals[:, 0]
        point_data['ny'] = normals[:, 1]
        point_data['nz'] = normals[:, 2]
        
    if colors is not None:
        point_data['red']   = colors[:, 0]
        point_data['green'] = colors[:, 1]
        point_data['blue']  = colors[:, 2]
        if colors.shape[1] == 4: 
            point_data['alpha']  = colors[:, 3]

    for k, v in eva.iteritems():
        point_data[k] = v
    
    t_dt = tris.dtype
    tri_data = np.empty(len(tris), [('num_vert', 'u1'), ('v1', t_dt), ('v2', t_dt), ('v3', t_dt)])
    tri_data['num_vert'] = 3
    tri_data['v1'] = tris[:, 0]
    tri_data['v2'] = tris[:, 1]
    tri_data['v3'] = tris[:, 2]

    return '\n'.join(header) + '\n' + point_data.tostring() + tri_data.tostring()

