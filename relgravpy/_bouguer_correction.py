# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:51:25 2019

@author: lzampa
"""

# -----------------------------------------------------------------------------
# Import libraries

from . import utils as utl
from . import shp_tools as shp
from . import raster_tools as rt
from . import grav_modelling as gm

import multiprocessing as mpr
import warnings
import argparse
import functools
import platform
import matplotlib as mplt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------------------------------------------------------
# Set the aliases for some libraries from the utils and other module

np = utl.np
os = utl.os
sys = utl.sys
ogr = utl.ogr
osr = utl.osr
gdal = utl.gdal
plt = utl.plt
time = utl.time
copy = utl.copy

# -----------------------------------------------------------------------------
# Alias for the current directory (main dir.)
mdir = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Constants
G = utl.G
M = utl.M
a_wgs84 = utl.a_wgs84
c_wgs84 = utl.c_wgs84
R_wgs84 = utl.R_wgs84
J2_wgs84 = utl.J2_wgs84
w_wgs84 = utl.w_wgs84 
  

# -----------------------------------------------------------------------------
# Spherical polyhedral approximation of the outer terrain.
#
# The observation frame is local Cartesian ENU, with origin at the reference
# sphere directly below the station. Horizontal coordinates come from the
# station-centred orthographic projection; z is positive upwards.
#
# IMPORTANT: gravicalc2_full returns g_z positive UP, whereas the existing
# Nagy prism kernel returns gravitational attraction positive DOWN in mGal.
# We reverse the polyhedron sign and account for the different G constants.
_CURVED_FACES = (
    (3, 3, 2, 1), (3, 3, 1, 0),   # triangulated lower surface
    (3, 4, 5, 6), (3, 4, 6, 7),   # triangulated upper surface
    (3, 0, 1, 5), (3, 0, 5, 4), # south side
    (3, 1, 2, 6), (3, 1, 6, 5), # east side
    (3, 2, 3, 6), (3, 3, 7, 6), # north side (shared diagonal)
    (3, 3, 0, 7), (3, 0, 4, 7), # west side (shared diagonal)
)
_POLY_G_RATIO = 1.0  # compatibility with existing numerical test harness


def _curved_cell_vertices(prism, radius):
    """Convert a terrain cell to a closed, triangulated spherical polyhedron.

    Parameters
    ----------
    prism : (6,) array
        West, east, south, north, bottom, top in station-centred orthographic
        metres. Bottom/top are heights relative to the reference sphere.
    radius : float
        Radius of the local reference sphere, in metres.

    Notes
    -----
    Surface vertices lie on spheres of radius (radius + height).
    The lower and upper surfaces are *chordal* triangles, not tangent planes:
    this yields a consistent polygonal approximation of the reference sphere.
    At shared grid nodes, neighbouring cells reuse the same lower vertices.
    """
    west, east, south, north, bottom, top = np.asarray(prism, dtype=float)
    if east <= west or north <= south or top <= bottom:
        raise ValueError("Terrain polyhedron must have positive dimensions.")
    xy = np.array([
        [west, south], [east, south], [east, north], [west, north]
    ], dtype=float)
    radial_sq = radius**2 - np.sum(xy**2, axis=1)
    if np.any(radial_sq <= 0):
        raise ValueError("Orthographic cell lies outside the reference hemisphere.")
    radial_z = np.sqrt(radial_sq)

    def surface(height):
        scale = 1.0 + height / radius
        return np.column_stack((xy * scale, radial_z * scale - radius))

    return np.vstack((surface(bottom), surface(top)))


def _curved_cell_vertices_batch(prisms, radius):
    """Vectorized eight-vertex representation of spherical DEM cells."""
    prisms = np.asarray(prisms, dtype=float).reshape(-1, 6)
    if len(prisms) == 0:
        return np.empty((0, 8, 3), dtype=float)
    w, e, south, north, bottom, top = prisms.T
    xy = np.stack((
        np.stack((w, south), axis=-1),
        np.stack((e, south), axis=-1),
        np.stack((e, north), axis=-1),
        np.stack((w, north), axis=-1),
    ), axis=1)
    radicand = radius**2 - np.sum(xy**2, axis=-1)
    if np.any(radicand <= 0):
        raise ValueError("Orthographic cell lies outside reference hemisphere.")
    radial_z = np.sqrt(radicand)
    scale_b = 1 + bottom[:, None]/radius
    scale_t = 1 + top[:, None]/radius
    vb = np.concatenate((xy*scale_b[..., None],
                         (radial_z*scale_b-radius)[..., None]), axis=-1)
    vt = np.concatenate((xy*scale_t[..., None],
                         (radial_z*scale_t-radius)[..., None]), axis=-1)
    return np.concatenate((vb, vt), axis=1)


def _surface_integral_vertices_gz(vertices, densities, observation):
    """Numerically stable gz for arbitrary closed 8-vertex cells.

    Use a 7-point Dunavant rule on the outward-oriented triangular faces.
    The same triangulation is used for ordinary spherical polyhedra and
    for the transition polyhedra anchored to outer Nagy prism faces.
    """
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 8, 3)
    densities = np.asarray(densities, dtype=float).ravel()
    if len(vertices) != len(densities):
        raise ValueError("Expected one density per polyhedron.")
    if not len(vertices):
        return 0.0
    faces_indices = np.array([
        [3, 2, 1], [3, 1, 0], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 6], [3, 7, 6], [3, 0, 7], [0, 4, 7],
    ], dtype=int)
    faces = vertices[:, faces_indices, :]
    z_area = np.cross(faces[:, :, 1]-faces[:, :, 0],
                      faces[:, :, 2]-faces[:, :, 0])[:, :, 2]/2
    a = 0.059715871789770
    b = 0.470142064105115
    c = 0.797426985353087
    d = 0.101286507323456
    bary = np.array([
        [1/3, 1/3, 1/3],
        [a, b, b], [b, a, b], [b, b, a],
        [c, d, d], [d, c, d], [d, d, c],
    ])
    weights = np.array([0.225]+[0.132394152788506]*3
                       +[0.125939180544827]*3)
    samples = np.einsum("qc,ntcj->ntqj", bary, faces)
    distance = np.linalg.norm(
        samples-np.asarray(observation, dtype=float).reshape(1, 1, 1, 3),
        axis=-1)
    if np.any(distance == 0):
        raise ValueError("Observation is on a quadrature node.")
    return float(gm.G*1e5*np.sum(
        densities[:, None]*z_area*
        np.sum(weights[None, None, :]/distance, axis=-1)
    ))


def _surface_integral_polyhedral_gz(prisms, densities, observation, radius):
    """Stable far-field spherical polyhedra, down-positive in mGal."""
    vertices = _curved_cell_vertices_batch(prisms, radius)
    return _surface_integral_vertices_gz(vertices, densities, observation)


def _polyhedral_gz(prisms, densities, observation, radius):
    """Down-positive mGal from curvature-following polyhedra.

    For nearby cells reuse grav_modelling.polydron_grav's analytic kernel. For sufficiently
    distant cells use a stable face-integral kernel. The original solver can
    underestimate small distant elements because its line/angle terms cancel
    numerically; the face integral avoids that catastrophic loss of precision.
    """
    if np.asarray(prisms).size == 0:
        return 0.0
    prisms = np.atleast_2d(np.asarray(prisms, dtype=float))
    densities = np.atleast_1d(np.asarray(densities, dtype=float))
    obs = np.asarray(observation, dtype=float).ravel()
    if prisms.ndim != 2 or prisms.shape[1] != 6 or len(prisms) != len(densities):
        raise ValueError("Expected one density per (west,east,south,north,bottom,top) cell.")
    if obs.size != 3:
        raise ValueError("Expected a 3-component observation vector.")
    dx = prisms[:,1] - prisms[:,0]
    dy = prisms[:,3] - prisms[:,2]
    dz = prisms[:,5] - prisms[:,4]
    if np.any(dx <= 0) or np.any(dy <= 0) or np.any(dz <= 0):
        raise ValueError("Terrain polyhedra must have positive dimensions.")
    centroids = np.column_stack((
        (prisms[:,0] + prisms[:,1])/2,
        (prisms[:,2] + prisms[:,3])/2,
        (prisms[:,4] + prisms[:,5])/2
    ))
    distances = np.linalg.norm(centroids - obs, axis=1)
    span = np.maximum.reduce((dx, dy, dz))
    # Use the existing analytic formulation only in its tested near range.
    near = distances <= 1.75*span
    total = 0.0
    for prism, den in zip(prisms[near], densities[near]):
        if den == 0:
            continue
        vertices = _curved_cell_vertices(prism, radius)
        total += gm.polydron_grav(
            vertices, [list(face) for face in _CURVED_FACES], den
        ).compute_gz(obs, method="analytic")
    if np.any(~near):
        total += _surface_integral_polyhedral_gz(
            prisms[~near], densities[~near], obs, radius)
    return float(total)


# =============================================================================
# R1 PRISM / SPHERICAL-POLYHEDRON CONFORMAL TRANSITION
# -----------------------------------------------------------------------------
# Single coarse-cell transition mesh, including special R1 corner cells.
# Functions are private to Bouguer correction. Inner faces share Nagy
# prism vertices; outer faces share the unchanged far polyhedra triangles.
# =============================================================================

def _outward_triangle_winding(vertices, triangles):
    """Make a closed triangular surface consistently oriented, not centre-fanned.

    The centroid/normal test is *incorrect* on the re-entrant step faces
    introduced by independently centre-shifted adjacent Nagy prisms.
    """
    from collections import defaultdict, deque
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=int).copy()
    incidence = defaultdict(list)
    for i, (a, b, c) in enumerate(triangles):
        for u, v in ((a, b), (b, c), (c, a)):
            incidence[(min(u, v), max(u, v))].append((i, u, v))
    if not incidence or any(len(faces) != 2 for faces in incidence.values()):
        raise ValueError("R1 transition polyhedron has open/non-manifold edges.")
    adjacent = [[] for _ in triangles]
    for shared in incidence.values():
        (i, a, b), (j, c, d) = shared
        relative_flip = (a == c and b == d)
        adjacent[i].append((j, relative_flip))
        adjacent[j].append((i, relative_flip))
    flipped = {0: False}
    queue = deque((0,))
    while queue:
        i = queue.popleft()
        for j, different in adjacent[i]:
            expected = bool(flipped[i] ^ different)
            if j in flipped:
                if flipped[j] != expected:
                    raise ValueError("R1 transition face winding is non-orientable.")
            else:
                flipped[j] = expected
                queue.append(j)
    if len(flipped) != len(triangles):
        raise ValueError("R1 transition contains disconnected surfaces.")
    for i, reverse in flipped.items():
        if reverse:
            triangles[i] = triangles[i, ::-1]
    # Signed oriented volume is translation-invariant for a closed surface.
    centre = vertices.mean(axis=0)
    p = vertices[triangles] - centre
    volume = float(np.einsum(
        "ni,ni->n", p[:, 0], np.cross(p[:, 1], p[:, 2])
    ).sum() / 6.0)
    if volume < 0:
        triangles = triangles[:, ::-1]
        volume = -volume
    if volume <= 0:
        raise ValueError("R1 transition polyhedron has non-positive volume.")
    return triangles, volume


def _bridge_mesh(cell, near_prisms, side, radius, tol=1e-6):
    """Return (vertices, outward triangular faces) for one far parent cell.

    The entire near-facing side is a mosaic of actual Nagy prism outer
    rectangles. All its near grid knots and stepped top/bottom heights
    become explicit edges. The outer face has only the four original far
    spherical vertices, shared exactly with the next ordinary far cell.
    """
    w, e, s, n, bottom, top = np.asarray(cell, dtype=float)
    near = np.asarray(near_prisms, dtype=float).reshape(-1, 6)
    if side == "east":
        u0, u1, v0, v1 = w, e, s, n
        transform = lambda u, v, z: (u, v, z)
        matching = abs(near[:, 1] - w) < tol
        starts, stops = near[:, 2], near[:, 3]
    elif side == "west":
        u0, u1, v0, v1 = -e, -w, s, n
        transform = lambda u, v, z: (-u, v, z)
        matching = abs(near[:, 0] - e) < tol
        starts, stops = near[:, 2], near[:, 3]
    elif side == "north":
        u0, u1, v0, v1 = s, n, w, e
        transform = lambda u, v, z: (v, u, z)
        matching = abs(near[:, 3] - s) < tol
        starts, stops = near[:, 0], near[:, 1]
    elif side == "south":
        u0, u1, v0, v1 = -n, -s, w, e
        transform = lambda u, v, z: (v, -u, z)
        matching = abs(near[:, 2] - n) < tol
        starts, stops = near[:, 0], near[:, 1]
    else:
        raise ValueError("Invalid transition side.")
    if u1 <= u0 or v1 <= v0 or top <= bottom:
        raise ValueError("Bridge parent cell has invalid bounds.")
    segments = []
    for p, a, b, include in zip(near, starts, stops, matching):
        if not include:
            continue
        a, b = max(float(a), v0), min(float(b), v1)
        if b - a > tol:
            if p[5] <= p[4]:
                raise ValueError("Nagy prism height must be positive.")
            segments.append((a, b, float(p[4]), float(p[5])))
    segments.sort()
    if not segments or abs(segments[0][0] - v0) > tol or abs(
            segments[-1][1] - v1) > tol:
        raise ValueError("R1 bridge has incomplete Nagy prism face coverage.")
    for left, right in zip(segments[:-1], segments[1:]):
        if abs(left[1] - right[0]) > tol:
            raise ValueError("R1 bridge has overlapping or missing near faces.")
        if min(left[3], right[3]) <= max(left[2], right[2]):
            raise ValueError(
                "R1 bridge near-prism height intervals are disjoint "
                "within one material; cannot construct a valid connected "
                "single polyhedron.")

    xyz, lookup, faces = [], {}, []
    def add_xyz(p):
        point = tuple(float(q) for q in p)
        key = tuple(round(q, 9) for q in point)
        if key not in lookup:
            lookup[key] = len(xyz)
            xyz.append(point)
        return lookup[key]

    def inner(u, v, z):
        return add_xyz(transform(u, v, z))

    def outer(v, height):
        x, y, _ = transform(u1, v, 0.)
        scale = 1. + height/radius
        z = np.sqrt(radius**2 - x*x - y*y)*scale - radius
        return add_xyz((x*scale, y*scale, z))

    def tri(a, b, c):
        if len({a, b, c}) != 3:
            return
        p, q, r = np.asarray([xyz[a], xyz[b], xyz[c]])
        if np.linalg.norm(np.cross(q - p, r - p)) <= 1e-13:
            return
        faces.append((a, b, c))

    # Four untouched far-grid vertices, with an identical face diagonal to
    # the neighbouring ordinary spherical polyhedron.
    out_b0, out_b1 = outer(v0, bottom), outer(v1, bottom)
    out_t0, out_t1 = outer(v0, top), outer(v1, top)
    in_b0 = inner(u0, v0, segments[0][2])
    in_t0 = inner(u0, v0, segments[0][3])
    in_b1 = inner(u0, v1, segments[-1][2])
    in_t1 = inner(u0, v1, segments[-1][3])
    tri(out_b0, out_b1, out_t1)
    tri(out_b0, out_t1, out_t0)

    # Inner patch: each actual Nagy prism exterior rectangle. Split the
    # edges at neighbouring step-height endpoints to obtain a conformal
    # triangulation when adjacent prisms have different z-shifts/heights.
    for j, (a, b, zb, zt) in enumerate(segments):
        centre = inner(u0, (a+b)/2, (zb+zt)/2)
        start_cuts = {zb, zt}
        end_cuts = {zb, zt}
        if j:
            start_cuts.update(z for z in segments[j-1][2:]
                              if zb < z < zt)
        if j+1 < len(segments):
            end_cuts.update(z for z in segments[j+1][2:]
                            if zb < z < zt)
        start_cuts = sorted(start_cuts)
        end_cuts = sorted(end_cuts)
        boundary = [inner(u0, a, zb), inner(u0, b, zb)]
        boundary += [inner(u0, b, z) for z in end_cuts[1:]]
        boundary += [inner(u0, a, zt)]
        boundary += [inner(u0, a, z) for z in start_cuts[-2::-1]]
        boundary = [vertex for i, vertex in enumerate(boundary)
                    if i == 0 or vertex != boundary[i-1]]
        if boundary[0] == boundary[-1]:
            boundary.pop()
        for p, q in zip(boundary, boundary[1:] + boundary[:1]):
            tri(centre, p, q)

    # Upper and lower surfaces: triangle fans that interpolate every
    # staircase segment to the original coarse-grid outside edge. The
    # zero-width XY triangles at height steps are genuine vertical
    # sections of the closed 3D surface.
    upper, lower = [], []
    for a, b, zb, zt in segments:
        upper.extend((inner(u0, a, zt), inner(u0, b, zt)))
        lower.extend((inner(u0, a, zb), inner(u0, b, zb)))
    for chain, first, last in ((upper, out_t0, out_t1),
                               (lower, out_b0, out_b1)):
        chain = [v for i, v in enumerate(chain)
                 if i == 0 or v != chain[i-1]]
        for p, q in zip(chain[:-1], chain[1:]):
            tri(first, p, q)
        tri(first, chain[-1], last)

    # Two tangential ends, same diagonal on either side for a matching
    # facet with the neighbouring coarse cell/corner.
    for inner_bottom, inner_top, outer_bottom, outer_top in (
        (in_b0, in_t0, out_b0, out_t0),
        (in_b1, in_t1, out_b1, out_t1),
    ):
        tri(inner_bottom, outer_bottom, outer_top)
        tri(inner_bottom, outer_top, inner_top)

    vertices = np.asarray(xyz, dtype=float)
    oriented, volume = _outward_triangle_winding(vertices, faces)
    return vertices, [[3, *map(int, face)] for face in oriented]


def _corner_mesh(cell, near_prisms, radius, R1, tol=1e-6):
    """Anchor the one inner diagonal vertex of a first-ring corner cell.

    Matching the centre-shifted corner near prism at (±R1, ±R1) removes
    the otherwise overlooked dangling edge at the four corners.
    """
    w, e, s, n, bottom, top = np.asarray(cell, dtype=float)
    near = np.asarray(near_prisms, dtype=float).reshape(-1, 6)
    x = R1 if w >= R1-tol else -R1
    y = R1 if s >= R1-tol else -R1
    select = (
        (abs(near[:,1]-x) < tol if x > 0 else abs(near[:,0]-x) < tol)
        & (abs(near[:,3]-y) < tol if y > 0 else abs(near[:,2]-y) < tol)
    )
    matched = near[select]
    if len(matched) != 1:
        raise ValueError("R1 corner must match precisely one Nagy prism.")
    prism = matched[0]
    xy = np.array([[w,s],[e,s],[e,n],[w,n]],float)
    rr = radius**2-np.sum(xy**2,axis=1)
    if np.any(rr <= 0):
        raise ValueError("Corner cell lies beyond sphere.")
    root = np.sqrt(rr)
    def shell(height):
        scale = 1.+height/radius
        return np.column_stack((xy*scale,root*scale-radius))
    vertices = np.vstack((shell(bottom),shell(top)))
    idx = int(np.flatnonzero(
        (abs(xy[:,0]-x)<tol)&(abs(xy[:,1]-y)<tol))[0])
    vertices[idx]=(x,y,prism[4])
    vertices[idx+4]=(x,y,prism[5])
    indices = [
        (3,2,1),(3,1,0),(4,5,6),(4,6,7),
        (0,1,5),(0,5,4),(1,2,6),(1,6,5),
        (2,3,6),(3,7,6),(3,0,7),(0,4,7),
    ]
    # At R1's corners the two adjoining SIDE bridges always use a
    # diagonal from their inner-bottom vertex to the corresponding
    # outer-top vertex. Adjust the corner's shared diagonals in the
    # other three quadrants to match exactly. Do not modify its outward
    # coarse faces: they must retain the standard far-grid diagonal.
    if idx == 1:       # NW corner: inner vertex is bottom SE
        indices[4:6] = [(0,1,4),(1,5,4)]      # south
    elif idx == 3:     # SE corner: inner vertex is bottom NW
        indices[10:12] = [(3,0,4),(3,4,7)]   # west
    elif idx == 2:     # SW corner: inner vertex is bottom NE
        indices[6:8] = [(1,2,5),(2,6,5)]     # east
        indices[8:10] = [(2,3,7),(2,7,6)]    # north
    outward, volume = _outward_triangle_winding(vertices,indices)
    return vertices, [[3,*map(int,f)] for f in outward]


def _transition_polyhedral_gz(far_cells, far_density, near_prisms,
                              near_density, observation, radius, R1):
    """Sum coarse far cells and R1 bridge polyhedra without reclassifying.

    Land/sea assignment and density come exclusively from the existing
    GDAL-derived meshes. Join fine-to-coarse geometry only where adjacent
    near prisms represent the SAME density as the original far cell.
    Where land/sea classification changes, keep the original far
    polyhedron completely unchanged: do not split coastal cells, invent
    mixed-density volumes, or force non-manifold transition facets.
    Material boundaries retain their pre-existing representation; no
    diagnostic counts or coastline warnings are generated here.
    """
    far = np.asarray(far_cells,float).reshape(-1,6)
    fd = np.asarray(far_density,float).ravel()
    near = np.asarray(near_prisms,float).reshape(-1,6)
    nd = np.asarray(near_density,float).ravel()
    if len(far)!=len(fd) or len(near)!=len(nd):
        raise ValueError("Density array and cell count mismatch.")
    tol=max(1e-6,abs(R1)*1e-10)
    total=0.
    ordinary=[]
    ordinary_density=[]
    for cell,density in zip(far,fd):
        w,e,s,n=cell[:4]
        mx,my=(w+e)/2,(s+n)/2
        side=None
        if abs(w-R1)<tol and -R1<my<R1:side="east"
        elif abs(e+R1)<tol and -R1<my<R1:side="west"
        elif abs(s-R1)<tol and -R1<mx<R1:side="north"
        elif abs(n+R1)<tol and -R1<mx<R1:side="south"
        corner=(abs(w-R1)<tol or abs(e+R1)<tol) and (
            abs(s-R1)<tol or abs(n+R1)<tol)
        if side or corner:
            if side:
                if side in ("east","west"):
                    a,b=s,n
                    on_edge=(abs(near[:,1]-R1)<tol if side=="east"
                             else abs(near[:,0]+R1)<tol)
                    lo,hi=near[:,2],near[:,3]
                else:
                    a,b=w,e
                    on_edge=(abs(near[:,3]-R1)<tol if side=="north"
                             else abs(near[:,2]+R1)<tol)
                    lo,hi=near[:,0],near[:,1]
                overlapping=on_edge & (lo<b-tol) & (hi>a+tol)
                # Crossing a GDAL land/sea boundary is NOT a reason to
                # change the density or split the original coarse cell.
                # The original far cell is retained and integrated by
                # the standard polyhedron solver instead of forcing a
                # bridge that would smear one material into the other.
                if (not np.any(overlapping) or
                        not np.all(np.isclose(
                            nd[overlapping],density,rtol=0,atol=1e-8))):
                    ordinary.append(cell)
                    ordinary_density.append(density)
                    continue
                vertices,facets=_bridge_mesh(
                    cell,near[overlapping],side,radius,tol)
            else:
                # A corner uses the near rectangle only for geometry.
                # Its material is whatever GDAL assigned to the far cell.
                # Multiple vertically stacked near compartments remain a
                # separate geometric case.
                x=R1 if w>=R1-tol else -R1
                y=R1 if s>=R1-tol else -R1
                select=(
                    (abs(near[:,1]-x)<tol if x>0 else abs(near[:,0]-x)<tol)
                    & (abs(near[:,3]-y)<tol if y>0 else abs(near[:,2]-y)<tol))
                if (len(near[select])!=1 or
                        not np.isclose(
                            nd[select][0],density,rtol=0,atol=1e-8)):
                    ordinary.append(cell)
                    ordinary_density.append(density)
                    continue
                vertices,facets=_corner_mesh(cell,near[select],radius,R1,tol)
            total+=gm.polydron_grav(
                vertices,facets,density).compute_gz(
                    observation,method="surface")
        else:
            ordinary.append(cell)
            ordinary_density.append(density)
    return total, np.asarray(ordinary,float).reshape(-1,6), np.asarray(
        ordinary_density,float)


def _point_mass_gz(points, masses, observation, radius):
    """Down-positive mGal from masses in orthographic x,y and radial height."""
    points = np.asarray(points, dtype=float).reshape(3, -1)
    masses = np.asarray(masses, dtype=float).ravel()
    if len(masses) != points.shape[1]:
        raise ValueError("Points and masses must have the same length.")
    if masses.size == 0:
        return 0.0
    x, y, height = points
    radial_sq = radius**2 - x*x - y*y
    if np.any(radial_sq <= 0):
        raise ValueError("Point lies outside the reference hemisphere.")
    scale = 1.0 + height / radius
    sources = np.column_stack((x*scale, y*scale, np.sqrt(radial_sq)*scale-radius))
    difference = np.asarray(observation, dtype=float).reshape(1, 3) - sources
    distance = np.linalg.norm(difference, axis=1)
    if np.any(distance == 0):
        raise ValueError("Observation coincides with a point mass.")
    return float(gm.G*1e5 * np.sum(masses * difference[:, 2] / distance**3))


def _ring_polyhedral_effect(ring_bounds, densities, radius, height):
    """Approximate the legacy ring-estimation boxes by spherical polyhedra.

    Bounds use (west,east,south,north,r_bottom,r_top) with angles in degrees.
    Longitude/latitude are centred at the equator in the empirical ring tests.
    """
    cells = np.asarray(ring_bounds, dtype=float).copy()
    cells[:, :4] = np.deg2rad(cells[:, :4]) * radius
    cells[:, 4:] -= radius
    return _polyhedral_gz(cells, densities, [0.0, 0.0, height], radius)


# =============================================================================
# LIGHTWEIGHT SURVEY-WIDE BOUGUER MESH OPTIMIZER
# -----------------------------------------------------------------------------
# This is a fast, morphology-based ERROR ESTIMATOR, not a numerical proof of
# convergence to the true Bouguer correction. It examines EVERY station, but
# only samples a tiny local raster window and tests a small candidate grid.
# No full gravity forward calculations or 167-km rasters are generated.
# =============================================================================

def _bgc_terrain_features(patches, fine_spacing, broad_spacing):
    """Extract (slope, roughness, relief, mean_abs_height) from DTM windows.

    patches : pair of equally-sized 2D arrays, fine and broader scales.
    Spacings are in metres per output pixel. Output is a conservative,
    dimensional predictor for tuning, *not* a certified gravity error.
    """
    fine, broad = [np.asarray(a, dtype=float) for a in patches]
    if (fine.ndim != 2 or broad.ndim != 2 or
            min(fine.shape) < 3 or min(broad.shape) < 3):
        raise ValueError("DTM terrain previews must be at least 3x3 arrays.")
    if fine_spacing <= 0 or broad_spacing <= 0:
        raise ValueError("DTM preview spacings must be positive.")
    def valid(a):
        if np.count_nonzero(np.isfinite(a)) < 9:
            raise ValueError("Insufficient valid topographic preview pixels.")
        return np.where(np.isfinite(a), a, np.nanmedian(a))
    fine, broad = valid(fine), valid(broad)
    dx = np.abs(np.diff(fine, axis=1)) / fine_spacing
    dy = np.abs(np.diff(fine, axis=0)) / fine_spacing
    dx_b = np.abs(np.diff(broad, axis=1)) / broad_spacing
    dy_b = np.abs(np.diff(broad, axis=0)) / broad_spacing
    slope = max(float(np.percentile(dx, 90)),
                float(np.percentile(dy, 90)),
                float(np.percentile(dx_b, 90)),
                float(np.percentile(dy_b, 90)))
    detail = fine[1:-1, 1:-1] - 0.25*(
        fine[1:-1, :-2] + fine[1:-1, 2:] +
        fine[:-2, 1:-1] + fine[2:, 1:-1])
    roughness = float(np.sqrt(np.mean(detail**2)))
    relief = float(np.percentile(broad, 95)-np.percentile(broad, 5))
    thickness = float(np.mean(np.abs(broad)))
    return np.array([slope, roughness, relief, thickness])


def _bgc_station_terrain_features(dtm, x, y, prjcode_in,
                                  prjcode_m=None):
    """Sample two small GDAL raster windows for EVERY input station.

    A native-resolution 33x33 window detects short-scale roughness;
    a 256-pixel window reduced to 25x25 captures broad relief. Raster
    reads are bounded in size, independent of R2 and candidate count.
    Geographic DEM pixel sizes are measured after coordinate projection.
    """
    xy = np.column_stack((np.asarray(x, float).ravel(),
                          np.asarray(y, float).ravel()))
    if not len(xy) or not np.all(np.isfinite(xy)):
        raise ValueError("At least one finite station coordinate is required.")
    gt = dtm.GetGeoTransform()
    prj = dtm.GetProjection()
    if not prj:
        raise ValueError("DTM must have a defined coordinate reference system.")
    if prjcode_m is None:
        cx, cy = np.mean(xy, axis=0)
        gx, gy = utl.prjxy(prjcode_in, 4326, cx, cy)
        prjcode_m = (
            f"+proj=aeqd +lat_0={gy} +lon_0={gx} "
            "+datum=WGS84 +units=m")
    src_x, src_y = utl.prjxy(prjcode_in, prj, xy[:, 0], xy[:, 1])
    src_x, src_y = np.asarray(src_x), np.asarray(src_y)
    # Metres per source raster pixel in the survey's metric projection.
    a0, b0 = utl.prjxy(prj, prjcode_m, src_x, src_y)
    a1, b1 = utl.prjxy(prj, prjcode_m,
                        src_x+gt[1], src_y+gt[4])
    a2, b2 = utl.prjxy(prj, prjcode_m,
                        src_x+gt[2], src_y+gt[5])
    resolution = np.maximum(
        np.hypot(np.asarray(a1)-a0, np.asarray(b1)-b0),
        np.hypot(np.asarray(a2)-a0, np.asarray(b2)-b0))
    linear = np.array([[gt[1], gt[2]], [gt[4], gt[5]]], dtype=float)
    pixels = np.linalg.solve(
        linear, np.vstack((src_x-gt[0], src_y-gt[3]))).T
    band = dtm.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    results = []
    degraded = 0
    for (px, py), pitch in zip(pixels, resolution):
        if not np.isfinite(pitch) or pitch <= 0:
            raise ValueError("Invalid projected raster pixel dimensions.")
        windows = []
        for radius_pixels, output_pixels in ((16, 33), (128, 25)):
            centre_x, centre_y = int(np.floor(px)), int(np.floor(py))
            x0 = max(0, centre_x-radius_pixels)
            y0 = max(0, centre_y-radius_pixels)
            x1 = min(dtm.RasterXSize, centre_x+radius_pixels+1)
            y1 = min(dtm.RasterYSize, centre_y+radius_pixels+1)
            if x1-x0 < 3 or y1-y0 < 3:
                windows = []
                break
            arr = band.ReadAsArray(
                xoff=x0, yoff=y0, win_xsize=x1-x0,
                win_ysize=y1-y0, buf_xsize=output_pixels,
                buf_ysize=output_pixels)
            if arr is None:
                windows = []
                break
            arr = np.asarray(arr, float)
            if nodata is not None:
                arr[np.isclose(arr, nodata)] = np.nan
            arr[~np.isfinite(arr)] = np.nan
            actual_pitch = pitch * max(
                (x1-x0)/max(1, output_pixels-1),
                (y1-y0)/max(1, output_pixels-1))
            windows.append((arr, actual_pitch))
        if len(windows) == 2:
            try:
                results.append(_bgc_terrain_features(
                    (windows[0][0], windows[1][0]),
                    windows[0][1], windows[1][1]))
                continue
            except ValueError:
                pass
        # A missing preview must never be interpreted as flat terrain.
        # Assign a deliberately rugged fallback so the optimizer moves
        # towards fine meshes and clearly warns the user.
        degraded += 1
        results.append([1.5, 25.0, 1500.0, 1000.0])
    if degraded:
        warnings.warn(
            f"{degraded} station(s) lack valid DTM previews; "
            "conservative rugged-terrain proxies were used. "
            "The target error is NOT certified.", RuntimeWarning,
            stacklevel=2)
    return np.asarray(results, float), float(np.min(resolution))


def _bgc_mesh_error_proxy(features, gs1, R1, gs2, density, radius,
                          native_resolution=None):
    """Estimate error (mGal) for EACH station with cheap terrain features.

    Terms model fine-grid smoothing and far-field quadrature / relief
    attenuation. Coefficients are heuristic; independent DEM validation
    is required before interpreting them as error bounds.
    """
    features = np.asarray(features, float)
    if features.ndim != 2 or features.shape[1] != 4:
        raise ValueError("Features must have four columns.")
    slope, roughness, relief, thickness = features.T
    k = 2*np.pi*gm.G*abs(density)*1e5
    near = k*(0.004*slope*gs1
              + 0.010*roughness
              + 0.005*thickness*(gs1/max(R1,1.))**2)
    far = k*0.015*(thickness+0.20*relief)*(
        gs2/max(R1,gs2,1.))**2
    curvature = k*0.002*thickness*(R1/radius)**2
    # High relief aliased by a near-grid coarser than the source DEM is
    # missed by the linear smoothing estimate. Full synthetic forward
    # gravity revealed a factor-of-three UNDERestimate at 2x native
    # resolution: gs1=60 m gave 0.126 mGal actual error while the
    # previous proxy predicted 0.038 mGal. Penalize strongly nonlinear
    # aliasing ONLY when gs1 exceeds native DEM resolution.
    #
    # This is a provisional, empirical safeguard, NOT a certified bound.
    alias = 0.
    if native_resolution is not None:
        native = float(native_resolution)
        if not np.isfinite(native) or native <= 0:
            raise ValueError("native_resolution must be positive.")
        excess = max(gs1/native-1., 0.)
        alias = k*0.06*np.maximum(slope-0.25, 0.)*native*excess**2
    # Small uncertainty margin calibrated against controlled synthetic
    # mountain/coastal forward comparisons. This does NOT make the
    # predictor an analytical upper bound.
    return 1.5*(near+far+curvature+alias)


def optimize_bgc_mesh(x, y, dtm1, gs1=None, R1=None, gs2=None,
                      R2=167000., target_error=0.05, herr=5.,
                      dc=2670., dw=1030., R=R_wgs84, R3=50000.,
                      prjcode_in=4326, prjcode_m=None,
                      terrain_features=None, raster_resolution=None):
    """Cheap one-time mesh selection for all stations in a campaign.

    Only gs1, R1, gs2 set to None are searched; all four returned mesh
    parameters are integer metres. User-supplied parameters must be
    positive integers; explicit gs1 must be EVEN for a centred grid.
    adj_mesh can increase radii slightly and adjust gs2 by <=5%. The nominal R2 DEFAULT is
    167 km, never optimized; adj_mesh can raise it slightly to an aligned
    grid boundary.

    target_error is the MAXIMUM allowed *estimated* error in mGal
    across ALL input stations, never an average. This is a lightweight
    morphology-based estimator, not a rigorous error certificate.
    No full gravity modelling is done during tuning.
    If no candidate is estimated to meet target_error, a RuntimeWarning
    is issued and the least-error candidate is returned nonetheless.

    Returns a dictionary including gs1, R1, gs2, R2, estimated_error_mgal,
    meets_target, estimated_cost, critical_station_index, and candidates.
    """
    xs, ys = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if len(xs) != len(ys) or not len(xs):
        raise ValueError("x and y must have the same nonzero station count.")
    auto = (gs1 is None or R1 is None or gs2 is None)
    # Manual values are exact metre counts; NEVER silently round user input.
    fixed = {"gs1": gs1, "R1": R1, "gs2": gs2,
             "R2": 167000 if R2 is None else R2}
    for name, value in fixed.items():
        if value is None:
            continue
        v = float(value)
        if not np.isfinite(v) or v <= 0 or abs(v-round(v)) > 1e-8:
            raise ValueError(f"{name} must be a positive integer number of metres.")
        fixed[name] = int(round(v))
    if gs1 is not None and fixed["gs1"] % 2:
        raise ValueError(
            "gs1 must be EVEN for an integer R1 and odd-centred mesh.")
    gs1, R1, gs2, R2 = (
        fixed[name] for name in ("gs1", "R1", "gs2", "R2"))
    if not auto:
        return {"gs1": gs1, "R1": R1, "gs2": gs2,
                "R2": R2, "optimized": False,
                "estimated_error_mgal": None, "meets_target": None,
                "candidates": 0}
    if not np.isfinite(target_error) or target_error <= 0:
        raise ValueError("target_error must be positive and finite (mGal).")
    if terrain_features is None:
        if dtm1 is None:
            raise ValueError("A DTM or terrain_features is required.")
        features, native = _bgc_station_terrain_features(
            dtm1, xs, ys, prjcode_in, prjcode_m)
    else:
        features = np.asarray(terrain_features, float)
        native = (30. if raster_resolution is None else
                  float(raster_resolution))
    if (features.shape != (len(xs), 4) or
            np.any(~np.isfinite(features)) or np.any(features < 0)):
        raise ValueError("Need four finite nonnegative features per station.")
    if not np.isfinite(native) or native <= 0:
        raise ValueError("raster_resolution must be finite and positive.")
    # Small bounded Cartesian search; cheap scalar operations only. R1
    # candidates are informed by the geometric sagitta estimate (herr).
    # Even cell widths are mathematically necessary: (odd * step)/2
    # must itself be an integer radius.
    ceil_even = lambda v: 2*int(np.ceil(float(v)/2.))
    g1_base = ceil_even(max(5., float(native)))
    g1_values = ([g1_base*f for f in (1, 2, 4, 8)]
                 if gs1 is None else [gs1])
    r_base = max(500., np.sqrt(2.*R*max(float(herr), 0.1)))
    r1_values = ([int(np.ceil(max(4*g1_base, r_base*f)))
                  for f in (0.25, 0.5, 1., 1.5)]
                 if R1 is None else [R1])
    g2_values = ([ceil_even(max(2*g1_base, g)) for g in
                  (250, 500, 1000, 2000, 4000)]
                 if gs2 is None else [gs2])
    # A conservative density factor for terrestrial and marine contrasts.
    rho = max(abs(float(dc)), abs(float(dw-float(dc))),
              abs(float(dw)))
    scores = []
    for near_step in sorted(set(g1_values)):
        for radius1 in sorted(set(r1_values)):
            for far_step in sorted(set(g2_values)):
                if far_step < near_step:
                    continue
                try:
                    a, b, c, d = adj_mesh(
                        near_step, radius1, far_step, R2,
                        max_R1_increase=(
                            max(4*near_step, 2*far_step)
                            if R1 is None else None))
                except ValueError:
                    continue
                if R3 is not None and float(R3) <= b+c:
                    continue
                errors = _bgc_mesh_error_proxy(
                    features,a,b,c,rho,R,native_resolution=native)
                worst = float(np.max(errors))
                # Proxy wall-clock cost: far-polyhedron work is batched,
                # near Nagy prisms and seam facets cost more per element.
                far_limit = min(d,float(R3)) if R3 is not None else d
                near_count = 4.*(b/a)**2
                far_count = 4.*max(0.,far_limit**2-b**2)/c**2
                seam_count = 4.*b/a
                work = near_count + 0.35*far_count + 2.*seam_count
                scores.append((worst,work,a,b,c,d,int(np.argmax(errors))))
    if not scores:
        # This is a geometry incompatibility, NOT a target error failure:
        # no valid parameters can be passed safely to the modelling code.
        raise ValueError("No geometrically compatible mesh candidates.")
    acceptable = [item for item in scores if item[0] <= target_error]
    if acceptable:
        # Cheapest admissible result; tie-break by smaller error.
        best = min(acceptable, key=lambda p: (p[1],p[0]))
    else:
        best = min(scores, key=lambda p: (p[0],p[1]))
        warnings.warn(
            "No mesh candidate reached the requested MAXIMUM estimated "
            f"error {target_error:.6g} mGal across all stations; "
            f"using the least-error candidate ({best[0]:.6g} mGal). "
            "These are morphology-based estimates, NOT certified bounds.",
            RuntimeWarning, stacklevel=2)
    worst,work,a,b,c,d,critical = best
    return {"gs1":a, "R1":b, "gs2":c, "R2":d,
            "estimated_error_mgal":worst, "meets_target":bool(acceptable),
            "estimated_cost":work, "critical_station_index":critical,
            "candidates":len(scores), "optimized":True,
            "error_type":"morphology_proxy_not_certified"}

# -----------------------------------------------------------------------------
def Radi1( herr=5, R=R_wgs84 ): 
    
    R1 = np.sqrt( herr * 2 * R ) 
    
    return R1

# -----------------------------------------------------------------------------
def GrdStep2( R1, h, errg, gs=25, R=R_wgs84, dc=2670, 
              plot=False, xmax=None, ymax=None, fs=14 ):  
    """
    Empirical estimate of far-field gridstep. 
    Computation starts from a distance 
    R1 and calculates the grav. effect of a squared 3D ring.
    h = constant heigh of the ring
    gs = incremental step to extend the ring outward.
    errg = accuracy of gravity data.
    The ring estimate uses spherical polyhedra.
    """  
             
    R1d = utl.m2deg(R1)
    gd = utl.m2deg(gs)
    rd = np.arange(gd, 3*R1d, gd)
    bottom, top = R, R+h
    t_st = (0, 0, top)
    den = [dc,dc,dc,dc]  
    r_eff = np.zeros(np.size(rd))
    d_eff = np.zeros(np.size(rd))
    for i, ri in enumerate(rd):
        e1 = [[-R1d, R1d, R1d, R1d+ri, bottom, top]]
        e2 = [[R1d, R1d+ri, -R1d-ri, R1d+ri, bottom, top]]
        e3 = [[-R1d, R1d, -R1d-ri, -R1d, bottom, top]]
        e4 = [[-R1d-ri, -R1d, -R1d-ri, R1d+ri, bottom, top]]
        e = e1+e2+e3+e4
        r_eff[i] = _ring_polyhedral_effect(e, den, R, h)
        n=i-1
        if i==0: n=0     
        d_eff[i] = r_eff[i]-r_eff[n]
    
    r = utl.deg2m(rd)    
    gs2d=np.interp(errg, r_eff, rd)
    gs2=np.interp(errg, r_eff, r)
    
    # Plot --------------------------------------------------------------------
    if plot==True:
        rstr = f'''gs2 = {np.round(gs2/1e3, 2)} km 
        g = {np.round(errg, 4)} mGal
        h = {np.round(h,0)} m'''
        plt.plot(np.array(r)/1e3,r_eff)
        plt.xlabel('gs2 [km]',fontsize=fs); plt.ylabel('g [mGal]',fontsize=fs)
        plt.vlines(gs2/1e3, 0, errg, linestyle="dashed")
        plt.hlines(errg, 0, gs2/1e3, linestyle="dashed")
        plt.scatter(gs2/1e3, errg, zorder=2);plt.xlim(left=0);plt.ylim(bottom=0)
        box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        if xmax== None: xmax=np.mean(r)/1e3
        if ymax== None: ymax=np.mean(r_eff)
        plt.xlim(0,xmax); plt.ylim(0,ymax);         
        plt.text(0.05, 0.95, rstr, fontsize=fs, transform=plt.gca().transAxes,
                 ha='left', va='top', linespacing=2, bbox=box_props)
        plt.tick_params(labelsize=fs)     

    return (gs2,gs2d), r_eff, d_eff  

# -----------------------------------------------------------------------------
def Radi2(R1, h, errg, gs=100, Rmax=166735, R=R_wgs84, dc=2670, 
          plot=False, xmax=None, ymax=None, font_size=14):
    """
    Empirical estimate of the far-field R2 radius, starting from a distance 
    Rmax and calculating the grav. effect of a squared 3D-ring
    h = constant heigh of the ring
    gs = incremental step to extend the ring inward.
    errg = accuracy of gravity data.
    The ring estimate uses spherical polyhedra.
    """  
       
    Rmaxd = utl.m2deg(Rmax)
    gsd = utl.m2deg(gs)
    R1d = utl.m2deg(R1)
    rd = np.arange(R1d, Rmaxd-gsd, gsd)
    bottom, top = R, R+h
    t_st = (0, 0, top)
    den = [dc,dc,dc,dc]  
    r_eff = np.zeros(np.size(rd))
    d_eff = np.zeros(np.size(rd))
    
    for i, ri in enumerate(rd):
        e1 = [[ -ri, ri, ri, Rmaxd, bottom, top ]]
        e2 = [[ ri, Rmaxd, -Rmaxd, Rmaxd, bottom, top ]]
        e3 = [[ -ri, +ri, -Rmaxd, -ri, bottom, top ]]
        e4 = [[ -Rmaxd, -ri, -Rmaxd, Rmaxd, bottom, top ]]
        e = e1+e2+e3+e4
        r_eff[i] = _ring_polyhedral_effect(e, den, R, h)
        n = i-1
        if i == 0: 
            n = 0     
        d_eff[i] = r_eff[i]-r_eff[n]
    
    r = utl.deg2m(rd)  
    R2d = np.interp(errg, r_eff[::-1], rd[::-1])
    R2 = np.interp(errg, r_eff[::-1], r[::-1])
    
    # -------------------------------------------------------------------------
    # Plot Result
    if plot == True:
        rstr = f'''R2 = {np.round(R2/1e3, 2)} km 
        g = {np.round(errg, 4)} mGal
        h = {np.round(h,0)} m'''
        plt.plot( np.array( r ) / 1e3, r_eff )
        plt.xlabel( 'R2 [km]', fontsize = font_size )  
        plt.ylabel('g [mGal]', fontsize = font_size )
        plt.vlines(R2/1e3, 0, errg, linestyle="dashed")
        plt.hlines(errg, 0, R2/1e3, linestyle="dashed")
        plt.scatter(R2/1e3, errg, zorder=2);plt.xlim(left=0);plt.ylim(bottom=0)
        box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        if xmax == None: 
            xmax = np.max( r ) / 1e3
        
        if ymax== None: 
            ymax = np.max( r_eff )
            
        plt.xlim( 0, xmax ) 
        plt.ylim( 0, ymax )
        
        plt.text( 0.95, 
                  0.95, 
                  rstr, 
                  fontsize = font_size, 
                  transform = plt.gca().transAxes,
                  ha = 'right', 
                  va = 'top', 
                  linespacing = 2, 
                  ma = 'left', 
                  bbox = box_props )  
        
        plt.tick_params( labelsize = font_size )  

    return (R2, R2d), r_eff, d_eff  

# -----------------------------------------------------------------------------  
def adj_mesh(gs1, R1, gs2, R2, max_gs2_change=0.05,
             max_R1_increase=None):
    """Exactly align two station-centred square grids using INTEGER metres.

    Both grids have an odd number of cells per side, centred on a cell.
    Their inner / outer common boundary obeys

        2*R1 = (2*n+1)*gs1 = (2*m+1)*gs2,
        R2 = R1 + k*gs2, k >= 1.

    An INTEGER R1 therefore requires an EVEN gs1 and gs2. gs1 is
    unchanged; odd gs1 or any noninteger input is rejected explicitly.
    The supplied integer gs2 is a target and may change by at most
    max_gs2_change to another EVEN integer. R1 and R2 grow only as
    needed to make all the identities hold exactly. No float rounding
    of a fractionally-aligned mesh is ever performed.

    By default R1 may increase by at most max(gs1, gs2/2);
    max_R1_increase may be explicitly enlarged. All four outputs are
    Python ints and the operation is idempotent.
    """
    requested = np.asarray([gs1, R1, gs2, R2], dtype=float)
    if not np.all(np.isfinite(requested)) or np.any(requested <= 0):
        raise ValueError("gs1, R1, gs2, R2 must be finite and positive.")
    if np.any(np.abs(requested-np.rint(requested)) > 1e-8):
        raise ValueError(
            "gs1, R1, gs2 and R2 must all be integer numbers of metres; "
            "fractional values cannot preserve the shared grid interface.")
    gs1, R1, gs2, R2 = (int(v) for v in np.rint(requested))
    if gs1 % 2:
        raise ValueError(
            "gs1 must be EVEN to obtain integer R1 with an odd number "
            "of station-centred cells. Supply an even near-grid step.")
    if not np.isfinite(max_gs2_change) or max_gs2_change < 0:
        raise ValueError("max_gs2_change must be finite and nonnegative.")
    if max_R1_increase is None:
        max_R1_increase = max(gs1, 0.5*gs2)
    if not np.isfinite(max_R1_increase) or max_R1_increase < 0:
        raise ValueError("max_R1_increase must be finite and nonnegative.")
    if R1 < gs1//2:
        raise ValueError("R1 must include at least one near-grid cell.")

    # Enumerate only a few possible near boundaries in nondecreasing R1.
    # A near boundary is (2*n+1)*gs1//2, always exactly integral.
    first = max(0, int(np.ceil((2.*R1/gs1-1.)/2.-1e-12)))
    last_radius = R1+max_R1_increase
    best = None
    result = None
    for n in range(first, int(np.floor(last_radius/gs1+1.5))+2):
        boundary = (2*n+1)*(gs1//2)
        if boundary < R1:
            continue
        if boundary > last_radius+1e-9:
            break
        total_width = 2*boundary
        # Range of possible far-grid ODD cell counts whose integer
        # steps lie within the requested relative tolerance.
        lower_step = max(2, int(np.ceil(
            gs2*(1.-max_gs2_change)-1e-10)))
        upper_step = int(np.floor(
            gs2*(1.+max_gs2_change)+1e-10))
        if upper_step < lower_step:
            continue
        m_min = max(1, (total_width+upper_step-1)//upper_step)
        m_max = total_width//lower_step
        if m_min % 2 == 0:
            m_min += 1
        if m_max % 2 == 0:
            m_max -= 1
        nearest = None
        for far_cells in range(m_min,m_max+1,2):
            if total_width % far_cells:
                continue
            candidate = total_width//far_cells
            if candidate % 2 or not (lower_step <= candidate <= upper_step):
                continue
            err = abs(candidate/gs2-1.)
            if nearest is None or err < nearest[0]:
                nearest = (err,candidate)
        if nearest is not None:
            result = (boundary,nearest[1])
            break

    if result is None:
        raise ValueError(
            "No EXACT integer-metre aligned R1/gs2 within the allowed "
            f"gs2 change ({100*max_gs2_change:.1f}%) and R1 increase "
            f"({max_R1_increase:g} m). Station-centred odd-cell grids "
            "require even integer gs1/gs2 and exact common boundaries. "
            "Try another even gs1/gs2 or increase max_R1_increase.")
    r1_new,g2_new = result
    steps = max(1, (max(R2,r1_new+g2_new)-r1_new+g2_new-1)//g2_new)
    r2_new = r1_new+steps*g2_new
    return gs1, r1_new, g2_new, r2_new


def _refine_interface_cells(prisms, densities, opposite_x, opposite_y,
                            R1, inside):
    """Conform a square R1 interface to the union of BOTH grids' nodes.

    The fine near grid and coarser far grid need not have commensurate cell
    widths. Boundary cells on *both* sides must be split at all opposite-grid
    knots that fall inside their tangential spans. Corner near cells may
    require both x- and y-splits.
    """
    prisms = np.asarray(prisms, dtype=float).reshape(-1, 6)
    densities = np.asarray(densities, dtype=float).ravel()
    if len(prisms) != len(densities):
        raise ValueError("Expected one density per interface cell.")
    if not len(prisms):
        return prisms, densities
    opposite_x = np.unique(np.asarray(opposite_x, dtype=float).ravel())
    opposite_y = np.unique(np.asarray(opposite_y, dtype=float).ravel())
    tol = max(1e-7, 1e-11*abs(float(R1)))
    cells_out, densities_out = [], []
    for cell, density in zip(prisms, densities):
        west, east, south, north, bottom, top = cell
        overlaps_y = south < R1-tol and north > -R1+tol
        overlaps_x = west < R1-tol and east > -R1+tol
        if inside:
            on_x = overlaps_y and (
                abs(east-R1) < tol or abs(west+R1) < tol)
            on_y = overlaps_x and (
                abs(north-R1) < tol or abs(south+R1) < tol)
        else:
            on_x = overlaps_y and (
                abs(west-R1) < tol or abs(east+R1) < tol)
            on_y = overlaps_x and (
                abs(south-R1) < tol or abs(north+R1) < tol)
        xs = [west, east]
        ys = [south, north]
        if on_y:
            knots = opposite_x[(opposite_x>west+tol)&(opposite_x<east-tol)]
            xs = [west, *knots, east]
        if on_x:
            knots = opposite_y[(opposite_y>south+tol)&(opposite_y<north-tol)]
            ys = [south, *knots, north]
        for x0, x1 in zip(xs[:-1],xs[1:]):
            for y0, y1 in zip(ys[:-1],ys[1:]):
                if x1-x0 <= tol or y1-y0 <= tol:
                    continue
                c = cell.copy()
                c[0], c[1] = x0, x1
                c[2], c[3] = y0, y1
                cells_out.append(c)
                densities_out.append(density)
    return np.asarray(cells_out, dtype=float).reshape(-1, 6), np.asarray(
        densities_out, dtype=float)


def _refine_far_interface(prisms, densities, near_x, near_y, R1):
    """Refine the far side of R1 to the tangential fine-grid nodes."""
    return _refine_interface_cells(prisms, densities, near_x, near_y,
                                   R1, inside=False)


def _refine_near_interface(prisms, densities, far_x, far_y, R1):
    """Refine the near side of R1 to the tangential coarse-grid nodes."""
    return _refine_interface_cells(prisms, densities, far_x, far_y,
                                   R1, inside=True)


# -----------------------------------------------------------------------------
def mesh_R3( pr2, R3, prjcode_g=None, prjcode_m=None, R=utl.R_wgs84 ) :
    
    if ( prjcode_g != None ) and ( prjcode_m != None ) :
        x1, y1 = utl.prjxy( prjcode_g, prjcode_m, pr2[0][:,0], pr2[0][:,2] )
        x2, y2 = utl.prjxy( prjcode_g, prjcode_m, pr2[0][:,1], pr2[0][:,3] ) 
    else :
       x1, y1 = pr2[0][:,0], pr2[0][:,2] 
       x2, y2 = pr2[0][:,1], pr2[0][:,3] 

    idx = ( x2 >= R3 ) | ( x2 <= -R3 ) | ( y2 >= R3 ) | ( y2 <= -R3 ) 
        
    dx = x2[idx] - x1[idx]
    dy = y2[idx] - y1[idx]
    dz = pr2[0][idx,5] - pr2[0][idx,4]   
    
    den = pr2[1][idx]
    
    xp = x1[idx] + ( dx / 2 )
    yp = y1[idx] + ( dy / 2 )
    zp = pr2[0][idx,4] + ( dz / 2 )  
    
    mass = den * dx * dy * dz
    points = np.vstack( ( xp, yp, zp ) )
    
    pr2 = ( pr2[0][~idx], pr2[1][~idx] )
    pr3 = ( points, mass )
    
    return pr2, pr3
    
# -----------------------------------------------------------------------------
def mesh_xyz( x, y, R1, R2, gs1, gs2, dtm1, dtm2, st_num=0, z=9999, 
              prjcode_in=4326, prjcode_m=None, R=R_wgs84, adjust_mesh=True,
              plot=None, vmin=None, vmax=None, adjust_dtm=True ) :
    """
    Creates the near-field and far-field array starting from a central point (x,y). 
    Returns:
    Mx = lon. of elements borders
    My = lat. of elements borders
    Zc = heght of elemrnts centre
    hst = dtm height of the central point
    """
    
    # Set reference codes obgect for coordinates transformation ---------------
    if prjcode_m is None:
        prjcode_m = f'+proj=ortho +lat_0={y} +lon_0={x} +ellps=sphere +R={R}'
    xm, ym = utl.prjxy( prjcode_in, prjcode_m, x, y ) # metric coordinates
    
    r_val = rt.xy2rasterVal( dtm1, x, y, prjcode=prjcode_in, close=False )[0]
    
    # Adjust mesh borders and step2 to prevent overlapping or gaps between near 
    # and far field     
    if adjust_mesh == True : 
        gs1, R1, gs2, R2 = adj_mesh( gs1, R1, gs2, R2 )  
    
    # Near field lat lon grid -------------------------------------------------
    lim1 = [ -R1, R1, -R1, R1 ] # borders coordinates [m]
    dtm1m = rt.raster_warp( dtm1, 
                            lim = lim1, 
                            method = gdal.GRA_Average, 
                            width = int(round(R1*2/gs1)), 
                            height = int(round(R1*2/gs1)), 
                            out_prjcode = prjcode_m, 
                            lim_prjcode = prjcode_m, 
                            new_name = 'mesh_dtm1_' + str( int( st_num ) ),
                            new_path = None,
                            extension = 'tif' ) 
    
    Zc1 = np.flipud( dtm1m.GetRasterBand(1).ReadAsArray() ) # elevetion grid [m]
    cpx = Zc1.shape[0] // 2  # centre cell in an odd-size near grid 
    
    # dtm station height ------------------------------------------------------
    if ( z != 9999 ) and ( adjust_dtm == True )  : 
         Zc1[cpx, cpx] = z 
    else :
         Zc1[cpx, cpx] = r_val
    
    if dtm1m.RasterCount == 2 :
        msk1 = np.round( np.flipud( dtm1m.GetRasterBand(2).ReadAsArray() ) )
    else: 
        msk1 = None    
    
    # Far field lat lon grid --------------------------------------------------
    lim2 = [ -R2, R2, -R2, R2 ] # borders coordinates [m]
    dtm2m = rt.raster_warp( dtm2, 
                            lim = lim2, 
                            method = gdal.GRA_Average,
                            width = int(round(R2*2/gs2)), 
                            height = int(round(R2*2/gs2)), 
                            out_prjcode = prjcode_m, 
                            lim_prjcode = prjcode_m, 
                            new_name = 'mesh_dtm2_' + str( int( st_num ) ),
                            new_path = None, 
                            extension = 'tif' )
    Zc2 =  np.flipud( dtm2m.GetRasterBand(1).ReadAsArray() ) # elevetion grid [m]

    # set nodata values to 0
    Zc2[ Zc2 == dtm2m.GetRasterBand(1).GetNoDataValue() ] = 0
    Zc1[ Zc1 == dtm1m.GetRasterBand(1).GetNoDataValue() ] = 0

    grid1 = np.linspace(-R1, R1, int(round(R1*2/gs1)) + 1)
    # grid1 = np.arange( -R1, R1, gs1 )
    grid2 = np.linspace(-R2, R2, int(round(R2*2/gs2)) + 1)
    # grid2 = np.arange( -R2, R2, gs2 )
    Mx_1m, My_1m = np.meshgrid( grid1, grid1 ) # near field mesh
    Mx_2m, My_2m = np.meshgrid( grid2, grid2 ) # far field mesh



    if dtm2m.RasterCount == 2 :
        msk2 = np.round( np.flipud( dtm2m.GetRasterBand(2).ReadAsArray() ) )
    else: 
        msk2 = None
  
    # Plot mesh ---------------------------------------------------------------
    if plot is not None:
        print( '====================================================================\n'     
               f'R1={round(R1)} | R2={round(R2)} | gs1={round(gs1)} | gs2={round(gs2)}\n'
               '====================================================================' )
        plt.figure()
        
        if plot == 1: 
            plot_1, plot_2, n = True, False, 0
            
        if plot == 2: 
            plot_1, plot_2, n = False, True, 0
        
        if plot == 12: 
            plot_1, plot_2, n = True, True, 1
            
        if plot_1 == True:
            p1 = plt.subplot( 1, 1 + n, 1 )            
            for i in range( Mx_2m.shape[1] ) :
                p1.plot( Mx_2m[:,i], My_2m[:,i], c='k' )
            for i in range( Mx_2m.shape[0] ) :
                p1.plot( Mx_2m[i,:], My_2m[i,:], c='k' )    
            Mxc, Myc = Mx_2m[ 1:, 1:] - gs2/2, My_2m[ 1:, 1: ] - gs2/2 # conrdinates of prism's center (array)
            p1.scatter( Mxc, Myc, c='b', marker='o' )
            rect = mplt.patches.Rectangle( ( Mx_1m.min(), My_1m.min()),
                                      (Mx_1m.shape[1]-1)*gs1, (Mx_1m.shape[0]-1)*gs1,
                                      linewidth=1, edgecolor='k', facecolor='w',
                                      zorder=2 )
            p1.add_patch( rect )
            for i in range( Mx_1m.shape[1] ) :
                p1.plot( Mx_1m[:,i], My_1m[:,i], c='k' )
            for i in range( Mx_1m.shape[0] ) :
                p1.plot( Mx_1m[i,:], My_1m[i,:], c='k' )    
            Mxc, Myc = Mx_1m[ 1:, 1:] - gs1/2, My_1m[ 1:, 1: ] - gs1/2 # conrdinates of prism's center (array)
            p1.scatter( Mxc, Myc, c='r', marker='o', zorder=2 )
            p1.scatter( 0, 0, c='k', marker='^', zorder=2 )
            xticks = p1.get_xticks() 
            yticks = p1.get_yticks()
            p1.xaxis.set_ticks( xticks )
            p1.set_xticklabels( xticks/1e3 ) 
            p1.yaxis.set_ticks( yticks )
            p1.set_yticklabels( yticks/1e3 ) 
            p1.set_aspect( 'equal', 'box' )
            plt.xlabel( 'Longitude [km]' ) 
            plt.ylabel( 'Latitude [km]' )
               
        if plot_2 == True:
            p2 = plt.subplot(1,1+n,1+n)
            if vmin == None: 
                vmin=np.mean(Zc2)-2*np.std(Zc2)
            if vmax == None: 
                vmax=np.mean(Zc2)+2*np.std(Zc2)    
            im = p2.pcolor(Mx_2m, My_2m, Zc2, 
                           cmap='terrain', edgecolors='white',
                           vmin=vmin,vmax=vmax)
            p2.pcolor(Mx_1m, My_1m, Zc1, cmap='terrain', 
                      edgecolors='white', vmin=vmin, vmax=vmax)
            p2.scatter(0, 0, s=gs2/15, c='m', marker='^')
            xticks = p2.get_xticks() 
            yticks = p2.get_yticks()
            p2.xaxis.set_ticks( xticks )
            p2.set_xticklabels( xticks/1e3 ) 
            p2.yaxis.set_ticks( yticks )
            p2.set_yticklabels( yticks/1e3 ) 
            p2.set_aspect( 'equal', 'box' )
            plt.xlabel('Longitude [km]') 
            plt.ylabel('Latitude [km]')
                
        if plot_2 == True:
            divider = make_axes_locatable(p2)
            cax = divider.append_axes('right', size='3%', pad=0.1)
            plt.gcf().colorbar(im, cax=cax, label='m')
        
        plt.tight_layout()
    # -------------------------------------------------------------------------
    # Remove temporary gdal .vrt files (from both python RAM and disk memory)
    dtm1m_path_name = dtm1m.GetDescription()
    dtm2m_path_name = dtm2m.GetDescription()
    dtm1m = None 
    dtm2m = None
    os.remove( dtm1m_path_name )
    os.remove( dtm2m_path_name )
    
    # -------------------------------------------------------------------------
    # return   
    return ( Mx_1m, Mx_2m ), ( My_1m, My_2m ), ( Zc1, Zc2 ), ( msk1, msk2 )
   
# -----------------------------------------------------------------------------
def mesh_ls( Mx, My, Zc, msk=None, R1=None, R=R_wgs84, dc=2670, dw=1030, z_shift=True ) :
    """
    Crate a prisms mesh valid for both Land and Sea surface stations.
    R1 = limit between near and far field
    R = earth radius (spherical approximation)
    dc, dw = water and crustal densities
    """    
    
    dhx, dhy = np.mean( np.diff( Mx ) ) / 2, np.mean( np.diff( My, axis=0 ) ) / 2
    Mxc, Myc = Mx[ 1:, 1:] - dhx, My[ 1:, 1: ] - dhy # conrdinates of prism's center (array)
    
    if R1 != None :
        Indx = ( Mxc < -R1 ) | ( Mxc > R1 ) | ( Myc < -R1 ) | ( Myc > R1 )
    else: 
        Indx = Zc == Zc 
    
    Me, Mw = Mxc-dhx, Mxc+dhx
    Ms, Mn = Myc-dhy, Myc+dhy

    Z_pos = ( Zc >= 0 ) & Indx
    Z_neg = ( Zc < 0 ) & Indx	
    
    # (A,B) compartment sorting -----------------------------------------------
    if msk is not None : # with coastline mask array
        IA = ( msk == 0 ) & Indx
        IB = ( msk == 1 ) & Indx
    if msk is None : # without coastline array mask
        IA = Z_pos
        IB = Z_neg

    # Bottom 
    MbA, MtA = np.zeros( Zc.shape ), np.zeros( Zc.shape )
    # Top
    MbB, MtB = np.zeros( Zc.shape ), np.zeros( Zc.shape )
    
    MbA[Z_pos & IA], MtA[Z_pos & IA] = 0, Zc[Z_pos & IA] # compartment A Z_positive
    MbA[Z_neg & IA], MtA[Z_neg & IA] = 0, 0 # compartment A Z_negative
    MbB[Z_pos & IB], MtB[Z_pos & IB] = 0, 0 # compartment B Z_positive
    MbB[Z_neg & IB], MtB[Z_neg & IB] = Zc[Z_neg & IB], 0 # compartment B Z_negative
    
    # desnsity array 
    DA = np.full(Zc.shape, dc) 
    DB = np.full(Zc.shape, -dc+dw) 
    
    # z_shift (curvature correction for prisms) -------------------------------
    if z_shift == True:
        # Preserve horizontal Nagy prisms; their base centre follows
        # the spherical reference surface (not the outermost corner).
        radial = R**2 - Mxc**2 - Myc**2
        if np.any(radial < 0):
            raise ValueError("Prism centre lies outside reference hemisphere.")
        Zsh = R - np.sqrt(radial)
        MbA, MtA = MbA-Zsh, MtA-Zsh
        MbB, MtB = MbB-Zsh, MtB-Zsh
        
    ma = np.column_stack((Me[IA].ravel(),Mw[IA].ravel(),
                          Ms[IA].ravel(),Mn[IA].ravel(),
                          MbA[IA].ravel(),MtA[IA].ravel()))
    
    mb = np.column_stack((Me[IB].ravel(),Mw[IB].ravel(),
                          Ms[IB].ravel(),Mn[IB].ravel(),
                          MbB[IB].ravel(),MtB[IB].ravel()))    
    
    m = np.vstack( ( ma, mb ) )
    d = np.hstack( ( DA[IA].ravel(),
                     DB[IB].ravel() ) )
    
    # Delete far field thin elements
    if R1 is not None :
        zero_h = np.abs( m[:,5] - m[:,4] ) < 0.1
        m = m[ ~zero_h ]
        d = d[ ~zero_h ]

    return m, d

# -----------------------------------------------------------------------------
def mesh_sb(Mx, My, Zc, hst, msk=None, R1=None, dc=2670, dw=1030, z_shift=True, R=R_wgs84):
    """
    Crate prism mesh_array valid for sea-bottom stations.
    hst = station depth (negative)
    R1 = limit between near and far field
    R = earth radius (spherical approximation)
    dc, dw = water and crustal densities
    """  
      
    dhx, dhy = np.mean( np.diff( Mx ) ) / 2, np.mean( np.diff( My, axis=0 ) ) / 2
    Mxc, Myc = Mx[ 1:, 1: ] - dhx, My[ 1:, 1: ] - dhy # center prism coordinates (array)  
    
    if R1!=None:
        Indx = ( Mxc < -R1 ) | ( Mxc > R1 ) | ( Myc < -R1 ) | ( Myc > R1 )
    else: 
        Indx = Zc == Zc # i.e. all array == True
    
#    Zc[ int(Zc.shape[0]/2)+1, int(Zc.shape[1]/2)+1 ] = hst
    
    Me, Mw = Mxc-dhx, Mxc+dhx
    Ms, Mn = Myc-dhy, Myc+dhy 

    # (A,B,C) compartment sorting ---------------------------------------------    
    if msk is not None : # with coastline mask array
        IA = ( Zc > 0 ) & ( msk == 0 ) & Indx
        IB = ( Zc > hst ) & ( msk == 1 ) & ( Zc < 0 ) & Indx
        IC = ( Zc < hst ) & ( msk == 1 ) & ( Zc < 0 ) & Indx
        ID = ( Zc == hst ) & ( msk == 1 ) & ( Zc < 0 ) & Indx

    if msk is None : # with coastline mask array
        IA = ( Zc > 0 ) & Indx
        IB = ( Zc > hst ) & ( Zc < 0 ) & Indx
        IC = ( Zc < hst ) & ( Zc < 0 ) & Indx
        ID = ( Zc == hst ) & ( Zc < 0 ) & Indx
    
    # Bottom 
    Zshp = Zc.shape 
    MbA, MbB1, MbB2, MbC1, MbC2, MbD = np.zeros(Zshp), np.zeros(Zshp), np.zeros(Zshp), np.zeros(Zshp), np.zeros(Zshp), np.zeros(Zshp)
    # Top
    MtA, MtB1, MtB2, MtC1, MtC2, MtD = np.zeros(Zshp), np.zeros(Zshp), np.zeros(Zshp), np.zeros(Zshp), np.zeros(Zshp), np.zeros(Zshp)
    
    MbA[IA],  MtA[IA] = hst, Zc[IA] 
    MbB1[IB],  MtB1[IB] = hst, Zc[IB]
    MbB2[IB], MtB2[IB] = Zc[IB], 0  
    MbC1[IC], MtC1[IC] = Zc[IC], hst 
    MbC2[IC], MtC2[IC] = hst, 0 
    MbD[ID], MtD[ID] = hst, 0


    # Check if top-bottom are all positive
    if ( MtA[IA]-MbA[IA] < 0 ).any() or ( MtB1[IB]-MbB1[IB] < 0 ).any() or ( MtB2[IB]-MbB2[IB] < 0 ).any() or ( MtC1[IC]-MbC1[IC] < 0 ).any() or ( MtC2[IC]-MbC2[IC] < 0 ).any() :
        print( 'Error: top-bottom are not all positive' )
        # Check if negative differernces are in compartment A, B1, B2, C1 or C2
        if ( MtA[IA]-MbA[IA] < 0 ).any():
            print( 'Error: negative top-bottom differences in compartment A' )
        if ( MtB1[IB]-MbB1[IB] < 0 ).any():
            print( 'Error: negative top-bottom differences in compartment B1' )
        if ( MtB2[IB]-MbB2[IB] < 0 ).any():
            print( 'Error: negative top-bottom differences in compartment B2' )
        if ( MtC1[IC]-MbC1[IC] < 0 ).any():
            print( 'Error: negative top-bottom differences in compartment C1' )
        if ( MtC2[IC]-MbC2[IC] < 0 ).any():
            print( 'Error: negative top-bottom differences in compartment C2' )
    
        # sys.exit()
    
    # desnsity array  
    DA = np.full( Zshp, dc ) 
    DB1 = np.full( Zshp, dc ) 
    DB2 = np.full( Zshp, dw )
    DC1 = np.full( Zshp, -dc + dw )
    DC2 = np.full( Zshp, dw )
    DD = np.full( Zshp, dw )
    
    # z_shift (curvature correction of prisms) --------------------------------
    if z_shift==True:
        # Preserve horizontal Nagy prisms; align each base centre
        # with the exact spherical reference surface.
        radial = R**2 - Mxc**2 - Myc**2
        if np.any(radial < 0):
            raise ValueError("Prism centre lies outside reference hemisphere.")
        Zsh = R - np.sqrt(radial)
        MbA, MtA = MbA - Zsh, MtA - Zsh
        MbB1, MtB1 = MbB1 - Zsh, MtB1 - Zsh
        MbB2, MtB2 = MbB2 - Zsh, MtB2 - Zsh
        MbC1, MtC1 = MbC1 - Zsh, MtC1 - Zsh
        MbC2, MtC2 = MbC2 - Zsh, MtC2 - Zsh
        MbD, MtD = MbD - Zsh, MtD - Zsh

    ma = np.column_stack(( Me[IA].ravel(),  Mw[IA].ravel(),
                           Ms[IA].ravel(),  Mn[IA].ravel(),
                          MbA[IA].ravel(), MtA[IA].ravel() ))
    
    mb1 = np.column_stack((  Me[IB].ravel(),   Mw[IB].ravel(),
                             Ms[IB].ravel(),   Mn[IB].ravel(),
                           MbB1[IB].ravel(), MtB1[IB].ravel() ))
    
    mb2 = np.column_stack((  Me[IB].ravel(),   Mw[IB].ravel(),
                             Ms[IB].ravel(),   Mn[IB].ravel(),
                           MbB2[IB].ravel(), MtB2[IB].ravel() ))
    
    mc1 = np.column_stack((  Me[IC].ravel(),   Mw[IC].ravel(),
                             Ms[IC].ravel(),   Mn[IC].ravel(),
                           MbC1[IC].ravel(), MtC1[IC].ravel() ))
    
    mc2 = np.column_stack((  Me[IC].ravel(),   Mw[IC].ravel(),
                             Ms[IC].ravel(),   Mn[IC].ravel(),
                           MbC2[IC].ravel(), MtC2[IC].ravel() ))
    
    md = np.column_stack((  Me[ID].ravel(),   Mw[ID].ravel(),
                             Ms[ID].ravel(),   Mn[ID].ravel(),
                           MbD[ID].ravel(), MtD[ID].ravel() ))
    
    m = np.vstack( ( ma, mb1, mb2, mc1, mc2, md ) )
    

    d = np.hstack( ( DA[IA].ravel(),
                     DB1[IB].ravel(),
                     DB2[IB].ravel(),
                     DC1[IC].ravel(),
                     DC2[IC].ravel(),
                     DD[ID].ravel() ) )

    # Delete far field thin elements
    if R1 is not None :
        zero_h = np.abs( m[:,5] - m[:,4] ) < 0.1
        m = m[ ~zero_h ]
        d = d[ ~zero_h ]
    
    return m, d 

# -----------------------------------------------------------------------------   
def st_loop( constant_arg, iterable_arg ) :  
    
    # Split constant arguments ------------------------------------------------
    R1, R2, R3, gs1, gs2, dtm1, dtm2, prjcode_g, curved, R, dc, dw, disable_checks, tot_len, \
    adjust_dtm, output_file, z_shift, prjcode_m, near_geometry, refine_interface = constant_arg
    
    # Split iterables arguments -----------------------------------------------     
    xg = iterable_arg[0]
    yg = iterable_arg[1]
    z = iterable_arg[2]
    st_num = iterable_arg[3] 
    st_type = iterable_arg[4]
    bgc = iterable_arg[5]
    bgc_near = iterable_arg[6]
    bgc_far = iterable_arg[7]
    zdtm = iterable_arg[8]
    st_num = int(st_num.item())
    # Retain the reference radius supplied to bgc(): the same radius must
    # define the station-centred orthographic grid, the Nagy centre shift,
    # and the outer spherical polyhedra. Never replace R inside this loop.
    R = float(R)

    # Create mesh ( m1 = Near_Field, m2 = Far_Field ) -------------------------
    prjcode_m = f'+proj=ortho +lon_0={xg} +lat_0={yg} +ellps=sphere +R={R}'
    Mx, My, Zc, msk = mesh_xyz( xg, yg, R1, R2, gs1, gs2, dtm1, dtm2, st_num=st_num,
                                z=z, prjcode_in=prjcode_g, adjust_dtm=adjust_dtm, 
                                R=R, prjcode_m=None, plot=None )
                         
    # If the station is on the sea surface, zs is set to 0
    # i.e. computational point attached to the prism's top
    if st_type == 1:
        z = 0
        
    # Default: centre-shifted horizontal Nagy prisms. The optional
    # 'spherical' mode uses the unified polydron_grav kernel.
    spherical_near = curved and near_geometry == "spherical"
    
    if ( st_type == 0 ) or ( st_type == 1 ) :
        pr1 = mesh_ls( Mx[0], My[0], Zc[0], msk[0], R=R, dc=dc, dw=dw,
                       z_shift=(z_shift and not spherical_near) )
        if curved is False :
            pr2 = mesh_ls( Mx[1], My[1], Zc[1], msk[1], R1=R1, R=R, dc=dc, dw=dw, z_shift=z_shift )
        else :
            pr2 = mesh_ls( Mx[1], My[1], Zc[1], msk[1], R1=R1, R=R, dc=dc, dw=dw, z_shift=False )
    
    if st_type == 2 :
        pr1 = mesh_sb( Mx[0], My[0], Zc[0], hst=z, msk=msk[0], R=R, dc=dc, dw=dw,
            z_shift=(z_shift and not spherical_near) )
        if curved == False :
            pr2 = mesh_sb( Mx[1], My[1], Zc[1], hst=z, msk=msk[1], R1=R1, R=R, dc=dc, dw=dw, 
                z_shift=z_shift ) 
        else :
            pr2 = mesh_sb( Mx[1], My[1], Zc[1], hst=z, msk=msk[1], R1=R1, R=R, dc=dc, dw=dw, 
                z_shift=False ) 
       
    # In default hybrid mode keep coarse far parent cells UNSPLIT. Their
    # entire R1-facing boundary is triangulated against the fine Nagy
    # prisms; the outer face preserves exactly the ordinary far-cell nodes.
    # No hanging nodes propagate into the second coarse ring.
    if curved and refine_interface and spherical_near:
        pr2 = _refine_far_interface(pr2[0], pr2[1], Mx[0][0, :],
                                    My[0][:, 0], R1)
        pr1 = _refine_near_interface(pr1[0], pr1[1], Mx[1][0, :],
                                     My[1][:, 0], R1)

    if R3 is not None:
        # mesh_R3 removes cells by their outer boundary. The first R1
        # coarse ring is the explicit prism-to-sphere bridge: it must
        # remain a polyhedron, never be replaced with a point mass.
        if curved and refine_interface and not spherical_near and (
                R3 <= R1 + gs2 + 1e-8):
            raise ValueError(
                "R3 must exceed R1 + gs2 to preserve the complete "
                "arbitrary-facet transition ring.")
        pr2, pr3 = mesh_R3(pr2, R3)
    
    # Coordinates of computational point (center of the mesh)
    st = [0, 0, z]
    
    # Evaluate near-field Bouguer correction component at station (st) ---------------------   
    
    if prjcode_m is None : 
        prjcode_m = f'+proj=ortho +lon_0=0 +lat_0=0 +ellps=sphere +R={R}'
    if spherical_near:
        # Remove zero-volume compartments (e.g. no land above sea level).
        # The Nagy prism formula permits these; polyhedra do not.
        finite_mass = (pr1[0][:, 5] - pr1[0][:, 4]) > 1e-10
        bgc_near = _polyhedral_gz(pr1[0][finite_mass],
                                  pr1[1][finite_mass], st, R)
    else:
        nag_near = gm.prism_grav(st, pr1[0], density=pr1[1])
        bgc_near = nag_near.gz()

    # xpt = np.concatenate( ( pr1[0][:,0], pr1[0][:,1] ) )
    # ypt = np.concatenate( ( pr1[0][:,2], pr1[0][:,3] ) )
    # plt.scatter( xpt, ypt, c='b', s=1, label='prism_near' )
    # shp.write_points( x=xpt, y=ypt, prj_code=prjcode_m, name='orhto_prism', path=mdir )

    if curved is True:
        # Build the R1 transition from the *external prism vertices*.
        if refine_interface and not spherical_near:
            bridge_gz, ordinary, ordinary_density = _transition_polyhedral_gz(
                pr2[0], pr2[1], pr1[0], pr1[1], st, R, R1)
            bgc_far = bridge_gz + _polyhedral_gz(
                ordinary, ordinary_density, st, R)
        else:
            bgc_far = _polyhedral_gz(pr2[0], pr2[1], st, R)

    else :
        nag_far = gm.prism_grav( st, pr2[0], density=pr2[1] )
        bgc_far = nag_far.gz()

        # xpt = np.concatenate( ( pr2[0][:,0], pr2[0][:,1] ) )
        # ypt = np.concatenate( ( pr2[0][:,2], pr2[0][:,3] ) )
        # plt.scatter( xpt, ypt, c='r', s=1, label='prism_far' )
        # shp.write_points( x=xpt, y=ypt, prj_code=prjcode_m, name='orhto_prism_far', path=mdir )

    if R3 is not None:
        bgc_far += _point_mass_gz(pr3[0], pr3[1], st, R)

    # Combine near-field and far-field Bouguer correction components
    bgc = bgc_near + bgc_far

    # Convert Bouguer correction outputs and station attributes to scalar types
    if isinstance( bgc, np.ndarray ) == True :
        bgc = bgc.item()
    if isinstance( bgc_near, np.ndarray ) == True :
        bgc_near = bgc_near.item()
    if isinstance( bgc_far, np.ndarray ) == True :
        bgc_far = bgc_far.item()
    if isinstance( z, np.ndarray ) == True :
        z = z.item()
    if isinstance( zdtm, np.ndarray ) == True :
        zdtm = zdtm.item()
    if isinstance( st_num, np.ndarray ) == True :
        st_num = st_num.item()
    if isinstance( st_type, np.ndarray ) == True :
        st_type = st_type.item()
    if isinstance( xg, np.ndarray ) == True :
        xg = xg.item()
    if isinstance( yg, np.ndarray ) == True :
        yg = yg.item()

    with open(output_file, 'a+') as f:
        
        line = "% 10d % 10f % 10f % 10.2f % 10.2f % 10.5f % 10.5f % 10.5f % 10d\n" % ( 
               st_num, np.round(xg, 6), np.round(yg, 6), np.round(z, 3), np.round(zdtm, 3),
               np.round(bgc_near, 6), np.round(bgc_far, 6), np.round(bgc, 6), st_type)
        
        f.write( line )
        
        f.seek(0)
        n_lines = sum(1 for line in f) - 6
        progress = str( round( 100 * n_lines / tot_len, 2 ) )
        print_line = line[:-1]+'  '+progress+' % \n'
        print( print_line )
        
        f.close() 

    pr1 = None
    pr2 = None
     
    # Return
    return bgc, bgc_near, bgc_far, z, st_num, pr1, pr2

# -----------------------------------------------------------------------------  
def bgc( x, y, dtm1, z='dtm', gs1=None, dtm2=None, R1=None, R2=167000., gs2=None, 
        errg=0.5, herr=5, dc=2670, dw=1030, R=R_wgs84, prjcode_in=4326, curved=True,
        prjcode_m=None, st_type=0, plot=None, output_file='Bouguer_correction', 
        disable_checks=True, cpu=1, ply_coast=None, new_dtm=False, adjust_dtm=False,
        R3=50000, z_shift=True, near_geometry="auto",
        refine_interface=True, target_error=0.05 ):
    """
    Compute the Bouguer correction model with Nagy prisms in the near field
    and curvature-following polyhedra in the far field (curved=True).
    Harmonica is not required. Returns the modeled gravitational mass
    contribution in mGal, positive downwards. The correction to be ADDED
    to measured gravity to remove this modeled contribution has opposite sign.
    This function alone does not compute a complete Bouguer anomaly.
    Set curved=False to use the far-field prism approximation.
    Default near_geometry="auto" retains HORIZONTAL Nagy prisms within R1,
    shifted vertically to align each base centre with the spherical surface.
    The first far polyhedral row joins their actual exterior prism vertices.
    refine_interface=True builds arbitrary-facet R1 polyhedra.
    An automatic, survey-wide mesh is selected only if any of
    gs1/R1/gs2 is None. target_error is the max *estimated* error in
    mGal and is ignored for fully explicit meshes. Nominal R2=167 km.
    
    INPUT:
    x,y,z ----->  coordinates of computational points (z can be as x,y or just a sigle value or z='dtm')
    gs1, gs2 --> grid-step of near&far field mesh (if gs2=None, gs2 is empirically calculated)
    dtm1, dtm2 > gdal dataset objects (DTM) of 1near&2far field (if dtm2=None, dtm2=dtm1)
    prjcode_in > input projection coordinate code from proj library eg.'epsg:4326' or even 4326 as integer
    errg ------> legacy tuning accuracy, unused by new auto tuner.
    target_error --> maximum allowable ESTIMATED error [mGal] across
                     all stations (not a certified accuracy bound).
    herr ------> accuracy of dtm data [m]
    st_t ------> station_type: 0 for land st. 1 for sea-surface st., 0 for sea-bottom st.
    """

    if near_geometry not in ("auto", "spherical", "prism"):
        raise ValueError("near_geometry must be 'auto', 'spherical', or 'prism'.")
    if near_geometry == "spherical" and not curved:
        raise ValueError("near_geometry='spherical' requires curved=True.")

    # Create new directory with computation's resaults ------------------------
    cwd = os.getcwd().split( os.sep )[-1]
    if cwd != output_file.split( os.sep )[-1] :
        ndir = str( np.copy( output_file ) )
        os.makedirs( ndir, exist_ok=True )
    else :
        ndir = os.getcwd()
    output_file = ndir + os.sep + output_file.split( os.sep )[-1]
    
    if os.path.isfile( output_file ) == True :
        os.remove( output_file )

    # Transform input data type -----------------------------------------------    
    x,y,z = np.asarray(x).ravel(), np.asarray(y).ravel(), np.asarray(z).ravel()
    st_type = np.asarray(st_type).ravel()
    
    # Stations  coordinates conversions ---------------------------------------
    # Convert to geographic coordinates (spherical approx.)
    prjcode_g = f'+proj=longlat +ellps=sphere +R={R}' 
    xg, yg = utl.prjxy( prjcode_in, prjcode_g, x, y )  # geo coordinates 
    xrmg ,yrmg = np.mean( xg ), np.mean( yg )  # geo coordinates of middle point
    # Convert to metric coordinates using orthographic projection
    if prjcode_m is None : 
        prjcode_m = f'+proj=ortho +lon_0={xrmg} +lat_0={yrmg} +ellps=sphere +R={R}'
    xm, ym = utl.prjxy( prjcode_g, prjcode_m, xg, yg ) # metric coordinates
    
    # First "large crop/resempling operation" on raster input files -----------
    # This will reduces the dtm model to the chosen limits and grid_step.
    # NB. A second "small crop/sempling operation" is done within the function 'mesh_xyz',
    # around each station point.
    # NB.NB. We decided to use Gdal functions because they are SUPER-FAST, 
    # when compared to scipy griddata functions. 
    # Gdal functions make use of '.vrt' file-formats to create intermidiate-step datasets.
    # The vrt format allowes superfast on-grid operations.
    # Therefore, our chice is motivated by the fact that Gdal is much more performing 
    # than any other pythonic method when dealing with large dtm arrays. 
    # More info https://gdal.org/user/virtual_file_systems.html.
    
    if isinstance(dtm1, str):
        dtm1 = gdal.Open(dtm1)
        if dtm1 is None:
            raise FileNotFoundError("Cannot open near-field DTM")

    if isinstance(dtm2, str):
        dtm2 = gdal.Open(dtm2)
        if dtm2 is None:
            raise FileNotFoundError("Cannot open far-field DTM")
    
    if dtm2 == None : 
        dtm2 = dtm1
        
    # Optimize exactly once for the ENTIRE input station set.
    # With all three mesh inputs specified, target_error is ignored.
    if R2 is None:
        R2 = 167000.
    if gs1 is None or R1 is None or gs2 is None:
        chosen = optimize_bgc_mesh(
            x, y, dtm1, gs1=gs1, R1=R1, gs2=gs2, R2=R2,
            target_error=target_error, herr=herr, dc=dc, dw=dw,
            R=R, R3=R3, prjcode_in=prjcode_in, prjcode_m=prjcode_m)
        gs1, R1, gs2, R2 = (chosen[k] for k in
                             ("gs1", "R1", "gs2", "R2"))
        print("SURVEY-WIDE MESH ERROR PROXY:",
              f"max={chosen['estimated_error_mgal']:.5g} mGal,",
              f"target={target_error:.5g} mGal,",
              f"station={chosen['critical_station_index']},",
              f"candidates={chosen['candidates']}",
              "(not a certified error bound)")

    # DTM elevations of the stations ------------------------------------------
    zdtm = rt.xy2rasterVal(dtm1, x, y, prjcode=prjcode_in)[0]

    # Align the radii BEFORE cropping the near DTM whenever gs2 is given.
    # Otherwise an upward R1 adjustment can leave an undersized raster.
    if gs2 is not None:
        original_R2 = R2
        gs1, R1, gs2, aligned_R2 = adj_mesh(
            gs1, R1, gs2, R2 if R2 is not None else R1+gs2)
        if original_R2 is not None:
            R2 = aligned_R2
        near_crop_margin = gs1
    else:
        # gs2 is estimated from the cropped near raster later; prepare an
        # expanded input buffer for the subsequent R1 alignment.
        R1 = (max(0, int(np.ceil(R1/gs1-0.5-1e-10)))+0.5)*gs1
        near_crop_margin = max(gs1, 0.25*R1)

    # Define lim1 -------------------------------------------------------------
    minx, maxx, miny, maxy = np.min(xm), np.max(xm), np.min(ym), np.max(ym)
    near_crop_radius = R1 + near_crop_margin
    lim1 = [minx-near_crop_radius, maxx+near_crop_radius,
            miny-near_crop_radius, maxy+near_crop_radius]
    
    # Create dtm1 -------------------------------------------------------------
    print( "LOADING/CREATING DTM_1 ...")        
    if ( os.path.isfile( ndir + os.sep + 'dtm1' + os.sep + 'dtm1.tif' ) is False ) or \
       ( new_dtm is True ) :       
        if gs1 == None: # Take gs1 from maximum gridstep of dtm1
            res1 = rt.raster_res( dtm1 )
            gs1 = np.mean( res1 )  
        # Replace nan values with -99999.0 ---------------------------------------
        dtm1 = rt.set_raster_nodata( dtm1, -32768.0 )
        # Crop-Sample to lim1 -------------------------------------------------
        dtm_t1 = rt.raster_warp( dtm1, 
                                 lim1, 
                                 xRes = gs1, 
                                 yRes = gs1, 
                                 out_prjcode = prjcode_m, 
                                 lim_prjcode = prjcode_m, 
                                 method = gdal.GRA_Average, 
                                 new_name = 'dtmt1', 
                                 new_path = ndir + os.sep + 'dtm1', 
                                 extension = 'tif',
                                 close = True )  

        # Adjust dtm1 values at station points --------------------------------
        if ( adjust_dtm is True ) :
            if ( type( z[0].astype(object) ) == str ) and ( z[0].astype(object) == 'dtm' ) :
                z = zdtm
            dtm_t1= rt.raster_edit( dtm_t1, x, y, z, prjcode=prjcode_in, band=1, close=True,
                                    path=ndir+os.sep+'dtm1', new_name='dtmt1', suffix='_e' ) 
              
        # Create mask for dtm_r1 from coastline polygon (if exist) ------------
        if ply_coast != None :    
            print('Add coast band')
            dtm1_new = rt.add_coastline_band( dtm_t1, ply_coast, 
                                              new_path=ndir+os.sep+'dtm1', new_name='dtm1' )
        else:
            dtm1_new = rt.raster_save( dtm_t1, new_path=ndir+os.sep+'dtm1', new_name='dtm1', 
                                       extension='tif', close=True)

        # Remove temporary rasters --------------------------------------------
        del dtm1, dtm_t1
        
        if os.path.exists( ndir + os.sep + 'dtm1' + os.sep + 'dtmt1.tif' ):
            os.remove( ndir + os.sep + 'dtm1' + os.sep + 'dtmt1.tif' ) 
            
        if os.path.exists( ndir + os.sep + 'dtm1' + os.sep + 'dtmt1_e.tif' ):
            os.remove( ndir + os.sep + 'dtm1' + os.sep + 'dtmt1_e.tif' )         
    else :
        dtm1_new = ndir + os.sep + 'dtm1' + os.sep + 'dtm1.tif' 
        
    # Parameters resolved. The old empirical gs2/R2 gravity loops
    # remain for reference but are no longer used by bgc().
    print("LOADING/CREATING DTM_2 ...")

    # Re-apply the exact alignment when gs2 was estimated from the DTM.
    # The function is idempotent when invoked earlier with explicit gs2.
    gs1, R1, gs2, R2 = adj_mesh(gs1, R1, gs2, R2)
    if R1+gs1 > near_crop_radius+1e-6:
        raise ValueError(
            "Estimated gs2 requires R1 to expand beyond the initial near "
            "DTM crop. Pass gs2 explicitly, or increase the near-DTM "
            "preprocessing buffer instead of sampling missing elevations.")

    # Create dtm2 ------------------------------------------------------------- 
    if ( os.path.isfile( ndir + os.sep + 'dtm2' + os.sep + 'dtm2.tif' ) is False ) or \
       ( new_dtm is True ) : 

        # Replace nan values with -9999.0
        dtm2 = rt.set_raster_nodata( dtm2, -32768.0 )
        # Crop-Resample dtm2 based on R2 and gs2 ------------------------------
        lim2 = [ minx-R2-gs2, maxx+R2+gs2, miny-R2-gs2, maxy+R2+gs2 ]          
        dtm_t2 = rt.raster_warp( dtm2, 
                                 lim = lim2, 
                                 xRes = gs2, 
                                 yRes = gs2, 
                                 out_prjcode = prjcode_m, 
                                 lim_prjcode = prjcode_m, 
                                 method = gdal.GRA_Average, 
                                 new_name ='dtmt2', 
                                 new_path = ndir+os.sep+'dtm2', 
                                 extension = 'tif',
                                 close = False ) 

        # Create mask for dtm_r1 from coastline polygon (if exist) ------------
        if ply_coast != None :    
            dtm2_new = rt.add_coastline_band( dtm_t2, ply_coast, 
                                              new_path=ndir+os.sep+'dtm2', new_name='dtm2' )
        
        else:
            dtm2_new = rt.raster_save( dtm_t2, new_path=ndir+os.sep+'dtm2', new_name='dtm2', 
                                       extension='tif', close=True)

        # Remove temporary rasters --------------------------------------------
        del dtm2, dtm_t2 
        os.remove( ndir + os.sep + 'dtm2' + os.sep + 'dtmt2.tif' )
        
    else:
        dtm2_new = ndir + os.sep + 'dtm2' + os.sep + 'dtm2.tif' 
                
    # Convert station attributes into iterable objects ------------------------
    if np.size( st_type ) == 1 :
       st_type = np.repeat( int( st_type[0] ), np.size( xg ) )
    if ( np.size( xg ) > 1 ) and ( np.size( z ) == 1 ) and ( type( z ) in [float,int] ) :
       z = np.repeat( z, np.size( xg ) ) # if z is only one constant variable
    if ( type( z[0].astype(object) ) == str ) and ( z[0].astype(object) == 'dtm' ) :
       z = zdtm
    # Allocate near-, far- and total Bouguer correction contributions    
    bgc_near = np.zeros( np.size (xg ) ) #(1)
    bgc_far = np.zeros( np.size (xg ) ) #(2)
    bgc = np.zeros( np.size( xg ) ) #(3) 
    
    print( 'Computational Parameters, gs1 gs2 R1 R2 :' )
    print( round(gs1), round(gs2), round(R1), round(R2) )
    
    # Check mesh: plot dtbm mesh to control if the model is OkyDoky 
    if plot != None :
            mesh_xyz( xrmg, yrmg, R1, R2, gs1, gs2, dtm1_new, dtm2_new, prjcode_in=prjcode_g, plot=plot )           
    
    pcol = [ 'st_num', 'Lon', 'Lat', 'h_st', 'h_dtm', 'bgc_near', 'bgc_far', 'bgc_tot', 'st_type', 'prog.%' ]
    
    tot_len = np.size( x )
    st_num = np.arange( 0, np.size(xg), dtype=int )
    
    time_count = time.time()
    header=( '==================================================================================================\n'
             '==================================================================================================\n'     
            f'Computational parameters [m]:\n'
            f'R1 = {round(R1)} | R2 = {round(R2)} | gs1 = {round(gs1)} | gs2 = {round(gs2)}\n'
             '==================================================================================================\n'
            f'{pcol[0]:>10} {pcol[1]:>10} {pcol[2]:>10} {pcol[3]:>10} {pcol[4]:>10}'+
            f' {pcol[5]:>10} {pcol[6]:>10} {pcol[7]:>10} {pcol[8]:>10}') 
    
    with open(output_file, 'w') as f:
        f.write(header + '\n')
        f.close()
        
        # START ITERATION =====================================================
        # Iterative computation over each input grav. station (index st_num) --
        if (R3 is not None ) and ( R3 >= R2 ) : 
            R3 = None
        constant_arg = ( R1, R2, R3, gs1, gs2, 
                         dtm1_new, dtm2_new, prjcode_g, curved, R, dc, dw, 
                         disable_checks, tot_len, adjust_dtm, output_file, 
                         z_shift, prjcode_m, near_geometry, refine_interface )
        
        iterable_arg = np.column_stack( ( xg, yg, z, st_num, st_type, bgc, bgc_near, bgc_far, zdtm ) ) 
                                          
        patial = functools.partial( st_loop, constant_arg ) 

        if cpu is None :
            cpu = mpr.cpu_count()

        print( "LOOP STARTED ( #cpu:",cpu,") ...")
        print( header )
        if cpu > 1 :
            with mpr.Pool(cpu) as p :
                Bgc_list = list( zip( *p.map( patial, iterable_arg ) ) )
                
        if cpu == 1:
            Bgc_list = list( zip ( *map( patial, iterable_arg ) ) ) 
               

        bgc = np.array( ( Bgc_list[0] ) ) 
        bgc_near = np.array( ( Bgc_list[1] ) ) 
        bgc_far = np.array( ( Bgc_list[2] ) ) 
        z = np.array( ( Bgc_list[3] ) ) 
        st_num = np.array( ( Bgc_list[4] ) )
        pr1 = Bgc_list[5]
        pr2 = Bgc_list[6]
            
        # END ITERATION =======================================================   
        End = ( '==================================================================================================\n'   
                '==================================================================================================\n'
               f'Total_time: {time.time()-time_count} sec.' )
        
    with open(output_file, 'w') as f:  
        f.write(header + '\n')
        
        array = np.column_stack( ( st_num,
                                   np.round(xg, 6),
                                   np.round(yg, 6),
                                   np.round(z, 3),
                                   np.round(zdtm, 3),
                                   np.round(bgc_near, 5),
                                   np.round(bgc_far, 5),
                                   np.round(bgc, 5),
                                   st_type ) )
        
        # Sort array according to st_num ( it is necessary when run parallel)
        array = array[ np.argsort( array [ :, 0 ] ) ] 
        fmt = '% 10d % 10f % 10f % 10.2f % 10.2f % 10.4f % 10.4f % 10.4f % 10d'
        np.savetxt( f, array, fmt = fmt  )
        
        f.write(End)
        print( End )
        f.close() 
        
    return bgc, bgc_near, bgc_far, R1, R2, gs1, gs2, pr1, pr2 

# -----------------------------------------------------------------------------
def load_bgc( file, ds_type='dict' ) :
    """
    Import results
    """
    
    load_resoult = np.genfromtxt( file, skip_header=6, skip_footer=3 ) 
    
    array = load_resoult[ np.argsort(load_resoult[:,0]) ] 
    
    st_num, Lon, Lat, h_st, h_dtm, bgc_near, bgc_far, bgc_tot, st_type \
        = np.hsplit(array, array.shape[1])
        
    d = { 'st_num' : st_num.ravel(),
          'Lon' : Lon.ravel(), 
          'Lat' : Lat.ravel(),
          'h_st' : h_st.ravel(),
          'h_dtm' : h_dtm.ravel(),
          'bgc_near' : bgc_near.ravel(), 
          'bgc_far' : bgc_far.ravel(), 
          'bgc_tot' : bgc_tot.ravel(), 
          'st_type' : st_type.ravel()
        }
    
    # Return 
    if ds_type == 'dict' :
        
        return d
    
    if ds_type == 'array' :
        
        return array
    
    if ds_type == 'all' :
        
        return array, d

# -----------------------------------------------------------------------------
def run_bgc( x, y, dtm1, z='dtm', gs1=None, dtm2=None, R1=None, R2=167000., gs2=None, 
            errg=0.5, herr=5, dc=2670, dw=1030, R=R_wgs84, prjcode_in=4326, curved=True,
            prjcode_m=None, st_type=0, plot=None, output_file='Bouguer_correction', 
            disable_checks=True, cpu=None, ply_coast=None, new_dtm=False, R3=50000,
            near_geometry="auto", refine_interface=True, target_error=0.05 ) :
    
    # Create new directory for computation's resaults -------------------------
    cwd = os.getcwd().split( os.sep )[-1]
    if cwd != output_file.split( os.sep )[-1] :
        ndir = str( np.copy( output_file ) )
        os.makedirs( ndir, exist_ok=True )
    else :
        ndir = os.getcwd()

    xyz = np.column_stack( ( x, y ) )
    
    if type( z ) != str:
        if np.size( z ) > 1 :
            xyz = np.column_stack( ( xyz, z ) )
        else:
            xyz = np.column_stack( ( xyz, np.repeat( z, np.size( x ) ) ) ) 
    else:
        xyz = np.column_stack( ( xyz, np.repeat( 9999, np.size( x ) ) ) )              

    if np.size( st_type ) > 1 :
        xyz = np.column_stack( ( xyz, st_type ) )
    else:
        xyz = np.column_stack( ( xyz, np.repeat( st_type, np.size( x ) ) ) )                
        
    np.savetxt( ndir + os.sep + 'xyz', xyz, fmt='%.6f %.6f %.2f %.0f' )
    
    if type( dtm1 ) != str : dtm1 = dtm1.GetDescription()
    if type( dtm1 ) != str : dtm1 = dtm1.GetDescription()
    if type( gs1 ) != str : gs1 = str( gs1 )
    if type( R1 ) != str : R1 = str( R1 )
    if type( gs2 ) != str : gs2 = str( gs2 )
    if type( R2 ) != str : R2 = str( R2 )
    if type( prjcode_in ) != str : prjcode_in = str( prjcode_in )
    if type( prjcode_m ) != str : prjcode_m = str( prjcode_m )  
    if type( output_file ) != str : output_file = str( output_file ) 
    if type( cpu ) != str : cpu = str( cpu ) 
    if type( ply_coast ) != str : ply_coast = str( ply_coast ) 
    if type( new_dtm ) != str : new_dtm = str( new_dtm ) 
    if type( R3 ) != str : R3 = str( R3 )
    if type( near_geometry ) != str : near_geometry = str( near_geometry )
    if type( refine_interface ) != str : refine_interface = str( refine_interface )
    if type( target_error ) != str : target_error = str( target_error )
      
    irs = prjcode_in
    pcs = prjcode_m
    out = output_file 
    
    cfn = os.path.abspath(__file__)
    
    # Do not hide warnings from the estimator.
    run = f'python' + \
          f' -m relgravpy._bouguer_correction' + \
          f' -xyz {ndir + os.sep + "xyz"}' + \
          f' -dtm1 {dtm1}' + \
          f' -gs1 {gs1}' + \
          f' -R1 {R1}' + \
          f' -dtm2 {dtm2}' + \
          f' -gs2 {gs2}' + \
          f' -R2 {R2}' + \
          f' -irs {irs}' + \
          f' -pcs {pcs}' + \
          f' -out {out}' + \
          f' -cpu {cpu}' + \
          f' -ply_coast {ply_coast}' + \
          f' -curved {curved}' + \
          f' -new_dtm {new_dtm}' + \
          f' -R3 {R3}' + \
          f' -near_geometry {near_geometry}' + \
          f' -refine_interface {refine_interface}' + \
          f' -target_error {target_error}'
              
    if platform.system() == 'Linux' :
        run_str = ndir + os.sep + 'run.sh'
    if platform.system() == 'Windows' :
        run_str = ndir + os.sep + 'run.bat'
        
    with open( run_str , 'w') as f :
              f.write(run)
              f.close()        
 
    os.system( run_str )
    
    out_file = ndir + os.sep + output_file.split( os.sep )[-1]
    
    result = load_bgc( out_file, ds_type='all' )
    
    with open( out_file, 'r' ) as f :
        print( f.read() )    
    
    return result
    
# -----------------------------------------------------------------------------
if __name__ == '__main__' :  
        
    p = argparse.ArgumentParser()
    p.add_argument( '-xyz', '--xyz', help='input file with x, y coordinates \
                    of computational points, z is optional', type=str )
    p.add_argument( '-dtm1', '--dtm1', type=str )
    p.add_argument( '-gs1', '--gs1', default=None, type=str )
    p.add_argument( '-R1', '--R1', default=None, type=str )
    p.add_argument( '-dtm2', '--dtm2', default=None, type=str )
    p.add_argument( '-gs2', '--gs2', default=None, type=str )
    p.add_argument( '-R2', '--R2', default='167000.', type=str )
    p.add_argument( '-R3', '--R3', default=None, type=str )
    p.add_argument( '-near_geometry', '--near_geometry', default='auto', type=str )
    p.add_argument( '-refine_interface', '--refine_interface', default='True', type=str )
    p.add_argument( '-target_error', '--target_error', default=0.05, type=float )
    p.add_argument( '-irs', '--irs', default=4326, type=str )
    p.add_argument( '-pcs', '--pcs', default=None, type=str )
    p.add_argument( '-del_xyz', '--del_xyz', default=str )
    p.add_argument( '-out', '--out', default='Bouguer_correction', type=str )
    p.add_argument( '-cpu', '--cpu', default=None, type=str )
    p.add_argument( '-ply_coast', '--ply_coast', default=None, type=str )
    p.add_argument( '-curved', '--curved', default=True, type=str )
    p.add_argument( '-new_dtm', '--new_dtm', default=False, type=str )
                
    arg = p.parse_args()
    xyz = np.loadtxt( arg.xyz )

    if xyz.shape[1] == 2 :
        x, y = np.hsplit( xyz, 2 )
        z = 'dtm'
        st_type = 0
    if xyz.shape[1] == 3 :
        x, y, z = np.hsplit( xyz, 3 )
        st = 0
    if xyz.shape[1] == 4 :
        x, y, z, st_type = np.hsplit( xyz, 4 )   
    if np.any( ( z == 9999 ) ) :
        z = 'dtm'             
              
    arg.irs = eval( arg.irs )  
    arg.pcs = eval ( arg.pcs )   
    arg.cpu = eval ( arg.cpu )
    arg.R1 = eval ( arg.R1 )
    arg.R2 = eval ( arg.R2 )
    arg.gs1 = eval ( arg.gs1 )
    arg.gs2 = eval ( arg.gs2 )
    arg.curved = eval ( arg.curved )
    arg.ply_coast = eval ( arg.ply_coast )
    arg.new_dtm = eval ( arg.new_dtm )
    arg.R3 = eval ( arg.R3 )
    arg.refine_interface = arg.refine_interface.lower() == 'true'

    bgc( x, 
        y, 
        dtm1 = arg.dtm1, 
        z = z, 
        gs1 = arg.gs1, 
        dtm2 = arg.dtm2, 
        R1 = arg.R1, 
        R2 = arg.R2, 
        gs2 = arg.gs2, 
        prjcode_in = arg.irs,
        prjcode_m = arg.pcs, 
        st_type = st_type, 
        output_file = arg.out,
        cpu = arg.cpu,
        ply_coast = arg.ply_coast,
        curved = arg.curved,
        new_dtm = arg.new_dtm,
        R3 = arg.R3,
        near_geometry = arg.near_geometry,
        refine_interface = arg.refine_interface,
        target_error = arg.target_error )
    
    if arg.del_xyz == True:
        os.remove( arg.xyz )
        
    
        