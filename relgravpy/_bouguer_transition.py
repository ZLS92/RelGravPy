"""Conformal arbitrary-facet polyhedra joining shifted Nagy prisms to R1.

One polyhedron per ORIGINAL far-field cell: fine stair-step prism geometry
on the R1-facing side, coarse untouched spherical vertices on the exterior.
Unlike subdividing coarse cells, this does not introduce hanging vertices
on the next far ring. Works in the station-centred ENU Cartesian frame.
"""
import numpy as np

from . import grav_modelling as gm


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
            raise ValueError("R1 bridge near-prism height intervals are disjoint.")

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
    outward, volume = _outward_triangle_winding(vertices,indices)
    return vertices, [[3,*map(int,f)] for f in outward]


def _transition_polyhedral_gz(far_cells, far_density, near_prisms,
                              near_density, observation, radius, R1):
    """Sum coarse uncut far cells plus arbitrary-facet R1 bridge polyhedra.

    Density mismatch at the near/far interface remains an explicit error.
    The fast ordinary far-field kernel is provided by the caller.
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
                if not np.any(overlapping) or not np.all(
                    np.isclose(nd[overlapping],density,rtol=0,atol=1e-8)):
                    raise ValueError(
                        "R1 bridge has no matching Nagy prism density "
                        "for far cell; reconcile coastal compartments.")
                vertices,facets=_bridge_mesh(
                    cell,near[overlapping],side,radius,tol)
            else:
                # The corner density should match that of the adjoining
                # near prism. A unique near rectangle may carry multiple
                # density compartments; reject this for now.
                x=R1 if w>=R1-tol else -R1
                y=R1 if s>=R1-tol else -R1
                select=(
                    (abs(near[:,1]-x)<tol if x>0 else abs(near[:,0]-x)<tol)
                    & (abs(near[:,3]-y)<tol if y>0 else abs(near[:,2]-y)<tol))
                if len(near[select])!=1 or not np.isclose(
                        nd[select][0],density,rtol=0,atol=1e-8):
                    raise ValueError("R1 corner lacks matching Nagy density.")
                vertices,facets=_corner_mesh(cell,near[select],radius,R1,tol)
            total+=gm.polydron_grav(
                vertices,facets,density).compute_gz(
                    observation,method="surface")
        else:
            ordinary.append(cell)
            ordinary_density.append(density)
    return total, np.asarray(ordinary,float).reshape(-1,6), np.asarray(
        ordinary_density,float)
