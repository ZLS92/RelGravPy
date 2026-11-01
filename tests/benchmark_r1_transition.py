"""Independent spherical benchmark for R1 near/far resolution transition.

Near field uses horizontal centre-shifted Nagy prisms. The first ring
of coarse spherical far cells is joined to the actual outer prism vertices.
The far cells are not subdivided; their inner faces have arbitrary facets. Compare with the converged
uniform spherical-volume Gauss-Legendre integral (not Harmonica).

Run: PYTHONPATH=tests:. python -u tests/benchmark_r1_transition.py
"""
import json
import numpy as np

from test_bouguer_polyhedra import BC, GM
from benchmark_spherical_reference import (
    spherical_volume_gravity, RADIUS, DOMAIN, DENSITY,
    THICKNESS, OBS_Z,
)


def square_cells(extent, n):
    edges = np.linspace(-extent, extent, n+1)
    west, south = np.meshgrid(edges[:-1], edges[:-1], indexing="xy")
    east, north = np.meshgrid(edges[1:], edges[1:], indexing="xy")
    cells = np.column_stack((west.ravel(), east.ravel(),
                             south.ravel(), north.ravel(),
                             np.zeros(n*n), np.full(n*n,THICKNESS)))
    return cells, edges


def spherical_wedge_reference(horizontal_half_width, rbottom, rtop,
                              station_height, nangles=96, nradial=32):
    """Independent high-order radial/lat/lon integral for one spherical wedge.

    The angular footprint is longitude/latitude bounded, not an exactly
    orthographic square. The <centimetre edge discrepancy is a separate
    small geometry approximation in these 20-km tests.
    """
    angle=np.arcsin(horizontal_half_width/RADIUS)
    x,w=np.polynomial.legendre.leggauss(nangles)
    x, w=(x+1)/2, w/2
    a=angle*x**3
    aw=w*3*angle*x*x
    lat,lon=a[:,None],a[None,:]
    q=4*(np.sin(lat/2)**2+np.cos(lat)*np.sin(lon/2)**2)
    sw=4*aw[:,None]*aw[None,:]*np.cos(lat)
    xr,wr=np.polynomial.legendre.leggauss(nradial)
    radii=RADIUS+rbottom+(rtop-rbottom)*(xr+1)/2
    radial_weights=(rtop-rbottom)*wr/2
    ro=RADIUS+station_height
    total=0.
    for rs,weight in zip(radii,radial_weights):
        delta=ro-rs
        dist_sq=delta*delta+ro*rs*q
        total+=weight*rs*rs*np.sum(sw*(delta+rs*q/2)/dist_sq**1.5)
    return BC.gm.G*DENSITY*1e5*float(total)


def independent_polyhedron_volume_gz(vertices, density, observation, order=4, faces=None):
    """Independent 3-D volume integral over tetrahedra of an eight-vertex cell.

    Decompose each face triangle with the internal centroid and apply a
    tensor-product Gauss-Legendre cubature after a Duffy transformation.
    This uses Newton's 1/r**3 field directly, not the surface integral in
    the RelGravPy kernel. It is stable for these distant R1 boundary cells.
    """
    v = np.asarray(vertices, float)
    centre = v.mean(axis=0)
    if faces is None:
        faces=[list(f) for f in BC._CURVED_FACES]
    faces=np.asarray([face[1:] for face in faces],dtype=int)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes=(nodes+1)/2
    weights=weights/2
    U,V,W=np.meshgrid(nodes,nodes,nodes,indexing="ij")
    wu,wv,ww=np.meshgrid(weights,weights,weights,indexing="ij")
    u,vp,w=U.ravel(),V.ravel(),W.ravel()
    factor=(wu*wv*ww).ravel()*(1-u)**2*(1-vp)
    obs=np.asarray(observation,float)
    integral=0.
    for a,b,c in faces:
        e1,e2,e3 = v[a]-centre,v[b]-centre,v[c]-centre
        jac=np.linalg.det(np.stack((e1,e2,e3),axis=1))
        xyz=(centre[None,:]+u[:,None]*e1
             +((1-u)*vp)[:,None]*e2
             +((1-u)*(1-vp)*w)[:,None]*e3)
        diff=obs[None,:]-xyz
        distance=np.linalg.norm(diff,axis=1)
        integral+=jac*np.sum(factor*diff[:,2]/distance**3)
    return float(BC.gm.G*1e5*density*integral)


def independent_bridge_mass_difference(far_cells, near, observation, R1):
    """Independent 3D bridge volume correction for CURRENT coarse parents.

    Each bridge is integrated as a general triangular surface via signed
    tetrahedra. In contrast to the legacy implementation, far parent
    cells remain unsplit, and the four corner polyhedra also bridge.
    """
    from test_bouguer_transition import TR
    near=np.asarray(near,float)
    delta=0.
    tol=1e-6
    spherical_faces=[list(f) for f in BC._CURVED_FACES]
    for cell in far_cells:
        w,e,s,n=cell[:4]
        mx,my=(w+e)/2,(s+n)/2
        side=None
        if abs(w-R1)<tol and -R1<my<R1:side="east"
        elif abs(e+R1)<tol and -R1<my<R1:side="west"
        elif abs(s-R1)<tol and -R1<mx<R1:side="north"
        elif abs(n+R1)<tol and -R1<mx<R1:side="south"
        corner=(abs(w-R1)<tol or abs(e+R1)<tol) and (
            abs(s-R1)<tol or abs(n+R1)<tol)
        if side:
            bridge,faces=TR._bridge_mesh(cell,near,side,RADIUS)
        elif corner:
            bridge,faces=TR._corner_mesh(cell,near,RADIUS,R1)
        else:
            continue
        sphere=BC._curved_cell_vertices(cell,RADIUS)
        delta+=independent_polyhedron_volume_gz(
            bridge,DENSITY,observation,faces=faces)
        delta-=independent_polyhedron_volume_gz(
            sphere,DENSITY,observation,faces=spherical_faces)
    return delta


def run_case(near_n=80, far_n=64, R1=5000.0,
             near_top=THICKNESS, far_top=THICKNESS, station_height=None):
    # Fine mesh [-R1, R1]^2; coarse mesh [-10km,10km]^2.
    # For far_n=64: R1 is an exact coarse grid line (16 cells from centre).
    near, near_edges = square_cells(R1, near_n)
    all_far, far_edges = square_cells(DOMAIN/2, far_n)
    cx = (all_far[:,0]+all_far[:,1])/2
    cy = (all_far[:,2]+all_far[:,3])/2
    exterior = (abs(cx)>=R1) | (abs(cy)>=R1)
    far = all_far[exterior]
    near[:,5] = near_top
    far[:,5] = far_top
    if station_height is None:
        station_height = max(near_top,far_top)+20.
    observation=[0,0,station_height]
    far_dense = np.full(len(far),DENSITY)
    fine_dense = np.full(len(near),DENSITY)

    # Actual method: HORIZONTAL Nagy prisms with centres on the sphere.
    # The first far polyhedra have their INNER faces attached exactly to the
    # outer face of the corresponding centre-shifted near prism.
    shifted_near = near.copy()
    cx=(near[:,0]+near[:,1])/2
    cy=(near[:,2]+near[:,3])/2
    sag = RADIUS - np.sqrt(RADIUS**2 - cx**2 - cy**2)
    shifted_near[:,4:] -= sag[:,None]
    near_g = float(GM.prism_grav([observation],shifted_near,fine_dense).gz()[0])
    from test_bouguer_transition import TR
    bridge_g,ordinary,ordinary_density=TR._transition_polyhedral_gz(
        far,far_dense,shifted_near,fine_dense,observation,RADIUS,R1)
    far_g=bridge_g+BC._polyhedral_gz(
        ordinary,ordinary_density,observation,RADIUS)
    unrefined_far = BC._polyhedral_gz(far,far_dense,observation,RADIUS)

    # Legacy prism approximation: near-field cell shifted by the farthest
    # horizontal corner; far-field spherical unchanged.
    shifted = near.copy()
    xymax = np.maximum(abs(near[:,0]),abs(near[:,1]))**2
    xymax += np.maximum(abs(near[:,2]),abs(near[:,3]))**2
    shifted[:,4:] -= (xymax/(2*RADIUS))[:,None]
    near_nagy = float(GM.prism_grav([observation],shifted,fine_dense).gz()[0])

    if near_top == far_top == THICKNESS and station_height == OBS_Z:
        exact = spherical_volume_gravity(96,32)
    else:
        exact = spherical_wedge_reference(DOMAIN/2,0,far_top,station_height)
        if near_top > far_top:
            exact += spherical_wedge_reference(R1,far_top,near_top,station_height)
        elif near_top < far_top:
            exact -= spherical_wedge_reference(R1,near_top,far_top,station_height)
    # The sharp-step spherical reference is *not* the same body as the
    # prism-anchored bridge when near/far DEM heights differ. Evaluate the
    # bridge's additional mass independently with direct 3D volume cubature.
    bridge_delta=independent_bridge_mass_difference(
        far,shifted_near,observation,R1)
    hybrid_reference=exact+bridge_delta
    data = {
        "R1_m":R1,
        "near_top_m":near_top,
        "far_top_m":far_top,
        "station_height_m":station_height,
        "near_cells_original":len(near),
        "near_cells_refined":len(shifted_near),
        "far_cells_original":len(far),
        "far_cells_refined":len(far),
        "bridge_parent_cells":len(far)-len(ordinary),
        "near_cell_width_m":2*R1/near_n,
        "far_cell_width_m":DOMAIN/far_n,
        "sharp_step_reference_mGal":exact,
        "independent_bridge_delta_uGal":1000*bridge_delta,
        "hybrid_reference_mGal":hybrid_reference,
        "numerical_residual_uGal":1000*(near_g+far_g-hybrid_reference),
        "new_combined_mGal":near_g+far_g,
        "legacy_combined_mGal":near_nagy+unrefined_far,
        "unjoined_prism_spherical_mGal":near_g+unrefined_far,
        "new_error_uGal":1000*(near_g+far_g-exact),
        "legacy_error_uGal":1000*(near_nagy+unrefined_far-exact),
        "transition_wedge_effect_uGal":1000*(far_g-unrefined_far)
    }
    print("R1_BENCHMARK",json.dumps(data),flush=True)
    # The sharp step measures a different body. Only the corrected
    # reference, including the *independently integrated* bridge mass,
    # is an appropriate numerical-error comparator.
    assert abs(data["numerical_residual_uGal"]) < 0.2, data
    return data


if __name__ == "__main__":
    run_case()
    run_case(near_top=350.,far_top=250.,station_height=370.)
