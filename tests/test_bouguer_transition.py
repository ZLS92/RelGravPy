"""Geometry and gravity tests for the single-corona arbitrary-face R1 bridge.

Keep the tests GDAL-free by extracting only the tested numerical functions.
"""
import warnings
import numpy as np
import pytest

from test_bouguer_polyhedra import GM, BC
from pathlib import Path
import ast

# The R1 bridge is internal to _bouguer_correction.py.
TR = BC

R = 6371000.
R1 = 1125.
GS1 = 150.
GS2 = 250.
R2 = 1625.


def square_cells(edges, bottom=0., top=300.):
    w,s=np.meshgrid(edges[:-1],edges[:-1],indexing="xy")
    e,n=np.meshgrid(edges[1:],edges[1:],indexing="xy")
    return np.column_stack((w.ravel(),e.ravel(),s.ravel(),n.ravel(),
                            np.full(w.size,bottom),np.full(w.size,top)))


def scenario(top=300.,shift=True,step=False):
    near=square_cells(np.arange(-R1,R1+GS1*.5,GS1),top=top)
    if shift:
        xc=(near[:,0]+near[:,1])/2
        yc=(near[:,2]+near[:,3])/2
        sag=R-np.sqrt(R**2-xc**2-yc**2)
        near[:,4:]-=sag[:,None]
    if step:
        yc=(near[:,2]+near[:,3])/2
        # Terrain changes across strip; near/far DEMs genuinely differ.
        near[:,5]+=5.*np.sin(yc/250.)
    far=square_cells(np.arange(-R2,R2+GS2*.5,GS2),top=top)
    cx=(far[:,0]+far[:,1])/2
    cy=(far[:,2]+far[:,3])/2
    far=far[(abs(cx)>=R1)|(abs(cy)>=R1)]
    return near,far


@pytest.mark.parametrize("side,cell,filter_axis",[
    ("east",[R1,R1+GS2,-125,125,0,300],1),
    ("west",[-R1-GS2,-R1,-125,125,0,300],0),
    ("north",[-125,125,R1,R1+GS2,0,300],3),
    ("south",[-125,125,-R1-GS2,-R1,0,300],2),
])
@pytest.mark.parametrize("stepped", [False, True])
def test_variable_face_bridge_is_closed_with_correct_winding(side,cell,filter_axis,stepped):
    near, _ = scenario(step=stepped)
    vertices, faces = TR._bridge_mesh(cell, near, side, R)
    assert len(vertices)>8
    assert len(faces)>12
    model=GM.polydron_grav(vertices,faces,2670.)
    assert np.isfinite(model.compute_gz([0.,0.,350.],method="surface"))
    triangles=np.asarray([f[1:] for f in faces])
    from collections import Counter
    edges=Counter(tuple(sorted((a,b))) for face in triangles
                  for a,b in zip(face,(face[1],face[2],face[0])))
    assert all(count==2 for count in edges.values())
    # Each directed edge is used exactly once in each direction.
    oriented=Counter((a,b) for face in triangles
                     for a,b in zip(face,(face[1],face[2],face[0])))
    assert all(count==1 and oriented.get((b,a))==1
               for (a,b),count in oriented.items())


def test_outer_side_matches_next_coarse_polyhedron_face():
    near,_=scenario()
    bridge=[R1,R1+GS2,-125,125,0,300]
    next_far=[R1+GS2,R1+2*GS2,-125,125,0,300]
    xyz,faces=TR._bridge_mesh(bridge,near,"east",R)
    vf=BC._curved_cell_vertices(next_far,R)
    # The west face of the next ring uses vertices 0,3,4,7.
    target=vf[[0,3,4,7]]
    on_boundary=[i for i,p in enumerate(xyz)
                 if np.any(np.linalg.norm(target-p,axis=1)<1e-8)]
    assert len(on_boundary)==4
    parent_faces=[tuple(sorted(f[1:])) for f in faces
                  if all(x in on_boundary for x in f[1:])]
    assert len(parent_faces)==2
    # Standard far west-facet triangles (3,0,7),(0,4,7).
    for indices in ((3,0,7),(0,4,7)):
        coords=vf[list(indices)]
        ids=[next(i for i,p in enumerate(xyz)
                  if np.linalg.norm(p-v)<1e-8) for v in coords]
        assert tuple(sorted(ids)) in parent_faces


def test_corner_attached_to_both_neighbouring_bridges():
    near,_=scenario()
    cell=[R1,R1+GS2,R1,R1+GS2,0,300]
    corner,faces=TR._corner_mesh(cell,near,R,R1)
    model=GM.polydron_grav(corner,faces,2670.)
    assert np.isfinite(model.compute_gz([0,0,350]))
    near_corner=near[(abs(near[:,1]-R1)<1e-8)&
                     (abs(near[:,3]-R1)<1e-8)][0]
    np.testing.assert_allclose(corner[0],[R1,R1,near_corner[4]],atol=1e-9)
    np.testing.assert_allclose(corner[4],[R1,R1,near_corner[5]],atol=1e-9)


def test_whole_ring_uses_original_coarse_cells_without_subdivision():
    near,far=scenario()
    density=2670.
    total,ordinary,ordinary_density=TR._transition_polyhedral_gz(
        far,np.full(len(far),density),
        near,np.full(len(near),density),
        [0,0,350],R,R1)
    # 4 sides x 9 coarse cells + 4 corners = 40; 48 ordinary cells.
    assert len(ordinary)==len(far)-40
    assert len(ordinary_density)==len(ordinary)
    assert np.isfinite(total)


def test_bridge_mismatching_density_keeps_existing_cell_classification():
    # GDAL has already classified each cell. Even when a coastline falls
    # within a coarse far cell, do not split or replace that assignment.
    near,far=scenario()
    near_before,far_before=near.copy(),far.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        total,ordinary,remaining_density=TR._transition_polyhedral_gz(
            far,np.full(len(far),1030.-2670.),
            near,np.full(len(near),2670.),
            [0,0,350],R,R1)
    assert total==0.
    assert len(ordinary)==len(far)
    np.testing.assert_array_equal(ordinary,far)
    assert np.all(remaining_density==1030.-2670.)
    np.testing.assert_array_equal(near,near_before)
    np.testing.assert_array_equal(far,far_before)
    # Near density must never overwrite a far cell's GDAL classification.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        alternate,alt_ordinary,alt_d=TR._transition_polyhedral_gz(
            far,np.full(len(far),2670.),
            near,np.full(len(near),-500.),
            [0,0,350],R,R1)
    assert alternate==0.
    np.testing.assert_array_equal(alt_ordinary,far)
    assert np.all(alt_d==2670.)



@pytest.mark.parametrize("xsign,ysign", [
    (+1,+1),(+1,-1),(-1,+1),(-1,-1)
])
def test_all_four_corner_cells_share_identical_triangles_with_side_bridges(xsign,ysign):
    near,_=scenario()
    def bounds(sign):
        return (R1,R1+GS2) if sign>0 else (-R1-GS2,-R1)
    x0,x1=bounds(xsign)
    y0,y1=bounds(ysign)
    corner=[x0,x1,y0,y1,0,300]
    corner_xyz,corner_faces=TR._corner_mesh(corner,near,R,R1)
    # One side cell on the x side of each corner, one on its y side.
    yinner=(R1-GS2,R1) if ysign>0 else (-R1,-R1+GS2)
    xinner=(R1-GS2,R1) if xsign>0 else (-R1,-R1+GS2)
    neighbours=[
        ([x0,x1,*yinner,0,300],"east" if xsign>0 else "west"),
        ([*xinner,y0,y1,0,300],"north" if ysign>0 else "south"),
    ]
    def triangle_as_coords(xyz,faces):
        return [xyz[np.asarray(f[1:],dtype=int)] for f in faces]
    corner_tri=triangle_as_coords(corner_xyz,corner_faces)
    for side_cell,side in neighbours:
        side_xyz,side_faces=TR._bridge_mesh(side_cell,near,side,R)
        side_tri=triangle_as_coords(side_xyz,side_faces)
        # Exactly two triangles on the plane between corner and side.
        common=[
            point for point in corner_xyz
            if any(np.linalg.norm(point-q)<1e-8 for q in side_xyz)
        ]
        assert len(common)==4
        def internal(mesh):
            return [t for t in mesh if all(any(
                np.linalg.norm(p-q)<1e-8 for q in common) for p in t)]
        ta,tb=internal(corner_tri),internal(side_tri)
        assert len(ta)==len(tb)==2
        assert all(any(all(any(np.linalg.norm(p-q)<1e-8 for q in t2)
                           for p in t1) for t2 in tb) for t1 in ta)


def test_transition_rejects_missing_near_prism_edge_segments():
    near,_=scenario()
    bridge=[R1,R1+GS2,-125,125,0,300]
    subset=near[near[:,3]<-50]  # excludes several required intervals
    with pytest.raises(ValueError,match="incomplete|missing"):
        TR._bridge_mesh(bridge,subset,"east",R)


def test_bridge_is_internal_to_bouguer_correction():
    """Keep the R1 geometry colocated with Bouguer correction."""
    root = Path(__file__).resolve().parents[1] / "relgravpy"
    module = root / "_bouguer_correction.py"
    assert module.is_file()
    assert not (root / "_bouguer_transition.py").exists()
    source = module.read_text(encoding="utf-8")
    assert "R1 PRISM / SPHERICAL-POLYHEDRON CONFORMAL TRANSITION" in source
    names = {node.name for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef)}
    assert {"_outward_triangle_winding", "_bridge_mesh",
            "_corner_mesh", "_transition_polyhedral_gz"}.issubset(names)
    assert "tr._transition_polyhedral_gz" not in source
