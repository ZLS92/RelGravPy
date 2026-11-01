"""Verify the exact mesh interface between the new R1 ring and normal far cells.

A conformal arbitrary-facet bridge uses the original four vertices and
two triangles on the outer face of every coarse far parent cell; no new
coarse-ring nodes occur. This replaces the former 78-hanging-node result
from subdividing the first ring.
"""
import json
import numpy as np

from test_bouguer_polyhedra import BC
from test_bouguer_transition import TR

RADIUS=6371000.


def square_cells(edges, top=300.):
    w,s=np.meshgrid(edges[:-1],edges[:-1],indexing="xy")
    e,n=np.meshgrid(edges[1:],edges[1:],indexing="xy")
    return np.column_stack((w.ravel(),e.ravel(),s.ravel(),n.ravel(),
                            np.zeros(w.size),np.full(w.size,top)))


def expected_spherical_face(cell, side):
    """Four vertices and exact two standard triangles on an outer face."""
    v=BC._curved_cell_vertices(cell,RADIUS)
    if side=="east":
        tris=((1,2,6),(1,6,5))
    elif side=="west":
        tris=((3,0,7),(0,4,7))
    elif side=="north":
        tris=((2,3,6),(3,7,6))
    elif side=="south":
        tris=((0,1,5),(0,5,4))
    return [v[list(t)] for t in tris]


def close_triangle(a,b,tol=2e-8):
    return all(any(np.linalg.norm(vertex-candidate)<tol
                   for candidate in b) for vertex in a)


def diagnose(gs1=100, R1_requested=5000, gs2_requested=500,
             R2_requested=10000.):
    gs1,R1,gs2,R2=BC.adj_mesh(
        gs1,R1_requested,gs2_requested,R2_requested)
    fine=np.linspace(-R1,R1,int(round(2*R1/gs1))+1)
    coarse=np.linspace(-R2,R2,int(round(2*R2/gs2))+1)
    near=square_cells(fine)
    xc=(near[:,0]+near[:,1])/2
    yc=(near[:,2]+near[:,3])/2
    shift=RADIUS-np.sqrt(RADIUS**2-xc**2-yc**2)
    near[:,4:]-=shift[:,None]
    all_far=square_cells(coarse)
    x=(all_far[:,0]+all_far[:,1])/2
    y=(all_far[:,2]+all_far[:,3])/2
    far=all_far[(abs(x)>=R1)|(abs(y)>=R1)]
    errors=0
    bridges=0
    corners=0
    total_triangles=0
    for cell in far:
        w,e,s,n=cell[:4]
        mx,my=(w+e)/2,(s+n)/2
        side=None
        if abs(w-R1)<1e-6 and -R1<my<R1:side="east"
        elif abs(e+R1)<1e-6 and -R1<my<R1:side="west"
        elif abs(s-R1)<1e-6 and -R1<mx<R1:side="north"
        elif abs(n+R1)<1e-6 and -R1<mx<R1:side="south"
        corner=(abs(w-R1)<1e-6 or abs(e+R1)<1e-6) and (
            abs(s-R1)<1e-6 or abs(n+R1)<1e-6)
        if side is None and not corner:
            continue
        if corner:
            corners+=1
            vertices,faces=TR._corner_mesh(cell,near,RADIUS,R1)
        else:
            bridges+=1
            vertices,faces=TR._bridge_mesh(cell,near,side,RADIUS)
        triangles=[vertices[np.asarray(f[1:],dtype=int)] for f in faces]
        total_triangles+=len(triangles)
        outward=("east","west","north","south") if corner else (side,)
        for exterior in outward:
            if corner:
                # For corner cells two of the four faces are truly outer.
                if exterior=="east" and abs(e-R1-gs2)>1e-5:continue
                if exterior=="west" and abs(w+R1+gs2)>1e-5:continue
                if exterior=="north" and abs(n-R1-gs2)>1e-5:continue
                if exterior=="south" and abs(s+R1+gs2)>1e-5:continue
            for expected in expected_spherical_face(cell,exterior):
                if not any(close_triangle(expected,tri) for tri in triangles):
                    errors+=1
    result=dict(
        requested=dict(gs1_m=gs1,R1_m=R1_requested,
                       gs2_m=gs2_requested,R2_m=R2_requested),
        adjusted=dict(gs1_m=gs1,R1_m=R1,gs2_m=gs2,R2_m=R2),
        transition_parent_cells=bridges,
        corner_parent_cells=corners,
        total_transition_triangles=total_triangles,
        unmatched_outer_triangles=errors,
        external_boundary_conformal=(errors==0),
    )
    print("R1_OUTER_SEAM",json.dumps(result),flush=True)
    assert bridges>0 and corners==4 and errors==0
    return result


if __name__=="__main__":
    diagnose()
