"""Quantify the remaining T-junctions at the OUTER edge of the R1 strip.

Run without GDAL:
PYTHONPATH=tests:. python -u tests/diagnose_r1_outer_seam.py

This is a diagnostic, not a closure-certifying test. It should explicitly
report nonconforming vertices until a coarsening / transition cell is built.
"""
import json
import numpy as np

from test_bouguer_polyhedra import BC


def square_cells(edges, top=300.):
    w,s=np.meshgrid(edges[:-1],edges[:-1],indexing="xy")
    e,n=np.meshgrid(edges[1:],edges[1:],indexing="xy")
    return np.column_stack((w.ravel(),e.ravel(),s.ravel(),n.ravel(),
                            np.zeros(w.size),np.full(w.size,top)))


def diagnose(gs1=125., R1_requested=5000., gs2_requested=312.5,
             R2_requested=10000., radius=6371000.):
    gs1,R1,gs2,R2 = BC.adj_mesh(gs1,R1_requested,gs2_requested,R2_requested)
    fine = np.linspace(-R1,R1,int(round(2*R1/gs1))+1)
    coarse=np.linspace(-R2,R2,int(round(2*R2/gs2))+1)
    near=square_cells(fine)
    far_all=square_cells(coarse)
    cx=(far_all[:,0]+far_all[:,1])/2
    cy=(far_all[:,2]+far_all[:,3])/2
    far=far_all[(abs(cx)>=R1) | (abs(cy)>=R1)]
    refined,_=BC._refine_far_interface(far,np.ones(len(far)),fine,fine,R1)
    # East-facing portion of the outer seam x=R1+gs2, y between +/-R1.
    x_outer=R1+gs2
    tol=1e-6
    transitional=refined[(abs(refined[:,0]-R1)<tol)&
                          (abs(refined[:,1]-x_outer)<tol)&
                          ((refined[:,2]+refined[:,3])/2 <R1)&
                          ((refined[:,2]+refined[:,3])/2 >-R1)]
    next_ring=far[(abs(far[:,0]-x_outer)<tol)&
                  (abs((far[:,2]+far[:,3])/2)<R1)]
    child_nodes=np.unique(np.r_[transitional[:,2],transitional[:,3]])
    parent_nodes=np.unique(np.r_[next_ring[:,2],next_ring[:,3]])
    unmatched=np.array([t for t in child_nodes
                        if np.min(abs(parent_nodes-t))>tol])
    largest_offset=0.
    for y in unmatched:
        match=next_ring[(next_ring[:,2]<y)&(y<next_ring[:,3])]
        if not len(match):
            continue
        parent=match[0]
        v=BC._curved_cell_vertices(parent,radius)
        fraction=(y-parent[2])/(parent[3]-parent[2])
        # Parent's west face (inner side) bottom edge is a straight segment;
        # refined child has a curved-sphere vertex at precisely (x_outer,y).
        chord=(1-fraction)*v[0]+fraction*v[3]
        source=BC._curved_cell_vertices(
            [x_outer,x_outer+gs2,y,y+gs1,0,300],radius)[0]
        largest_offset=max(largest_offset,float(np.linalg.norm(source-chord)))
    report=dict(
        requested=dict(gs1_m=gs1,R1_m=R1_requested,
                       gs2_m=gs2_requested,R2_m=R2_requested),
        adjusted=dict(gs1_m=gs1,R1_m=R1,gs2_m=gs2,R2_m=R2),
        transitional_outer_segments=len(transitional),
        next_ring_segments=len(next_ring),
        unmatched_outer_vertices=len(unmatched),
        maximum_chord_mismatch_mm=1000*largest_offset,
        certified_watertight=(len(unmatched)==0 and largest_offset<1e-9),
    )
    print("R1_OUTER_SEAM",json.dumps(report),flush=True)
    return report


if __name__=="__main__":
    diagnose()
