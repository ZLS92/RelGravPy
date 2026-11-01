"""Independent 3D volume check of new arbitrary-facet R1 transition.

The reference integrates Newton's law directly over signed tetrahedra of
the complete arbitrary polygonal surface; the production code uses a
surface integral, so these are independent integration strategies.
"""
import json
import numpy as np

from test_bouguer_transition import TR, GM, R, R1, GS1, GS2, scenario

def independent_volume_gz(vertices, faces, obs, density=2670., order=5):
    v=np.asarray(vertices,float)
    facets=np.asarray([face[1:] for face in faces],dtype=int)
    centre=np.mean(v,axis=0)
    nodes,weights=np.polynomial.legendre.leggauss(order)
    nodes=(nodes+1)/2
    weights=weights/2
    u,vv,w=np.meshgrid(nodes,nodes,nodes,indexing="ij")
    wu,wv,ww=np.meshgrid(weights,weights,weights,indexing="ij")
    u,vv,w=u.ravel(),vv.ravel(),w.ravel()
    q=(wu*wv*ww).ravel()*(1-u)**2*(1-vv)
    station=np.asarray(obs,float)
    result=0.
    for ia,ib,ic in facets:
        a,b,c=v[[ia,ib,ic]]-centre
        signed_det=float(np.dot(a,np.cross(b,c)))
        xyz=(centre[None,:]+u[:,None]*a
             +((1-u)*vv)[:,None]*b
             +((1-u)*(1-vv)*w)[:,None]*c)
        disp=station[None,:]-xyz
        length=np.linalg.norm(disp,axis=1)
        result+=signed_det*np.sum(q*disp[:,2]/length**3)
    return float(GM.G*density*1e5*result)


def run():
    obs=np.array([0.,0.,350.])
    near,far=scenario(step=True)
    comparisons=[]
    for side,cell in (
        ("east",[R1,R1+GS2,-125,125,0,300]),
        ("west",[-R1-GS2,-R1,-125,125,0,300]),
        ("north",[-125,125,R1,R1+GS2,0,300]),
        ("south",[-125,125,-R1-GS2,-R1,0,300]),
    ):
        vertices,faces=TR._bridge_mesh(cell,near,side,R)
        actual=GM.polydron_grav(vertices,faces,2670).compute_gz(
            obs,method="surface")
        independent=independent_volume_gz(vertices,faces,obs,order=6)
        comparisons.append(dict(
            side=side,vertices=len(vertices),triangles=len(faces),
            numerical_mGal=actual,volume_mGal=independent,
            difference_uGal=1000*(actual-independent)
        ))
    corner=[R1,R1+GS2,R1,R1+GS2,0,300]
    vertices,faces=TR._corner_mesh(corner,near,R,R1)
    actual=GM.polydron_grav(vertices,faces,2670).compute_gz(
        obs,method="surface")
    independent=independent_volume_gz(vertices,faces,obs,order=6)
    comparisons.append(dict(
        side="NE-corner",vertices=len(vertices),triangles=len(faces),
        numerical_mGal=actual,volume_mGal=independent,
        difference_uGal=1000*(actual-independent)
    ))
    worst=max(abs(c["difference_uGal"]) for c in comparisons)
    report=dict(cases=comparisons,max_abs_diff_uGal=worst)
    print("R1_CONFORMAL_VOLUME_BENCHMARK",json.dumps(report),flush=True)
    assert worst < 0.05, report

if __name__=="__main__":
    run()
