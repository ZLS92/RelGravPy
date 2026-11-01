"""Validation of one Polygrav solver against exact Nagy prisms.

Near observations use the deterministic analytic polyhedron kernel;
distant observations use stable outward-triangle surface quadrature.
All reported gravity components are DOWN-positive in mGal.
Run: PYTHONPATH=tests:. pytest -v tests/test_polygrav_nagy.py
"""
import itertools
import numpy as np
import pytest

from test_bouguer_polyhedra import GM


@pytest.mark.parametrize("length,thickness", [
    (20.,10.),(200.,100.),(1000.,300.),(5000.,1500.),(10000.,200.),
])
@pytest.mark.parametrize("offset_factor", [0.,0.8,2.,8.,30.])
@pytest.mark.parametrize("density", [0.,1000.,2670.,-1800.])
def test_polygrav_matches_nagy_over_scales_distance_density(
        length,thickness,offset_factor,density):
    prism=np.array([[-length/2,length/2,-length/2,length/2,
                     -thickness,0.]],float)
    verts,faces=GM.prism_to_corners_faces(prism[0])
    observation=np.array([length*offset_factor,
                          -length*offset_factor*0.21,30.])
    reference=float(GM.nagy([observation],prism,density).gz()[0])
    poly=GM.Polygrav(verts,faces,density).compute_gz(observation)
    assert np.isfinite(poly)
    assert np.isfinite(reference)
    # Absolute floor 0.002 microgal protects far small sources; otherwise
    # require 0.015% relative agreement with Nagy.
    np.testing.assert_allclose(poly,reference,rtol=1.5e-4,atol=2e-6)


def test_polygrav_multi_observation_no_cached_edge_contamination():
    prism=[-500.,500.,-250.,250.,-300.,0.]
    vertices,faces=GM.prism_to_corners_faces(prism)
    model=GM.Polygrav(vertices,faces,2670.)
    stations=np.array([[0.,0.,20.],[0.,0.,300.],
                       [1500.,250.,150.],[10.,90.,60.]])
    combined=model.compute_gz(stations,method="analytic")
    individual=np.array([model.compute_gz(st,method="analytic")
                         for st in stations])
    reference=GM.nagy(stations,np.array([prism]),2670.).gz()
    np.testing.assert_allclose(combined,individual,atol=1e-9,rtol=1e-12)
    np.testing.assert_allclose(combined,reference,atol=2e-6,rtol=1e-4)


def test_polygrav_analytic_surface_agree_at_safe_distance():
    prism=[-100.,100.,-100.,100.,-200.,0.]
    corners,faces=GM.prism_to_corners_faces(prism)
    model=GM.Polygrav(corners,faces,2670)
    obs=np.array([1200.,500.,400.])
    analytic=model.compute_gz(obs,method="analytic")
    surface=model.compute_gz(obs,method="surface")
    reference=GM.nagy([obs],np.array([prism]),2670).gz()[0]
    np.testing.assert_allclose(surface,reference,atol=3e-7,rtol=5e-4)
    np.testing.assert_allclose(analytic,reference,atol=2e-6,rtol=1e-4)


def test_polygrav_linear_density_and_mirror_symmetry():
    v,f=GM.prism_to_corners_faces([-100,100,-100,100,-100,0])
    obs=[350.,0.,60.]
    g1=GM.Polygrav(v,f,2670).compute_gz(obs)
    g2=GM.Polygrav(v,f,5340).compute_gz(obs)
    gneg=GM.Polygrav(v,f,-2670).compute_gz(obs)
    np.testing.assert_allclose(g2,2*g1,atol=1e-10)
    np.testing.assert_allclose(gneg,-g1,atol=1e-10)
    np.testing.assert_allclose(GM.Polygrav(v,f,2670).compute_gz([-350.,0.,60.]),
                               g1,atol=1e-8)


def test_polygrav_arbitrary_facets_tessellated_cube():
    # Add a node on each top edge, deliberately increasing the number of
    # polygon vertices/facets without changing the represented prism volume.
    a=np.array([[-100,-100,-100],[100,-100,-100],[100,100,-100],
                [-100,100,-100],[-100,-100,0],[100,-100,0],
                [100,100,0],[-100,100,0],
                [0,-100,0],[100,0,0],[0,100,0],[-100,0,0]],float)
    faces=[
        [4,3,2,1,0],
        [8,4,8,5,9,6,10,7,11],
        [5,0,1,5,8,4],
        [5,1,2,6,9,5],
        [5,2,3,7,10,6],
        [5,3,0,4,11,7]
    ]
    # Each added node belongs to exactly two adjacent oriented faces.
    model=GM.Polygrav(a,faces,2670)
    stations=np.array([[0.,0.,30.],[350.,150.,300.]])
    actual=model.compute_gz(stations)
    prism=np.array([[-100.,100.,-100.,100.,-100.,0.]])
    expected=GM.nagy(stations,prism,2670).gz()
    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-6)


def test_polygrav_invalid_open_mesh_rejected():
    v,f=GM.prism_to_corners_faces([-100,100,-100,100,-100,0])
    with pytest.raises(ValueError,match="not closed"):
        GM.Polygrav(v,f[:-1],2670)



@pytest.mark.parametrize("station", [
    [0.,0.,80.], [25.,40.,-70.], [30.,-20.,-220.],
    [20.,70.,-10.], [400.,-50.,-100.], [1000.,0.,-300.]
])
def test_polygrav_inside_and_below_body_vs_nagy(station):
    prism=np.array([[-150.,150.,-120.,120.,-260.,0.]])
    vertices,faces=GM.prism_to_corners_faces(prism[0])
    actual=GM.Polygrav(vertices,faces,2670.).compute_gz(
        station,method="analytic")
    reference=float(GM.nagy([station],prism,2670.).gz()[0])
    np.testing.assert_allclose(actual,reference,rtol=1e-4,atol=2e-6)


def test_nagy_new_compute_gz_method_matches_existing_gz():
    prism=np.array([[-100.,100.,-100.,100.,-200.,0.]])
    model=GM.nagy([[0,0,50.],[200,0,30.]],prism,2670.)
    np.testing.assert_allclose(model.compute_gz(),model.gz(),rtol=0,atol=0)


def test_polygrav_far_field_point_mass_limit():
    prism=[-20.,20.,-30.,30.,-40.,0.]
    density=2670.
    v,f=GM.prism_to_corners_faces(prism)
    model=GM.Polygrav(v,f,density)
    obs=[120000.,-90000.,60000.]
    actual=model.compute_gz(obs)
    xyz=np.array([0.,0.,-20.])
    diff=np.asarray(obs)-xyz
    reference=(GM.G*density*(40*60*40)*1e5*
               diff[2]/np.linalg.norm(diff)**3)
    np.testing.assert_allclose(actual,reference,rtol=5e-7,atol=1e-13)
