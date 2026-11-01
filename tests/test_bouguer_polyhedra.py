"""Numerical checks for the independent curved-polyhedron Bouguer backend.

Load only the numerical functions with AST to avoid optional GDAL/OGR imports
during continuous integration. This follows the existing min-curvature tests.
"""
import ast
from pathlib import Path
import types
import warnings

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_numerical_subset(path, function_names, class_names=(), gm=None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            nodes.append(node)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"_CURVED_FACES", "_POLY_G_RATIO"}
            for target in node.targets
        ):
            nodes.append(node)
    assert len(nodes) == len(function_names) + len(class_names) + (
        2 if gm is not None else 0
    )
    module = types.ModuleType("_isolated_polyhedral_test")
    module.np = np
    module.warnings = warnings
    module.R_wgs84 = 6371000.
    if gm is not None:
        module.gm = gm
    else:
        module.G = 6.6743e-11
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
         module.__dict__)
    return module


GM = _load_numerical_subset(
    ROOT / "relgravpy" / "grav_modelling.py",
    {"anglegravi_exact", "build_edge_table", "prism_to_corners_faces",
     "nagy_fz", "nagy_gz"},
    class_names={"prism_grav", "polydron_grav"},
)
BC = _load_numerical_subset(
    ROOT / "relgravpy" / "_bouguer_correction.py",
    {"_curved_cell_vertices", "_curved_cell_vertices_batch",
     "_surface_integral_vertices_gz", "_surface_integral_polyhedral_gz",
     "_polyhedral_gz", "_outward_triangle_winding", "_bridge_mesh",
     "_corner_mesh", "_transition_polyhedral_gz",
     "_point_mass_gz", "_ring_polyhedral_effect",
     "adj_mesh", "_bgc_terrain_features", "_bgc_station_terrain_features",
     "_bgc_mesh_error_proxy", "optimize_bgc_mesh",
     "_refine_interface_cells", "_refine_far_interface",
     "_refine_near_interface"},
    gm=GM,
)


def test_curved_vertices_follow_reference_sphere():
    radius = 6_371_000.0
    p = [1000, 1250, 500, 750, 0, 140]
    vertices = BC._curved_cell_vertices(p, radius)
    assert vertices.shape == (8, 3)
    np.testing.assert_allclose(
        np.linalg.norm(vertices[:4] + np.array([0, 0, radius]), axis=1),
        radius, atol=1e-6
    )
    np.testing.assert_allclose(
        np.linalg.norm(vertices[4:] + np.array([0, 0, radius]), axis=1),
        radius + 140, atol=1e-6
    )


def test_neighbouring_curved_cells_share_lower_edge():
    radius = 6_371_000.0
    left = BC._curved_cell_vertices([0, 100, 200, 300, 0, 20], radius)
    right = BC._curved_cell_vertices([100, 200, 200, 300, 0, 30], radius)
    np.testing.assert_allclose(left[[1, 2]], right[[0, 3]], atol=1e-10)


@pytest.mark.parametrize("observation", [(0, 0, 10), (0, 0, 100), (200, 50, 10)])
def test_large_radius_polyhedron_agrees_with_nagy(observation):
    prism = np.array([[-100, 100, -100, 100, -100, 0]], dtype=float)
    density = 2670.0
    obs = np.array(observation, float)
    rectangular = GM.prism_grav([obs], prism, density).gz()[0]
    curved = BC._polyhedral_gz(prism, [density], obs, 1.0e10)
    np.testing.assert_allclose(curved, rectangular, rtol=2e-5, atol=2e-5)
    assert curved > 0  # Down-positive field for mass below the observation


def test_negative_density_reverses_gravity_sign():
    prism = [[500, 1000, 300, 800, 0, 100]]
    gpos = BC._polyhedral_gz(prism, [2670.0], [0, 0, 120], 6371000.0)
    gneg = BC._polyhedral_gz(prism, [-2670.0], [0, 0, 120], 6371000.0)
    np.testing.assert_allclose(gpos, -gneg, rtol=0, atol=1e-8)


def test_point_mass_local_up_down_convention():
    radius = 6_371_000.0
    sources = np.array([[8000.0], [0.0], [0.0]])
    masses = np.array([1e11])
    g = BC._point_mass_gz(sources, masses, [0, 0, 100], radius)
    assert np.isfinite(g)
    assert g > 0


def test_invalid_cell_rejected():
    with pytest.raises(ValueError, match="positive dimensions"):
        BC._curved_cell_vertices([200, 100, 0, 100, 0, 50], 6371000.0)


def test_empirical_ring_is_finite_without_harmonica():
    radius = 6_371_000.0
    ring = [[-0.10, 0.10, 0.10, 0.12, radius, radius + 200]]
    g = BC._ring_polyhedral_effect(ring, [2670], radius, 200)
    assert np.isfinite(g)


def test_small_far_cell_no_longer_suffers_cancellation():
    """The original polyhedral solver nearly erased this 0.38-uGal signal."""
    radius = 6_371_000.0
    prism = np.array([[7750, 8000, -125, 125, 0, 300.0]], float)
    obs = np.array([0, 0, 700.0], float)
    sag = (8000**2 + 125**2)/(2*radius)
    shifted = prism.copy()
    shifted[:,4:] -= sag
    nagy = GM.prism_grav([obs], shifted, 2670.0).gz()[0]
    curved = BC._polyhedral_gz(prism, [2670.0], obs, radius)
    assert nagy > 0.0002
    assert curved > 0.0002
    np.testing.assert_allclose(curved, nagy, rtol=0.03, atol=2e-6)


def test_surface_quadrature_matches_nagy_at_large_radius():
    """Ensure numerical surface integration gives a stable distant field."""
    prism = [[7750, 8000, -125, 125, 0, 300.0]]
    obs = [0, 0, 700.0]
    nagy = GM.prism_grav([obs], np.array(prism), 2670.0).gz()[0]
    surf = BC._surface_integral_polyhedral_gz(
        np.array(prism), np.array([2670.0]), obs, 1e10)
    np.testing.assert_allclose(surf, nagy, rtol=3e-4, atol=2e-7)


def test_bgc_public_api_names_and_output_columns():
    """Experimental Bouguer correction API must be internally consistent."""
    path = ROOT / "relgravpy" / "_bouguer_correction.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"bgc", "run_bgc", "load_bgc"}.issubset(names)
    assert not {"te", "run_te", "load_te"}.intersection(names)
    assert "bgc(" in src
    assert "result = load_bgc(" in src
    for field in ("bgc_near", "bgc_far", "bgc_tot"):
        assert field in src
    assert "'te_near'" not in src
    assert "'te_far'" not in src
    assert "'te_tot'" not in src
    assert "'Bouguer_correction'" in src
    assert "'Te_correction'" not in src


def test_bgc_cli_entrypoint_calls_correct_function():
    path = ROOT / "relgravpy" / "_bouguer_correction.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    main_blocks = [node for node in tree.body if isinstance(node, ast.If)
                   and isinstance(node.test, ast.Compare)]
    assert main_blocks
    main = main_blocks[-1]
    calls = [n.func.id for n in ast.walk(main)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "bgc" in calls
    assert "te" not in calls



def test_r1_interface_splits_coarse_face_at_near_grid_nodes():
    radius = 6_371_000.
    near_x = np.arange(-1000., 1000.1, 250.)
    near_y = np.arange(-1000., 1000.1, 250.)
    far = np.array([
        [1000., 1500., -1000., -500., 0., 300.],
        [1000., 1500., -500., 0., 0., 300.],
        [1000., 1500., 1000., 1500., 0., 300.],  # corner only
        [-500., 0., 1000., 1500., 0., 300.],     # north, split x
    ])
    density = np.array([2670., 2670., 2670., -1300.])
    new, dens = BC._refine_far_interface(far, density, near_x, near_y, 1000.)
    # Two east seam cells -> 4; north seam cell -> 2; corner -> 1.
    assert len(new) == 7
    assert len(dens) == 7
    np.testing.assert_allclose(
        np.sum(np.prod(new[:, [1,3,5]]-new[:, [0,2,4]], axis=1) * dens),
        np.sum(np.prod(far[:, [1,3,5]]-far[:, [0,2,4]], axis=1) * density),
        rtol=1e-12
    )
    right = new[
        (np.abs(new[:,0]-1000) < 1e-9) &
        (np.abs(new[:,2]+1000) < 1e-9) &
        (np.abs(new[:,3]+750) < 1e-9)
    ][0]
    left = np.array([750.,1000.,-1000.,-750.,0.,300.])
    vnear = BC._curved_cell_vertices(left, radius)
    vfar = BC._curved_cell_vertices(right, radius)
    np.testing.assert_allclose(
        vnear[[1,2,5,6]], vfar[[0,3,4,7]], rtol=0, atol=1e-9
    )


def test_r1_interface_subdivision_is_idempotent():
    far = np.array([[1000., 1500., 0., 500., 0., 300.]])
    edge = np.linspace(-1000,1000,9)
    once, den1 = BC._refine_far_interface(far,[2670],edge,edge,1000)
    twice, den2 = BC._refine_far_interface(once,den1,edge,edge,1000)
    np.testing.assert_allclose(once,twice)
    np.testing.assert_allclose(den1,den2)


def test_curved_near_switch_exposes_nagy_fallback():
    path = ROOT / "relgravpy" / "_bouguer_correction.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    bgc_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "bgc")
    args = {arg.arg: default for arg, default in zip(
        bgc_fn.args.args[-len(bgc_fn.args.defaults):], bgc_fn.args.defaults)}
    assert "near_geometry" in args
    assert "refine_interface" in args
    assert '"auto"' in src
    assert 'spherical_near = curved and near_geometry == "spherical"' in src
    assert "pr1[0][finite_mass]" in src



def test_r1_both_sides_share_full_triangulated_interface():
    """Non-commensurate 250-m / 375-m grids share same radial triangles."""
    radius = 6_371_000.
    near_edges = np.arange(-1000., 1000.1, 250.)
    far_edges = np.array([-1000., -625., -250., 125., 500., 875., 1250.])
    near = np.array([
        [750, 1000, -1000, -750, 0, 300],
        [750, 1000, -750, -500, 0, 300],
    ],float)
    far = np.array([
        [1000, 1500, -1000, -625, 0, 300],
        [1000, 1500, -625, -250, 0, 300],
    ],float)
    a, ad = BC._refine_near_interface(near,[2670,2670],far_edges,far_edges,1000)
    b, bd = BC._refine_far_interface(far,[2670,2670],near_edges,near_edges,1000)
    # Only compare portions covered by BOTH sources: y=-1000 to -500.
    near_intervals = sorted((round(p[2],8),round(p[3],8)) for p in a
                            if abs(p[1]-1000.) < 1e-9)
    far_intervals = sorted((round(p[2],8),round(p[3],8)) for p in b
                           if p[3]<=-500.+1e-9)
    assert near_intervals == far_intervals
    assert len(near_intervals)>2
    for na, fa in zip(sorted((p for p in a if abs(p[1]-1000.)<1e-9),
                             key=lambda p:p[2]),
                      sorted((p for p in b if p[3]<=-500.+1e-9),
                             key=lambda p:p[2])):
        vn = BC._curved_cell_vertices(na,radius)
        vf = BC._curved_cell_vertices(fa,radius)
        np.testing.assert_allclose(vn[[1,2,5,6]],vf[[0,3,4,7]],atol=1e-9,rtol=0)
        # Canonical east/west diagonals are the same 3-D segment.
        np.testing.assert_allclose(vn[[1,6]],vf[[0,7]],atol=1e-9,rtol=0)


def test_r1_corner_near_cell_splits_on_both_axes():
    near = [[750,1000,750,1000,0,300]]
    far_knots = np.array([800,900.])
    cells,_=BC._refine_near_interface(near,[2670],far_knots,far_knots,1000.)
    assert len(cells)==9
    assert np.isclose(sum((p[1]-p[0])*(p[3]-p[2]) for p in cells),
                      250.**2)



@pytest.mark.parametrize("station_height", [300., 320., 250.])
def test_spherical_near_cell_at_surface_above_and_inside(station_height):
    """Avoid singular/incorrect near-cell gravity at realistic station heights."""
    prism = np.array([[-50., 50., -50., 50., 0., 300.]])
    station = np.array([0., 0., station_height])
    curved = BC._polyhedral_gz(prism, [2670.], station, 6_371_000.)
    planar = GM.prism_grav([station], prism, [2670.]).gz()[0]
    assert np.isfinite(curved)
    np.testing.assert_allclose(curved, planar, rtol=1e-3, atol=1e-4)


def test_r1_north_south_radial_faces_share_diagonal():
    radius=6_371_000.
    southern=BC._curved_cell_vertices([-100,100,750,1000,0,300],radius)
    northern=BC._curved_cell_vertices([-100,100,1000,1250,0,300],radius)
    # Northern side of southern cell: left bottom idx3 -> right top idx6
    # Southern side of northern cell: left bottom idx0 -> right top idx5
    np.testing.assert_allclose(southern[[3,6]],northern[[0,5]],
                               atol=1e-9,rtol=0)


def test_r1_transition_density_mismatch_rejected_in_single_module():
    near = [[750,1000,-125,125,-0.08,299.92]]
    far = [[1000,1375,-125,125,0,300]]
    with pytest.raises(ValueError, match="density"):
        BC._transition_polyhedral_gz(
            far, [1030], near, [2670],
            [0, 0, 320], 6371000, 1000)


def test_near_prism_shift_uses_cell_center():
    code = (ROOT/"relgravpy"/"_bouguer_correction.py").read_text()
    tree = ast.parse(code)
    for name in ("mesh_ls", "mesh_sb"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name==name)
        snippet = ast.get_source_segment(code,fn)
        assert "R - np.sqrt(radial)" in snippet
        assert "Mxc**2 - Myc**2" in snippet
        assert "Mxb[Mxb<0]" not in snippet


def test_nagy_prism_centre_base_intersects_spherical_reference():
    path = ROOT / "relgravpy" / "_bouguer_correction.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "mesh_ls")
    ns = {"np": np, "R_wgs84": 6371000.0}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    x = np.array([[5000., 5100.], [5000., 5100.]])
    y = np.array([[1000., 1000.], [1100., 1100.]])
    cells, density = ns["mesh_ls"](
        x, y, np.array([[300.]]), R=6371000., z_shift=True)
    assert cells.shape == (1, 6)
    xc, yc = (cells[0,0]+cells[0,1])/2, (cells[0,2]+cells[0,3])/2
    spherical_base = np.sqrt(6371000.**2 - xc**2 - yc**2) - 6371000.
    np.testing.assert_allclose(cells[0,4], spherical_base, atol=1e-9)
    np.testing.assert_allclose(cells[0,5] - cells[0,4], 300., atol=1e-10)


def test_station_uses_caller_reference_radius_for_transition():
    src=(ROOT/"relgravpy"/"_bouguer_correction.py").read_text()
    tree=ast.parse(src)
    fn=next(n for n in tree.body
            if isinstance(n,ast.FunctionDef) and n.name=="st_loop")
    body=ast.get_source_segment(src,fn)
    assert "R = float(R)" in body
    assert "R = utl.local_sph_raduis(yg)" not in body


@pytest.mark.parametrize("gs1,r1,gs2,r2", [
    (50, 2000, 250, 167000),
    (50, 5000, 250, 10000),
    (50, 3000, 250, 166735),
    (100, 8000, 500, 167000),
])
def test_adj_mesh_minimal_upward_radii_and_small_step_adjustment(gs1,r1,gs2,r2):
    a,R1,b,R2=BC.adj_mesh(gs1,r1,gs2,r2)
    assert a == gs1
    assert all(isinstance(v,int) for v in (a,R1,b,R2))
    assert all(v==int(v) for v in (a,R1,b,R2))
    assert gs1%2 == b%2 == 0
    assert R1 >= r1-1e-7
    assert R2 >= r2-1e-7
    assert R2 > R1
    assert abs(b/gs2-1.0) <= 0.0500001
    assert np.isclose(2*R1/gs1, round(2*R1/gs1), atol=1e-9)
    assert round(2*R1/gs1) % 2 == 1
    assert np.isclose(2*R1/b, round(2*R1/b), atol=1e-9)
    assert round(2*R1/b) % 2 == 1
    assert np.isclose(2*R2/b, round(2*R2/b), atol=1e-9)
    assert round(2*R2/b) % 2 == 1
    assert R2 - R1 >= b - 1e-8
    np.testing.assert_allclose(BC.adj_mesh(a,R1,b,R2), (a,R1,b,R2),
                               atol=1e-7,rtol=1e-12)


def test_adj_mesh_smallest_radius_satisfying_step_tolerance():
    gs1, r1, gs2, r2 = 50, 2000, 250, 167000
    _,R1,s2,_=BC.adj_mesh(gs1,r1,gs2,r2)
    assert abs(s2/gs2-1) <= 0.05+1e-12
    for r_before in np.arange(gs1/2,R1-1e-6,gs1):
        if r_before<r1:
            continue
        odd_base=max(1,int(np.floor((2*r_before/gs2-1)/2))*2+1)
        possible=[2*r_before/m for m in (odd_base,odd_base+2)]
        assert all(abs(v/gs2-1)>0.05-1e-10 for v in possible)


def test_adj_mesh_rejects_bad_steps_and_tolerances():
    for args in ((0,2000,500,167000),(-25,2000,500,167000),
                 (25,0,500,167000)):
        with pytest.raises(ValueError):
            BC.adj_mesh(*args)
    with pytest.raises(ValueError):
        BC.adj_mesh(25,2000,500,167000,max_gs2_change=-0.01)


def test_adj_mesh_avoids_excessive_r1_growth_on_extreme_coarse_input():
    with pytest.raises(ValueError, match="No EXACT integer-metre aligned"):
        BC.adj_mesh(7000.,42000.,25000.,167000.)
    # Relaxation is explicit, never a silent 17.5 km R1 expansion.
    a,R1,b,R2=BC.adj_mesh(7000.,42000.,25000.,167000.,
                          max_R1_increase=25000.)
    assert a==7000.
    assert R1>=42000.
    assert R2>=167000.
    assert abs(b/25000.-1)<0.05


@pytest.mark.parametrize("invalid", [
    (125,5000,312,10000),       # odd gs1 incompatible with integer R1
    (100,5000,312.5,10000),     # decimal gs2 forbidden
    (100,5000.5,312,10000),     # decimal R1 forbidden
    (100,5000,312,10000.5),     # decimal R2 forbidden
    (100.5,5000,312,10000),     # decimal gs1 forbidden
])
def test_adj_mesh_rejects_noninteger_or_odd_near_steps(invalid):
    with pytest.raises(ValueError,match="integer|EVEN"):
        BC.adj_mesh(*invalid)


def test_adj_mesh_keeps_exact_centred_integer_r1_r2():
    gs1,R1,gs2,R2=BC.adj_mesh(50,5000,250,10000)
    assert (gs1,R1,gs2,R2)==(50,5125,250,10125)
    assert 2*R1//gs1 % 2 == 1
    assert 2*R1//gs2 % 2 == 1
    assert 2*R2//gs2 % 2 == 1
    assert (2*R1)%gs1==(2*R1)%gs2==0
    assert BC.adj_mesh(gs1,R1,gs2,R2)==(gs1,R1,gs2,R2)
