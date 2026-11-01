"""Numerical checks for the independent curved-polyhedron Bouguer backend.

Load only the numerical functions with AST to avoid optional GDAL/OGR imports
during continuous integration. This follows the existing min-curvature tests.
"""
import ast
from pathlib import Path
import types

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
    if gm is not None:
        module.gm = gm
    else:
        module.G = 6.6743e-11
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
         module.__dict__)
    return module


GM = _load_numerical_subset(
    ROOT / "relgravpy" / "grav_modelling.py",
    {"anglegravi_exact", "build_edge_table", "gravicalc2_full",
     "nagy_fz", "nagy_gz"},
    class_names={"nagy"},
)
BC = _load_numerical_subset(
    ROOT / "relgravpy" / "_bouguer_correction.py",
    {"_curved_cell_vertices", "_surface_integral_polyhedral_gz",
     "_polyhedral_gz", "_point_mass_gz", "_ring_polyhedral_effect"},
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
    rectangular = GM.nagy([obs], prism, density).gz()[0]
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
    nagy = GM.nagy([obs], shifted, 2670.0).gz()[0]
    curved = BC._polyhedral_gz(prism, [2670.0], obs, radius)
    assert nagy > 0.0002
    assert curved > 0.0002
    np.testing.assert_allclose(curved, nagy, rtol=0.03, atol=2e-6)


def test_surface_quadrature_matches_nagy_at_large_radius():
    """Ensure numerical surface integration gives a stable distant field."""
    prism = [[7750, 8000, -125, 125, 0, 300.0]]
    obs = [0, 0, 700.0]
    nagy = GM.nagy([obs], np.array(prism), 2670.0).gz()[0]
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
