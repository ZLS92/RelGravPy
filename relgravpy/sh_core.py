
"""Independent spherical harmonic utilities for RelGravPy.

Uses NumPy only. Fully-normalized (4-pi) geodetic coefficients, without
Condon-Shortley phase. No pyshtools or Numba dependency.

The high-degree kernel is deliberately a reference implementation; benchmark
memory and runtime before using global degree-10,800 synthesis.
"""
from pathlib import Path
from urllib.request import urlopen
import warnings
import numpy as np

A_WGS84 = 6378137.0
F_WGS84 = 1.0 / 298.257223563
GM_WGS84 = 3.986004418e14
OMEGA_WGS84 = 7.292115e-5

MODEL_URLS = {
    "EIGEN-6C4": "https://icgem.gfz.de/getmodel/gfc/7fd8fe44aa1518cd79ca84300aef4b41ddb2364aef9e82b7cdaabdb60a9053f1/EIGEN-6C4.gfc",
    "EIGENC4": "https://icgem.gfz.de/getmodel/gfc/7fd8fe44aa1518cd79ca84300aef4b41ddb2364aef9e82b7cdaabdb60a9053f1/EIGEN-6C4.gfc",
    "WHU-CASM-UGM2025_2159": "https://icgem.gfz.de/getmodel/gfc/82962bb8156bfbf074b8f97bf558ac0e0ee5e481c864a29441fb0f68ddb26be2/WHU-CASM-UGM2025_2159.gfc",
}
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "shm_data"



def sh_read_gfc(path, lmax=None):
    """Read a static, fully normalized ICGEM gravity model from a .gfc file.
    
    Only 'gfc' records are supported. The presence of time-varying
    'gfct', 'trnd', 'asin', or 'acos' records raises NotImplementedError.
    Coefficients are read incrementally from the text file.
    
    Parameters
    ----------
    path : str or pathlib.Path
        Path to a local ICGEM .gfc file.
    lmax : int or None, optional
        Highest degree retained. None uses the file's max_degree;
        requests above that limit are clipped with a warning.
    
    Returns
    -------
    dict
        'c', 's' : ndarray, shape (L+1, L+1)
            Dimensionless fully normalized cosine/sine coefficients,
            with m > n entries zero.
        'gm' : float
            Model GM in m**3/s**2.
        'radius' : float
            Model reference radius in metres.
        'lmax', 'max_degree' : int
            Retained and advertised maximum degree.
        'modelname' : str
            Header model name, or the file stem.
        'header' : dict[str, str]
            Original header values indexed by lowercase keys.
    
    Raises
    ------
    ValueError
        When 'end_of_head' is missing or normalization is not
        'fully_normalized'.
    NotImplementedError
        If time-dependent coefficient records are present.
    
    Examples
    --------
    >>> from relgravpy.sh_core import sh_read_gfc
    >>> model = sh_read_gfc("shm_data/EIGEN-6C4.gfc", lmax=60)
    >>> model["c"].shape
    (61, 61)
    
    Notes
    -----
    This parser does not reconcile tide systems or reference frames."""
    # Read the ICGEM header before allocating degree/order coefficient arrays.
    path=Path(path)
    header={}
    in_head=True
    c=s=None
    available=None
    degree=None
    dynamic=False
    with path.open("rt",encoding="utf-8",errors="replace") as stream:
        for line in stream:
            parts=line.strip().split()
            if not parts or parts[0].startswith(("#","%")):
                continue
            if in_head:
                if parts[0].lower()=="end_of_head":
                    in_head=False
                    norm=header.get("norm","").lower().replace("-","_")
                    if norm not in ("fully_normalized","fully normalized",
                                    "fullynormalized"):
                        raise ValueError("Only fully_normalized ICGEM supported")
                    available=int(header["max_degree"])
                    degree=sh_resolve_degree(lmax,available)
                    c=np.zeros((degree+1,degree+1),dtype=float)
                    s=np.zeros_like(c)
                elif len(parts)>1:
                    header[parts[0].lower()]=" ".join(parts[1:])
                continue
            tag=parts[0].lower()
            # Never silently interpret time-dependent coefficients as static.
            if tag in ("gfct","trnd","asin","acos"):
                dynamic=True
            if tag!="gfc" or len(parts)<5:
                continue
            n,m=int(parts[1]),int(parts[2])
            if n<=degree and 0<=m<=n:
                c[n,m]=float(parts[3].replace("D","E").replace("d","e"))
                s[n,m]=float(parts[4].replace("D","E").replace("d","e"))
    if in_head:
        raise ValueError("ICGEM file missing end_of_head")
    if dynamic:
        raise NotImplementedError("Time-varying ICGEM coefficients require an epoch")
    return {"c":c,"s":s,
            "gm":float(header["earth_gravity_constant"].replace("D","E")),
            "radius":float(header["radius"].replace("D","E")),
            "lmax":degree,"max_degree":available,
            "modelname":header.get("modelname",path.stem),"header":header}


def sh_download_model(name, directory=None, overwrite=False):
    """Explicitly download a registered ICGEM model to an offline cache.
    
    The transfer first writes a temporary .part file and then renames
    it to the destination. No checksum validation is performed.
    
    Parameters
    ----------
    name : str
        Registered MODEL_URLS key, e.g. 'EIGEN-6C4', its alias
        'EIGENC4', or 'WHU-CASM-UGM2025_2159'.
    directory : str, pathlib.Path, or None, optional
        Cache directory; defaults to DEFAULT_DATA_DIR.
    overwrite : bool, optional
        Whether to replace an already cached file. Default False.
    
    Returns
    -------
    pathlib.Path
        Location of the .gfc file. 'EIGENC4' uses the canonical
        filename 'EIGEN-6C4.gfc'.
    
    Raises
    ------
    KeyError
        If no download URL is registered for name.
    OSError
        On filesystem or download failure.
    
    Examples
    --------
    >>> from relgravpy.sh_core import sh_download_model
    >>> path = sh_download_model("EIGEN-6C4", "shm_data")
    >>> path.name
    'EIGEN-6C4.gfc'
    
    Notes
    -----
    The first download requires internet access. Reusing a cached file
    does not establish its authenticity or check its completeness."""
    name = str(name)
    if name not in MODEL_URLS:
        raise KeyError(f"No verified direct download URL registered for {name!r}")
    directory = Path(directory) if directory is not None else DEFAULT_DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    canonical = "EIGEN-6C4" if name == "EIGENC4" else name
    dest = directory / (canonical + ".gfc")
    if dest.exists() and not overwrite:
        return dest
    # Keep an interrupted download separate from an existing valid cache file.
    temp = dest.with_suffix(".gfc.part")
    try:
        with urlopen(MODEL_URLS[name], timeout=120) as response, temp.open("wb") as out:
            while True:
                buf = response.read(1024*1024)
                if not buf:
                    break
                out.write(buf)
        if not temp.stat().st_size:
            raise IOError("Empty download")
        temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)
    return dest


def sh_resolve_degree(requested, available):
    """Limit a requested maximum degree to the available model degree.
    
    Parameters
    ----------
    requested : int or None
        Requested maximum degree; None selects all available degrees.
        Converted to int when supplied.
    available : int
        Largest harmonic degree offered by the model.
    
    Returns
    -------
    int
        Effective degree, min(requested, available).
    
    Raises
    ------
    ValueError
        If requested is negative.
    
    Warns
    -----
    UserWarning
        If requested exceeds available, before clipping it.
    
    Examples
    --------
    >>> from relgravpy.sh_core import sh_resolve_degree
    >>> sh_resolve_degree(120, 360)
    120
    >>> sh_resolve_degree(None, 360)
    360"""
    if requested is None:
        return int(available)
    requested = int(requested)
    if requested < 0:
        raise ValueError("lmax must be non-negative")
    if requested > available:
        warnings.warn(f"Requested degree {requested} exceeds {available}; using {available}",
                      UserWarning, stacklevel=2)
    return min(requested, int(available))


def sh_taper(lmax, gentle_cut=None):
    """Create a cosine taper across spherical harmonic degrees.
    
    Weights are one through gentle_cut, then smoothly decay to
    zero at lmax. With gentle_cut=None, all weights equal one.
    
    Parameters
    ----------
    lmax : int
        Highest degree; output indices range from 0 to lmax.
    gentle_cut : int or None, optional
        Last untapered degree, satisfying 0 <= gentle_cut < lmax.
    
    Returns
    -------
    numpy.ndarray, shape (lmax+1,)
        Dimensionless spectral weights indexed by degree.
    
    Raises
    ------
    ValueError
        If gentle_cut is outside its valid range.
    
    Examples
    --------
    >>> from relgravpy.sh_core import sh_taper
    >>> weights = sh_taper(10, gentle_cut=6)
    >>> weights[6], weights[10]
    (1.0, 0.0)"""
    w = np.ones(lmax+1)
    if gentle_cut is None:
        return w
    cut = int(gentle_cut)
    if not 0 <= cut < lmax:
        raise ValueError("gentle_cut must be between 0 and lmax-1")
    degrees = np.arange(cut+1, lmax+1, dtype=float)
    w[cut+1:] = 0.5*(1+np.cos(np.pi*(degrees-cut)/(lmax-cut)))
    return w


def sh_legendre(latitude_deg, lmax):
    """Evaluate 4-pi fully normalized associated Legendre functions.
    
    The recurrence does NOT apply the Condon-Shortley (-1)**m
    phase. It avoids factorial ratios, but the full triangular
    array still scales quadratically with the maximum degree.
    
    Parameters
    ----------
    latitude_deg : float
        Angular latitude in degrees, geodetic or geocentric as
        appropriate to the surrounding synthesis geometry.
    lmax : int
        Maximum polynomial degree and order.
    
    Returns
    -------
    numpy.ndarray, shape (lmax+1, lmax+1)
        P[n, m] (dimensionless), for 0 <= m <= n <= lmax.
        Entries with m > n are zero. For example,
        P[1, 1] = sqrt(3)*cos(latitude).
    
    Examples
    --------
    >>> import numpy as np
    >>> from relgravpy.sh_core import sh_legendre
    >>> np.isclose(sh_legendre(0., 1)[1, 1], np.sqrt(3.))
    True"""
    u = np.sin(np.deg2rad(float(latitude_deg)))
    v = np.cos(np.deg2rad(float(latitude_deg)))
    p = np.zeros((lmax+1, lmax+1), dtype=float)
    p[0,0] = 1.
    if lmax == 0:
        return p
    p[1,0] = np.sqrt(3.)*u
    p[1,1] = np.sqrt(3.)*v
    # First compute sectorial terms P[m,m], then off-diagonal terms.
    for m in range(2, lmax+1):
        p[m,m] = np.sqrt((2*m+1)/(2*m))*v*p[m-1,m-1]
    for m in range(1,lmax):
        p[m+1,m] = np.sqrt(2*m+3)*u*p[m,m]
    # Apply the zonal recurrence independently of the tesseral degrees.
    for n in range(2,lmax+1):
        a = np.sqrt((2*n+1)*(2*n-1)/(n*n))
        b = np.sqrt((2*n+1)*(n-1)*(n-1)/((2*n-3)*n*n))
        p[n,0] = a*u*p[n-1,0]-b*p[n-2,0]
    for m in range(1,lmax+1):
        for n in range(m+2,lmax+1):
            a = np.sqrt((2*n+1)*(2*n-1)/((n-m)*(n+m)))
            b = np.sqrt((2*n+1)*(n+m-1)*(n-m-1)/
                        ((2*n-3)*(n-m)*(n+m)))
            p[n,m] = a*u*p[n-1,m]-b*p[n-2,m]
    return p


def sh_grid(limits, step):
    """Construct a regular geographic longitude/latitude meshgrid.
    
    Parameters
    ----------
    limits : sequence of four numbers
        Bounds [lon_min, lon_max, lat_min, lat_max], in degrees.
        Latitude must lie within [-90, 90]; bounds must be ordered.
    step : float or sequence of two floats
        Positive spacing in degrees, either a scalar or
        (longitude_step, latitude_step).
    
    Returns
    -------
    lon, lat : tuple of numpy.ndarray
        Matching 2-D meshgrids (nlat, nlon), in degrees. Rows
        share latitude and columns share longitude. The upper
        bound is included if within the grid endpoint tolerance.
    
    Raises
    ------
    ValueError
        For invalid limits or spacing.
    
    Examples
    --------
    >>> from relgravpy.sh_core import sh_grid
    >>> lon, lat = sh_grid([10, 12, 44, 45], (1, 0.5))
    >>> lon.shape, lat.shape
    ((3, 3), (3, 3))"""
    lon0,lon1,lat0,lat1 = map(float, limits)
    if not (-90 <= lat0 <= lat1 <= 90 and lon0 <= lon1):
        raise ValueError("Invalid geographic limits")
    dl = (float(step),float(step)) if np.isscalar(step) else tuple(step)
    if len(dl) != 2 or min(dl) <= 0:
        raise ValueError("step must be positive")
    lon = np.arange(lon0,lon1+dl[0]*0.25,dl[0])
    lat = np.arange(lat0,lat1+dl[1]*0.25,dl[1])
    return np.meshgrid(lon,lat)


def sh_ellipsoid_radius(lat):
    """Compute geocentric distance to the WGS84 ellipsoid surface.
    
    Parameters
    ----------
    lat : float or array_like
        WGS84 geodetic latitude in degrees.
    
    Returns
    -------
    numpy.ndarray or numpy.float64
        Geocentric radius of the ellipsoid at lat, in metres,
        retaining the input array shape.
    
    Examples
    --------
    >>> from relgravpy.sh_core import sh_ellipsoid_radius, A_WGS84
    >>> round(float(sh_ellipsoid_radius(0.))) == round(A_WGS84)
    True
    
    Notes
    -----
    This is not the prime-vertical radius of curvature N and
    does not include ellipsoidal height."""
    phi = np.deg2rad(np.asarray(lat, dtype=float))
    b = A_WGS84*(1-F_WGS84)
    cp,sp = np.cos(phi),np.sin(phi)
    return np.sqrt(((A_WGS84*A_WGS84*cp)**2+(b*b*sp)**2)/
                   ((A_WGS84*cp)**2+(b*sp)**2))


def sh_normal_gravity(lat):
    """Compute normal gravity on WGS84 using Somigliana's formula.
    
    Parameters
    ----------
    lat : float or array_like
        Geodetic latitude in degrees.
    
    Returns
    -------
    numpy.ndarray or numpy.float64
        Normal gravity at zero ellipsoidal height, m/s**2,
        with the same shape as lat.
    
    Examples
    --------
    >>> from relgravpy.sh_core import sh_normal_gravity
    >>> round(float(sh_normal_gravity(0.)), 6)
    9.780325
    
    See Also
    --------
    sh_normal_gravity_height : Second-order gravity continuation in height."""
    phi = np.deg2rad(np.asarray(lat, dtype=float))
    e2 = F_WGS84*(2-F_WGS84)
    k = 0.00193185265241
    return 9.7803253359*(1+k*np.sin(phi)**2)/np.sqrt(1-e2*np.sin(phi)**2)



def sh_wgs84_normal_zonals(gm, radius, lmax):
    """Generate WGS84 even-zonal normal-potential SH coefficients.
    
    The WGS84 J2n expansion is converted to fully normalized
    C[2n,0] and rescaled to an external gravity model's GM and
    reference radius. These terms can be subtracted from the
    model zonals to calculate the disturbing potential.
    
    Parameters
    ----------
    gm : float
        External gravity model GM in m**3/s**2.
    radius : float
        External model reference radius in metres.
    lmax : int
        Maximum degree to generate.
    
    Returns
    -------
    numpy.ndarray, shape (lmax+1,)
        Dimensionless 4-pi normalized C[n,0] terms. Index zero
        and all odd-degree indices equal zero.
    
    Examples
    --------
    >>> from relgravpy.sh_core import (
    ...     sh_wgs84_normal_zonals, GM_WGS84, A_WGS84)
    >>> c = sh_wgs84_normal_zonals(GM_WGS84, A_WGS84, 2)
    >>> round(c[2], 9)
    -0.000484167
    
    Notes
    -----
    The reference coefficients follow the constants hardcoded in
    this module and do not correct model-specific tide conventions."""
    j2=1.08262982131e-3
    e2=F_WGS84*(2-F_WGS84)
    out=np.zeros(lmax+1)
    for n in range(1,lmax//2+1):
        k=2*n
        j2n=((-1)**(n+1) * 3*(e2**n)/((k+1)*(k+3)) *
             (1-n+5*n*j2/e2))
        out[k]=-j2n/np.sqrt(2*k+1) * (GM_WGS84/gm) * (A_WGS84/radius)**k
    return out


def sh_geocentric_coordinates(lat, height):
    """Convert WGS84 geodetic latitude and ellipsoidal height to spherical coordinates.
    
    Parameters
    ----------
    lat : float or array_like
        Geodetic latitude in degrees.
    height : float or array_like
        Height above the WGS84 ellipsoid, in metres; broadcastable
        against lat.
    
    Returns
    -------
    latitude_gc : numpy.ndarray or numpy.float64
        Geocentric latitude in degrees.
    radius : numpy.ndarray or numpy.float64
        Distance from Earth's centre, in metres. Both outputs
        have the shape broadcast from lat and height.
    
    Examples
    --------
    >>> from relgravpy.sh_core import sh_geocentric_coordinates
    >>> phi, radius = sh_geocentric_coordinates(45., 0.)
    >>> bool(phi < 45.)
    True
    
    Notes
    -----
    The longitude coordinate is unchanged by this conversion."""
    ph=np.deg2rad(np.asarray(lat,dtype=float))
    e2=F_WGS84*(2-F_WGS84)
    N=A_WGS84/np.sqrt(1-e2*np.sin(ph)**2)
    x=(N+height)*np.cos(ph)
    z=(N*(1-e2)+height)*np.sin(ph)
    return np.rad2deg(np.arctan2(z,x)),np.hypot(x,z)


def sh_normal_gravity_height(lat, height):
    """Approximate WGS84 normal gravity at nonzero ellipsoidal height.
    
    Applies the second-order height continuation of Somigliana
    normal gravity, rather than an exact high-altitude solution.
    
    Parameters
    ----------
    lat : float or array_like
        WGS84 geodetic latitude, in degrees.
    height : float or array_like
        Ellipsoidal height, in metres, broadcastable with lat.
    
    Returns
    -------
    numpy.ndarray or numpy.float64
        Normal gravity at height, in m/s**2.
    
    Examples
    --------
    >>> from relgravpy.sh_core import sh_normal_gravity_height
    >>> bool(sh_normal_gravity_height(45., 100.) <
    ...      sh_normal_gravity_height(45., 0.))
    True
    
    Notes
    -----
    This near-surface approximation is not an exact ellipsoidal
    gravity solution at arbitrary altitude."""
    ph=np.deg2rad(np.asarray(lat,dtype=float))
    m=OMEGA_WGS84**2*A_WGS84**2*(A_WGS84*(1-F_WGS84))/GM_WGS84
    q=1+F_WGS84+m-2*F_WGS84*np.sin(ph)**2
    h=np.asarray(height,dtype=float)
    return sh_normal_gravity(lat)*(1-2*q*h/A_WGS84+3*(h/A_WGS84)**2)


def sh_synthesis_core(model, limits=None, step=1., lmax=None, gentle_cut=None,
                      fields=("grav_dist",), height=0.0, surface="ellipsoid",
<<<<<<< HEAD
                      topo=None, lon=None, lat=None, height_reference="geoid",
                      geoid_undulation=0.0):
    """Spherical harmonic synthesis with WGS84 even-zonal normal-field removal.

    Supports both regular grids from limits/step and user-defined lon/lat arrays.
    The default height convention is geoid-referenced (H), converted internally to
    ellipsoidal height h = H + N when height_reference='geoid'.
    """
=======
                      topo=None):
    """Synthesize gravity-related fields from normalized spherical harmonics.
    
    Subtracts WGS84 degree-zero and even-zonal normal-field
    coefficients, evaluates the disturbing potential T, and
    computes spherical radial derivatives. Uses Bruns' approximation
    for geoid undulation and second-order WGS84 normal-gravity
    continuation for an approximate absolute gravity.
    
    Parameters
    ----------
    model : dict
        Coefficient dictionary from sh_read_gfc(), with 'c', 's'
        dimensionless 4-pi fully normalized arrays (no
        Condon-Shortley phase), 'gm' in m**3/s**2, 'radius'
        in metres, and 'lmax' as an integer.
    limits : sequence of four floats
        Geographic [lon_min, lon_max, lat_min, lat_max] in degrees.
    step : float or (float, float)
        Longitude/latitude grid spacing in degrees.
    lmax : int or None, optional
        Maximum synthesized degree; None uses model['lmax'].
    gentle_cut : int or None, optional
        Last untapered degree before cosine roll-off; None
        disables tapering.
    fields : str or sequence of str, optional
        Any subset of 'potential', 'geoid', 'abs_grav',
        'grav_dist', 'free_air' and 'grav_gradient'.
    height : float or array_like, optional
        Height offset in metres, broadcastable to the output grid.
    surface : {'ellipsoid', 'sphere', 'mean_sea_level', 'topography'}
        'ellipsoid': WGS84 ellipsoid plus height;
        'sphere': model reference sphere plus height (grid latitude
        treated as geocentric);
        'mean_sea_level': currently approximated by the ellipsoid;
        'topography': ellipsoid plus height plus topo (approximate).
    topo : array_like or None, optional
        Topographic heights in metres, broadcastable to the grid.
        Required for surface='topography'. Heights are treated as
        approximate ellipsoidal heights (no geoid conversion).
    
    Returns
    -------
    dict[str, numpy.ndarray]
        'lon', 'lat' : arrays of shape (nlat, nlon)
            Geographic meshgrid coordinates in degrees.
        Requested field arrays, all with matching 2-D shape:
        'potential' : disturbing potential T, m**2/s**2.
        'geoid' : Bruns approximation T/gamma, metres.
        'grav_dist' : spherical radial gravity disturbance, mGal.
        'free_air' : spherical anomaly (dg - 2T/r), mGal.
        'abs_grav' : gamma(height) + dg, in mGal (approximate).
        'grav_gradient' : radial derivative of the disturbance,
            in Eotvos (1 E = 1e-9 s**-2).
    
    Raises
    ------
    ValueError
        For unsupported fields or surface types, invalid geographic
        bounds, or missing topography when required.
    
    Examples
    --------
    >>> import numpy as np
    >>> from relgravpy.sh_core import (
    ...     sh_synthesis_core, A_WGS84, GM_WGS84)
    >>> c = np.array([[1.]])
    >>> model = {"c": c, "s": np.zeros_like(c), "gm": GM_WGS84,
    ...          "radius": A_WGS84, "lmax": 0}
    >>> result = sh_synthesis_core(
    ...     model, [10, 12, 45, 45], 1, lmax=0,
    ...     fields=("potential", "grav_dist"), surface="sphere")
    >>> np.allclose(result["potential"], 0.0)
    True
    
    Notes
    -----
    This is a reference spherical/approximate ellipsoidal kernel,
    NOT an exact ICGEM geodetic gravity calculator. In particular,
    the radial disturbance is not an ellipsoidal component
    projection; geoid zero-degree conventions and tide systems
    are not fully implemented. Full degree-10800 synthesis can
    be computationally expensive."""
>>>>>>> 9b2cf5ca54d3847a2269d9c6498bd4aeb63b3624
    if isinstance(fields,str):
        fields=(fields,)
    allowed={"potential","geoid","abs_grav","grav_dist","free_air","grav_gradient"}
    if set(fields)-allowed:
        raise ValueError(f"Unsupported fields: {sorted(set(fields)-allowed)}")

    if lon is not None or lat is not None:
        if lon is None or lat is None:
            raise ValueError("Both lon and lat must be provided for a custom grid")
        lon=np.asarray(lon,dtype=float)
        lat=np.asarray(lat,dtype=float)
        if lon.ndim == 1 and lat.ndim == 1:
            lon,lat=np.meshgrid(lon,lat)
        elif lon.shape != lat.shape:
            raise ValueError("lon and lat must have matching shapes")
    else:
        if limits is None:
            raise ValueError("Either limits or explicit lon/lat must be provided")
        lon,lat=sh_grid(limits,step)

    degree=sh_resolve_degree(lmax,model["lmax"])
    w=sh_taper(degree,gentle_cut)

    h=np.broadcast_to(np.asarray(height,dtype=float),lat.shape).copy()
    if surface == "topography":
        if topo is None:
            raise ValueError("topography surface requires explicit topo heights")
        topo_arr=np.asarray(topo,dtype=float)
        if topo_arr.shape == ():
            h = h + topo_arr
        else:
            h = h + np.broadcast_to(topo_arr,lat.shape)
    elif surface not in ("ellipsoid","sphere","mean_sea_level"):
        raise ValueError("Unknown surface")

    if height_reference == "geoid":
        N=np.asarray(geoid_undulation,dtype=float)
        if N.shape == ():
            N=np.full_like(h, N, dtype=float)
        elif N.shape != h.shape:
            N=np.broadcast_to(N, h.shape)
        h = h + N
    elif height_reference != "ellipsoid":
        raise ValueError("height_reference must be 'geoid' or 'ellipsoid'")

    if surface == "sphere":
        r=np.full_like(lat,model["radius"])+h
        phi_gc=lat.copy()
    else:
        phi_gc,r=sh_geocentric_coordinates(lat,h)

    # Remove the WGS84 normal potential to obtain disturbing SH coefficients;
    # retain the original model array unchanged for future syntheses.
    delta_c=model["c"][:degree+1,:degree+1].copy()
    delta_s=model["s"][:degree+1,:degree+1]
    delta_c[0,0] -= GM_WGS84/model["gm"]
    reference=sh_wgs84_normal_zonals(model["gm"],model["radius"],degree)
    for n in range(2,degree+1,2):
        delta_c[n,0]-=reference[n]
    results={name:np.zeros_like(lon) for name in fields}
<<<<<<< HEAD
=======
    longs=np.deg2rad(lon[0])
    order=np.arange(degree+1)
    # Trigonometric factors depend only on longitude and can be reused by row.
    trigc=np.cos(order[:,None]*longs[None,:])
    trigs=np.sin(order[:,None]*longs[None,:])
>>>>>>> 9b2cf5ca54d3847a2269d9c6498bd4aeb63b3624
    pot=np.zeros_like(lon)
    dg=np.zeros_like(lon)
    grad=np.zeros_like(lon)

    if lon.ndim == 2 and lon.shape[0] > 0 and lon.shape[1] > 0:
        for i in range(lat.shape[0]):
            if np.all(phi_gc[i] == phi_gc[i,0]):
                p=sh_legendre(phi_gc[i,0],degree)
                radial=np.ones(lon.shape[1])
                x=model["radius"]/r[i]
                longs=np.deg2rad(lon[0])
                order=np.arange(degree+1)
                trigc=np.cos(order[:,None]*longs[None,:])
                trigs=np.sin(order[:,None]*longs[None,:])
                for n in range(degree+1):
                    if n:
<<<<<<< HEAD
                        radial *= x
                    hnm=p[n,:n+1] @ (
                        delta_c[n,:n+1,None]*trigc[:n+1] +
                        delta_s[n,:n+1,None]*trigs[:n+1]
                    )
                    hnm=w[n]*radial*hnm
                    pot[i]+=hnm
                    dg[i]+=(n+1)*hnm
                    grad[i]+=(n+1)*(n+2)*hnm
            else:
                for j in range(lon.shape[1]):
                    p=sh_legendre(phi_gc[i,j],degree)
                    x=model["radius"]/r[i,j]
                    radial=1.0
                    lam=np.deg2rad(lon[i,j])
                    m=np.arange(degree+1)
                    trigc=np.cos(m*lam)
                    trigs=np.sin(m*lam)
                    for n in range(degree+1):
                        if n:
                            radial*=x
                        harmonics=(delta_c[n,:n+1]*trigc[:n+1] +
                                   delta_s[n,:n+1]*trigs[:n+1])
                        hnm=w[n]*radial*np.dot(p[n,:n+1],harmonics)
                        pot[i,j]+=hnm
                        dg[i,j]+=(n+1)*hnm
                        grad[i,j]+=(n+1)*(n+2)*hnm
            pot[i]*=model["gm"]/r[i]
            dg[i]*=model["gm"]/r[i]**2
            grad[i]*=-model["gm"]/r[i]**3
    else:
        for idx in np.ndindex(lon.shape):
            lam=np.deg2rad(lon[idx])
            phi=lat[idx]
            h0=h[idx]
            phi_gc0,r0=sh_geocentric_coordinates(phi,h0)
            p=sh_legendre(phi_gc0,degree)
            radial=1.0
            x=model["radius"]/r0
            pot0=0.0; dg0=0.0; grad0=0.0
            for n in range(degree+1):
                if n:
                    radial*=x
                m=np.arange(n+1)
                trigc=np.cos(m*lam)
                trigs=np.sin(m*lam)
                harmonics=(delta_c[n,:n+1]*trigc[:n+1] +
                           delta_s[n,:n+1]*trigs[:n+1])
                hnm=w[n]*radial*np.dot(p[n,:n+1],harmonics)
                pot0 += hnm
                dg0 += (n+1)*hnm
                grad0 += (n+1)*(n+2)*hnm
            pot[idx] = pot0*model["gm"]/r0
            dg[idx] = dg0*model["gm"]/r0**2
            grad[idx] = -model["gm"]/r0**3

=======
                        radial*=x
                    harmonics=(delta_c[n,:n+1]*trigc[:n+1,j]+
                               delta_s[n,:n+1]*trigs[:n+1,j])
                    hnm=w[n]*radial*np.dot(p[n,:n+1],harmonics)
                    pot[i,j]+=hnm
                    dg[i,j]+=(n+1)*hnm
                    grad[i,j]+=(n+1)*(n+2)*hnm
        # Convert degree-weighted harmonic sums to potential and radial
        # derivatives in SI units after completing the latitude row.
        pot[i]*=model["gm"]/r[i]
        dg[i]*=model["gm"]/r[i]**2
        grad[i]*=-model["gm"]/r[i]**3
>>>>>>> 9b2cf5ca54d3847a2269d9c6498bd4aeb63b3624
    if "potential" in results:
        results["potential"]=pot
    if "grav_dist" in results:
        results["grav_dist"]=dg*1e5
    if "geoid" in results:
        results["geoid"]=pot/sh_normal_gravity(lat)
    if "free_air" in results:
        results["free_air"]=(dg-2*pot/r)*1e5
    if "abs_grav" in results:
        results["abs_grav"]=(sh_normal_gravity_height(lat,h)+dg)*1e5
    if "grav_gradient" in results:
        results["grav_gradient"]=grad*1e9
    return {"lon":lon,"lat":lat,**results}



def sh_analysis_fft(lon, lat, values, lmax):
    """Analyze a global scalar field with longitude FFT and latitude quadrature.
    
    Longitude harmonics are extracted using a real FFT. Latitude
    integration is trapezoidal in sin(latitude), so results on
    equiangular grids are APPROXIMATE and not equivalent to exact
    Gauss-Legendre spherical harmonic analysis.
    
    Parameters
    ----------
    lon, lat : array_like, shape (nlat, nlon)
        Matching separable 2-D geographic meshgrids in degrees.
        Longitude must have uniform spacing covering exactly 360
        degrees, without a duplicated endpoint. Latitude must
        reach both -90 and +90 degrees.
    values : array_like, shape (nlat, nlon)
        Scalar values on the grid, all finite; their units carry
        over to the recovered harmonic coefficients.
    lmax : int
        Largest estimated degree. Requires nlon >= 2*lmax+1
        and nlat >= lmax+1.
    
    Returns
    -------
    dict
        'c', 's' : ndarray, shape (lmax+1, lmax+1)
            Fully normalized, no-Condon-Shortley cosine and sine
            coefficients in the units of values; m > n entries are zero.
        'lmax' : int
            Analyzed maximum degree.
        'method' : str
            Always 'fft_trapezoid'.
        'latitude_quadrature' : str
            Always 'trapezoid'.
    
    Raises
    ------
    ValueError
        For dimension/shape mismatches, insufficient sampling,
        nonfinite values, missing poles, non-separable meshgrids,
        or an incomplete/nonuniform longitude period.
    
    Examples
    --------
    >>> import numpy as np
    >>> from relgravpy.sh_core import sh_analysis_fft
    >>> lon, lat = np.meshgrid(np.arange(-180., 180., 1.),
    ...                        np.linspace(-90., 90., 181))
    >>> fit = sh_analysis_fft(lon, lat, np.full(lon.shape, 5.), 0)
    >>> np.isclose(fit["c"][0, 0], 5.)
    True
    
    Notes
    -----
    Use sh_analysis_core(method='lstsq') for regional grids.
    The FFT approximation is not an exact spectral inversion."""
    lon=np.asarray(lon,dtype=float)
    lat=np.asarray(lat,dtype=float)
    values=np.asarray(values,dtype=float)
    if not (lon.ndim==lat.ndim==values.ndim==2 and
            lon.shape==lat.shape==values.shape):
        raise ValueError("lon, lat and values must be matching 2-D grids")
    ny,nx=values.shape
    if nx < 2*lmax+1 or ny < lmax+1:
        raise ValueError("Grid undersampled for requested degree")
    if np.any(~np.isfinite(values)):
        raise ValueError("FFT analysis does not accept NaNs")
    lons=lon[0]
    lats=lat[:,0]
    step=np.diff(lons)
    if not np.allclose(step,step[0],rtol=1e-7,atol=1e-10):
        raise ValueError("Longitude grid must be regular")
    if abs(abs(step[0])*nx-360)>1e-6:
        raise ValueError("Longitude must span a complete 360-degree period")
    if not (abs(np.min(lats)+90)<1e-6 and
            abs(np.max(lats)-90)<1e-6):
        raise ValueError("Latitude range must cover both poles")
    if not np.allclose(lon,lons[None,:]) or not np.allclose(lat,lats[:,None]):
        raise ValueError("Longitude/latitude must be separable meshgrid")
    # Integrate over sin(latitude) to include the spherical area element.
    u=np.sin(np.deg2rad(lats))
    du=np.abs(np.diff(u))
    ww=np.zeros(ny)
    ww[:-1]+=du/2.
    ww[1:]+=du/2.
    # The factor exp(-i m lambda0) accounts for arbitrary longitude start.
    # Normalize by longitude sample count before latitude integration.
    ft=np.fft.rfft(values,axis=1)/nx
    mm=np.arange(lmax+1)
    ft=ft[:,:lmax+1]*np.exp(-1j*mm[None,:]*np.deg2rad(lons[0]))
    cc=np.zeros((lmax+1,lmax+1))
    ss=np.zeros_like(cc)
    for i,phi in enumerate(lats):
        p=sh_legendre(phi,lmax)
        for n in range(lmax+1):
            cc[n,:n+1]+=.5*ww[i]*p[n,:n+1]*ft[i,:n+1].real
            ss[n,1:n+1]-=.5*ww[i]*p[n,1:n+1]*ft[i,1:n+1].imag
    return {"c":cc,"s":ss,"lmax":lmax,
            "method":"fft_trapezoid","latitude_quadrature":"trapezoid"}



def sh_analysis_core(lon, lat, values, lmax, radius=6378136.3,
                     normalization="fully_normalized", method="auto"):
    """Estimate normalized real scalar SH coefficients from grid samples.
    
    Selects a global FFT/quadrature analysis when possible, or
    solves a dense linear least-squares system. This analyzes an
    arbitrary scalar field: it does not apply gravity-specific
    radial continuation or reference-potential subtraction.
    
    Parameters
    ----------
    lon, lat, values : array_like
        Matching arrays of geographic longitude, latitude
        (degrees) and scalar samples. Least squares uses only
        entries where all three are finite. Coefficients retain
        the physical units of values.
    lmax : int
        Maximum SH degree. Least squares fits (lmax+1)**2
        real sine/cosine coefficients.
    radius : float, optional
        Reference radius metadata, in metres (default 6378136.3).
        Does NOT enter the scalar design matrix.
    normalization : str, optional
        Only 'fully_normalized' is supported.
    method : {'auto', 'fft', 'lstsq'}, optional
        'fft' requires a full global, regular, pole-to-pole grid;
        'lstsq' uses the sample design matrix;
        'auto' first tries FFT and falls back on least squares
        when the grid does not satisfy FFT geometry requirements.
    
    Returns
    -------
    dict
        'c', 's' : ndarray, shape (lmax+1, lmax+1)
            Dimensionful fully normalized scalar SH coefficients.
        'lmax' : int
            Maximum degree used.
        'radius' : float
            Provided reference-radius metadata, in metres.
        For least squares, also returns 'rank' (int),
        'residuals' (ndarray) and 'condition' (float estimated
        from the design-matrix singular values).
        For FFT, also returns 'method'='fft_trapezoid' and
        'latitude_quadrature'='trapezoid'.
    
    Raises
    ------
    ValueError
        For inconsistent shapes, unsupported settings, no finite
        samples, too few samples, or invalid FFT grid geometry.
    MemoryError
        If the dense least-squares design matrix exceeds 512 MiB.
    
    Warns
    -----
    UserWarning
        If the least-squares design matrix is rank deficient.
    
    Examples
    --------
    >>> import numpy as np
    >>> from relgravpy.sh_core import sh_analysis_core, sh_grid
    >>> lon, lat = sh_grid([-180, 150, -60, 60], 30)
    >>> fit = sh_analysis_core(lon, lat, np.full(lon.shape, 7.),
    ...                        0, method="lstsq")
    >>> np.isclose(fit["c"][0, 0], 7.)
    True
    
    Notes
    -----
    Regional SH analysis may be nonunique or severely ill-conditioned
    even with more data than unknowns. The FFT method provides an
    approximate, not exact, equiangular-latitude transform."""
    if normalization!="fully_normalized":
        raise ValueError("Only fully_normalized supported")
    lon=np.asarray(lon,dtype=float)
    lat=np.asarray(lat,dtype=float)
    values=np.asarray(values,dtype=float)
    if not (lon.shape==lat.shape==values.shape):
        raise ValueError("lon/lat/values shapes differ")
    if method not in ("auto","lstsq","fft"):
        raise ValueError("method must be 'auto', 'lstsq' or 'fft'")
    if method in ("auto","fft"):
        try:
            result=sh_analysis_fft(lon,lat,values,int(lmax))
        except ValueError:
            if method=="fft":
                raise
        else:
            result["radius"]=radius
            return result
    mask=np.isfinite(lon)&np.isfinite(lat)&np.isfinite(values)
    if not np.any(mask):
        raise ValueError("No finite data")
    lmax=int(lmax)
    k=(lmax+1)**2
    if mask.sum()<k:
        raise ValueError(f"At least {k} samples needed for degree {lmax}")
    ylon=lon[mask]; ylat=lat[mask]; yy=values[mask]
    p_cache={}
    terms=[]
    for n in range(lmax+1):
        for m in range(n+1):
            terms.append((n,m,"c"))
            if m:
                terms.append((n,m,"s"))
    # Limit dense least-squares memory: rows are samples, columns are SH terms.
    if yy.size*k*8>512*1024**2:
        raise MemoryError(
            "Least-squares design matrix exceeds 512 MiB. Use a full "
            "global grid and method='fft', or reduce lmax.")
    A=np.zeros((yy.size,k),float)
    # Cache Legendre functions across samples sharing the same latitude.
    for ii,latitude in enumerate(ylat):
        if latitude not in p_cache:
            p_cache[latitude]=sh_legendre(latitude,lmax)
        p=p_cache[latitude]
        ll=np.deg2rad(ylon[ii])
        for j,(n,m,cs) in enumerate(terms):
            A[ii,j]=p[n,m]*(np.cos(m*ll) if cs=="c" else np.sin(m*ll))
    coeff,residuals,rank,singular=np.linalg.lstsq(A,yy,rcond=None)
    if rank<k:
        warnings.warn("SH analysis design matrix is rank deficient",UserWarning,stacklevel=2)
    c=np.zeros((lmax+1,lmax+1));s=np.zeros_like(c)
    for a,(n,m,cs) in zip(coeff,terms):
        (c if cs=="c" else s)[n,m]=a
    return {"c":c,"s":s,"lmax":lmax,"radius":radius,
            "rank":rank,"residuals":residuals,"condition":(singular[0]/singular[-1]
                          if singular[-1] else np.inf)}



EARTH2014_SUR_URL = ("https://ddfe.blazejbucha.com/models/Earth2014/"
                     "data_1min/shcs_to10800/Earth2014.SUR2014.degree10800.bshc")


def sh_download_topography(directory=None, overwrite=False):
    """Explicitly download Earth2014 SUR2014 binary topography coefficients.
    
    Downloads the original high-degree .bshc model (about 933 MB)
    to the offline cache. A temporary .part file is used to avoid
    replacing an existing dataset before the transfer completes.
    
    Parameters
    ----------
    directory : str, pathlib.Path, or None, optional
        Destination cache directory; None means DEFAULT_DATA_DIR.
    overwrite : bool, optional
        Whether to replace an existing file. Default False.
    
    Returns
    -------
    pathlib.Path
        Path to 'Earth2014.SUR2014.degree10800.bshc'.
    
    Raises
    ------
    OSError
        On a network or filesystem failure.
    
    Examples
    --------
    >>> from relgravpy.sh_core import sh_download_topography
    >>> path = sh_download_topography(directory="shm_data")
    >>> path.suffix
    '.bshc'
    
    Notes
    -----
    This call performs network I/O unless the file is already cached.
    There is no automatic download from synthesis functions.
    Source: Hirt and Rexer (2015), DOI 10.1016/j.jag.2015.03.001.
    The implementation checks only a minimum file size, not a checksum."""
    directory = Path(directory) if directory is not None else DEFAULT_DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory/"Earth2014.SUR2014.degree10800.bshc"
    if dest.exists() and not overwrite:
        return dest
    tmp = dest.with_suffix(".bshc.part")
    try:
        with urlopen(EARTH2014_SUR_URL, timeout=120) as response, tmp.open("wb") as out:
            while True:
                chunk=response.read(1024*1024)
                if not chunk:
                    break
                out.write(chunk)
        if tmp.stat().st_size < 16:
            raise IOError("Earth2014 download is empty or truncated")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def sh_read_bshc(path, lmax=None):
    """Load Earth2014 SUR2014 scalar harmonics from its binary .bshc format.
    
    The little-endian binary file stores (nmin, nmax) as two float64
    numbers followed by packed triangular C coefficients and then S
    coefficients, both ordered by ascending degree and order. A
    memory map avoids eagerly reading the full source file.
    
    Parameters
    ----------
    path : str or pathlib.Path
        Local Earth2014 SUR2014 .bshc file.
    lmax : int or None, optional
        Maximum degree to retain; None uses the file maximum.
        Larger requests are clipped with a UserWarning.
    
    Returns
    -------
    dict
        'c', 's' : ndarray, shape (L+1, L+1)
            Fully normalized scalar height coefficients in metres.
            Entries with order m > degree n are zero.
        'lmax', 'max_degree' : int
            Retained and file-maximum degrees.
        'modelname' : str
            Always 'Earth2014 SUR2014'.
        'unit' : str
            Always 'm'.
    
    Raises
    ------
    ValueError
        If the binary header is invalid or its expected byte count
        disagrees with the actual file size.
    
    Examples
    --------
    >>> from relgravpy.sh_core import sh_read_bshc
    >>> topo = sh_read_bshc(
    ...     "shm_data/Earth2014.SUR2014.degree10800.bshc",
    ...     lmax=120)
    >>> topo["c"].shape
    (121, 121)
    
    Notes
    -----
    This is a topographic HEIGHT model, not an ICGEM gravity model.
    The file format is incompatible with .gfc gravity coefficients."""
    path=Path(path)
    header=np.fromfile(path,dtype="<f8",count=2)
    if len(header)!=2:
        raise ValueError("Invalid .bshc header")
    nmin,nmax=map(int,header)
    if (nmin,nmax)!=(0,round(float(header[1]))) or nmax<0 or nmax>50000:
        raise ValueError("Unexpected .bshc header or byte order")
    total=(nmax+1)*(nmax+2)//2
    expected=(2+2*total)*8
    if path.stat().st_size!=expected:
        raise ValueError(f"Wrong Earth2014 size: expected {expected} bytes")
    degree=sh_resolve_degree(lmax,nmax)
    # Binary coefficients are packed triangularly: all C terms, then all S.
    raw=np.memmap(path,dtype="<f8",mode="r",offset=16,shape=(2*total,))
    c=np.zeros((degree+1,degree+1))
    s=np.zeros_like(c)
    for n in range(degree+1):
        i=n*(n+1)//2
        c[n,:n+1]=raw[i:i+n+1]
        s[n,:n+1]=raw[total+i:total+i+n+1]
    return {"c":c,"s":s,"lmax":degree,"max_degree":nmax,
            "modelname":"Earth2014 SUR2014","unit":"m"}


def sh_scalar_synthesis(coeff, lon, lat, lmax=None, gentle_cut=None):
    """Synthesize a scalar field from dimensionful SH coefficients.
    
    Computes a real spherical harmonic expansion using 4-pi
    normalization and no Condon-Shortley phase. Coefficient
    units are preserved; Earth2014 SUR2014 yields elevations in m.
    
    Parameters
    ----------
    coeff : dict
        'c', 's' 2-D coefficient arrays and 'lmax' available degree.
        Coefficients carry the intended output unit.
    lon, lat : array_like, shape (nlat, nlon)
        Matching 2-D geographic grids in degrees. Latitude is
        assumed constant across each row; longitude must have
        the same columns for every row.
    lmax : int or None, optional
        Maximum synthesized degree; None uses coeff['lmax'].
    gentle_cut : int or None, optional
        Last untapered degree for an optional cosine taper.
    
    Returns
    -------
    numpy.ndarray, shape (nlat, nlon)
        Scalar synthesis in coefficient units, e.g. metres
        when coeff contains Earth2014 height harmonics.
    
    Raises
    ------
    ValueError
        If lon and lat are not equally shaped 2-D arrays.
    
    Examples
    --------
    >>> import numpy as np
    >>> from relgravpy.sh_core import sh_scalar_synthesis, sh_grid
    >>> lon, lat = sh_grid([0, 2, 0, 1], 1)
    >>> coeff = {"c": np.array([[100.]]),
    ...          "s": np.array([[0.]]), "lmax": 0}
    >>> np.allclose(sh_scalar_synthesis(coeff, lon, lat), 100.)
    True
    
    Notes
    -----
    Meshgrid row/column separability is assumed, not independently
    validated. This computes no gravity-specific radial derivatives."""
    lon=np.asarray(lon,dtype=float)
    lat=np.asarray(lat,dtype=float)
    if lon.shape!=lat.shape or lon.ndim!=2:
        raise ValueError("lon/lat must be matching 2-D grids")
    degree=sh_resolve_degree(lmax,coeff["lmax"])
    w=sh_taper(degree,gentle_cut)
    out=np.empty_like(lon)
    m=np.arange(degree+1)
    cos_m=np.cos(m[:,None]*np.deg2rad(lon[0])[None,:])
    sin_m=np.sin(m[:,None]*np.deg2rad(lon[0])[None,:])
    cc,ss=coeff["c"],coeff["s"]
    for i,phi in enumerate(lat[:,0]):
        p=sh_legendre(phi,degree)
        a=np.sum(p[:degree+1,:degree+1]*cc[:degree+1,:degree+1]*
                 w[:,None],axis=0)
        b=np.sum(p[:degree+1,:degree+1]*ss[:degree+1,:degree+1]*
                 w[:,None],axis=0)
        out[i]=a@cos_m+b@sin_m
    return out


<<<<<<< HEAD
def sh_synthesis_with_topography(model, limits=None, step=1., fields=("grav_dist",),
                                 lmax=None, gentle_cut=None, height=0., surface="ellipsoid",
                                 topo=None, directory=None, topo_lmax=None, lon=None, lat=None,
                                 height_reference="geoid", geoid_undulation=0.0):
    """Synthesize available gravity fields, optionally adding Earth2014 topo.

    Both the legacy limits/step grid and a user-defined lon/lat custom grid are
    supported. When lon/lat are provided, z is interpreted with the selected
    height_reference convention (default: geoid).
    """
    if isinstance(fields,str):
        fields=(fields,)
    fields=tuple(fields)

    if lon is not None or lat is not None:
        if lon is None or lat is None:
            raise ValueError("Both lon and lat must be provided for a custom grid")
        if topo is None and (surface=="topography" or "topo" in fields):
            raise ValueError("topography surface requires explicit topo values for custom grids")
        grav_fields=tuple(x for x in fields if x!="topo")
        if grav_fields:
            out=sh_synthesis_core(model, limits=limits, step=step, lmax=lmax,
                                  gentle_cut=gentle_cut, fields=grav_fields,
                                  height=height, surface=surface, topo=topo,
                                  lon=np.asarray(lon,dtype=float),
                                  lat=np.asarray(lat,dtype=float),
                                  height_reference=height_reference,
                                  geoid_undulation=geoid_undulation)
        else:
            lon_arr=np.asarray(lon,dtype=float)
            lat_arr=np.asarray(lat,dtype=float)
            if lon_arr.ndim == 1 and lat_arr.ndim == 1:
                lon_arr,lat_arr=np.meshgrid(lon_arr,lat_arr)
            out={"lon":lon_arr,"lat":lat_arr}
        if "topo" in fields:
            topo_arr=np.asarray(topo,dtype=float)
            if topo_arr.shape == ():
                topo_arr=np.full_like(np.asarray(lon,dtype=float), topo_arr, dtype=float)
            if topo_arr.shape != out["lon"].shape:
                topo_arr=np.broadcast_to(topo_arr, out["lon"].shape)
            out["topo"]=topo_arr.copy()
        return out

=======
def sh_synthesis_with_topography(model, limits, step, fields, lmax,
                                 gentle_cut, height, surface, topo,
                                 directory=None, topo_lmax=None):
    """Synthesize gravity outputs while optionally including Earth2014 relief.
    
    If the caller requests the 'topo' field or sets the surface
    to 'topography', missing heights are synthesized from a locally
    cached Earth2014 SUR2014 .bshc file. No implicit download occurs.
    
    Parameters
    ----------
    model : dict
        Gravity coefficients and metadata accepted by
        sh_synthesis_core().
    limits : sequence of four floats
        [lon_min, lon_max, lat_min, lat_max] in degrees.
    step : float or (float, float)
        Geographic grid spacing in degrees.
    fields : str or sequence of str
        Gravity field names accepted by sh_synthesis_core(),
        optionally with 'topo' for elevation in metres.
    lmax : int or None
        Maximum gravity-model synthesis degree.
    gentle_cut : int or None
        Last untapered gravity-model harmonic degree.
    height : float or array_like
        Height offset in metres, broadcastable to the grid.
    surface : str
        Surface accepted by sh_synthesis_core(), including
        'ellipsoid', 'sphere', 'mean_sea_level', or 'topography'.
    topo : array_like or None
        Explicit topographic heights in metres, broadcastable to
        the geographic grid, or None to use the local Earth2014 cache.
    directory : str, pathlib.Path, or None, optional
        Cache directory for Earth2014, default DEFAULT_DATA_DIR.
    topo_lmax : int or None, optional
        Maximum degree of Earth2014 height harmonics, independent
        of the gravity model degree lmax.
    
    Returns
    -------
    dict[str, numpy.ndarray]
        'lon', 'lat' grids in geographic degrees; all requested
        gravity fields in sh_synthesis_core() units; additionally
        'topo' in metres when requested. All values are 2-D grids.
    
    Raises
    ------
    FileNotFoundError
        If Earth2014 is required but absent from the offline cache.
    ValueError
        For invalid gravity field names or surface geometry.
    
    Examples
    --------
    >>> import numpy as np
    >>> from relgravpy.sh_core import (
    ...     sh_synthesis_with_topography, GM_WGS84, A_WGS84)
    >>> model = {"c": np.array([[1.]]), "s": np.array([[0.]]),
    ...          "gm": GM_WGS84, "radius": A_WGS84, "lmax": 0}
    >>> out = sh_synthesis_with_topography(
    ...     model, [10, 11, 45, 45], 1, ("topo",), 0,
    ...     None, 0., "topography", [[100., 101.]])
    >>> out["topo"].tolist()
    [[100.0, 101.0]]
    
    Notes
    -----
    Earth2014 SUR heights are nominally mean-sea-level elevations;
    this kernel presently treats them as approximate ellipsoidal
    heights without a geoid-to-ellipsoid correction."""
    if isinstance(fields,str):
        fields=(fields,)
    fields=tuple(fields)
    # Resolve missing relief from the local cache only; never fetch implicitly.
>>>>>>> 9b2cf5ca54d3847a2269d9c6498bd4aeb63b3624
    if topo is None and (surface=="topography" or "topo" in fields):
        directory=Path(directory) if directory is not None else DEFAULT_DATA_DIR
        path=directory/"Earth2014.SUR2014.degree10800.bshc"
        if not path.exists():
            raise FileNotFoundError(
                f"Earth2014 missing: {path}. Call sh_download_topography() once.")
        lon,lat=sh_grid(limits,step)
        elevation=sh_read_bshc(path,lmax=topo_lmax)
        topo=sh_scalar_synthesis(elevation,lon,lat)
    grav_fields=tuple(x for x in fields if x!="topo")
    if grav_fields:
        out=sh_synthesis_core(model,limits,step,lmax,gentle_cut,
                              grav_fields,height,surface,topo,
                              height_reference=height_reference,
                              geoid_undulation=geoid_undulation)
    else:
        lon,lat=sh_grid(limits,step)
        out={"lon":lon,"lat":lat}
    if "topo" in fields:
        out["topo"]=np.broadcast_to(topo,out["lon"].shape).copy()
    return out

def gravity_on_points(
    model,
    lon,
    lat,
    z,
    *,
    lmax=None,
    gentle_cut=None,
    field="grav_dist",
    height_reference="geoid",
    geoid_undulation=0.0
):
    """
    Calcola gravità da un modello SH su una griglia di punti definiti manualmente.

    Parameters
    ----------
    model : dict
        Dati del modello SH (es. model.data)
    lon, lat, z : array-like
        Coordinate geografiche e quota. shape compatibile.
    height_reference : {"geoid", "ellipsoid"}
        "geoid" => z è quota ortometrica H, viene convertita in h = H + N
        "ellipsoid" => z è quota ellissoidica h
    geoid_undulation : float or array-like
        Ondulazione del geoide N in metri. default=0
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    z = np.asarray(z, dtype=float)

    if lon.shape != lat.shape or lon.shape != z.shape:
        raise ValueError("lon, lat e z devono avere la stessa forma")

    N = np.asarray(geoid_undulation, dtype=float)
    if N.shape == ():
        N = np.full_like(z, N, dtype=float)
    elif N.shape != z.shape:
        raise ValueError("geoid_undulation deve essere scalare o avere forma uguale a z")

    degree = min(lmax if lmax is not None else model["lmax"], model["lmax"])
    w = sh_taper(degree, gentle_cut)

    out = np.zeros_like(lon, dtype=float)

    c = model["c"][:degree + 1, :degree + 1].copy()
    s = model["s"][:degree + 1, :degree + 1].copy()

    for idx in np.ndindex(lon.shape):
        lam = lon[idx]
        phi = lat[idx]

        # converte quota in quota ellissoidica h
        if height_reference == "geoid":
            H = z[idx]
            h = H + N[idx]
        elif height_reference == "ellipsoid":
            h = z[idx]
        else:
            raise ValueError("height_reference deve essere 'geoid' o 'ellipsoid'")

        phi_gc, r = sh_geocentric_coordinates(phi, h)

        pot = 0.0
        dg = 0.0

        for n in range(degree + 1):
            p = sh_legendre(phi_gc, degree)[n, :n + 1]
            m = np.arange(n + 1)
            trig_c = np.cos(m * np.deg2rad(lam))
            trig_s = np.sin(m * np.deg2rad(lam))
            harm = c[n, :n + 1] * trig_c + s[n, :n + 1] * trig_s

            radial = (model["radius"] / r) ** n if n else 1.0
            hnm = w[n] * radial * np.dot(p, harm)
            pot += hnm
            dg += (n + 1) * hnm

        if field == "grav_dist":
            out[idx] = dg * 1e5
        elif field == "potential":
            out[idx] = pot * model["gm"] / r
        elif field == "abs_grav":
            # approssimazione elementare: normal gravity + disturbance
            out[idx] = (dg * 1e5) + 9.81e5
        else:
            raise ValueError("field non supportato: grav_dist, potential, abs_grav")

    return out


class shm:
    """High-level interface for a locally cached, static ICGEM gravity model.
    
    Loads fully normalized gravity coefficients from a .gfc file.
    Network access is permitted only when download=True is requested.
    
    Parameters
    ----------
    model : str or pathlib.Path
        Local .gfc path or a registered model name, for example
        'EIGEN-6C4' or 'WHU-CASM-UGM2025_2159'.
    directory : str, pathlib.Path or None, optional
        Local model/cache directory; default DEFAULT_DATA_DIR.
    download : bool, optional
        Whether to download a missing registered model.
    lmax : int or None, optional
        Maximum degree loaded into memory. Synthesis cannot
        recover higher degrees excluded here.
    
    Attributes
    ----------
    directory : pathlib.Path
        Local model and topography cache directory.
    path : pathlib.Path
        Path to the loaded .gfc gravity model.
    data : dict
        Coefficients and metadata returned by sh_read_gfc().
    
    Examples
    --------
    >>> from relgravpy.sh_core import shm
    >>> model = shm("EIGEN-6C4", directory="shm_data")
    >>> result = model.synthesis([10, 12, 45, 46], 1,
    ...                          fields=("geoid", "grav_dist"), lmax=60)
    >>> result["grav_dist"].shape
    (2, 3)
    
    Notes
    -----
    The synthesis kernel uses spherical/approximate ellipsoidal
    formulas and is not an exact ICGEM ellipsoidal solution."""
    def __init__(self, model, directory=None, download=False, lmax=None):
        """Initialize a model reader with optional explicit download.
        
        Parameters
        ----------
        model : str or pathlib.Path
            Existing .gfc path or known model identifier.
        directory : str, pathlib.Path or None, optional
            Local coefficient cache directory.
        download : bool, optional
            Request download only when the model is not cached.
        lmax : int or None, optional
            Highest degree to retain when parsing coefficients.
        
        Returns
        -------
        None
            Sets the directory, path and data attributes.
        
        Raises
        ------
        FileNotFoundError
            If model cannot be located.
        KeyError
            When download=True and model is not registered.
        
        Examples
        --------
        >>> from relgravpy.sh_core import shm
        >>> model = shm("shm_data/EIGEN-6C4.gfc", lmax=60)
        >>> model.data["lmax"]
        60"""
        self.directory=Path(directory) if directory is not None else DEFAULT_DATA_DIR
        self.path=Path(model)
        if not self.path.is_file():
            self.path=self.directory/(("EIGEN-6C4" if str(model) == "EIGENC4" else str(model))+".gfc")
        if not self.path.is_file() and download:
            self.path=sh_download_model(str(model),self.directory)
        if not self.path.is_file():
            raise FileNotFoundError(f"Model unavailable offline: {self.path}")
        self.data=sh_read_gfc(self.path,lmax=lmax)

    def synthesis(self, limits=None, step=1., fields=("grav_dist",), lmax=None,
                  gentle_cut=None, height=0., surface="ellipsoid", topo=None,
<<<<<<< HEAD
                  topo_lmax=None, lon=None, lat=None, height_reference="geoid",
                  geoid_undulation=0.0):

=======
                  topo_lmax=None):
        """Synthesize gravity and optional topographic fields on a grid.
        
        Parameters
        ----------
        limits : sequence of four numbers
            [lon_min, lon_max, lat_min, lat_max], degrees.
        step : float or (float, float), optional
            Geographic grid step in longitude/latitude degrees.
        fields : str or sequence of str, optional
            'potential' (m**2/s**2), 'geoid' (m), 'abs_grav'
            (mGal), 'grav_dist' (mGal), 'free_air' (mGal),
            'grav_gradient' (Eotvos), and/or 'topo' (m).
        lmax : int or None, optional
            Highest gravity-model harmonic degree used.
        gentle_cut : int or None, optional
            Last degree before cosine tapering.
        height : float or array_like, optional
            Height offset in metres, broadcastable to the grid.
        surface : {'ellipsoid', 'sphere', 'mean_sea_level', 'topography'}
            'mean_sea_level' currently approximates the WGS84
            ellipsoid; topography uses approximate ellipsoidal heights.
        topo : array_like or None, optional
            Topographic heights, in metres. If needed and missing,
            Earth2014 coefficients are sought in the offline cache.
        topo_lmax : int or None, optional
            Independent maximum degree for Earth2014 topography.
        
        Returns
        -------
        dict[str, numpy.ndarray]
            2-D 'lon', 'lat' grids in degrees and requested output
            arrays in the units above, all of shape (nlat, nlon).
        
        Raises
        ------
        FileNotFoundError
            When Earth2014 is required but not present locally.
        ValueError
            For unsupported fields or invalid geometry.
        
        Examples
        --------
        >>> from relgravpy.sh_core import shm
        >>> model = shm("EIGEN-6C4", directory="shm_data")
        >>> out = model.synthesis([10, 11, 45, 45], step=1,
        ...                       lmax=60, fields=("grav_dist",))
        >>> out["grav_dist"].shape
        (1, 2)
        
        See Also
        --------
        sh_synthesis_core : Physical approximations and output units.
        sh_synthesis_with_topography : Offline topography workflow."""
>>>>>>> 9b2cf5ca54d3847a2269d9c6498bd4aeb63b3624
        return sh_synthesis_with_topography(
            self.data, limits=limits, step=step, fields=fields, lmax=lmax,
            gentle_cut=gentle_cut, height=height, surface=surface, topo=topo,
            directory=self.directory, topo_lmax=topo_lmax, lon=lon, lat=lat,
            height_reference=height_reference, geoid_undulation=geoid_undulation)


class shm_analysis:
    """Object-oriented scalar spherical harmonic analysis.
    
    Stores geographical sample arrays and delegates SH coefficient
    estimation to sh_analysis_core(). The scalar expansion does not
    apply the gravitational normal potential or radial derivatives.
    
    Parameters
    ----------
    lon, lat : array_like
        Longitude and latitude arrays in geographic degrees.
    values : array_like
        Corresponding scalar samples, in arbitrary consistent units.
    
    Attributes
    ----------
    lon, lat, values : numpy.ndarray
        Stored coordinate and sample arrays.
    
    Examples
    --------
    >>> import numpy as np
    >>> from relgravpy.sh_core import shm_analysis, sh_grid
    >>> lon, lat = sh_grid([-180, 150, -60, 60], 30)
    >>> analysis = shm_analysis(lon, lat, np.ones(lon.shape))
    >>> np.isclose(analysis.analyze(0, "lstsq")["c"][0, 0], 1.)
    True
    
    Notes
    -----
    Regional inversions may be non-unique or ill-conditioned."""
    def __init__(self, lon, lat, values):
        """Store the scalar samples to be analyzed.
        
        Parameters
        ----------
        lon, lat : array_like
            Matching longitude/latitude coordinates, degrees.
        values : array_like
            Scalar samples in consistent physical units.
        
        Returns
        -------
        None
            Sets self.lon, self.lat and self.values.
        
        Examples
        --------
        >>> from relgravpy.sh_core import shm_analysis
        >>> analysis = shm_analysis([[0.]], [[45.]], [[2.]])
        >>> analysis.values.shape
        (1, 1)"""
        self.lon=np.asarray(lon)
        self.lat=np.asarray(lat)
        self.values=np.asarray(values)

    def analyze(self, lmax, method="auto"):
        """Fit fully normalized SH coefficients to the stored scalar field.
        
        Parameters
        ----------
        lmax : int
            Maximum degree, giving (lmax+1)**2 real unknowns
            for least-squares fitting.
        method : {'auto', 'lstsq', 'fft'}, optional
            'auto' attempts global FFT then falls back to least
            squares; 'fft' requires global sampling; 'lstsq'
            supports finite regional samples.
        
        Returns
        -------
        dict
            'c', 's' coefficient arrays in the units of self.values,
            'lmax', reference 'radius' and either least-squares
            diagnostics ('rank', 'residuals', 'condition') or
            FFT method/quadrature metadata.
        
        Raises
        ------
        ValueError
            For unsupported methods or invalid sampling.
        MemoryError
            If the least-squares matrix exceeds 512 MiB.
        
        Examples
        --------
        >>> import numpy as np
        >>> from relgravpy.sh_core import shm_analysis, sh_grid
        >>> lon, lat = sh_grid([-180, 150, -60, 60], 30)
        >>> fit = shm_analysis(lon, lat, np.ones(lon.shape))
        >>> np.isclose(fit.analyze(0, "lstsq")["c"][0, 0], 1.)
        True"""
        return sh_analysis_core(self.lon,self.lat,self.values,lmax,
                                method=method)


