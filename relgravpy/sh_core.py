
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
    for m in range(2, lmax+1):
        p[m,m] = np.sqrt((2*m+1)/(2*m))*v*p[m-1,m-1]
    for m in range(1,lmax):
        p[m+1,m] = np.sqrt(2*m+3)*u*p[m,m]
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


def sh_synthesis_core(model, limits, step, lmax=None, gentle_cut=None,
                      fields=("grav_dist",), height=0.0, surface="ellipsoid",
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
    if isinstance(fields,str):
        fields=(fields,)
    allowed={"potential","geoid","abs_grav","grav_dist","free_air","grav_gradient"}
    if set(fields)-allowed:
        raise ValueError(f"Unsupported fields: {sorted(set(fields)-allowed)}")
    lon,lat=sh_grid(limits,step)
    degree=sh_resolve_degree(lmax,model["lmax"])
    w=sh_taper(degree,gentle_cut)
    h=np.broadcast_to(np.asarray(height,dtype=float),lat.shape).copy()
    if surface == "topography":
        if topo is None:
            raise ValueError("topography surface requires explicit topo heights")
        h += np.broadcast_to(topo,lat.shape)
    elif surface not in ("ellipsoid","sphere","mean_sea_level"):
        raise ValueError("Unknown surface")
    if surface == "sphere":
        r=np.full_like(lat,model["radius"])+h
        phi_gc=lat.copy()
    else:
        # Approximate MSL by WGS84 ellipsoid: precise geoid is not obtained
        # until the iterative surface geometry has been implemented.
        phi_gc,r=sh_geocentric_coordinates(lat,h)

    delta_c=model["c"][:degree+1,:degree+1].copy()
    delta_s=model["s"][:degree+1,:degree+1]
    delta_c[0,0] -= GM_WGS84/model["gm"]
    reference=sh_wgs84_normal_zonals(model["gm"],model["radius"],degree)
    for n in range(2,degree+1,2):
        delta_c[n,0]-=reference[n]
    # Degree 1 is kept if the ICGEM model's reference origin differs.
    results={name:np.zeros_like(lon) for name in fields}
    longs=np.deg2rad(lon[0])
    order=np.arange(degree+1)
    trigc=np.cos(order[:,None]*longs[None,:])
    trigs=np.sin(order[:,None]*longs[None,:])
    pot=np.zeros_like(lon)
    dg=np.zeros_like(lon)
    grad=np.zeros_like(lon)
    for i in range(lat.shape[0]):
        if np.all(phi_gc[i] == phi_gc[i,0]):
            # Constant geocentric latitude across the row: compute Legendre
            # functions only once, then vectorize over longitude.
            p=sh_legendre(phi_gc[i,0],degree)
            radial=np.ones(lon.shape[1])
            x=model["radius"]/r[i]
            for n in range(degree+1):
                if n:
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
            # Longitude-varying topography requires a distinct radius and
            # geocentric latitude at each evaluation point.
            for j in range(lon.shape[1]):
                p=sh_legendre(phi_gc[i,j],degree)
                x=model["radius"]/r[i,j]
                radial=1.0
                for n in range(degree+1):
                    if n:
                        radial*=x
                    harmonics=(delta_c[n,:n+1]*trigc[:n+1,j]+
                               delta_s[n,:n+1]*trigs[:n+1,j])
                    hnm=w[n]*radial*np.dot(p[n,:n+1],harmonics)
                    pot[i,j]+=hnm
                    dg[i,j]+=(n+1)*hnm
                    grad[i,j]+=(n+1)*(n+2)*hnm
        pot[i]*=model["gm"]/r[i]
        dg[i]*=model["gm"]/r[i]**2
        grad[i]*=-model["gm"]/r[i]**3
    if "potential" in results:
        results["potential"]=pot  # m2/s2, disturbing potential
    if "grav_dist" in results:
        results["grav_dist"]=dg*1e5
    if "geoid" in results:
        results["geoid"]=pot/sh_normal_gravity(lat)
    if "free_air" in results:
        # Spherical approximation to the gravity anomaly, not identical
        # to the ICGEM ellipsoidal free-air computation.
        results["free_air"]=(dg-2*pot/r)*1e5
    if "abs_grav" in results:
        results["abs_grav"]=(sh_normal_gravity_height(lat,h)+dg)*1e5
    if "grav_gradient" in results:
        results["grav_gradient"]=grad*1e9
    return {"lon":lon,"lat":lat,**results}



def sh_analysis_fft(lon, lat, values, lmax):
    """Global scalar SH analysis using longitude FFT and latitude quadrature.

    Requires a full 360-degree periodic longitude grid with no duplicate
    endpoint, and latitude coverage from pole to pole. On equiangular
    latitudes the area integral uses trapezoidal integration in sin(lat)
    and is approximate, not an exact Gauss-Legendre transform.
    """
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
    u=np.sin(np.deg2rad(lats))
    du=np.abs(np.diff(u))
    ww=np.zeros(ny)
    ww[:-1]+=du/2.
    ww[1:]+=du/2.
    # The factor exp(-i m lambda0) accounts for arbitrary longitude start.
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
    """Least-squares scalar SH analysis on a gridded surface.

    For regional grids this solution is non-unique and potentially ill-
    conditioned. Use global sampling for a defensible transform.
    """
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
    if yy.size*k*8>512*1024**2:
        raise MemoryError(
            "Least-squares design matrix exceeds 512 MiB. Use a full "
            "global grid and method='fft', or reduce lmax.")
    A=np.zeros((yy.size,k),float)
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
    """Retrieve the original Earth2014 SUR spherical harmonics (~933 MB).

    This is a one-time explicit download for later offline work.
    Dataset: Hirt & Rexer (2015), DOI 10.1016/j.jag.2015.03.001.
    """
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
    """Read Curtin Earth2014 4-pi-normalized SHCs from a binary .bshc.

    The file stores two float64 values (nmin,nmax), followed by all C(n,m)
    and all S(n,m), each in ascending degree then order. Arrays are returned
    in metres. Memmapping avoids loading the entire ~933MB file.
    """
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
    """Synthesize a fully normalized scalar field from dimensionful SHCs.

    Longitude and latitude are 2-D meshgrids; return matching 2-D output.
    This is appropriate for Earth2014 elevation coefficients (metres).
    """
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


def sh_synthesis_with_topography(model, limits, step, fields, lmax,
                                 gentle_cut, height, surface, topo,
                                 directory=None, topo_lmax=None):
    """Synthesize available gravity fields, optionally adding Earth2014 topo.

    Earth2014 is sought locally only; downloading it always requires a
    separate explicit call. No hidden network I/O is performed.
    """
    if isinstance(fields,str):
        fields=(fields,)
    fields=tuple(fields)
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
                              grav_fields,height,surface,topo)
    else:
        lon,lat=sh_grid(limits,step)
        out={"lon":lon,"lat":lat}
    if "topo" in fields:
        out["topo"]=np.broadcast_to(topo,out["lon"].shape).copy()
    return out


class shm:
    """Read a local gravity SH model, or explicitly download an ICGEM .gfc."""
    def __init__(self, model, directory=None, download=False, lmax=None):
        self.directory=Path(directory) if directory is not None else DEFAULT_DATA_DIR
        self.path=Path(model)
        if not self.path.is_file():
            self.path=self.directory/(("EIGEN-6C4" if str(model) == "EIGENC4" else str(model))+".gfc")
        if not self.path.is_file() and download:
            self.path=sh_download_model(str(model),self.directory)
        if not self.path.is_file():
            raise FileNotFoundError(f"Model unavailable offline: {self.path}")
        self.data=sh_read_gfc(self.path,lmax=lmax)

    def synthesis(self, limits, step=1., fields=("grav_dist",), lmax=None,
                  gentle_cut=None, height=0., surface="ellipsoid", topo=None,
                  topo_lmax=None):
        return sh_synthesis_with_topography(
            self.data, limits, step, fields, lmax, gentle_cut,
            height, surface, topo, directory=self.directory,
            topo_lmax=topo_lmax)


class shm_analysis:
    """Estimate fully-normalized scalar SH coefficients by least squares."""
    def __init__(self, lon, lat, values):
        self.lon=np.asarray(lon)
        self.lat=np.asarray(lat)
        self.values=np.asarray(values)

    def analyze(self, lmax, method="auto"):
        return sh_analysis_core(self.lon,self.lat,self.values,lmax,
                                method=method)
