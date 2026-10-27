"""Spherical-harmonic gravity utilities for RelGravPy.

This module centralizes the low-level gravity-field utilities used to read,
validate, download, and synthesize spherical-harmonic models such as EIGEN-6C4.
It provides functions to read ICGEM ``.gfc`` files, resolve the maximum degree,
and perform spherical-harmonic synthesis for gravity-related fields.

The module supports:
- reading ICGEM ``.gfc`` coefficient files;
- downloading official model files when needed;
- applying normal-field corrections and WGS84 conventions;
- synthesizing gravity-related fields such as ``grav_dist`` and ``abs_grav``;
- evaluating models on regular grids or custom lon/lat arrays;
- loading Earth2014 topography files when available.

Examples
--------
>>> from relgravpy.shm_tools import shm
>>> model = shm("EIGEN-6C4", directory="./shm_data", download=False)
>>> result = model.synthesis(
...     limits=[7.0, 7.2, 45.0, 45.2],
...     step=0.05,
...     fields=("abs_grav",),
...     lmax=120,
... )
>>> "abs_grav" in result
True
"""

# Import standard libraries
from pathlib import Path
from urllib.request import urlopen
import warnings
import numpy as np

# WGS84 reference constants
A_WGS84 = 6378137.0
F_WGS84 = 1.0 / 298.257223563
GM_WGS84 = 3.986004418e14
OMEGA_WGS84 = 7.292115e-5

# Verified ICGEM model URLs and default data directory
MODEL_URLS = {
    "EIGEN-6C4": "https://icgem.gfz.de/getmodel/gfc/7fd8fe44aa1518cd79ca84300aef4b41ddb2364aef9e82b7cdaabdb60a9053f1/EIGEN-6C4.gfc",
    "EIGENC4": "https://icgem.gfz.de/getmodel/gfc/7fd8fe44aa1518cd79ca84300aef4b41ddb2364aef9e82b7cdaabdb60a9053f1/EIGEN-6C4.gfc",
    "WHU-CASM-UGM2025_2159": "https://icgem.gfz.de/getmodel/gfc/82962bb8156bfbf074b8f97bf558ac0e0ee5e481c864a29441fb0f68ddb26be2/WHU-CASM-UGM2025_2159.gfc",
}
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "shm_data"

EARTH2014_SUR_URL = ("https://ddfe.blazejbucha.com/models/Earth2014/"
                     "data_1min/shcs_to10800/Earth2014.SUR2014.degree10800.bshc")

# -----------------------------------------------------------------------------
def sh_read_gfc(path, lmax=None):
    """Read a fully normalized ICGEM ``.gfc`` gravity model into coefficient arrays.

    This function parses the static coefficient block of an ICGEM gravity model
    and returns a dictionary that contains the cosine and sine coefficient arrays
    plus the metadata required for later synthesis. It supports the standard
    ``gfc`` format used by global gravity models such as EIGEN-6C4.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ICGEM coefficient file.
    lmax : int or None, optional
        Maximum spherical-harmonic degree to read. If ``None``, the file maximum
        degree is used.

    Returns
    -------
    dict
        Dictionary with keys ``"c"``, ``"s"``, ``"gm"``, ``"radius"``,
        ``"lmax"``, ``"max_degree"``, ``"modelname"`` and ``"header"``.

    Raises
    ------
    ValueError
        If the input file is missing the required ``end_of_head`` block or uses a
        non-supported normalization.
    NotImplementedError
        If time-varying coefficients are encountered.

    Examples
    --------
    >>> model = sh_read_gfc("./shm_data/EIGEN-6C4.gfc", lmax=120)
    >>> model["c"].shape
    (121, 121)
    """
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

    out_dict = {
        "c": c,
        "s": s,
        "gm": float(header["earth_gravity_constant"].replace("D","E")),
        "radius": float(header["radius"].replace("D","E")),
        "lmax": degree,
        "max_degree": available,
        "modelname": header.get("modelname", path.stem),
        "header": header
    }

    return out_dict


# -----------------------------------------------------------------------------
def sh_download_model(name, directory=None, overwrite=False):
    """Download a verified ICGEM model file and cache it locally.

    Parameters
    ----------
    name : str
        Registered model name. Supported values are defined in ``MODEL_URLS``.
    directory : str or pathlib.Path, optional
        Destination directory where the model file will be stored.
    overwrite : bool, optional
        If ``True``, overwrite an existing cached file.

    Returns
    -------
    pathlib.Path
        Local filesystem path of the downloaded model file.

    Raises
    ------
    KeyError
        If the requested model name is not in the verified registry.
    IOError
        If the download completes but the file content is empty or invalid.
    """
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


# -----------------------------------------------------------------------------
def sh_resolve_degree(requested, available):
    """Return the valid spherical-harmonic degree within the available model range.

    Parameters
    ----------
    requested : int or None
        Requested maximum degree.
    available : int
        Maximum degree available in the loaded model.

    Returns
    -------
    int
        The effective degree, clipped to the maximum supported by the model.

    Raises
    ------
    ValueError
        If the requested degree is negative.
    """
    if requested is None:
        return int(available)
    requested = int(requested)
    if requested < 0:
        raise ValueError("lmax must be non-negative")
    if requested > available:
        warnings.warn(f"Requested degree {requested} exceeds {available}; using {available}",
                      UserWarning, stacklevel=2)
    return min(requested, int(available))


# -----------------------------------------------------------------------------
def sh_taper(lmax, gentle_cut=None):
    """Build a cosine taper for the spherical-harmonic spectrum.

    Parameters
    ----------
    lmax : int
        Maximum degree of the spectrum.
    gentle_cut : int or None, optional
        Degree beyond which the taper starts to decay. If ``None``, the taper is
        unity for all degrees.

    Returns
    -------
    ndarray
        Array of weights with values between 1 and 0.
    """
    w = np.ones(lmax+1)
    if gentle_cut is None:
        return w
    cut = int(gentle_cut)
    if not 0 <= cut < lmax:
        raise ValueError("gentle_cut must be between 0 and lmax-1")
    degrees = np.arange(cut+1, lmax+1, dtype=float)
    w[cut+1:] = 0.5*(1+np.cos(np.pi*(degrees-cut)/(lmax-cut)))
    return w


# -----------------------------------------------------------------------------
def sh_legendre(latitude_deg, lmax):
    """Compute fully normalized associated Legendre functions at a single latitude.

    Parameters
    ----------
    latitude_deg : float
        Geodetic latitude in degrees.
    lmax : int
        Maximum degree of the Legendre expansion.

    Returns
    -------
    ndarray
        Array of shape ``(lmax + 1, lmax + 1)`` containing ``P_n^m`` values for
        all degrees and orders up to ``lmax``.
    """
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


# -----------------------------------------------------------------------------
def sh_grid(limits, step):
    """Generate a regular geographic grid from a bounding box.

    Parameters
    ----------
    limits : sequence of float
        Geographic limits in the order ``(lon_min, lon_max, lat_min, lat_max)``.
    step : float or tuple of float
        Grid spacing in degrees. If a scalar is provided, the same value is used
        for both axes.

    Returns
    -------
    tuple of ndarray
        Longitude and latitude 2-D meshgrids with the requested spacing.

    Examples
    --------
    >>> lon, lat = sh_grid([0.0, 2.0, 45.0, 47.0], 0.5)
    >>> lon.shape == lat.shape
    True
    """
    lon0,lon1,lat0,lat1 = map(float, limits)
    if not (-90 <= lat0 <= lat1 <= 90 and lon0 <= lon1):
        raise ValueError("Invalid geographic limits")
    dl = (float(step),float(step)) if np.isscalar(step) else tuple(step)
    if len(dl) != 2 or min(dl) <= 0:
        raise ValueError("step must be positive")
    lon = np.arange(lon0,lon1+dl[0]*0.25,dl[0])
    lat = np.arange(lat0,lat1+dl[1]*0.25,dl[1])
    return np.meshgrid(lon,lat)


# -----------------------------------------------------------------------------
def sh_ellipsoid_radius(lat):
    """Return the WGS84 ellipsoid radius for a geodetic latitude.

    Parameters
    ----------
    lat : float or ndarray
        Geodetic latitude in degrees.

    Returns
    -------
    float or ndarray
        Ellipsoid radius in metres.
    """
    phi = np.deg2rad(np.asarray(lat, dtype=float))
    b = A_WGS84*(1-F_WGS84)
    cp,sp = np.cos(phi),np.sin(phi)
    return np.sqrt(((A_WGS84*A_WGS84*cp)**2+(b*b*sp)**2)/
                   ((A_WGS84*cp)**2+(b*sp)**2))


# -----------------------------------------------------------------------------
def sh_normal_gravity(lat):
    """Compute the WGS84 normal gravity on the ellipsoid.

    The formula follows the Somigliana approximation for the normal gravity of a
    reference ellipsoid.

    Parameters
    ----------
    lat : float or ndarray
        Geodetic latitude in degrees.

    Returns
    -------
    float or ndarray
        Normal gravity in m/s^2.
    """
    phi = np.deg2rad(np.asarray(lat, dtype=float))
    e2 = F_WGS84*(2-F_WGS84)
    k = 0.00193185265241
    return 9.7803253359*(1+k*np.sin(phi)**2)/np.sqrt(1-e2*np.sin(phi)**2)


# -----------------------------------------------------------------------------
def sh_wgs84_normal_zonals(gm, radius, lmax):
    """Compute the even zonal coefficients of the WGS84 normal potential.

    Parameters
    ----------
    gm : float
        Gravitational parameter of the model in m^3/s^2.
    radius : float
        Reference radius in metres.
    lmax : int
        Maximum degree of the zonal coefficients to compute.

    Returns
    -------
    ndarray
        Fully normalized even zonal coefficients up to degree ``lmax``.
    """
    j2=1.08262982131e-3
    e2=F_WGS84*(2-F_WGS84)
    out=np.zeros(lmax+1)
    for n in range(1,lmax//2+1):
        k=2*n
        j2n=((-1)**(n+1) * 3*(e2**n)/((k+1)*(k+3)) *
             (1-n+5*n*j2/e2))
        out[k]=-j2n/np.sqrt(2*k+1) * (GM_WGS84/gm) * (A_WGS84/radius)**k
    return out


# -----------------------------------------------------------------------------
def sh_geocentric_coordinates(lat, height):
    """Convert geodetic latitude and height to geocentric coordinates.

    Parameters
    ----------
    lat : float or ndarray
        Geodetic latitude in degrees.
    height : float or ndarray
        Ellipsoidal height in metres.

    Returns
    -------
    tuple of ndarray
        Geocentric latitude in degrees and geocentric radius in metres.
    """
    ph=np.deg2rad(np.asarray(lat,dtype=float))
    e2=F_WGS84*(2-F_WGS84)
    N=A_WGS84/np.sqrt(1-e2*np.sin(ph)**2)
    x=(N+height)*np.cos(ph)
    z=(N*(1-e2)+height)*np.sin(ph)
    return np.rad2deg(np.arctan2(z,x)),np.hypot(x,z)


# -----------------------------------------------------------------------------
def sh_normal_gravity_height(lat, height):
    """Approximate the normal gravity continuation above the ellipsoid.

    Parameters
    ----------
    lat : float or ndarray
        Geodetic latitude in degrees.
    height : float or ndarray
        Ellipsoidal height in metres.

    Returns
    -------
    float or ndarray
        Normal gravity at the requested altitude in m/s^2.
    """
    ph=np.deg2rad(np.asarray(lat,dtype=float))
    m=OMEGA_WGS84**2*A_WGS84**2*(A_WGS84*(1-F_WGS84))/GM_WGS84
    q=1+F_WGS84+m-2*F_WGS84*np.sin(ph)**2
    h=np.asarray(height,dtype=float)
    return sh_normal_gravity(lat)*(1-2*q*h/A_WGS84+3*(h/A_WGS84)**2)


# -----------------------------------------------------------------------------
def sh_synthesis_core(model, limits=None, step=1., lmax=None, gentle_cut=None,
                      fields=("grav_dist",), height=0.0, surface="ellipsoid",
                      topo=None, lon=None, lat=None, height_reference="geoid",
                      geoid_undulation=None):
    """Synthesize gravity quantities from a spherical-harmonic model.

    This is the core forward model of the module. It evaluates the selected
    spherical-harmonic coefficients on either a regular geographic grid or a
    custom lon/lat mesh, then reconstructs the requested geodetic quantities.

    Parameters
    ----------
    model : dict
        Model dictionary produced by :func:`sh_read_gfc`.
    limits : tuple or None, optional
        Bounding box in the form ``(lon_min, lon_max, lat_min, lat_max)``.
    step : float or tuple, optional
        Grid spacing in degrees.
    lmax : int or None, optional
        Degree cutoff for synthesis. If ``None``, the model maximum is used.
    gentle_cut : int or None, optional
        Optional taper cutoff used to smooth the highest-order terms.
    fields : tuple of str, optional
        Requested output fields, e.g. ``("grav_dist",)`` or
        ``("abs_grav", "geoid")``.
    height : float or ndarray, optional
        Height value(s) used in the synthesis.
    surface : {"ellipsoid", "sphere", "mean_sea_level", "topography"}
        Reference surface for the evaluation.
    topo : ndarray or None, optional
        Topography grid when a topographic surface is required.
    lon, lat : ndarray or None, optional
        Explicit evaluation mesh; if given, they override the automatic grid.
    height_reference : {"geoid", "ellipsoid"}, optional
        Interpretation of the supplied height values.
    geoid_undulation : float or ndarray, optional
        Geoid undulation in metres when ``height_reference='geoid'``.

    Returns
    -------
    dict
        Dictionary with keys ``"lon"``, ``"lat"`` and any requested field names.
    """
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
            if lon.shape != lat.shape:
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

        if geoid_undulation is None:
            # When the user does not provide an explicit geoid correction, compute
            # the undulation from the same model. This keeps the default behavior
            # consistent with the gravity model itself rather than assuming zero.
            geoid_reference = sh_synthesis_core(
                model,
                limits=limits,
                step=step,
                lmax=lmax,
                gentle_cut=gentle_cut,
                fields=("geoid",),
                height=0.0,
                surface="ellipsoid",
                topo=topo,
                lon=lon,
                lat=lat,
                height_reference="ellipsoid",
                geoid_undulation=0.0,
            )["geoid"]
            N = np.asarray(geoid_reference, dtype=float)
        else:
            N = np.asarray(geoid_undulation, dtype=float)
        if N.shape == ():
            N = np.full_like(h, N, dtype=float)
        elif N.shape != h.shape:
            N = np.broadcast_to(N, h.shape)
        h = h + N
    elif height_reference != "ellipsoid":
        raise ValueError("height_reference must be 'geoid' or 'ellipsoid'")

    if surface == "sphere":
        r=np.full_like(lat,model["radius"])+h
        phi_gc=lat.copy()
    else:
        phi_gc,r=sh_geocentric_coordinates(lat,h)

    delta_c=model["c"][:degree+1,:degree+1].copy()
    delta_s=model["s"][:degree+1,:degree+1]
    delta_c[0,0] -= GM_WGS84/model["gm"]
    reference=sh_wgs84_normal_zonals(model["gm"],model["radius"],degree)
    for n in range(2,degree+1,2):
        delta_c[n,0]-=reference[n]
    results={name:np.zeros_like(lon) for name in fields}
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

    # Store computed results in the output dictionary
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

    out_dict = {"lon":lon,"lat":lat,**results}
    return out_dict



# -----------------------------------------------------------------------------
def sh_analysis_fft(lon, lat, values, lmax):
    """Estimate spherical-harmonic coefficients from a full global grid via FFT.

    This routine is intended for global or near-global fields sampled on a regular
    longitude/latitude grid. It is substantially faster than the least-squares
    inversion and is suitable when the grid spans the full 360 degrees in
    longitude and covers the poles in latitude.

    Parameters
    ----------
    lon, lat, values : ndarray
        Two-dimensional arrays with matching shapes representing longitude,
        latitude and scalar field values.
    lmax : int
        Maximum spherical-harmonic degree to estimate.

    Returns
    -------
    dict
        Dictionary containing the cosine and sine coefficients as well as the
        field metadata.

    Raises
    ------
    ValueError
        If the grid is not regular or does not satisfy the assumptions required
        by the FFT-based harmonic analysis.
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



# -----------------------------------------------------------------------------
def sh_analysis_core(lon, lat, values, lmax, radius=6378136.3,
                     normalization="fully_normalized", method="auto"):
    """Estimate spherical-harmonic coefficients from a sampled scalar field.

    Parameters
    ----------
    lon, lat, values : ndarray
        Arrays describing the sampled scalar field on a regular or irregular grid.
    lmax : int
        Maximum spherical-harmonic degree to estimate.
    radius : float, optional
        Reference radius used in the design matrix, by default 6378136.3 m.
    normalization : str, optional
        Normalization convention. Only ``"fully_normalized"`` is supported.
    method : {"auto", "fft", "lstsq"}, optional
        Method used to estimate the coefficients.

    Returns
    -------
    dict
        Dictionary with the estimated cosine and sine coefficients and the metadata
        used by the synthesis routines.

    Notes
    -----
    For a regional grid, the solution can be ill-conditioned and therefore should
    be interpreted with care. The FFT-based estimate is preferred when the grid
    satisfies the global assumptions.
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

    out_dict ={ "c":c,"s":s,"lmax":lmax,"radius":radius,
                "rank":rank,"residuals":residuals,"condition":(singular[0]/singular[-1]
                              if singular[-1] else np.inf)}
    return out_dict

# -----------------------------------------------------------------------------
def sh_download_topography(directory=None, overwrite=False):
    """Download the Earth2014 topography spherical-harmonic coefficients.

    This helper fetches the official Earth2014 SUR model, which is used for
    synthetic topography grids when a surface is evaluated on a terrain model
    instead of on the ellipsoid or sphere.

    Parameters
    ----------
    directory : str or pathlib.Path, optional
        Destination directory for the cached model file. If omitted, the module's
        default data directory is used.
    overwrite : bool, optional
        If ``True``, force a fresh download even when the file already exists.

    Returns
    -------
    pathlib.Path
        Local path of the downloaded topography model.

    Notes
    -----
    The Earth2014 SUR dataset is large (~933 MB). The file is cached locally so
    repeated synthesis runs do not require re-downloading it unless the caller
    explicitly requests an overwrite.
    """
    directory = Path(directory) if directory is not None else DEFAULT_DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)

    # Final destination for the cached model.
    dest = directory / "Earth2014.SUR2014.degree10800.bshc"
    if dest.exists() and not overwrite:
        return dest

    # Write to a temporary file first so a partial/incomplete download is not
    # mistaken for a valid model file.
    tmp = dest.with_suffix(".bshc.part")
    try:
        with urlopen(EARTH2014_SUR_URL, timeout=120) as response, tmp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        if tmp.stat().st_size < 16:
            raise IOError("Earth2014 download is empty or truncated")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)

    return dest


# -----------------------------------------------------------------------------
def sh_read_bshc(path, lmax=None):
    """Read Earth2014 binary spherical-harmonic coefficients from a ``.bshc`` file.

    The Earth2014 SUR model stores coefficients in a compact binary layout: first
    the minimum and maximum degree, then the cosine and sine coefficients for all
    degrees and orders in ascending order. This routine loads the file with a
    memory map so only the requested harmonic range is materialized in RAM.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a valid Earth2014 ``.bshc`` file.
    lmax : int or None, optional
        Maximum degree to read. If omitted, the file maximum degree is used.

    Returns
    -------
    dict
        Dictionary containing the coefficient arrays and metadata required by the
        scalar topography synthesis routines.
    """
    path = Path(path)

    # The binary header stores only the minimum and maximum degree of the model.
    header = np.fromfile(path, dtype="<f8", count=2)
    if len(header) != 2:
        raise ValueError("Invalid .bshc header")

    nmin, nmax = map(int, header)
    if (nmin, nmax) != (0, round(float(header[1]))) or nmax < 0 or nmax > 50000:
        raise ValueError("Unexpected .bshc header or byte order")

    total = (nmax + 1) * (nmax + 2) // 2
    expected = (2 + 2 * total) * 8
    if path.stat().st_size != expected:
        raise ValueError(f"Wrong Earth2014 size: expected {expected} bytes")

    degree = sh_resolve_degree(lmax, nmax)

    # The coefficient block follows the header; each coefficient is stored in
    # degree-order order, compacted to the triangular storage pattern.
    raw = np.memmap(path, dtype="<f8", mode="r", offset=16, shape=(2 * total,))
    c = np.zeros((degree + 1, degree + 1))
    s = np.zeros_like(c)

    for n in range(degree + 1):
        i = n * (n + 1) // 2
        c[n, :n + 1] = raw[i:i + n + 1]
        s[n, :n + 1] = raw[total + i:total + i + n + 1]

    return {
        "c": c,
        "s": s,
        "lmax": degree,
        "max_degree": nmax,
        "modelname": "Earth2014 SUR2014",
        "unit": "m",
    }


# -----------------------------------------------------------------------------
def sh_scalar_synthesis(coeff, lon, lat, lmax=None, gentle_cut=None):
    """Synthesize a scalar field from a fully normalized spherical-harmonic set.

    This routine is used mainly for the Earth2014 topography coefficients, where
    the input coefficients are expressed in metres and the output is a scalar
    elevation field defined on a geographic grid.

    Parameters
    ----------
    coeff : dict
        Coefficient dictionary returned by :func:`sh_read_bshc` or a similarly
        structured spherical-harmonic model.
    lon, lat : ndarray
        2-D longitude and latitude meshgrids with matching shapes.
    lmax : int or None, optional
        Maximum harmonic degree to synthesize.
    gentle_cut : int or None, optional
        Optional taper used to smooth the highest-degree terms.

    Returns
    -------
    ndarray
        2-D scalar field sampled on the provided ``lon/lat`` grid.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    if lon.shape != lat.shape or lon.ndim != 2:
        raise ValueError("lon/lat must be matching 2-D grids")

    degree = sh_resolve_degree(lmax, coeff["lmax"])
    w = sh_taper(degree, gentle_cut)
    out = np.empty_like(lon)
    m = np.arange(degree + 1)

    # The longitude dependence is expressed as a trigonometric series in order m.
    cos_m = np.cos(m[:, None] * np.deg2rad(lon[0])[None, :])
    sin_m = np.sin(m[:, None] * np.deg2rad(lon[0])[None, :])
    cc, ss = coeff["c"], coeff["s"]

    # Evaluate latitude-by-latitude: for each row, compute the associated
    # Legendre functions and combine them with the cosine/sine series.
    for i, phi in enumerate(lat[:, 0]):
        p = sh_legendre(phi, degree)
        a = np.sum(p[:degree + 1, :degree + 1] * cc[:degree + 1, :degree + 1]
                   * w[:, None], axis=0)
        b = np.sum(p[:degree + 1, :degree + 1] * ss[:degree + 1, :degree + 1]
                   * w[:, None], axis=0)
        out[i] = a @ cos_m + b @ sin_m

    return out


# -----------------------------------------------------------------------------
def sh_synthesis(model, limits=None, step=1., fields=("grav_dist",),
                lmax=None, gentle_cut=None, height=0., surface="ellipsoid",
                topo=None, directory=None, topo_lmax=None, lon=None, lat=None,
                height_reference="geoid", geoid_undulation=None,
                x=None, y=None, z=None):

    """Unified public synthesis entry point for spherical-harmonic gravity models.

    This function centralizes the full workflow that previously lived in separate
    helpers: it accepts either a regular geographic box (``limits`` + ``step``),
    an explicit custom lon/lat mesh, or a direct ``x/y/z`` point set, then
    evaluates the requested gravity quantities on the selected geometry.

    Parameters
    ----------
    model : dict
        Dictionary returned by :func:`sh_read_gfc` and containing the spherical
        harmonic coefficients and metadata.
    limits : tuple or None, optional
        Geographic limits in the form ``(lon_min, lon_max, lat_min, lat_max)``.
    step : float or tuple, optional
        Grid spacing in degrees used when building a regular grid.
    fields : tuple of str, optional
        Gravity fields to synthesize, for example ``("grav_dist",)`` or
        ``("abs_grav", "geoid")``.
    lmax : int or None, optional
        Maximum harmonic degree to evaluate.
    gentle_cut : int or None, optional
        Optional taper used to smooth the highest-degree terms.
    height : float or ndarray, optional
        Height values used in the synthesis.
    surface : {"ellipsoid", "sphere", "mean_sea_level", "topography"}, optional
        Evaluation surface used for the gravity field.
    topo : ndarray or None, optional
        Topography field for a topographic surface evaluation.
    directory : pathlib.Path or None, optional
        Directory containing the optional Earth2014 topography file.
    topo_lmax : int or None, optional
        Degree used when generating a synthetic topography model.
    lon, lat : ndarray or None, optional
        Explicit longitude/latitude arrays or meshgrids.
    x, y, z : ndarray or scalar, optional
        Alternative direct evaluation interface. ``x`` and ``y`` are the
        longitude and latitude arrays or meshes, while ``z`` is the height.
    height_reference : {"geoid", "ellipsoid"}, optional
        Height convention for the supplied values.
    geoid_undulation : float or ndarray, optional
        Geoid undulation in metres when ``height_reference='geoid'``.

    Returns
    -------
    dict
        Dictionary keyed by the selected field names and containing the
        corresponding ``"lon"`` and ``"lat"`` grids.
    """
    # Normalize the requested field names to a tuple so the rest of the routine
    # can treat them uniformly regardless of whether the caller passed a single
    # string or a sequence of strings.
    if isinstance(fields, str):
        fields = (fields,)
    fields = tuple(fields)

    # -------------------------------------------------------------------------
    # 1) Custom point-based input: x/y/z arrays or meshgrids.
    # -------------------------------------------------------------------------
    # This branch is used when the caller already has the evaluation coordinates
    # in Python and wants to skip the definition of a regular bounding box.
    # It accepts both one-dimensional point arrays and full 2-D meshes.
    if any(v is not None for v in (x, y, z)):
        if lon is not None or lat is not None:
            raise ValueError("Use either lon/lat or x/y/z for custom evaluation points")
        if x is None or y is None:
            raise ValueError("x and y are required when providing explicit custom coordinates")

        # Interpret x/y as geographic coordinates and z as the height values.
        lon = np.asarray(x, dtype=float)
        lat = np.asarray(y, dtype=float)
        if z is None:
            z = height
        height = np.asarray(z, dtype=float)

    # -------------------------------------------------------------------------
    # 2) Explicit lon/lat evaluation mesh.
    # -------------------------------------------------------------------------
    # This is the usual path for user-defined grids that are not derived from a
    # geographic bounding box. One-dimensional arrays are expanded to a mesh only
    # when they are not already a matching pair of point coordinates.
    if lon is not None or lat is not None:
        if lon is None or lat is None:
            raise ValueError("Both lon and lat must be provided for a custom grid")

        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)

        if lon.ndim == 1 and lat.ndim == 1 and lon.shape != lat.shape:
            lon, lat = np.meshgrid(lon, lat)
        elif lon.shape != lat.shape:
            raise ValueError("lon and lat must have matching shapes")

        if topo is None and (surface == "topography" or "topo" in fields):
            raise ValueError("topography surface requires explicit topo values for custom grids")

        grav_fields = tuple(item for item in fields if item != "topo")

        if grav_fields:
            # Evaluate the selected gravity fields on the custom grid.
            out = sh_synthesis_core(model, limits=limits, step=step, lmax=lmax,
                                   gentle_cut=gentle_cut, fields=grav_fields,
                                   height=height, surface=surface, topo=topo,
                                   lon=lon, lat=lat,
                                   height_reference=height_reference,
                                   geoid_undulation=geoid_undulation)
        else:
            out = {"lon": lon, "lat": lat}

        # If the caller explicitly asked for a topography array, attach it to the
        # result dictionary with the same geometry as the evaluation grid.
        if "topo" in fields:

            topo_arr = np.asarray(topo, dtype=float)
            if topo_arr.shape == ():
                topo_arr = np.full_like(np.asarray(lon, dtype=float), topo_arr, dtype=float)
            if topo_arr.shape != out["lon"].shape:
                topo_arr = np.broadcast_to(topo_arr, out["lon"].shape)
            out["topo"] = topo_arr.copy()

        return out

    # -------------------------------------------------------------------------
    # 3) Regular geographic grid generated from a bounding box.
    # -------------------------------------------------------------------------
    # This is the default path for maps and regional synthesis. It builds a
    # regular lon/lat mesh from the limits and step values and then evaluates the
    # model on that grid.
    if topo is None and (surface == "topography" or "topo" in fields):
        directory = Path(directory) if directory is not None else DEFAULT_DATA_DIR
        path = directory / "Earth2014.SUR2014.degree10800.bshc"

        if not path.exists():
            raise FileNotFoundError(
                f"Earth2014 missing: {path}. Call sh_download_topography() once.")
        lon, lat = sh_grid(limits, step)
        elevation = sh_read_bshc(path, lmax=topo_lmax)
        topo = sh_scalar_synthesis(elevation, lon, lat)

    grav_fields = tuple(item for item in fields if item != "topo")
    if grav_fields:
        # All remaining gravity quantities are evaluated by the common synthesis
        # core, which keeps the numerical implementation in one place.
        out = sh_synthesis_core(model, limits, step, lmax, gentle_cut,
                               grav_fields, height, surface, topo,
                               height_reference=height_reference,
                               geoid_undulation=geoid_undulation)
    else:
        lon, lat = sh_grid(limits, step)
        out = {"lon": lon, "lat": lat}

    if "topo" in fields:
        out["topo"] = np.broadcast_to(topo, out["lon"].shape).copy()

    return out

# -----------------------------------------------------------------------------
class shm:
    """Read a local gravity SH model, or explicitly download an ICGEM .gfc.

    The class resolves a model name or file path, loads the coefficient set, and
    then exposes a synthesis API for producing regional gravity maps or pointwise
    values. It is the main user-facing entry point for the scientific workflows
    in this module.

    Parameters
    ----------
    model : str or pathlib.Path
        Model name or path to a local ``.gfc`` file.
    directory : pathlib.Path or str, optional
        Directory in which the local model should be looked up.
    download : bool, optional
        If ``True``, fetch the model from the official ICGEM archive when it is
        not present locally.
    lmax : int or None, optional
        Maximum degree to read from the model file.
    """

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

    def synthesis(self, limits=None, step=1., fields=("grav_dist",), lmax=None,
                  gentle_cut=None, height=0., surface="ellipsoid", topo=None,
                  topo_lmax=None, lon=None, lat=None, height_reference="geoid",
                  geoid_undulation=None, x=None, y=None, z=None):

        """Synthesize one or more gravity fields for the loaded model.

        Parameters
        ----------
        limits : tuple or None, optional
            Geographic limits in the form ``(lon_min, lon_max, lat_min, lat_max)``.
        step : float or tuple, optional
            Spatial resolution in degrees.
        fields : tuple of str, optional
            Fields to evaluate, typically ``("grav_dist",)`` or
            ``("abs_grav", "geoid")``.
        lmax : int or None, optional
            Maximum degree used for the synthesis.
        gentle_cut : int or None, optional
            Optional taper to smooth the highest degrees.
        height : float or ndarray, optional
            Height at which the field is evaluated.
        surface : str, optional
            Evaluation surface, e.g. ``"ellipsoid"`` or ``"sphere"``.
        topo : ndarray or None, optional
            Optional topography field for topographic surface evaluation.
        topo_lmax : int or None, optional
            Degree used when computing the topography model.
        lon, lat : ndarray or None, optional
            Optional custom evaluation mesh.
        x, y, z : ndarray or scalar, optional
            Alternative custom grid input. ``x`` and ``y`` are the longitude and
            latitude values or meshgrids, while ``z`` is the corresponding height
            array. This lets the caller evaluate the model directly on arbitrary
            point sets or 2D arrays without first building a regular box grid.
        height_reference : {"geoid", "ellipsoid"}, optional
            Interpretation of the supplied height values.
        geoid_undulation : float or ndarray, optional
            Geoid undulation to use when ``height_reference='geoid'``.

        Returns
        -------
        dict
            Dictionary containing the synthesized field arrays and the associated
            longitude/latitude grids.
        """

        grid = sh_synthesis(
            self.data, limits=limits, step=step, fields=fields, lmax=lmax,
            gentle_cut=gentle_cut, height=height, surface=surface, topo=topo,
            directory=self.directory, topo_lmax=topo_lmax, lon=lon, lat=lat,
            height_reference=height_reference, geoid_undulation=geoid_undulation,
            x=x, y=y, z=z)

        return grid


# -----------------------------------------------------------------------------
class shm_analysis:
    """Estimate fully-normalized scalar SH coefficients by least squares.

    This class is useful when a regional scalar field is known and one wants to
    recover the underlying spherical-harmonic content, or when validating the
    forward synthesis against a reference dataset.
    """
    def __init__(self, lon, lat, values):
        self.lon=np.asarray(lon)
        self.lat=np.asarray(lat)
        self.values=np.asarray(values)

    def analyze(self, lmax, method="auto"):
        """Estimate the coefficients up to the requested degree.

        Parameters
        ----------
        lmax : int
            Maximum degree to estimate.
        method : {"auto", "fft", "lstsq"}, optional
            Estimation method. The default behavior tries the FFT route first and
            then falls back to a least-squares solve if needed.

        Returns
        -------
        dict
            Dictionary containing the estimated coefficients and metadata.
        """
        return sh_analysis_core(self.lon,self.lat,self.values,lmax,
                                method=method)


