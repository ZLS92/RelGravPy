
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
    """Stream static fully normalized ICGEM .gfc into degree/order arrays.

    Dynamic components are not supported; use a time-epoch-aware model reader
    for gfct/trnd/asin/acos records. Reject unsupported normalization.
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
    return {"c":c,"s":s,
            "gm":float(header["earth_gravity_constant"].replace("D","E")),
            "radius":float(header["radius"].replace("D","E")),
            "lmax":degree,"max_degree":available,
            "modelname":header.get("modelname",path.stem),"header":header}


def sh_download_model(name, directory=None, overwrite=False):
    """Download a verified ICGEM model file for durable offline use."""
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
    """Cosine taper equal to one through gentle_cut, reaching zero at lmax."""
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
    """Fully normalized geodetic associated Legendre functions at one latitude.

    Pnm has the 4-pi normalization and no Condon-Shortley phase. The
    three-term recurrence avoids factorial ratios and their overflow.
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


def sh_grid(limits, step):
    """Geographic grid: [lon_min, lon_max, lat_min, lat_max], step degrees."""
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
    """Geocentric distance of WGS84 ellipsoid at geodetic latitude."""
    phi = np.deg2rad(np.asarray(lat, dtype=float))
    b = A_WGS84*(1-F_WGS84)
    cp,sp = np.cos(phi),np.sin(phi)
    return np.sqrt(((A_WGS84*A_WGS84*cp)**2+(b*b*sp)**2)/
                   ((A_WGS84*cp)**2+(b*sp)**2))


def sh_normal_gravity(lat):
    """Somigliana normal gravity on WGS84 ellipsoid, m/s2."""
    phi = np.deg2rad(np.asarray(lat, dtype=float))
    e2 = F_WGS84*(2-F_WGS84)
    k = 0.00193185265241
    return 9.7803253359*(1+k*np.sin(phi)**2)/np.sqrt(1-e2*np.sin(phi)**2)



def sh_wgs84_normal_zonals(gm, radius, lmax):
    """Fully-normalized zonal coefficients of the WGS84 normal potential.

    Coefficients follow the even-zonal J2n formula of physical geodesy.
    Scaled to the GM and reference radius of the external .gfc model.
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


def sh_geocentric_coordinates(lat, height):
    """Geocentric latitude/radius for WGS84 geodetic latitude and height."""
    ph=np.deg2rad(np.asarray(lat,dtype=float))
    e2=F_WGS84*(2-F_WGS84)
    N=A_WGS84/np.sqrt(1-e2*np.sin(ph)**2)
    x=(N+height)*np.cos(ph)
    z=(N*(1-e2)+height)*np.sin(ph)
    return np.rad2deg(np.arctan2(z,x)),np.hypot(x,z)


def sh_normal_gravity_height(lat, height):
    """Second-order WGS84 normal gravity continuation above ellipsoid [m/s2]."""
    ph=np.deg2rad(np.asarray(lat,dtype=float))
    m=OMEGA_WGS84**2*A_WGS84**2*(A_WGS84*(1-F_WGS84))/GM_WGS84
    q=1+F_WGS84+m-2*F_WGS84*np.sin(ph)**2
    h=np.asarray(height,dtype=float)
    return sh_normal_gravity(lat)*(1-2*q*h/A_WGS84+3*(h/A_WGS84)**2)


def sh_synthesis_core(model, limits=None, step=1., lmax=None, gentle_cut=None,
                      fields=("grav_dist",), height=0.0, surface="ellipsoid",
                      topo=None, lon=None, lat=None, height_reference="geoid",
                      geoid_undulation=0.0):
    """Spherical harmonic synthesis with WGS84 even-zonal normal-field removal.

    Supports both regular grids from limits/step and user-defined lon/lat arrays.
    The default height convention is geoid-referenced (H), converted internally to
    ellipsoidal height h = H + N when height_reference='geoid'.
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

    def synthesis(self, limits=None, step=1., fields=("grav_dist",), lmax=None,
                  gentle_cut=None, height=0., surface="ellipsoid", topo=None,
                  topo_lmax=None, lon=None, lat=None, height_reference="geoid",
                  geoid_undulation=0.0):

        return sh_synthesis_with_topography(
            self.data, limits=limits, step=step, fields=fields, lmax=lmax,
            gentle_cut=gentle_cut, height=height, surface=surface, topo=topo,
            directory=self.directory, topo_lmax=topo_lmax, lon=lon, lat=lat,
            height_reference=height_reference, geoid_undulation=geoid_undulation)


class shm_analysis:
    """Estimate fully-normalized scalar SH coefficients by least squares."""
    def __init__(self, lon, lat, values):
        self.lon=np.asarray(lon)
        self.lat=np.asarray(lat)
        self.values=np.asarray(values)

    def analyze(self, lmax, method="auto"):
        return sh_analysis_core(self.lon,self.lat,self.values,lmax,
                                method=method)


