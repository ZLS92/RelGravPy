"""Pure NumPy Tamura T87 solid-Earth gravity tide predictor.

Uses the 1200-wave Tamura (1987) potential catalogue in HW95
normalization (H.-G. Wenzel, 1995). Astronomical and geodetic
conventions follow the published ETERNA theory. No PyGTide dependency.
UTC epochs; positive downward gravity; excludes ocean loading,
polar motion and atmospheric effects. Model not yet TSoft-validated.
"""
from datetime import datetime, timezone
from functools import lru_cache
from math import factorial, sqrt
from pathlib import Path
import numpy as np

_PATH = Path(__file__).resolve().parent / "data" / "tamurahw.dat"
_LEAPS = (("1972-01-01",10),("1972-07-01",11),("1973-01-01",12),
 ("1974-01-01",13),("1975-01-01",14),("1976-01-01",15),
 ("1977-01-01",16),("1978-01-01",17),("1979-01-01",18),
 ("1980-01-01",19),("1981-07-01",20),("1982-07-01",21),
 ("1983-07-01",22),("1985-07-01",23),("1988-01-01",24),
 ("1990-01-01",25),("1991-01-01",26),("1992-07-01",27),
 ("1993-07-01",28),("1994-07-01",29),("1996-01-01",30),
 ("1997-07-01",31),("1999-01-01",32),("2006-01-01",33),
 ("2009-01-01",34),("2012-07-01",35),("2015-07-01",36),
 ("2017-01-01",37))

@lru_cache(maxsize=1)
def _catalogue():
    arguments, degree, coeffs = [], [], []
    with _PATH.open(encoding="ascii", errors="replace") as f:
        for line in f:
            try:
                k = int(line[:6])
            except ValueError:
                continue
            if k == 999999:
                break
            if not 1 <= k <= 1200:
                continue
            degree.append(int(line[9:11]))
            arguments.append([int(line[11+3*j:14+3*j]) for j in range(11)])
            coeffs.append([float(line[i:j]) for i,j in
                           ((44,56),(56,68),(68,80),(80,90),(90,100))])
    if len(arguments) != 1200:
        raise ValueError("Incomplete Tamura T87 catalogue: expected 1200 waves")
    return (np.asarray(arguments, dtype=int),
            np.asarray(degree, dtype=int), np.asarray(coeffs, dtype=float))

def _utc(value):
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(value, "ns")
    if isinstance(value, str):
        v = value.replace("Z", "+00:00")
        if "+" in v[10:] or "-" in v[10:]:
            return _utc(datetime.fromisoformat(v))
    return np.datetime64(value, "ns")

def _legendre(n, m, theta):
    # Fully normalized associated Legendre function, no Condon-Shortley sign.
    x, s = np.cos(theta), np.sin(theta)
    pmm = 1.
    for k in range(1, m+1):
        pmm *= (2*k-1)*s
    if n == m:
        p = pmm
    else:
        p0, p1 = pmm, (2*m+1)*x*pmm
        for k in range(m+2, n+1):
            p0, p1 = p1, ((2*k-1)*x*p1-(k+m-1)*p0)/(k-m)
        p = p1
    return (sqrt((2*n+1)*(1 if m == 0 else 2) *
                 factorial(n-m)/factorial(n+m)) * p)

@lru_cache(maxsize=128)
def _geodetic_factors(lat, elevation):
    a, ecc2 = 6378136.3, 6.69439795140e-3
    phi = np.deg2rad(lat)
    radius_normal = a/np.sqrt(1-ecc2*np.sin(phi)**2)
    x = (radius_normal+elevation)*np.cos(phi)
    z = (radius_normal*(1-ecc2)+elevation)*np.sin(phi)
    radius = np.hypot(x,z)
    psi = np.arctan2(z,x)
    theta = np.pi/2-psi
    north_proj, up_proj = np.sin(phi-psi), np.cos(phi-psi)
    values = {}
    for n in range(2,7):
        for m in range(n+1):
            p = _legendre(n,m,theta)
            eps = 1e-5
            dp = (_legendre(n,m,theta+eps)-_legendre(n,m,theta-eps))/(2*eps)
            scale = (radius/a)**n*1e9/radius
            # Vertical geodetic projection, positive downward.
            values[(n,m)] = -scale*(north_proj*(-dp)+up_proj*n*p)
    return values


def _elastic_amplitudes(lat, degrees, orders, freq_deg_h):
    """Wahr-Dehant-Zschau gravimetric factors including diurnal FCN."""
    # Based on Dehant (1987), coefficients reported by ETERNA ETLOVE.
    baseline=np.array((1.1576,1.1542,1.1600,
                       1.0728,1.0728,1.0728,1.0728,
                       1.0363,1.0363,1.0363,1.0363,1.0363))
    plus=np.array((-0.0016,-0.0018,-0.0010,
                   0.,0.,0.,-0.0010,
                   0.,0.,0.,0.,-0.000315))
    minus=np.array((0.0054,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.))
    # Geocentric angle from ETLOVE (GRS80).
    phi=np.deg2rad(lat)
    e2=6.69438002290e-3
    dn=6378137./np.sqrt(1.-e2*np.sin(phi)**2)
    psi=np.arctan2((dn*(1.-e2))*np.sin(phi),dn*np.cos(phi))
    c2=np.sin(psi)**2
    denom=3.*c2-1.
    if abs(denom)<0.1:
        denom=np.copysign(0.1,denom)
    lp=np.array((
       0.335410*(35*c2*c2-30*c2+3)/denom,
       0.612372*(7*c2-3),
       0.866025*(7*c2-1),
       0.,0.,0.,
       0.829156*(9*c2-1),
       0.,0.,0.,0.,
       0.806226*(11*c2-1)
    ))
    lm=np.zeros(12)
    lm[0]=0.894427/denom
    base=baseline+plus*lp+minus*lm
    indices=degrees*(degrees+1)//2-3+orders
    factors=base[indices]
    diurnal=(degrees==2)&(orders==1)
    w=freq_deg_h[diurnal]
    factors[diurnal]+=-0.000625*(w-13.943036)/(15.073729-w)
    return factors


def _astronomy(jd, delta_tt):
    utc_c = (jd-2451545.)/36525.
    t = (jd+delta_tt/86400.-2451545.)/36525.
    h = ((jd+0.5)%1.)*24.
    gm = 280.4606184+36000.7700536*utc_c+0.00038793*utc_c**2-0.0000000258*utc_c**3
    moon = 218.316656+481267.881342*t-0.001330*t*t
    sun = 280.466449+36000.769822*t+0.0003036*t*t
    angles = np.stack((
      gm-moon+15*h,
      moon+0.004*np.cos(np.deg2rad(29+133*t)),
      sun+0.0018*np.cos(np.deg2rad(159+19*t)),
      83.353243+4069.013711*t-0.010324*t*t,
      234.955444+1934.136185*t-0.002076*t*t,
      282.937348+1.719533*t+0.0004597*t*t,
      252.25090552+149474.07217223248*t+0.000303498417*t*t,
      181.97980085+58519.21295333027*t+0.000310139472*t*t,
      355.43299958+19141.69637029695*t+0.000310518722*t*t,
      34.35151874+3036.30277484806*t+0.000223297222*t*t,
      50.07744430+1223.51106862167*t+0.000519078250*t*t
    ), axis=-1)
    return angles, t

def _terrestrial_minus_utc(times):
    starts = np.array([np.datetime64(d,"ns") for d,_ in _LEAPS])
    index = np.searchsorted(starts,times,side="right")-1
    if np.any(index < 0):
        raise ValueError("Tamura T87 implemented for dates >= 1972-01-01 UTC")
    return np.array([n for _,n in _LEAPS])[index]+32.184

def earth_tides_tsoft(lat, lon, z=0, DateTime=None, *,
                      start=None, end=None, step=None, delta=1.16,
                      wavegroups=None, unit="mGal", correction=False):
    """Predict theoretical Tamura (1987) solid-Earth gravity tide.

    lat/lon: geodetic degrees, east positive; z: ellipsoidal height [m].
    DateTime: scalar, array or numpy datetime64 in UTC; naive means UTC.
    Alternatively specify start, end (inclusive), step (seconds).
    delta: uniform elastic gravimetric amplitude factor (default 1.16).
    wavegroups: optional Nx4 matrix (min_cpd,max_cpd,delta,lead_degrees),
        or "tsoft_wdd" to use the published TSoft Manual WDD 1999 example.
        Unlisted frequencies are omitted.
    unit: "nm/s2", "uGal" or "mGal" (default). Positive down denotes tide
        effect; correction=True changes its sign to a correction to add.
    Does not calculate ocean loading, polar motion or LOD tides.

    A range returns (UTC datetimes, values); a scalar instant returns float.
    """
    range_mode = start is not None or end is not None or step is not None
    if range_mode:
        if DateTime is not None or any(v is None for v in (start,end,step)):
            raise ValueError("Provide DateTime OR start/end/step")
        if not np.isfinite(step) or step <= 0:
            raise ValueError("step must be positive and finite seconds")
        nsec = int(round(step*1e9))
        if nsec <= 0:
            raise ValueError("step too small")
        a,b = _utc(start),_utc(end)
        if b<a:
            raise ValueError("end before start")
        dates = np.arange(a,b+np.timedelta64(1,"ns"),np.timedelta64(nsec,"ns"))
    else:
        if DateTime is None:
            raise ValueError("Specify DateTime or start/end/step")
        dates = np.asarray([_utc(v) for v in np.atleast_1d(DateTime)])
    scalar = not range_mode and np.ndim(DateTime)==0
    lat,lon,z,dates = np.broadcast_arrays(np.asarray(lat,float),
                                          np.asarray(lon,float),
                                          np.asarray(z,float),dates)
    if (np.any(~np.isfinite(lat)) or np.any(np.abs(lat)>90) or
        np.any(~np.isfinite(lon)) or np.any(~np.isfinite(z))):
        raise ValueError("Invalid station coordinates")
    if unit not in ("nm/s2","uGal","mGal"):
        raise ValueError("unit must be nm/s2, uGal or mGal")
    args, degree, c = _catalogue()
    if isinstance(wavegroups,str):
        if wavegroups.lower() != "tsoft_wdd":
            raise ValueError('Unknown wavegroups preset; use "tsoft_wdd"')
        # WDD parameters from TSoft Manual 2.2.4, Example 3,
        # Wahr-Dehant-Defraigne (1999), reference station 50.6092 N.
        wavegroups = [
            [0.000000,0.002427,1.00000,0.],
            [0.002428,0.249951,1.15787,0.],
            [0.721500,0.906315,1.15432,0.],
            [0.921941,0.940487,1.15430,0.],
            [0.958085,0.998028,1.14908,0.],
            [0.999853,1.000147,1.13448,0.],
            [1.005329,1.005623,1.27258,0.],
            [1.007595,1.011099,1.17072,0.],
            [1.013689,1.216397,1.15636,0.],
            [1.719381,2.182843,1.16200,0.],
            [2.753244,3.381478,1.07360,0.],
            [3.381379,4.347615,1.03900,0.],
        ]
    if wavegroups is None:
        if not np.isfinite(delta) or delta<=0:
            raise ValueError("delta must be positive and finite")
        factors = np.full(len(degree),float(delta))
        leads = np.zeros(len(degree))
    else:
        groups = np.asarray(wavegroups,float)
        if (groups.ndim!=2 or groups.shape[1]!=4 or not len(groups)
                or not np.all(np.isfinite(groups))):
            raise ValueError("wavegroups must be a finite Nx4 matrix")
        if (np.any(groups[:,0]>groups[:,1]) or
            np.any(groups[:,2]<=0) or
            np.any(groups[1:,0]<=groups[:-1,1])):
            raise ValueError("Invalid or overlapping wavegroups")
        freq = c[:,0]/15.
        factors = np.zeros(len(degree))
        leads = np.zeros(len(degree))
        for low,high,amp,lead in groups:
            keep = (freq>=low-1e-10)&(freq<=high+1e-10)
            factors[keep],leads[keep] = amp,lead

    result = np.empty(dates.size)
    la,lo,he = lat.ravel(),lon.ravel(),z.ravel()
    ts = dates.ravel()
    for site in dict.fromkeys(zip(la,lo,he)):
        indexes = np.flatnonzero((la==site[0])&(lo==site[1])&(he==site[2]))
        geo = _geodetic_factors(site[0],site[2])
        geometric = np.array([geo[(int(n),int(m))] for n,m
                              in zip(degree,args[:,0])])
        body = _elastic_amplitudes(site[0],degree,args[:,0],c[:,0])
        # ETERNA/TSoft convention: divide all constituent body-tide
        # factors in a group by its strongest constituent's factor.
        # The externally supplied delta is the desired factor at
        # that main constituent, not an arbitrary global multiplier.
        base_groups = (np.array([[0.,10.,float(delta),0.]])
                       if wavegroups is None else groups)
        body_normalized = np.zeros_like(body)
        amplitudes = np.hypot(c[:,1],c[:,2])*np.abs(geometric)
        for low,high,amp,lead in base_groups:
            mask=(c[:,0]/15.>=low-1e-10)&(c[:,0]/15.<=high+1e-10)
            if not np.any(mask):
                continue
            members=np.flatnonzero(mask)
            strongest=members[np.argmax(amplitudes[mask])]
            body_normalized[mask]=body[mask]/body[strongest]
        weights = geometric*factors*body_normalized
        for chunk in np.array_split(indexes,max(1,(len(indexes)+255)//256)):
            time = ts[chunk]
            jd = 2440587.5+time.astype("datetime64[ns]").astype("int64")/86400e9
            ang,t = _astronomy(jd,_terrestrial_minus_utc(time))
            ang[:,0] += site[1]
            phases = np.deg2rad((ang@args.T+leads)%360.)
            coeff_cos = c[:,1][None,:]+t[:,None]*c[:,3][None,:]
            coeff_sin = c[:,2][None,:]+t[:,None]*c[:,4][None,:]
            values = coeff_cos*np.cos(phases)+coeff_sin*np.sin(phases)
            result[chunk] = (values*weights[None,:]).sum(axis=1)*1e-10

    if correction:
        result = -result
    result /= {"nm/s2":1., "uGal":10., "mGal":10000.}[unit]
    result = result.reshape(dates.shape)
    if range_mode:
        return dates,result
    return float(result.item()) if scalar else result
