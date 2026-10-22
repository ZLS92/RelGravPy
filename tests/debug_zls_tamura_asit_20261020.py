"""Diagnostic: compare user-shared ZLS B-109 ET corrections vs Tamura T87.

This is a temporary diagnostic, deliberately not merged into main.
11 readings, Turin 20 Oct 2026; times as in notebook screenshot.
"""
from datetime import datetime, timedelta
import numpy as np
from relgravpy.earth_tides_t87 import earth_tides_tsoft

utc_text=[
"2026-10-20 07:29:53",
"2026-10-20 08:50:06",
"2026-10-20 09:14:39",
"2026-10-20 09:58:14",
"2026-10-20 10:47:59",
"2026-10-20 11:16:01",
"2026-10-20 11:41:41",
"2026-10-20 12:33:32",
"2026-10-20 13:40:53",
"2026-10-20 14:05:15",
"2026-10-20 14:28:43",
]
zls=np.array([0.050711,0.037241,0.030736,0.014904,-0.009120,
 -0.019216,-0.030215,-0.053240,-0.072838,-0.077398,-0.080235])
shown_tamura=np.array([-0.027323,-0.033605,-0.032598,-0.027823,
 -0.018512,-0.012185,-0.006101,0.005779,0.017245,0.019675,0.020996])
lats=np.array([45.011650,45.015990,45.015990,45.011650,45.015990,
 45.016084,45.015990,45.015990,45.011650,45.011650,45.011650])
lons=np.array([7.572245,7.641695,7.641695,7.572245,7.641695,
 7.641629,7.641695,7.641695,7.572245,7.572245,7.572245])
elev=np.array([0,285,285,0,285,0,285,285,0,0,0],dtype=float)
datetimes=[datetime.fromisoformat(s) for s in utc_text]
print('ZLS vs screenshot user-generated Tamura')
def metrics(x,y):
    d=x-y
    return tuple(round(float(v),6) for v in (np.sqrt(np.mean(d**2)),
        np.mean(d),np.max(np.abs(d)),np.corrcoef(x,y)[0,1]))
print('raw TAMURA vs ZLS: ', metrics(shown_tamura,zls))
print('negative TAMURA vs ZLS: ',metrics(-shown_tamura,zls))
for wave in [None,"tsoft_wdd"]:
    print('\nwavegroups:',wave)
    for offset in [0,-2,+2,-1,+1,-3,+3]:
        dates=[d+timedelta(hours=offset) for d in datetimes]
        g=earth_tides_tsoft(lats,lons,elev,DateTime=dates,
                            wavegroups=wave,unit="mGal")
        for sign in [1,-1]:
            print('offset',offset,'sign',sign,'RMSE,mean,max,r',
                  metrics(sign*g,zls))
        if offset==0:
            print(' values in mGal:',np.round(g,6).tolist())
            print(' vs screenshot Tamura RMS,mean,max,r',
                  metrics(g,shown_tamura))
    best=[]
    for minutes in range(-360,361,15):
        dates=[d+timedelta(minutes=minutes) for d in datetimes]
        g=earth_tides_tsoft(lats,lons,elev,DateTime=dates,
                            wavegroups=wave,unit="mGal")
        best.append((np.sqrt(np.mean((-g-zls)**2)),minutes))
    print('best sign - from -6..6h:', sorted(best)[:6])
