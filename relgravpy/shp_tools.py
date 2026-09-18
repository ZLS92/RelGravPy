# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:05:39 2020

@author: zampa
"""

# -----------------------------------------------------------------------------
# Import libraries

from . import utils as utl
from matplotlib import path as mpath

# -----------------------------------------------------------------------------
# Set the aliases for some libraries from the utils module

np = utl.np
os = utl.os
sys = utl.sys
ogr = utl.ogr
osr = utl.osr
gdal = utl.gdal
plt = utl.plt

# -----------------------------------------------------------------------------
# Alias for the current directory (main dir.)
mdir = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------------------------------------------------
# Define the dictionary of OGR types

OGRTypes = { int: ogr.OFTInteger, str: ogr.OFTString, float: ogr.OFTReal,
             'str_': ogr.OFTString, 'float64':ogr.OFTReal,
             'int64':ogr.OFTInteger }

# -----------------------------------------------------------------------------
def write_points( x, y, fields=None, name='points_shp', path=None, prj_code=4326,
                  kml=True, csv=False, write_xy=True, dir_suffix='shp' ):
    """
    Function to crete a .shp file of points given the points coordinates

    x, y --> coordinates of the points (it works if x and y are numpy_arrays, but it should works also with lists or tuples)

    fields --> dictionary with attributes of each point
               e.g fields={'height':heigh_of_points, 'height_error':h_err, ...}
               (with 'heigh_of_points' and 'h_err' being numpy_arrays... but it should works also with lists or tuples)

    name --> name of the new file (NB. the file is alwas created within a folder with its name)

    path --> path where the new/file_folder is created

    prj_code --> code from Proj library to identify the coordinate the reference systhem (default is geo. WGS84, i.e. 4326)
                 (if its an int. it will be recognized as an epsg code)

    kml, csv --> if True a copy of the file issaved also as GoogleEarth or Com.Sep.Val. formats

    NB. THIS FUNTION IS BASED ON GDAL/OGR Python API (first tested whith gdal version 3.0.0)

    """

    # Define Driver
    driver = ogr.GetDriverByName("ESRI Shapefile")

    if path is None:
        path = name
    else:
        path = path + os.sep + name + '_' + dir_suffix

    os.makedirs( path, exist_ok=True )

    pt_nmS = path + os.sep + name + '.shp'

    while True:
        try:
            if os.path.exists( pt_nmS ) :
                os.remove( pt_nmS )

            # Define Driver
            driver = ogr.GetDriverByName("ESRI Shapefile")
            # Create shp file

            point_data_source = driver.CreateDataSource(pt_nmS)
            # Define srs
            srs = osr.SpatialReference()
            srs.ImportFromWkt( utl.prj_(prj_code).to_wkt() )
            # Define layer name and type
            point = point_data_source.CreateLayer(name, srs, ogr.wkbPoint)
            # Add fields
            if write_xy == True :
                xField = ogr.FieldDefn('x', ogr.OFTReal)
                point.CreateField(xField)
                yField = ogr.FieldDefn('y', ogr.OFTReal)
                point.CreateField(yField)
            if fields is not None:
                for f in fields:
                    Field = ogr.FieldDefn(f, OGRTypes[type(fields[f][0]).__name__])
                    point.CreateField(Field)
            # Create the feature and set values
            featureDefn = point.GetLayerDefn()
            feature = ogr.Feature(featureDefn)
            pointi = ogr.Geometry(ogr.wkbPoint)

            for n, (i, j) in enumerate(zip(x,y)):
                pointi.AddPoint(float(i), float(j))
                feature.SetGeometry(pointi)
                if write_xy == True :
                    feature.SetField("x", i)
                    feature.SetField("y", j)
                if fields is not None:
                    for f in fields:
                       feature.SetField(f, fields[f][n])
                point.CreateFeature(feature)

            del point_data_source

            if kml==True: # Save as kml in the same folder
                points_kml = gdal.VectorTranslate(path + os.sep + name + '.kml' ,pt_nmS, format='KML')
                del points_kml

            if csv == True :

                fields['shp_x'] = x
                fields['shp_y'] = y
                utl.dict2csv( fields, path_name=path + os.sep + name + '.csv'  )

            return pt_nmS

        except Exception as e:
            print(e)
            del point_data_source

# -----------------------------------------------------------------------------
def write_ply(xy_rings, fields=None, name='points_shp', path=None, prj_code=4326, kml=True, csv=False):

    """
    Function to crete a .shp file of polygons given the polygons boudaries coordinates

    x, y --> coordinates of points (it works if x and y are numpy arrays, it should works also with lists or tuples)
    fields --> dictionary with attributes of each point
               e.g fields={'height':heigh_of_points, 'height_error':h_err, ...}
               (with 'heigh_of_points' and 'h_err' being numpy arrays, it should works also with lists or tuples)
    name --> name of the new file (NB. the file is alwas created within a folder with its name)
    path --> path where the new/file_folder is created
    prj_code --> code from proj library to identify the coordinate reference systhem
                 (if its an int. it will be recognized as an epsg code)
    kml, csv --> if True they allawed to save a copy of the file also in GoogleEarth or Com.Sep.Val. formats

    NB. THIS FUNTION IS BASED ON GDAL/OGR Python API (first tested whith gdal version 3.0.0)

    """

    if path is None:
        shp_ = name
    else:
        shp_ = path + os.sep + name

    ply_nmS = shp_ + os.sep + name + '.shp'
    os.makedirs( shp_, exist_ok=True )

    # Define Driver
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # Create shp file
    while True:
        try:
            os.remove( ply_nmS ) # check if file is used in another process
            ply_data_source = driver.CreateDataSource(ply_nmS)
            # Define srs
            srs = osr.SpatialReference()
            srs.ImportFromWkt(utl.prj_(prj_code).to_wkt())
            # Define layer name and type
            ply = ply_data_source.CreateLayer(name, srs, ogr.wkbMultiPolygon)
            # Add fields
            id_field = ogr.FieldDefn('id', ogr.OFTInteger)
            ply.CreateField(id_field)
            if fields is not None:
                for f in fields:
                    Field = ogr.FieldDefn(f, OGRTypes[type(fields[f][0]).__name__])
                    ply.CreateField(Field)

            # Create the feature and set values
            featureDefn = ply.GetLayerDefn()
            feature = ogr.Feature(featureDefn)
            feature_geom = ogr.Geometry(ogr.wkbPolygon)
            ring_geom = ogr.Geometry(ogr.wkbLinearRing)

            for n, ring in enumerate(xy_rings):
                for r in range(0,ring.shape[0]):
                    ring_geom.AddPoint(ring[r,0], ring[r,1])
                feature_geom.AddGeometry(ring_geom)
                feature.SetGeometry(feature_geom)
                feature.SetField('id', n)
                if fields is not None:
                    for f in fields:
                          feature.SetField(f, fields[f][n])
                ply.CreateFeature(feature)

            del ply_data_source
            if kml==True: # Save as kml in the same folder
                gdal.VectorTranslate(shp_ + os.sep + name + '.kml' ,ply_nmS, format='KML')

            return ply_nmS

            break
        except Exception as e:
            print(e)
            del ply_data_source

# -----------------------------------------------------------------------------
def xy_in_ply( x, y, input_ply, proj_xy=4326, within=True, plot=False ) :

    proj_ply = shp_prjcode( input_ply, p=False )
    xp, yp = utl.prjxy( proj_xy, proj_ply, x, y )

    lim = utl.xy2lim( xp, yp, extend=True, d=10 )
    input_ply = translate( input_ply, lim=lim )

    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds_in = driver.Open(input_ply)
    lyr_in = ds_in.GetLayer(0)

    mask = np.full(np.size(x), False, dtype=bool)
    pt = ogr.Geometry(ogr.wkbPoint)
    for feat_in in lyr_in:
      ply = feat_in.GetGeometryRef()
      if type(ply) != None.__class__:
          minx,maxx,miny,maxy = ply.GetEnvelope()
          menv = np.where( (xp>minx) & (xp<maxx) & (yp>miny) & (yp<maxy) )[0]
          for n in menv:
                mask[n] = pt.SetPoint_2D(0, xp.ravel()[n], yp.ravel()[n])
                mask[n] = pt.Within(ply)

    if within is True :
        xf, yf, xdiff, ydiff = x[mask], y[mask], x[~mask], y[~mask]
    if within is False :
        xf, yf, xdiff, ydiff = x[~mask], y[~mask], x[mask], y[mask]
        mask = ~mask

    del ds_in

    if plot==True:
        plt.scatter( xdiff, ydiff, c='b', label='Non_Selected' )
        plt.scatter( xf, yf, c='r', label='Selected' )
        plt.legend()

    return xf, yf, mask

# -----------------------------------------------------------------------------
def shp_lim( input_shp, layer=0, extend=10, method='percentage',
             sqr_area=False, plot=False, prj_code=None, p=False ) :

    """
    Return shapefile limits as [xmin, xmax, ymin, ymax].

    If prj_code is provided, limits are transformed from the shapefile CRS
    to the requested output CRS.

    Parameters
    ----------
    p : bool
        If True, print non-fatal warnings (e.g. missing CRS metadata).
    """

    ds_in = ogr.Open( input_shp )
    lyr_in = ds_in.GetLayer( layer )

    lim = lyr_in.GetExtent()

    if ( extend != 0 ) or ( extend is not False ) or ( extend is not None ) :
        lim = utl.extend_lim( lim, d=extend, method=method, sqr_area=sqr_area, plot=plot )

    if ( plot is True ) and ( ( extend == 0 ) or ( extend is False ) or ( extend is None ) ) :
        xplot = [lim[0], lim[0], lim[1], lim[1], lim[0]]
        yplot = [lim[2], lim[3], lim[3], lim[2], lim[2]]
        plt.plot(xplot, yplot, c='k')

    if prj_code is not None:
        shp_prj = lyr_in.GetSpatialRef()
        if shp_prj is not None:
            shp_prj = shp_prj.ExportToProj4()
            x_lim, y_lim = utl.prjxy(shp_prj, prj_code,
                                     [lim[0], lim[1]],
                                     [lim[2], lim[3]])
            lim = [x_lim[0], x_lim[1], y_lim[0], y_lim[1]]
        else:
            if p == True:
                print('Could not determine SRID')

    ds_in = None
    lyr_in = None

    return lim

# -----------------------------------------------------------------------------
def get_points(input_shp, fields=None, prj_code=None, in_prjcode=None,
    x_key='x', y_key='y', p=False, plot=False, ax=None,
    ps=10, pc='r', marker='o', alpha=0.9, label=None,
    legend=False, figsize=(8, 6), zorder=10):

    """
    Extract point features from a shapefile into a dictionary of 1D numpy arrays.

    Parameters
    ----------
    input_shp : str or OGR datasource
        Input point shapefile.
    fields : list[str] or str or None
        Fields to extract. If None, all attribute fields are extracted.
    prj_code : int/str or None
        Output CRS for coordinates. If None, coordinates are returned
        in the native layer CRS.
    in_prjcode : int/str or None
        Force input CRS for coordinate transformation.
        If None, CRS is read from layer metadata.
    x_key, y_key : str
        Dictionary keys used for coordinates.
    p : bool
        Print basic diagnostics.
    plot : bool
        If True, plot extracted points as a scatter plot.
    ax : matplotlib axis or None
        Existing axis. If None and plot=True, a new figure is created.
    ps, pc, marker, alpha, zorder : scatter style options.
    label : str or None
        Label for scatter points.
    legend : bool
        If True and label is set, draw legend.
    figsize : tuple
        Figure size used only when plot=True and ax is None.

    Returns
    -------
    dict
        Dictionary where each key contains a 1D numpy array.
        Coordinates are stored in data[x_key] and data[y_key].
    """

    close_ds = False
    if type(input_shp) == str:
        ds_in = ogr.Open(input_shp)
        close_ds = True
    else:
        ds_in = input_shp

    if ds_in is None:
        raise ValueError('Could not open input shapefile')

    lyr_in = ds_in.GetLayer(0)
    if lyr_in is None:
        raise ValueError('Could not read layer from input shapefile')

    layer_def = lyr_in.GetLayerDefn()
    if fields is None:
        fields = [layer_def.GetFieldDefn(i).GetNameRef()
                  for i in range(layer_def.GetFieldCount())]
    if type(fields) == str:
        fields = [fields]

    data = {x_key: [], y_key: []}
    for fld in fields:
        data[fld] = []

    lyr_in.ResetReading()
    for feat in lyr_in:

        geom = feat.GetGeometryRef()
        if type(geom) == None.__class__:
            continue

        gname = geom.GetGeometryName()
        if gname == 'POINT':
            geoms = [geom]
        elif gname == 'MULTIPOINT':
            geoms = [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())]
        else:
            if p == True:
                print(f'Skipping non-point geometry: {gname}')
            continue

        for g in geoms:
            data[x_key].append(g.GetX())
            data[y_key].append(g.GetY())
            for fld in fields:
                data[fld].append(feat.GetField(fld))

    data[x_key] = np.asarray(data[x_key], dtype=float)
    data[y_key] = np.asarray(data[y_key], dtype=float)

    if prj_code is not None:
        if in_prjcode is None:
            lyr_srs = lyr_in.GetSpatialRef()
            if lyr_srs is None:
                raise ValueError('Could not determine input CRS. Set in_prjcode explicitly.')
            in_prjcode = lyr_srs.ExportToProj4()

        x_out, y_out = utl.prjxy(in_prjcode, prj_code, data[x_key], data[y_key])
        data[x_key] = np.asarray(x_out, dtype=float)
        data[y_key] = np.asarray(y_out, dtype=float)

    for fld in fields:
        arr = np.asarray(data[fld])
        if arr.dtype == object:
            try:
                arr = arr.astype(float)
            except Exception:
                pass
        data[fld] = arr

    if p == True:
        print(f'Points extracted: {data[x_key].size}')
        print(f'Output keys: {list(data.keys())}')

    if plot is True:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.scatter(data[x_key], data[y_key], s=ps, c=pc, marker=marker,
                   alpha=alpha, label=label, zorder=zorder)
        if legend is True and label is not None:
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend()

    if close_ds == True:
        ds_in = None
    lyr_in = None

    return data

# -----------------------------------------------------------------------------
def ply_bounds( input_ply, ply_id=0, prj_code=4326 ):

    ds_in = ogr.Open(input_ply)
    lyr_in = ds_in.GetLayer(0)

    for n, feat_in in enumerate(lyr_in):
      ply = feat_in.GetGeometryRef()
      minx,maxx,miny,maxy = ply.GetEnvelope()
      if n==ply_id: break

    proj_ply = lyr_in.GetSpatialRef().ExportToProj4()
    print( proj_ply )
    xp, yp = utl.prjxy(proj_ply, prj_code, [minx,maxx], [miny,maxy])
    bounds = [xp[0], xp[1], yp[0], yp[1]]

    ds_in = None
    lyr_in = None

    return bounds

# -----------------------------------------------------------------------------
def lim2ring(lim, prj_in=4326, prj_out=4326):

    if prj_in!=prj_out:
        [lim[0], lim[1]], [lim[2],lim[3]] = utl.prjxy(prj_in, prj_out, [lim[0], lim[1]], [lim[2],lim[3]])

    ring = np.zeros((5,2))
    ring[0,:] = np.array((lim[0], lim[3]))
    ring[1,:] = np.array((lim[0], lim[2]))
    ring[2,:] = np.array((lim[1], lim[2]))
    ring[3,:] = np.array((lim[1], lim[3]))
    ring[4,:] = np.array((lim[0], lim[3]))

    return ring

# -----------------------------------------------------------------------------
def get_ply( shp_ply, lim=None, plot=True ):

    if lim is not None :
        shp_ply = translate( shp_ply, new_name=None, new_path='/vsimem/',
            suffix='_cut' , extension='shp', lim=lim )

    ds = ogr.Open( shp_ply )
    lyr = ds.GetLayer(0)

    plyco = []
    lyr.ResetReading()
    xply = []
    yply = []
    fply = []
    fply = []
    nply = []

    for fi, feat in enumerate(lyr):
        geom = feat.GetGeometryRef()
        if geom.GetGeometryName() == 'MULTIPOLYGON':
            for i in range(geom.GetGeometryCount()):
                sub_geom = geom.GetGeometryRef(i)
                plyco.append(extract_polygon_coordinates(sub_geom))
                for ni, _ in enumerate( plyco[-1][0] ):
                    xply.append( plyco[-1][0][ni])
                    yply.append(plyco[-1][1][ni])
                    nply.append(ni)
                    fply.append(fi)
        else:
            plyco.append(extract_polygon_coordinates(geom))
            i = 0
            for ni, _ in enumerate( plyco[-1][0] ):
                xply.append( plyco[-1][0][ni])
                yply.append(plyco[-1][1][ni])
                nply.append(ni)
                fply.append(fi)

    np_ply = np.column_stack((xply, yply, fply, nply))

    return np_ply

# -----------------------------------------------------------------------------
def plt_ply(
    shp_ply,
    ax=None,
    in_prjcode=None,
    out_prjcode=None,
    lim=None,
    lim_prjcode=None,
    color='k',
    linewidth=1.5,
    linestyle='-',
    alpha=1.0,
    zorder=10,
    label='Polygon',
    legend=False,
    set_limits=False,
    aspect=None,
    figsize=(8, 6),
    p=False,
):
    """
    Plot polygon boundaries from a shapefile on a new or existing axis.

    Parameters
    ----------
    shp_ply : str
        Input polygon shapefile path.
    ax : matplotlib axis or None
        Existing axis. If None, a new figure/axis is created.
    in_prjcode, out_prjcode : int/str or None
        Input and output CRS. If input is None, it is read from shapefile.
        If output is None, no reprojection is applied.
    lim : list/tuple or None
        Plot limits as [xmin, xmax, ymin, ymax].
    lim_prjcode : int/str or None
        CRS of lim. If different from output CRS, limits are transformed.
    color, linewidth, linestyle, alpha, zorder : plot style options.
    label : str or None
        Label used for the first polygon ring.
    legend : bool
        If True, draw legend when labeled artists are available.
    set_limits : bool
        If True and lim is provided, enforce axis limits.
    aspect : str/float or None
        Axis aspect passed to ax.set_aspect.
    figsize : tuple
        Figure size used only when ax is None.
    p : bool
        If True, print basic bounds diagnostics.

    Returns
    -------
    ax : matplotlib axis
    np_ply_out : ndarray
        Polygon coordinates with columns [x, y, feature_id, node_id] in output CRS.
    """

    np_ply = get_ply(shp_ply, plot=False)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if np_ply.size == 0:
        if p is True:
            print('No polygon vertices found.')
        return ax, np_ply

    x = np_ply[:, 0].astype(float)
    y = np_ply[:, 1].astype(float)

    if in_prjcode is None:
        in_prjcode = shp_prjcode(shp_ply)

    out_srs = None
    if out_prjcode is not None:
        out_srs = utl.prj_(out_prjcode).srs

    if (in_prjcode is not None) and (out_prjcode is not None):
        in_srs = utl.prj_(in_prjcode).srs
        if in_srs != out_srs:
            x, y = utl.prjxy(in_srs, out_srs, x, y)
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)

    lim_plot = lim
    if (lim is not None) and (lim_prjcode is not None) and (out_prjcode is not None):
        lim_in_srs = utl.prj_(lim_prjcode).srs
        if lim_in_srs != out_srs:
            lim_plot = utl.prj_lim(lim, lim_prjcode, out_prjcode)

    ring_start = np.where(np_ply[:, 3] == 0)[0]
    ring_start = np.append(ring_start, np_ply.shape[0])

    for i in range(len(ring_start) - 1):
        i0, i1 = ring_start[i], ring_start[i + 1]
        ilabel = label if i == 0 else None
        ax.plot(
            x[i0:i1],
            y[i0:i1],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
            label=ilabel,
        )

    if set_limits is True and lim_plot is not None:
        ax.set_xlim(lim_plot[0], lim_plot[1])
        ax.set_ylim(lim_plot[2], lim_plot[3])

    if aspect is not None:
        ax.set_aspect(aspect)

    if legend is True:
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend()

    if p is True:
        print(
            'Polygon bounds: '
            f'[{np.nanmin(x):.6f}, {np.nanmax(x):.6f}, '
            f'{np.nanmin(y):.6f}, {np.nanmax(y):.6f}]'
        )

    np_ply_out = np.column_stack((x, y, np_ply[:, 2], np_ply[:, 3]))

    return ax, np_ply_out

# -----------------------------------------------------------------------------
def extract_polygon_coordinates(geom) :

    codes = []
    all_x = []
    all_y = []

    for i in range( geom.GetGeometryCount() ):
        ref = geom.GetGeometryRef( i )
        x = [ ref.GetX(j) for j in range( ref.GetPointCount() ) ]
        y = [ ref.GetY(j) for j in range( ref.GetPointCount() ) ]
        codes += [ mpath.Path.MOVETO ] + ( len(x) - 1 ) * [ mpath.Path.LINETO ]
        all_x += x
        all_y += y

    return [all_x, all_y]

# -----------------------------------------------------------------------------
def get_lin(shp_lin, lim=None, plot=True):

    ogr.UseExceptions()
    gdal.SetConfigOption('SHAPE_ENCODING', "UTF-8")

    if lim is not None :
        shp_lin = translate( shp_lin, new_name=None, new_path='/vsimem/',
                             suffix='_cut' , extension='shp', lim=lim )

    ds = ogr.Open(shp_lin)
    lyr = ds.GetLayer(0)

    linco = []
    lyr.ResetReading()

    for feat in lyr:
        geom = feat.GetGeometryRef()
        if geom.GetGeometryName() == 'MULTILINESTRING':
            for i in range(geom.GetGeometryCount()):
                sub_geom = geom.GetGeometryRef(i)
                coord = extract_linestring_coordinates( sub_geom )
                coord[0] = np.array( coord[0], dtype=float )
                coord[1] = np.array( coord[1], dtype=float )
                linco.append( coord )
        else:
            coord = extract_linestring_coordinates( geom )
            coord[0] = np.array( coord[0], dtype=float )
            coord[1] = np.array( coord[1], dtype=float )
            linco.append( coord )

    if plot is True :
        for lin in linco:
            plt.gca().plot( lin[0], lin[1], c='k' )

    gdal.SetConfigOption('SHAPE_ENCODING', None)
    ds = None

    return linco

# -----------------------------------------------------------------------------
def extract_linestring_coordinates(geom):

    x = [ geom.GetX(i) for i in range(geom.GetPointCount()) ]
    y = [ geom.GetY(i) for i in range(geom.GetPointCount()) ]

    return [ x, y ]

# -----------------------------------------------------------------------------
def ogr2ogr(in_shp, arguments, new_name=None, new_path='/vsimem/', suffix='', extension='shp'):

    if new_path == None:
        new_path = os.path.dirname(os.path.realpath(in_shp))

    if new_path != '/vsimem/':
        os.makedirs(new_path, exist_ok=True)
        new_path = new_path + os.sep

    if new_name == None:
        new_name = in_shp.split(os.sep)[-1].split('.')[0] + '_new'

    out_shp = new_path + new_name + suffix +'.'+extension

    cmd = f'ogr2ogr {arguments} {out_shp} {in_shp}'

    os.system(cmd)

    return out_shp

# -----------------------------------------------------------------------------
def shp_prjcode( shp, p=False ) :

    if type( shp ) == str :
        shp = ogr.Open( shp )

    lyr = shp.GetLayer()
    code = lyr.GetSpatialRef()
    if code is None:
        print('Could not determine SRID')
        prjcode = None
    else :
        prjcode = utl.prj_( code.ExportToProj4() ).srs

    if p == True :
        print( prjcode )

    shp = None
    lyr = None

    return prjcode

# -----------------------------------------------------------------------------
def shp_epsg( shp, p=False ) :

    if type( shp ) == str :
        shp = ogr.Open( shp )

    lyr = shp.GetLayer()
    code = lyr.GetSpatialRef()

    # Try to determine the EPSG/SRID code
    if code.AutoIdentifyEPSG() == 0: # success
       epsg = int( code.GetAuthorityCode(None) )
    else:
        print('Could not determine SRID')

    if p == True :
        print( epsg )

    shp = None
    lyr = None

    return epsg

# -----------------------------------------------------------------------------
def shp_name( shp, p=False ) :

    if type( shp ) == str :
        shp = ogr.Open( shp )

    path_name = shp.GetDescription()
    path_name = path_name.replace("//", "/")
    path_name = path_name.replace("/", "\\")
    name = path_name.split('\\')[-1].split('.')[0]

    if p == True :
        print( name )

    shp = None

    return name

# -----------------------------------------------------------------------------
def shp_path( shp, p=False ) :

    if type( shp ) == str :
        shp = ogr.Open( shp )

    path_name = shp.GetDescription()
    path_name = path_name.replace( "//", "/" )
    path_name = path_name.replace( "/", "\\" )
    name = path_name.split( "\\" )[-1]
    pn_split = list(shp.GetDescription())
    n_split = list(name)
    n_len = len(n_split)
    del pn_split[-n_len::]

    while ( pn_split[-1] == '/' ) or ( pn_split[-1] == '\\' ) :
        del pn_split[-1]

    path = ''.join( pn_split )

    shp = None

    if p == True :
        print( path )

    return path

# -----------------------------------------------------------------------------
def translate( in_shp,
               new_name=None,
               new_path='/vsimem/',
               suffix='_new' ,
               extension='shp',
               lim=None,
               lim_prjcode=None,
               clip2lim = True,
               in_prjcode=None,
               out_prjcode=None,
               options = '',
               buffer=None ) :

    """
    in_shp --- a .shp Dataset object or a filename
    new_name --- name of the output file (if None, it will be the same of the in_shp + suffix = '_new')
    new_path --- path of the output file (default is the file system '/vsimem/'; if None, it will be the same of the in_shp )
    suffix --- suffix to add after the name (e.g., Coast.shp --> suffix = '_new' --> output_name = Coast_new.shp)
    lim --- spatial filter as (minX, minY, maxX, maxY) bounding box
    lim_prjcode --- SRS in which the lim is expressed. If not specified, it is assumed to be the one of the layer(s)
    options --- can be be an array of strings, a string or let empty and filled from other keywords.
    fmt --- output format ("ESRI Shapefile", etc...)
    accessMode --- None for creation, 'update', 'append', 'overwrite'
    in_prjcode --- source SRS
    out_prjcode --- output SRS (with reprojection if reproject = True)
    skipFailures --- whether to skip failures

    See also GDAL/OGR Python API ( https://gdal.org/python/ )
    """

    if type ( in_shp ) is not str :
        in_shp = in_shp.GetDescription()

    if new_name == None :
        new_name = shp_name( in_shp ) + suffix

    if new_path == None :
        new_path = shp_path( in_shp )
    else :
        if new_path != '/vsimem/' :
           new_path = new_path + os.sep + new_name
           os.makedirs( new_path, exist_ok=True )
           new_path = new_path + os.sep

    out_shp = new_path + new_name + '.' + extension

    if out_prjcode is None :
        out_prjcode = shp_prjcode( in_shp )
    else :
        out_prjcode = utl.prj_( out_prjcode ).srs

    options = options + f' -t_srs "{out_prjcode}"'

    if in_prjcode is None :
        in_prjcode = shp_prjcode( in_shp )
    else :
        in_prjcode = utl.prj_( in_prjcode ).srs

    options = options + f' -s_srs "{in_prjcode}"'

    if lim is not None :
        if lim_prjcode is None :
            lim_prjcode = in_prjcode
        if utl.prj_(lim_prjcode).srs != utl.prj_(out_prjcode).srs :
            lim = utl.prj_lim( lim, lim_prjcode, out_prjcode )
        lim = utl.lim_sort( lim )

    if ( lim is not None ) and ( clip2lim == True ) :
        options = options + f' -clipdst {lim[0]} {lim[1]} {lim[2]} {lim[3]} '

    if buffer is not None :
        in_shp_name = shp_name( in_shp )
        options = options + f' -dialect sqlite -sql "select ST_Buffer(geometry, {buffer}) from {in_shp_name}"'

#    print( options )
    new_shp = gdal.VectorTranslate( out_shp, in_shp, options = options )

    if new_path != '/vsimem/' :
        new_shp = None

    return out_shp


# -----------------------------------------------------------------------------
#def createBuffer( in_shp, bufferDist, new_name=None, new_path='/vsimem/', suffix='_new' ):
#
#    if type ( in_shp ) != str :
#        in_shp = in_shp.GetDescription()
#
#    if new_name == None :
#        new_name = shp_name( in_shp ) + suffix
#
#    if new_path == None :
#        new_path = shp_path( in_shp )
#    else :
#        if new_path != '/vsimem/' :
#           new_path = new_path + os.sep + new_name
#           os.makedirs( new_path, exist_ok=True )
#           new_path = new_path + os.sep
#
#    out_shp = new_path + new_name + '.shp'
#
#    if type ( in_shp ) != str :
#        inputds = ogr.Open(in_shp)
#
#    inputlyr = inputds.GetLayer()
#
#    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
#
#    if os.path.exists( out_shp ):
#        shpdriver.DeleteDataSource( out_shp )
#
#    outputBufferds = shpdriver.CreateDataSource(out_shp)
#    bufferlyr = outputBufferds.CreateLayer(out_shp, geom_type=ogr.wkbPolygon)
#    featureDefn = bufferlyr.GetLayerDefn()
#
#    for feature in inputlyr:
#        ingeom = feature.GetGeometryRef()
#        geomBuffer = ingeom.Buffer(bufferDist)
#
#        outFeature = ogr.Feature( featureDefn )
#        outFeature.SetGeometry( geomBuffer )
#        bufferlyr.
