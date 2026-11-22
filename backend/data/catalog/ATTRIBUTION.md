# Data attribution and licences

## Route data: OpenStreetMap

Route relations, way attributes and points of interest are from
OpenStreetMap, (c) OpenStreetMap contributors, available under the
Open Database Licence (ODbL) 1.0: https://www.openstreetmap.org/copyright

The files in this directory (`routes.jsonl.gz`, `geometry.jsonl.gz`, `details.jsonl.gz`) are a
Derivative Database of OpenStreetMap and are themselves licensed under the
ODbL 1.0. Any public use must credit "(c) OpenStreetMap contributors" and
keep the data under the ODbL. Extract obtained from Geofabrik GmbH
(https://download.geofabrik.de/); the exact file and checksum are in
`manifest.json`.

Generated descriptions are derived from the same data and are part of the
derivative database.

## Elevation: Mapzen / Tilezen Terrain Tiles (AWS Open Data)

Elevation (ascent, descent, min/max elevation, profiles) is sampled from the
Terrain Tiles "skadi" 1 arc-second HGT tiles
(https://registry.opendata.aws/terrain-tiles/), licence and attribution as
documented at https://github.com/tilezen/joerd/blob/master/docs/attribution.md.
The tiles over Bavaria are composited mainly from SRTM; the documented
attribution for the sources that cover this region is:

* SRTM terrain data courtesy of the U.S. Geological Survey (public domain).
* Europe terrain data produced using Copernicus data and information funded
  by the European Union - EU-DEM layers.
* Austria terrain data (c) offene Daten Oesterreichs - Digitales
  Gelaendemodell (DGM) Oesterreich.
* Global GMTED2010 terrain data courtesy of the U.S. Geological Survey;
  ETOPO1 U.S. National Oceanic and Atmospheric Administration.
