# Route catalogue (Phase 1 data)

A reproducible catalogue of outdoor routes in **Bavaria** built from OpenStreetMap
route relations and SRTM-derived elevation. It replaces the 111-route legacy
Outdooractive seed for evaluation (D3 in [DECISIONS.md](DECISIONS.md)).

Every number in this document is printed by

```bash
cd backend && python -m pipeline.stats
```

which reads only the committed files in `backend/data/catalog/`. Thresholds and
rules (e.g. "within 75 m") are pipeline parameters from `backend/pipeline/config.py`
and the rule modules, not measurements.

## 1. Files

| File | Content |
|---|---|
| `backend/data/catalog/routes.jsonl.gz` | one route per line (gzip, fixed mtime), sorted by `id`, byte-for-byte deterministic for the same input |
| `backend/data/catalog/geometry.jsonl.gz` | per route: simplified polyline `[[lon, lat], ...]` (Douglas-Peucker 10 m) and an elevation profile every 200 m |
| `backend/data/catalog/details.jsonl.gz` | per route: OSM ids of every listed feature, stitching diagnostics, member way ids of the main line |
| `backend/data/catalog/manifest.json` | input files (name, URL, md5, sha256, replication timestamp and sequence), DEM tiles with sha256, pipeline version, all parameters, counts, output checksums |
| `backend/data/catalog/build_report.json` | drop counts with example ids, stitching statistics, region-selection counts, validation results, DEM calibration grid |
| `backend/data/catalog/ATTRIBUTION.md` | licence and attribution text |

Downloads and intermediates live in `backend/data/raw/` (git-ignored).

## 2. Sources, licence, attribution

**OpenStreetMap.** Geofabrik extract `bayern-261101.osm.pbf` (the 1 November 2026
monthly snapshot; replication timestamp 2026-11-01T21:21:36Z, sequence 4952; md5
`e081102904686f1af63e61fdbb341c09`). Licence: ODbL 1.0, "© OpenStreetMap
contributors". The catalogue is a Derivative Database and is itself ODbL; any
public use must carry the attribution and keep the data under the ODbL.

**Elevation.** Mapzen/Tilezen Terrain Tiles on AWS Open Data, "skadi" format: 1° x 1°
HGT tiles, 1 arc-second (3601 x 3601 big-endian int16), gzip. The registry entry
points to the Tilezen attribution document for its licence. Over Bavaria the tiles
are composited mainly from SRTM (public domain, courtesy of the U.S. Geological
Survey); the same document lists EU-DEM (Copernicus) and Austria's DGM
(data.gv.at) as sources in this part of Europe, so all three are credited in
`ATTRIBUTION.md`. The tiles have been static since 2016 (S3 `Last-Modified`
2016); their sha256 is in the manifest. 19 tiles are used; there were no void
samples and no samples on missing tiles.

**Reproducibility.** `python -m pipeline.build` downloads the pinned dated extract
(md5-checked) and the DEM tiles into `backend/data/raw/` and rebuilds everything.
Two consecutive builds, one of them from a cold cache, produced identical
checksums for every output file. Geofabrik keeps daily snapshots for about ten
days, monthly snapshots (`YYMM01`) for a few months and the yearly 1 January
snapshots (`YY0101`) indefinitely. When `bayern-261101.osm.pbf` is no longer
downloadable, the build stops with an explanation: **the committed catalogue
plus `manifest.json` are then the artefact of record.** A rebuild from a newer
extract (`PIPELINE_OSM_EXTRACT=bayern-270101 python -m pipeline.build`) works but
yields different counts and needs the md5 check entry added in `config.py`.

## 3. Region and why

The first choice was the Bavarian Alps and foothills (Regierungsbezirke
Oberbayern + Schwaben). Real counts ruled it out. Routes that pass every quality
filter, counted by the Regierungsbezirk of their start (before the region
filter and dedup), are:

| Regierungsbezirk | routes |
|---|---|
| Oberbayern | 1818 |
| Unterfranken | 1773 |
| Oberfranken | 1628 |
| Mittelfranken | 1444 |
| Oberpfalz | 1363 |
| Niederbayern | 844 |
| Schwaben | 560 |

Oberbayern + Schwaben gives 2378, short of the 3,000 target before any dedup,
and many Allgäu routes cross into Austria or Baden-Württemberg and are cut by the
extract boundary. The region is therefore the **whole Free State of Bavaria**
(OSM `admin_level=4` "Bayern", 70994 km²): one coherent administrative and
tourism region with Alps, foothill lakes, Franconian Jura and low mountain
ranges, and one Geofabrik extract. A route belongs to the region when its start
lies inside the Bavarian boundary polygon and at least 75% of its length (100 m
samples) lies inside.

## 4. Pipeline, step by step

Code: `backend/pipeline/`. One command; PBF parsing dominates a cold run and is
cached as a pickle in `backend/data/raw/work/` afterwards.

1. **Download** (`download.py`): pinned extract, md5 verified; DEM tiles on demand.
2. **Extract** (`osm_extract.py`): five pyosmium passes, each pre-filtered in C++:
   route/superroute relations with `route` in hiking, foot, walking,
   nordic_walking, bicycle, mtb, running, fitness_trail; admin boundaries
   (levels 4–6), named water and protected-area multipolygons; their member
   ways; POI ways; POI nodes; node coordinates of all kept ways.
3. **Polygons** (`areas.py`): multipolygons assembled with `shapely.polygonize`
   (outer minus inner). Admin boundaries truncated by the extract (Austrian,
   Czech, Thuringian districts) are skipped.
4. **Feature index** (`features.py`): every POI is classified into one feature
   type (first matching rule) and put in an STRtree in a local metric projection.
5. **Per route** (`build.py`):
   1. lifecycle `state=proposed|disused|…` → drop;
   2. no `name` → drop (no title);
   3. expand members: ways with a main role (`""`, `main`, `forward`, `backward`,
      `route`, `section`) in relation order, recursing into child route
      relations (stages); roles such as `alternative`, `excursion`, `approach`
      are not part of the main line;
   4. any main member way or child relation missing from the extract → drop
      (the route crosses the extract boundary; no guessing);
   5. **stitch** (`geo.py`): see 4.1;
   6. length limits (hiking/running ≥ 1 km, cycling/MTB ≥ 3 km; maximum 400 km
      hiking, 60 km running, 800 km cycling, 300 km MTB);
   7. region test (section 3);
   8. elevation profile, way-attribute profile, features, locality, admin area.
6. **Dedup** (`dedup.py`), see section 9.
7. **DEM calibration**, see section 8.
8. **Attributes, tags, description**, then write outputs.

### 4.1 Way stitching

* **Ordered pass.** Follow the relation order, flipping ways where needed. A
  correctly ordered relation becomes one segment and is used as is (the mapper's
  direction and start are kept).
* **Graph pass** (otherwise). Split member ways at every node shared with another
  member way, which fixes ways that are not split where the route turns
  (T-junctions on interior nodes, overshooting ways). Merge edges through
  degree-2 nodes, add virtual gap edges between dangling ends and other chain
  ends closer than 50 m, then take the **longest trail** (no edge twice) by a
  bounded depth-first search. Unused chains (spurs, alternatives mapped with a
  main role, disconnected pieces) are dropped. A closed trail is rotated to start
  where the relation's first way starts.
* **Acceptance.** The main line must cover ≥ 90% of the member-way length and
  bridged gaps may total at most max(2% of the length, 100 m). Otherwise the
  relation is dropped as `stitch_failed`.

Of 11371 named, complete relations, 7263 (63.9%) assembled in relation order as
one segment; 4108 (36.1%) needed the graph pass, of which 3215 were accepted
(1055 with stray branches dropped, 1065 with bridged gaps); **893 (7.9%) failed**
and were dropped. Of the kept routes, 932 (10.0%) have a stitch coverage below
1.0 (median 0.984 among those). Inspection of failures shows mostly relations
that are genuinely incomplete in OSM (missing ways leave gaps of hundreds of
metres) and "network" relations that bundle a whole district's cycle network
(e.g. "Radverkehrsnetz Landkreis …").

## 5. Drop counts

| Step | dropped |
|---|---|
| relations considered | 14625 |
| `inactive_state` (proposed, disused, …) | 9 |
| `unnamed` | 2075 |
| `no_member_ways` | 50 |
| `incomplete_in_extract` (crosses the Bavaria extract boundary) | 1065 |
| `superroute_single_child` (wrapper around one stage) | 55 |
| `stitch_failed` | 893 |
| `too_short` | 990 |
| `too_long` | 2 |
| `outside_region` | 83 |
| `no_elevation` | 0 |
| `duplicate` | 56 |
| **final** | **9347** |

## 6. Schema (`routes.jsonl.gz`)

O = observed in OSM (copied or parsed), D = derived by the pipeline (geometry,
DEM, rules), G = generated text.

| Field | Type | Source | Meaning |
|---|---|---|---|
| `id` | int | O | OSM **relation** id (not a way id) |
| `source` | str | – | `"osm"` |
| `osm_type` | str | – | `"relation"` |
| `title` | str | O | `name` (else `name:de`, `name:en`), whitespace-normalised, mostly German |
| `description` | str | O+G | OSM `description:en`/`description`/`description:de` if present, followed by the generated text |
| `description_source` | str | – | `generated` or `osm+generated` |
| `category_name` | str | D | app vocabulary: `Hiking trail`, `Theme trail`, `Cycling`, `Long distance cycling`, `Mountainbiking`, `Jogging`, `Trail running` |
| `activity` | str | O→D | `hiking`, `cycling`, `mtb`, `running` from the relation's `route` |
| `osm_route` | str | O | raw `route` value (`superroute` for parents without one) |
| `tags` | list | D | controlled vocabulary, section 7 |
| `difficulty` | int 0–3 | D | section 10 |
| `difficulty_source` | str | D | `osm:sac_scale`, `osm:mtb:scale`, `derived:effort`, or `derived:effort (osm:… lower)` |
| `difficulty_technical`, `difficulty_effort` | int/null | D | the two components |
| `max_grade` | str/null | O | hardest SAC (`T1`–`T6`) or MTB (`S0`–`S6`) grade covering ≥ 100 m |
| `length_m` | int | D | haversine length of the stitched main line |
| `ascent_m`, `descent_m` | int | D | DEM, section 8 (in the mapped direction) |
| `min_ele_m`, `max_ele_m`, `start_ele_m` | int | D | smoothed DEM profile |
| `max_grade_200m_pct` | float | D | steepest 200 m stretch of the smoothed profile |
| `duration_min` | int | D | section 11 |
| `est_days` | int | D | days at 8 h (hiking), 6 h (cycling), 5 h (MTB), 3 h (running) |
| `start_lat/lon`, `end_lat/lon` | float | D | first/last vertex of the main line, WGS84, 5 decimals |
| `centroid_lat/lon` | float | D | centroid of the line |
| `is_loop` | bool | D | start and end within 300 m |
| `start_end_dist_m` | int | D | |
| `locality`, `locality_dist_m` | str, int | O+D | nearest `place=city|town|village` node to the start within 15 km (fallback suburb/hamlet within 5 km) |
| `end_locality` | str/null | O+D | same for the end, linear routes only |
| `admin_district` | str | O+D | Landkreis / kreisfreie Stadt (admin_level 6) containing the start |
| `admin_region` | str | O+D | Regierungsbezirk (admin_level 5) containing the start |
| `stage_of`, `stages` | list[int] | O | parent / child route relations that are also in the catalogue |
| `duplicates_merged` | list[int] | D | relation ids dropped as duplicates of this one |
| `features.on_route` | dict | O+D | per feature type, named items within the type's distance: `{name, dist_m, at_m, ele_m?, area_ha?}` (`at_m` = position along the line; rivers have `along_m`), max 8 per type |
| `features.on_route_counts` | dict | O+D | all items per type, named or not |
| `features.near_start`, `near_start_counts` | dict | O+D | stations, tram/subway, bus stops, parking near the start (nearest 3, plus counts); parking items carry `capacity` and `fee` when tagged |
| `features.near_end` | dict/null | O+D | nearest station/stop at the end of linear routes |
| `features.protected_areas` | list | O+D | protected areas covering ≥ 10% of the length: `{name, kind, share}` |
| `features.profile` | dict | O+D | length shares from member-way tags: `paved_share`, `unpaved_share`, `unknown_surface_share`, `surface_tagged_share`, `stroller_ok_share`, `road_share`, `busy_road_share`, `path_share`, `track_share`, `cycleway_share`, `steps_m`, `via_ferrata_m`, `sac_tagged_share`, `sac_m`, `mtb_tagged_share`, `mtb_scale_m`, `dog_no_share`, `dog_yes_share`, `lit_share` |
| `features.popularity_proxies` | dict | D | `other_routes_on_same_ways` (length-weighted mean number of other named route relations on the same ways), `pois_per_km`, `parking_capacity_near_start`, `has_wikidata` |
| `osm_tags` | dict | O | raw relation tags kept: name, ref, network, operator, osmc:symbol, symbol, colour, website, url, wikidata, wikipedia, from, to, via, roundtrip, distance, ascent, descent, description*, state, dog, wheelchair, … |
| `osm_length_m`, `osm_ascent_m`, `osm_descent_m` | int/null | O | parsed `distance`/`length`, `ascent`, `descent` tags (validation only) |
| `stitch_coverage` | float | D | share of member-way length on the main line |

Feature types and distance thresholds (to the route line, metres): peak 75,
saddle 50, alpine hut 100, wilderness hut 75, shelter 50, restaurant/cafe/pub/
ice cream 75, beer garden 75, viewpoint 50, waterfall 150, gorge 150, cave 100,
geological 100, lake ≥ 1 ha 100, river 60 *and followed for ≥ 1 km*, glacier 300,
spring 50, castle/palace/manor/fort 150, ruins 100, monastery 150, church 75,
museum 100, archaeological site 100, monument 75, attraction 75, playground 75,
swimming 150, zoo 200, bird hide 100, drinking water 50, picnic site 50, campsite
150, aerialway station 150, natural-monument tree 50. From the start point: train
station 1000, tram/subway 500, bus stop 300, parking 300.

## 7. Tag vocabulary and rules

The list and the rule text live in `backend/pipeline/tags.py` (`TAG_RULES`);
`derive_tags` implements exactly those rules. No tag is assigned without a fact.
Legacy tags (from the Outdooractive catalogue) keep their exact names so the
`v0_rules` Jaccard score keeps working.

| Tag | Rule | Routes |
|---|---|---|
| Round trip [legacy] | start/end within 300 m and not mostly retracing itself | 4617 (49.4%) |
| Out and back [legacy] | start/end within 300 m and ≥ 70% of the line within 30 m of another part ≥ 300 m away along it | 7 (0.1%) |
| Linear route [legacy] | start/end more than 300 m apart | 4723 (50.5%) |
| Scenic [legacy] | viewpoint, summit, lake, waterfall, gorge or glacier on the route | 3586 (38.4%) |
| Refreshment stops available [legacy] | restaurant, cafe, pub, beer garden, ice cream (75 m) or alpine hut (100 m) | 6167 (66.0%) |
| Family-friendly [legacy] | *heuristic*: hiking/running ≤ 7 km, ascent ≤ 250 m, difficulty 0, no SAC ≥ T3, < 10% busy roads, and a playground, swimming spot, lake, waterfall, zoo, picnic site, refreshment stop or hut on the route; cycling ≤ 30 km, ascent ≤ 300 m, difficulty ≤ 1, < 5% busy roads, same amenity condition; never MTB | 2354 (25.2%) |
| Cultural/historical interest [legacy] | castle, ruins, monastery, museum, archaeological site or historic monument on the route | 3458 (37.0%) |
| Geological highlights [legacy] | gorge, cave entrance, or named geological feature | 1473 (15.8%) |
| Flora [legacy] | ≥ 30% inside a nature reserve/national park, a named natural-monument tree, or a theme trail whose name is about plants/forest/moor | 1587 (17.0%) |
| Fauna [legacy] | zoo/wildlife park or bird hide, or a theme trail whose name is about animals | 388 (4.2%) |
| Summit route [legacy] | a named peak within 75 m | 1661 (17.8%) |
| Dog-friendly [legacy] | `dog=yes/leashed` on the relation or ≥ 50% of the ways, and `dog=no` nowhere | 15 (0.2%) |
| Suitable for strollers [legacy] | hiking/running, ≥ 90% on smooth surfaces, no steps, ascent ≤ 200 m, ≤ 15 km | 389 (4.2%) |
| Multi-stage route [legacy] | ≥ 2 stage sub-relations, or hiking time above one 8-hour day | 367 (3.9%) |
| Singletrail/Free ride [legacy] | MTB with ≥ 25% on paths | 19 (0.2%) |
| Cableway ascent/descent [legacy] | aerialway station within 150 m | 257 (2.7%) |
| Accessibility [legacy] | `wheelchair=yes` on the relation | 7 (0.1%) |
| Lake | named lake/reservoir/pond ≥ 1 ha within 100 m | 1551 (16.6%) |
| River | within 60 m of a named river for ≥ 1 km | 935 (10.0%) |
| Waterfall | waterfall within 150 m | 163 (1.7%) |
| Gorge | `natural=gorge` or named attraction/valley called *klamm/*schlucht/*tobel within 150 m | 111 (1.2%) |
| Cave | cave entrance within 100 m | 637 (6.8%) |
| Glacier | glacier within 300 m | 1 (0.0%) |
| Hut | alpine hut (100 m) or wilderness hut (75 m) | 323 (3.5%) |
| Beer garden | `amenity=biergarten` within 75 m | 1334 (14.3%) |
| Viewpoint | viewpoint within 50 m | 1283 (13.7%) |
| Castle | castle/palace/manor/fort within 150 m | 1859 (19.9%) |
| Ruins | ruins within 100 m | 540 (5.8%) |
| Monastery | monastery within 150 m | 354 (3.8%) |
| Museum | museum within 100 m | 1736 (18.6%) |
| Church/chapel | named place of worship within 75 m | 5342 (57.2%) |
| Playground | playground within 75 m | 1622 (17.4%) |
| Swimming spot | bathing place, swimming area or beach within 150 m | 262 (2.8%) |
| Picnic area | picnic site within 50 m | 421 (4.5%) |
| Drinking water | drinking-water point within 50 m | 306 (3.3%) |
| Nature reserve | ≥ 10% inside a nature reserve | 484 (5.2%) |
| National park | ≥ 10% inside a national park | 135 (1.4%) |
| Train access | railway station/halt within 1 km of the start | 1818 (19.5%) |
| Bus access | bus stop within 300 m of the start | 5022 (53.7%) |
| Public transport access | train ≤ 1 km, tram/subway ≤ 500 m or bus ≤ 300 m from the start | 5400 (57.8%) |
| Parking at start | public parking within 300 m of the start | 5532 (59.2%) |
| Alpine terrain | highest point ≥ 1600 m, a via ferrata, or SAC ≥ T3 on ≥ 100 m with highest point ≥ 1000 m | 193 (2.1%) |
| Via ferrata | `highway=via_ferrata` or `via_ferrata_scale` on the route | 23 (0.2%) |
| Mostly paved | ≥ 70% paved (explicit `surface`, else inferred from highway type) | 1677 (17.9%) |
| Mostly unpaved | ≥ 70% unpaved (same inference) | 4024 (43.1%) |
| Busy road sections | ≥ 10% on trunk/primary/secondary roads | 303 (3.2%) |
| Flat | ascent and descent both ≤ 150 m and ≤ 10 m/km | 1046 (11.2%) |
| Big climb | ascent or descent ≥ 1000 m | 332 (3.6%) |
| Waymarked | relation has `osmc:symbol` or `symbol` | 7335 (78.5%) |
| Pilgrimage route | name contains Jakobsweg, Pilgerweg, Wallfahrt, Via Romea/Claudia, Camino, Kreuzweg, or `pilgrimage=yes` | 87 (0.9%) |

Legacy tags **not** assigned: `Competition track` (no OSM fact distinguishes race
courses) and `Insider tip` (editorial). 13 routes carry no tag besides their shape tag.

Surface inference: explicit `surface` values map to paved/unpaved; without one,
roads and cycleways count as paved, `tracktype=grade1` as paved, other tracks
and paths as unpaved, footways/steps as unknown. Surface is explicitly tagged
on a median share of 0.85 of a route's length; 83.7% of routes have ≥ 50%
explicitly tagged.

## 8. Elevation

* Resample the main line every 20 m (linear interpolation between vertices),
  sample the DEM bilinearly. A point exactly on a tile's north/east edge falls
  back to the shared edge row/column of the neighbouring tile if needed; voids
  (-32768) would be linearly interpolated along the profile (none occurred).
* Smooth with a centred moving average, then count ascent/descent extreme to
  extreme with a hysteresis (dead-band): a peak or trough is confirmed only
  once the profile has moved back by the threshold, so real climbs are counted
  in full and wiggles below the threshold are ignored.
* **Parameters chosen on data.** The routes that carry an OSM `ascent` tag
  ≥ 50 m are split deterministically (md5 of the id) into a calibration and a
  validation half. A grid of smoothing windows {0, 60, 100, 160, 240, 320, 400} m
  x hysteresis {0, 2, 5, 10, 15} m is evaluated on the calibration half only;
  the minimum median absolute relative error is at **240 m window, 2 m
  hysteresis** (calibration MdAPE 0.144). Without any smoothing the raw DEM
  over-counts badly (MdAPE 0.617 on the calibration half).

Validation against OSM tags (`python -m pipeline.stats`, section "Validation"):

| Comparison | n | median rel. error | median abs. rel. error | p90 abs. rel. error | within 10% | within 20% | median abs. error |
|---|---|---|---|---|---|---|---|
| ascent vs OSM `ascent`, validation half | 101 | +0.019 | 0.171 | 0.622 | 0.30 | 0.55 | 57 m |
| ascent vs OSM `ascent`, calibration half | 114 | +0.002 | 0.145 | 0.582 | 0.31 | 0.60 | 45 m |
| raw DEM (no smoothing, no hysteresis), validation half | 101 | +0.542 | 0.569 | 1.776 | 0.03 | 0.13 | 262 m |
| length vs OSM `distance`/`length` | 4776 | −0.003 | 0.023 | 0.188 | 0.81 | 0.91 | 229 m |

Read this honestly: 97.6% of routes carry no ascent tag at all, the tags
themselves are of unknown quality (rounded, one direction only, copied from
signposts), and the error is large for individual routes (p90 above 60%). The
length agreement is good (81% within 10%); the tail comes from relations whose
mapping differs from the signposted route (variants included, incomplete
relations, stale tags).

Spot checks on well-known routes (printed by the stats command):

| id | title | length | ascent / descent | elevation |
|---|---|---|---|---|
| 1356555 | Eibsee Rundwanderweg | 7.0 km | +113 / −116 m | 983–1037 m |
| 26937 | Staffelsee-Rundweg | 20.6 km | +177 / −175 m | 648–686 m |
| 551290 | Rund um den Ammersee (cycling) | 46.3 km | +289 / −282 m | 533–618 m |
| 3190325 | Chiemsee Rundweg (cycling) | 58.2 km | +367 / −366 m | 516–592 m |
| 223215 | AV Weg 831A (Höllental) | 5.1 km | +616 / −13 m | 780–1385 m |
| 3201006 | Höllentalsteig (to the Zugspitze) | 4.6 km | +0 / −1532 m | 1388–2920 m |

Lengths and summit routes look right. The lake circuits show the main DEM
weakness: SRTM is a surface model (treetops, buildings, shoreline artefacts), so
flat routes collect spurious ascent (a lake shore at constant level shows
hundreds of metres over 58 km). The Höllentalsteig's highest point comes out at 2920 m although the
Zugspitze peak on the route has OSM `ele` 2962 m: SRTM at 30 m cannot resolve
sharp summits, so `max_ele_m` on summit routes is low.

## 9. Dedup

Two routes **of the same activity** are duplicates when ≥ 90% of each one's
50 m sample points lie within 40 m of the other's line (and lengths differ by
less than a factor 0.75). This catches the same trail mapped twice,
forward/backward variants, and stage/whole pairs that coincide. Superroutes that
only wrap a single stage are dropped earlier (55). The keeper is the relation with
an OSM description, then more tags, then the lower id; dropped ids are listed in
`duplicates_merged`. 10129 candidate pairs were compared; **56 duplicates**
dropped. A hiking and a cycling relation on the same lake shore are both kept
(different activities). Stages and their parent are both kept when they differ
(108 routes are stages of a kept parent; 27 parents have kept stages) and linked
via `stage_of` / `stages`.

## 10. Difficulty (0–3, one scale)

`difficulty = max(technical, effort)`; 0 easy, 1 moderate, 2 difficult, 3 very demanding.

* **Technical** (observed in OSM): the hardest grade covering ≥ 100 m of the
  route. Hiking/running SAC scale: T1 → 0, T2 → 1, T3 → 2, T4–T6 and via ferrata →
  3. MTB/cycling `mtb:scale`: S0 → 0, S1 → 1, S2 → 2, S3+ → 3.
* **Effort** (derived): index = km + max(ascent, descent)/K; levels at the
  thresholds below. Hiking uses the "Leistungskilometer" (K = 100).

| activity | K | level 1 from | level 2 from | level 3 from |
|---|---|---|---|---|
| hiking | 100 | 12 | 22 | 32 |
| running | 100 | 8 | 15 | 25 |
| cycling | 30 | 35 | 70 | 110 |
| mtb | 20 | 30 | 50 | 75 |

Distribution: 0: 5869 (62.8%), 1: 2329 (24.9%), 2: 623 (6.7%), 3: 526 (5.6%).
By source: `osm:sac_scale` 1694 (18.1%), `osm:mtb:scale` 169 (1.8%),
`derived:effort` 6332 (67.7%), effort dominating a tagged grade 885 (sac) +
267 (mtb). Only 2573 of 7648 hiking routes have a SAC grade on ≥ 100 m.

## 11. Duration

* Hiking: **DIN 33466**: 4 km/h horizontal, 300 m/h ascent, 500 m/h descent;
  the smaller of horizontal and vertical time is halved and added to the larger.
* Cycling: km / 18 km/h + ascent / 600 m/h. MTB: km / 12 + ascent / 400.
  Running: km / 9 + ascent / 600.
* No breaks are included. DIN 33466 is deliberately conservative (4 km/h).

## 12. Counts and distributions

| activity | routes |
|---|---|
| hiking | 7648 (81.8%) |
| cycling | 1428 (15.3%) |
| mtb | 216 (2.3%) |
| running | 55 (0.6%) |

| category_name | routes |
|---|---|
| Hiking trail | 7155 (76.5%) |
| Theme trail | 493 (5.3%) |
| Cycling | 1351 (14.5%) |
| Long distance cycling | 77 (0.8%) |
| Mountainbiking | 216 (2.3%) |
| Jogging | 52 (0.6%) |
| Trail running | 3 (0.0%) |

Category rules: hiking relations are `Theme trail` when the name matches a
theme pattern (Lehrpfad, Themenweg, Erlebnisweg, Kulturweg, Sagenweg,
Planetenweg, …; full regex in `attributes.py`) or `educational=yes`, else
`Hiking trail`. Cycling relations are `Long distance cycling` when ≥ 100 km or
`network=icn|ncn`, else `Cycling`. `route=mtb` → `Mountainbiking`. Running
relations are `Trail running` when ≥ 60% unpaved with ≥ 150 m ascent (or a
`route=running` that is ≥ 80% unpaved), else `Jogging`.

Loops: 4624 (49.5%). Distributions (min | p10 | p25 | median | p75 | p90 | max):

| | length km | ascent m | duration min |
|---|---|---|---|
| all | 1 · 3 · 4 · 7 · 13 · 28 · 390 | 0 · 27 · 70 · 144 · 273 · 522 · 6638 | 8 · 36 · 69 · 122 · 198 · 307 · 5604 |
| hiking | 1 · 2 · 4 · 7 · 11 · 17 · 331 | 0 · 28 · 72 · 137 · 242 · 426 · 6638 | 16 · 45 · 75 · 125 · 199 · 310 · 5604 |
| cycling | 3 · 4 · 7 · 21 · 40 · 65 · 390 | 0 · 24 · 61 · 204 · 421 · 674 · 4683 | 11 · 16 · 30 · 94 · 174 · 275 · 1659 |
| mtb | 3 · 6 · 11 · 21 · 33 · 47 · 148 | 0 · 119 · 318 · 588 · 840 · 1054 · 2823 | 17 · 56 · 117 · 203 · 284 · 388 · 1163 |
| running | 1 · 2 · 2 · 2 · 4 · 6 · 9 | 3 · 9 · 17 · 28 · 42 · 70 · 118 | 8 · 12 · 15 · 18 · 29 · 51 · 72 |

Highest point: median 522 m, p90 902 m, max 2920 m. Description length: median
536 characters (p10 355, p90 740). Tags per route: median 8 (p10 4, p90 14).

## 13. Missing values

| Field | missing |
|---|---|
| `locality`, `admin_district`, `ascent_m` | 0 (0.0%) |
| `end_locality` | 4624 (49.5%), i.e. exactly the loops; 0 of 4723 linear routes |
| `max_grade` / `difficulty_technical` | 6332 (67.7%) |
| `osm_length_m` | 4571 (48.9%) |
| `osm_ascent_m` | 9119 (97.6%) |

Raw OSM tags present: network 96.7%, osmc:symbol 71.9%, operator 65.5%, ref
54.7%, roundtrip 50.6%, description 42.4%, website 15.9%, from 14.8%, to 14.1%,
wikidata 3.7%. Descriptions: 5320 (56.9%) purely generated, 4027 (43.1%) quote an
OSM description first.

Feature coverage (routes with ≥ 1 item on the route): restaurant 64.5%, church
57.2%, castle 19.9%, museum 18.6%, peak 17.8%, playground 17.4%, lake 16.6%,
beer garden 14.3%, viewpoint 13.7%, geological 12.1%, river 10.0%, cave 6.8%,
ruins 5.8%, monastery 3.8%, drinking water 3.3%, alpine hut 2.9%, swimming 2.8%,
aerialway station 2.7%, waterfall 1.7%, gorge 1.2%, wilderness hut 0.8%. Near the
start: parking 59.2%, bus stop 53.7%, train station 19.5%, tram/subway 0.8%.
62.1% of routes run ≥ 10% through some protected area (mostly large
Landschaftsschutzgebiete; only nature reserves and national parks produce tags).

## 14. Descriptions

English text generated deterministically from the record (`describe.py`):
opening (length, shape, locality, district or from/to), elevation, named
natural highlights, huts and food, culture, protected areas, terrain and SAC/MTB
grade, difficulty and time, access from the start, the family/pushchair rules
when they apply, operator. Template variants and the order of middle sentences
are chosen from a SHA-256 hash of the relation id. Every fact comes from the
record; nothing is looked up elsewhere. Where OSM has a `description` it is
quoted first unchanged (often German, sometimes terse mapper text such as
"[LOC 37530] Rundwanderweg: …") and `description_source` is `osm+generated`.
OSM `note` tags are mapper notes, not route descriptions, and are not used.

## 15. Known weaknesses (blunt)

* **Descriptions are generated.** They are fluent but formulaic, repeat the
  same facts as the structured fields, and say nothing about atmosphere or
  views that OSM does not encode. Retrieval over them mostly rewards the
  structured facts. Titles are German, descriptions English.
* **Tags are derived from proximity**, not from route intent. A castle 140 m
  from a village street counts as "on the route"; a restaurant in a village the
  route passes makes it "Refreshment stops available" (66% of routes), and
  churches are everywhere (57%). Opening hours, seasons and whether a hut is
  staffed are ignored. "Family-friendly" and "Suitable for strollers" are
  heuristics over length/ascent/surface/amenities, not verified suitability.
  "Dog-friendly" is almost never assigned because OSM rarely records it.
* **DEM error.** SRTM is a 30 m surface model: flat lake shores and forest
  routes collect spurious ascent; sharp summits are cut (Höllentalsteig max
  2920 m vs Zugspitze `ele` 2962 m). Validation against OSM `ascent` tags (n=101) gives a median absolute
  error of 17%, with a long tail. Ascent/descent are in the mapped direction;
  for linear routes the reverse direction swaps them.
* **Mapping bias of OSM.** Coverage reflects mappers, not popularity: Franconia
  has dense waymarked-network mapping, the Allgäu (Schwaben) loses many routes
  that cross into Austria or Baden-Württemberg (`incomplete_in_extract` 1065),
  many Alpine paths exist as ways but not as named route relations, running
  routes are nearly absent (55), MTB routes are few (216). The catalogue is 82%
  hiking. Popularity proxies (`other_routes_on_same_ways`, POI density, parking
  capacity) are proxies only; there are no visitor counts.
* **Stitching.** Graph assembly takes the longest trail; when a relation maps
  variants with a main role, one variant is chosen and the others dropped;
  one-way split sections (forward/backward pairs) may both be counted. 10.0%
  of kept routes do not use all their member ways.
* **Direction and start.** The start of a loop is where the relation's first
  way starts, which is not necessarily the trailhead; `near_start` features
  inherit this.
* **Region edges.** Routes are dropped, not truncated, at the Bavarian border.
  Localities can be just across the border (nearest village, e.g. in Tyrol).
* **Snapshot.** OSM changes daily; this is the 1 Nov 2026 state.
* **File size.** `routes.jsonl.gz` is 4.8 MB compressed (31 MB of JSON lines
  uncompressed: 9347 routes with full structured features). `geometry.jsonl.gz`
  is 6.2 MB and `details.jsonl.gz` 3.0 MB.
