/*
 * gee_find_reaches.js - List the SWORD reach IDs near the gauge (Google Earth Engine).
 *
 * Usage:
 *   Paste into the Earth Engine Code Editor (code.earthengine.google.com) and click Run.
 *
 * SWOT RiverSP data are reported per SWORD reach (~10 km long), each with an
 * 11-digit reach ID; SWOT WSE is most reliable on rivers wider than ~100 m.
 * Earth Engine hosts SWORD v16. The script lists reaches within RADIUS_KM of
 * the gauge, sorted by distance, as a YAML list for swot_config.yaml.
 *   - On the map, cyan lines are reaches and the red point is the gauge. Use
 *     the Inspector tab to read the ID of a reach.
 *   - Keep only main-stem Athabasca reaches; remove tributaries such as the
 *     Clearwater.
 *   - The first reach listed is the nearest to the gauge and is the likely
 *     gauge_reach. Confirm on the map that the gauge lies on it.
 *
 * SWOT Version D uses SWORD v17b, in which some of these IDs changed;
 * 11b_find_version_d_ids.py handles the mapping.
 *
 * Inputs:  GAUGE_LAT, GAUGE_LON (printed by 10_fetch_gauge.py); SWORD v16 reaches
 * Outputs: console table of nearby reaches and a YAML list for swot_config.yaml
 */

// Gauge location (printed by 10_fetch_gauge.py)
var GAUGE_LON = -111.40;
var GAUGE_LAT = 56.78;
var RADIUS_KM = 80;          // search radius around the gauge (km)
var MIN_WIDTH_M = 100;       // SWOT WSE is less reliable on rivers narrower than ~100 m
var NAME_FILTER = 'Athabasca';   // '' disables the river-name filter

var gauge = ee.Geometry.Point([GAUGE_LON, GAUGE_LAT]);

var reaches = ee.FeatureCollection(
    'projects/sat-io/open-datasets/SWORD/reaches_merged')
  .filterBounds(gauge.buffer(RADIUS_KM * 1000))
  .filter(ee.Filter.gt('width', MIN_WIDTH_M));

if (NAME_FILTER) {
  reaches = reaches.filter(ee.Filter.stringContains('river_name', NAME_FILTER));
}

// Convert the IDs to strings; numeric IDs are printed in scientific notation
// and cannot be copied accurately
reaches = reaches.map(function(f) {
  var rid = f.get('reach_id');
  var ridStr = ee.Algorithms.If(
    ee.Algorithms.IsEqual(ee.Algorithms.ObjectType(rid), 'String'),
    rid,
    ee.Number(rid).format('%.0f'));
  return f.set({
    reach_id_str: ridStr,
    dist_to_gauge_km: f.geometry().distance(gauge, 10).divide(1000)
  });
}).sort('dist_to_gauge_km');

print('Reaches found:', reaches.size());
print('Properties available:', reaches.first().propertyNames());
print('Nearest reaches (first = likely gauge_reach):',
      reaches.limit(40).map(function(f) {
        return ee.Feature(null, {
          reach_id: f.get('reach_id_str'),
          river_name: f.get('river_name'),
          width_m: f.get('width'),
          km_from_gauge: f.get('dist_to_gauge_km')
        });
      }));

// YAML list to paste under reaches: in swot_config.yaml
var ids = ee.List(reaches.aggregate_array('reach_id_str'));
print('Paste under `reaches:` in swot_config.yaml:',
      ids.map(function(s) { return ee.String('  - "').cat(s).cat('"'); })
         .join('\n'));

Map.centerObject(gauge, 9);
Map.addLayer(reaches, {color: '00FFFF'}, 'SWOT reaches (SWORD v16)');
Map.addLayer(gauge, {color: 'FF0000'}, 'gauge');
