/*
 * Screening test: is a channel wide enough to be resolved by Sentinel-2
 * and Sentinel-1 at 10 m?
 *
 * Within a circular buffer around a point on the channel, the script
 * computes the Otsu threshold for MNDWI (optical) and for VV backscatter
 * (SAR), and reports Otsu's separability measure
 *
 *     eta = between-class variance / total variance
 *
 * eta close to 1 means water and land pixels form distinct classes. eta
 * close to 0 means the channel pixels are mixed with the banks, and no
 * width can be extracted.
 *
 * Usage: paste into the Earth Engine Code Editor and run. Inspect each
 * point on the map first (the layers are hidden by default) and confirm
 * that the buffer lies on the main channel, not on a lake, confluence,
 * estuary or bridge.
 */

// Hokkaido rivers, ordered approximately from narrowest to widest.
// lon/lat are approximate. widthGuess (m) holds a manually measured width;
// it is not used in the calculation and remains null until measured.
var reaches = [
  // Tested: neither sensor separates water from land reliably
  {name: 'Yubetsu (Engaru)',       lon: 143.530, lat: 44.060, widthGuess: null},
  {name: 'Yubetsu (mouth)',        lon: 143.600, lat: 44.200, widthGuess: null},

  // Expected to fail or be marginal
  {name: 'Biei R (Biei)',          lon: 142.470, lat: 43.590, widthGuess: null},
  {name: 'Chubetsu R (Asahikawa)', lon: 142.400, lat: 43.750, widthGuess: null},
  {name: 'Uryu R (Chippubetsu)',   lon: 141.850, lat: 43.720, widthGuess: null},
  {name: 'Sorachi R (Ashibetsu)',  lon: 142.180, lat: 43.510, widthGuess: null},

  // Expected to pass
  {name: 'Teshio R (Nayoro)',      lon: 142.460, lat: 44.360, widthGuess: null},
  {name: 'Teshio R (lower)',       lon: 141.800, lat: 44.850, widthGuess: null},
  {name: 'Tokachi R (Obihiro)',    lon: 143.200, lat: 42.920, widthGuess: null},
  {name: 'Tokachi R (Toyokoro)',   lon: 143.480, lat: 42.700, widthGuess: null},
  {name: 'Ishikari R (Takikawa)',  lon: 141.900, lat: 43.560, widthGuess: null},
  {name: 'Ishikari R (Ebetsu)',    lon: 141.570, lat: 43.110, widthGuess: null}
  // Ishikari mouth excluded: the buffer would include open sea
];

var BUFFER_M = 1200;   // buffer radius around each point (m)
var SCALE    = 10;

// Scene-level cloud filter, deliberately permissive because maskClouds()
// removes cloudy pixels. The first run returned no Sentinel-2 scenes (at a
// 10% threshold no Hokkaido summer scenes remained), so it was raised to 60%.
var MAX_CLOUD = 60;
var START     = '2024-06-01';
var END       = '2025-10-31';

// June to September only, because snow and ice also have high MNDWI
var MONTH_MIN = 6;
var MONTH_MAX = 9;

// Otsu threshold and between-class variance from an ee histogram.
// Otsu's method selects the threshold that maximises between-class variance.
function otsu(histogram) {
  histogram = ee.Dictionary(histogram);
  var counts = ee.Array(histogram.get('histogram'));
  var means  = ee.Array(histogram.get('bucketMeans'));
  var size   = means.length().get([0]);
  var total  = counts.reduce(ee.Reducer.sum(), [0]).get([0]);
  var sum    = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0]);
  var mean   = sum.divide(total);

  var indices = ee.List.sequence(1, size);

  var bss = indices.map(function(i) {
    var aCounts = counts.slice(0, 0, i);
    var aCount  = aCounts.reduce(ee.Reducer.sum(), [0]).get([0]);
    var aMeans  = means.slice(0, 0, i);
    var aMean   = aMeans.multiply(aCounts)
                    .reduce(ee.Reducer.sum(), [0]).get([0])
                    .divide(aCount);
    var bCount  = total.subtract(aCount);
    var bMean   = sum.subtract(aCount.multiply(aMean)).divide(bCount);
    return aCount.multiply(aMean.subtract(mean).pow(2))
             .add(bCount.multiply(bMean.subtract(mean).pow(2)))
             .divide(total);
  });

  var maxBss = ee.Array(bss).reduce(ee.Reducer.max(), [0]).get([0]);
  var idx    = ee.List(bss).indexOf(maxBss);
  return {threshold: means.get([idx]), between: maxBss};
}

function separability(image, bandName, region, waterIsLow) {
  var hist = image.reduceRegion({
    reducer: ee.Reducer.histogram(255, 0.001),
    geometry: region, scale: SCALE, bestEffort: true, maxPixels: 1e9
  }).get(bandName);

  var o = otsu(hist);
  var thr = ee.Number(o.threshold);

  var totalVar = ee.Number(image.reduceRegion({
    reducer: ee.Reducer.variance(),
    geometry: region, scale: SCALE, bestEffort: true, maxPixels: 1e9
  }).get(bandName));

  // Water has low VV backscatter and high MNDWI
  var water = waterIsLow ? image.lt(thr) : image.gt(thr);

  var frac = ee.Number(water.rename('w').reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region, scale: SCALE, bestEffort: true, maxPixels: 1e9
  }).get('w'));

  return {
    threshold: thr,
    eta: ee.Number(o.between).divide(totalVar),
    waterFrac: frac,
    waterMask: water
  };
}

function maskClouds(img) {
  var scl = img.select('SCL');
  var bad = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));
  return img.updateMask(bad.not());
}

// Main loop over the reaches
var rows = [];

reaches.forEach(function(r) {
  var pt  = ee.Geometry.Point([r.lon, r.lat]);
  var buf = pt.buffer(BUFFER_M);

  var s2col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(buf)
    .filterDate(START, END)
    .filter(ee.Filter.calendarRange(MONTH_MIN, MONTH_MAX, 'month'))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD))
    .map(maskClouds);

  var s1col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(buf)
    .filterDate(START, END)
    .filter(ee.Filter.calendarRange(MONTH_MIN, MONTH_MAX, 'month'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'));

  var nS2 = s2col.size();
  var nS1 = s1col.size();

  // Empty-collection guard. The first run returned no Sentinel-2 scenes and
  // failed on a missing band ("No band named B3"). A constant image is now
  // substituted, so the loop continues and reports a scene count of 0.
  var s2 = ee.Image(ee.Algorithms.If(
    nS2.gt(0),
    s2col.median().select(['B3', 'B11']),
    ee.Image.constant([0, 0]).rename(['B3', 'B11'])
  ));
  var s1 = ee.Image(ee.Algorithms.If(
    nS1.gt(0),
    s1col.select('VV').median(),
    ee.Image.constant(0)
  )).rename('vv');

  var mndwi   = s2.normalizedDifference(['B3', 'B11']).rename('mndwi');
  var optical = separability(mndwi, 'mndwi', buf, false);
  var radar   = separability(s1, 'vv', buf, true);

  print(r.name,
        '| S2:', nS2, 'S1:', nS1,
        '| MNDWI eta:', optical.eta, 'water:', optical.waterFrac,
        '| VV eta:', radar.eta, 'water:', radar.waterFrac);

  rows.push(ee.Feature(pt, {
    name: r.name, widthGuess: r.widthGuess,
    n_s2: nS2, n_s1: nS1,
    mndwi_eta: optical.eta, mndwi_thr: optical.threshold,
    mndwi_water: optical.waterFrac,
    vv_eta: radar.eta, vv_thr: radar.threshold,
    vv_water: radar.waterFrac
  }));

  Map.addLayer(buf, {color: 'yellow'}, r.name + ' buffer', false);
  Map.addLayer(s2, {bands: ['B3'], min: 0, max: 3000},
               r.name + ' green band', false);
  Map.addLayer(optical.waterMask.selfMask(), {palette: ['red']},
               r.name + ' water (MNDWI Otsu)', false);
});

Export.table.toDrive({
  collection: ee.FeatureCollection(rows),
  description: 'separability_table',
  folder: 'river_power',
  fileFormat: 'CSV'
});

print('--- interpretation ---');
print([
  'eta > 0.6   clear separation, method expected to work',
  'eta 0.3-0.6 marginal, expect large width error',
  'eta < 0.3   no clear separation, channel too narrow',
  '',
  'Also check waterFrac. A 1.2 km buffer around a river should be a few',
  'percent water. A value near 0.5 with high eta means the threshold has',
  'separated something else (shadow, sea, cloud edge). Check the red layer.',
  '',
  'S2: 0 means no optical scenes passed the filters. Raise MAX_CLOUD',
  'or widen the date range.'
].join('\n'));

Map.setCenter(142.0, 43.5, 7);
