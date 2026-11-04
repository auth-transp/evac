// Χάρτης OpenStreetMap (Leaflet): αναζήτηση τόπου και σχεδίαση του περιγράμματος της εγκατάστασης.
// Το πολύγωνο επιστρέφεται ως [[lat, lon], ...] και στέλνεται στο backend (/api/osm).

const $ = id => document.getElementById(id);

let map = null, polyLayer = null, draftLayer = null;
let drawing = false, pts = [], cursor = null, polygon = null, onPolygon = () => {};

export const mapAvailable = () => typeof L !== 'undefined';

export function onPolygonChange(fn) { onPolygon = fn; }
export const getPolygon = () => polygon;
export const getMap = () => map;                    // για έλεγχο/αποσφαλμάτωση

// Δημιουργεί τον χάρτη στο πρώτο άνοιγμα (και ξαναϋπολογίζει το μέγεθός του όταν εμφανίζεται)
export function showMap(visible) {
  $('stage').classList.toggle('maponly', visible);
  if (!visible || !mapAvailable()) return;
  if (!map) {
    map = L.map('map', { doubleClickZoom: false }).setView([25.9, 51.55], 11);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',
    }).addTo(map);
    map.on('click', e => { if (drawing) addVertex(e.latlng); });
    map.on('dblclick', () => { if (drawing) finish(); });
    map.on('mousemove', e => { if (drawing) { cursor = e.latlng; redrawDraft(); } });
    window.addEventListener('keydown', onKey);
  }
  setTimeout(() => map.invalidateSize(), 0);
}

// ---------- σχεδίαση περιοχής ----------
export function startDraw() {
  if (!map) return;
  clearPolygon();
  drawing = true; pts = []; cursor = null;
  map.getContainer().style.cursor = 'crosshair';
  setStatus('Κλικ στον χάρτη για κάθε κορυφή του περιγράμματος· διπλό κλικ ή Enter για κλείσιμο, Backspace αναιρεί κορυφή, Esc ακυρώνει.');
}

export function clearPolygon() {
  drawing = false; pts = []; cursor = null; polygon = null;
  if (polyLayer) { polyLayer.remove(); polyLayer = null; }
  if (draftLayer) { draftLayer.remove(); draftLayer = null; }
  if (map) map.getContainer().style.cursor = '';
  onPolygon(null);
}

function addVertex(ll) {
  const last = pts[pts.length - 1];
  if (last && map.latLngToContainerPoint(last).distanceTo(map.latLngToContainerPoint(ll)) < 5) return;      // διπλό κλικ
  if (pts.length >= 3 && map.latLngToContainerPoint(pts[0]).distanceTo(map.latLngToContainerPoint(ll)) < 10) return finish();
  pts.push(ll);
  redrawDraft();
}

function redrawDraft() {
  if (draftLayer) draftLayer.remove();
  const line = cursor ? [...pts, cursor] : pts;
  draftLayer = L.featureGroup([
    L.polyline(line, { color: '#ff7a00', weight: 2, dashArray: '5 4' }),
    ...pts.map((p, i) => L.circleMarker(p, { radius: i ? 3 : 5, color: '#ff7a00', fillColor: '#fff', fillOpacity: 1, weight: 2 })),
  ]).addTo(map);
}

function finish() {
  if (pts.length < 3) { setStatus('Χρειάζονται τουλάχιστον 3 κορυφές.'); return; }
  polygon = pts.map(p => [p.lat, p.lng]);
  drawing = false; pts = []; cursor = null;
  if (draftLayer) { draftLayer.remove(); draftLayer = null; }
  if (polyLayer) polyLayer.remove();
  polyLayer = L.polygon(polygon, { color: '#2ecc71', weight: 3, fillColor: '#2ecc71', fillOpacity: 0.12 }).addTo(map);
  map.getContainer().style.cursor = '';
  const [w, h] = extentMeters(polygon);
  setStatus(`Περιοχή ≈ ${Math.round(w)} × ${Math.round(h)} m. Πάτα «Εισαγωγή από OpenStreetMap».`);
  onPolygon(polygon);
}

function onKey(e) {
  if (!drawing || /INPUT|TEXTAREA/.test(document.activeElement.tagName)) return;
  if (e.key === 'Enter') finish();
  else if (e.key === 'Escape') { drawing = false; pts = []; cursor = null; if (draftLayer) { draftLayer.remove(); draftLayer = null; } map.getContainer().style.cursor = ''; setStatus(''); }
  else if (e.key === 'Backspace' && pts.length) { pts.pop(); redrawDraft(); }
}

// Διαστάσεις του bounding box του πολυγώνου σε μέτρα (ίδια προβολή με το backend)
export function extentMeters(poly) {
  const lats = poly.map(p => p[0]), lons = poly.map(p => p[1]);
  const lat0 = lats.reduce((a, b) => a + b, 0) / lats.length;
  return [(Math.max(...lons) - Math.min(...lons)) * Math.cos(lat0 * Math.PI / 180) * 111320, (Math.max(...lats) - Math.min(...lats)) * 110574];
}

function setStatus(t) { $('mapInfo').textContent = t; }

// ---------- αναζήτηση τόπου (Nominatim, μόνο κατόπιν ενέργειας του χρήστη) ----------
export async function searchPlaces(q) {
  const res = await fetch('https://nominatim.openstreetmap.org/search?format=jsonv2&limit=6&q=' + encodeURIComponent(q), { headers: { 'Accept-Language': 'el,en' } });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return (await res.json()).map(r => ({ name: r.display_name, lat: +r.lat, lon: +r.lon, box: r.boundingbox && r.boundingbox.map(Number) }));
}

export function goTo(place) {
  if (!map) return;
  map.invalidateSize();                              // αλλιώς, αν ο χάρτης δημιουργήθηκε όσο ήταν κρυμμένος, το fitBounds υπολογίζει μηδενικό μέγεθος (zoom 0)
  if (place.box) map.fitBounds([[place.box[0], place.box[2]], [place.box[1], place.box[3]]], { maxZoom: 17 });
  else map.setView([place.lat, place.lon], 16);
}
