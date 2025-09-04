// ---- Helpers ----
let lastTableRowsExport = [];   // crudo para CSV/XLSX (con DV y SI/NO)
let lastTableRowsRender = [];   // con chips HTML para la UI


let last_png_b64 = null;
let lastDrawState = null;
let network = null;
// valores heredados para compatibilidad con funciones no utilizadas
let scale = 1, translateX = 0, translateY = 0;
let initialScale = 1;
const MIN_SCALE = 0.4, MAX_SCALE = 4, ZOOM_STEP = 0.15;
const INITIAL_ZOOM_MODE = 'fit';
const INITIAL_ZOOM_MARGIN = 0.95;

function renderNetwork(drawState, tiposFiltrados){
  if(!drawState) return;
  const pos = drawState.pos_full || {};
  const metric = drawState.metric_map || {};
  const riesgo = new Set(drawState.riesgo_set || []);
  const patr = new Set(drawState.patrimonio_set || []);
  const conc = new Set(drawState.concentracion_set || []);
  const seed = String(drawState.seed || "");
  const maxMetric = Math.max(...Object.values(metric).map(v=>Number(v)||0), 0);
  const allowed = (tiposFiltrados && tiposFiltrados.length)
    ? new Set(tiposFiltrados.map(t=>String(t).toUpperCase()))
    : null;

  const nodes = [];
  (drawState.nodes_full || []).forEach(id => {
    const m = Number(metric[id] || 0);
    const norm = maxMetric ? m / maxMetric : 0;
    const size = 30;
    let colorBg = '#f0fff0';
    if (m > 0) {
      const tone = Math.round(255 * (1 - norm));
      colorBg = `rgb(255,${tone},${tone})`;
    }
    let border = '#444';
    let shape = 'dot';
    let bw = 1;
    if (patr.has(id)) { border = 'khaki'; }
    if (conc.has(id)) { border = 'gold'; shape = 'diamond'; }
    if (id === seed) { border = '#2b6cb0'; bw = 3; }
    if (riesgo.has(id)) { border = 'red'; bw = 3; }
    const p = pos[id] || [0,0];
    nodes.push({id, label:id, x:p[0]*300, y:-p[1]*300, size, shape,
                color:{background:colorBg, border}, borderWidth:bw});
  });

  const riesgoEdges = new Set((drawState.edges_riesgo||[]).map(e=>`${e[0]}|${e[1]}`));
  const concEdges = new Set((drawState.edges_concentracion||[]).map(e=>`${e[0]}|${e[1]}`));
  const edges = [];
  (drawState.edges_full || []).forEach(e => {
    const tipo = String(e.tipo || '');
    if (allowed && !allowed.has(tipo.toUpperCase())) return;
    const key1 = `${e.u}|${e.v}`;
    const key2 = `${e.v}|${e.u}`;
    let color = '#848484';
    let dashes = false;
    if (riesgoEdges.has(key1) || riesgoEdges.has(key2)) color = 'red';
    if (concEdges.has(key1) || concEdges.has(key2)) { color = 'gold'; dashes = true; }
    edges.push({from:e.u, to:e.v, color:{color}, dashes});
  });

  const data = {nodes:new vis.DataSet(nodes), edges:new vis.DataSet(edges)};
  const options = {physics:false, interaction:{hover:true, dragNodes:true}};
  if (network) network.destroy();
  const container = document.getElementById('network');
  network = new vis.Network(container, data, options);
  container.style.display = 'block';
}


// ==== helpers ====

// === DV de RUT (Módulo 11) ===
function rutDv(numStr) {
  const s = String(numStr || "").replace(/\D/g, "");

  // Si después de limpiar el string no es numérico, devuelve "X"
  if (s === "" || isNaN(s)) {
    return "X";
  }

  // Lógica de Módulo 11 para RUTs chilenos
  let mul = 2,
    sum = 0;
  for (let i = s.length - 1; i >= 0; i--) {
    sum += parseInt(s[i], 10) * mul;
    mul = (mul === 7 ? 2 : mul + 1);
  }
  const r = 11 - (sum % 11);
  return (r === 11) ? "0" : (r === 10 ? "K" : String(r));
}

function rutConDv(numStr){
  const body = rutSoloCuerpo(numStr);
  if (!body) return "";
  return `${body}-${rutDv(body)}`;
}

// === chips para tabla ===
function asSiNo(v){
  const s = String(v).trim().toUpperCase();
  return (s === "SI" || s === "TRUE" || s === "1") ? "SI" : "NO";
}
// En tabla: NO=verde (ok), SI=rojo (bad)
function chipBoolTabla(v){
  const t = asSiNo(v);
  const cls = (t === "SI") ? "bad" : "ok";
  return `<span class="seed-pill ${cls}">${t}</span>`;
}
function warnClassByCount(n){
  const k = Math.max(0, Math.min(5, Number(n) || 0));
  return k === 0 ? "num" : `warn${k}`;   // warn1..warn5
}
function chipCountTabla(n){
  const k = Number(n) || 0;
  const cls = warnClassByCount(k);
  return `<span class="seed-pill ${cls}">${k}</span>`;
}



function warnClassByCount(n){
  const k = Math.max(0, Math.min(5, Number(n) || 0));
  return k === 0 ? "num" : `warn${k}`;
}
function pillCount(label, n){
  const cls = warnClassByCount(n);
  const v = (n === undefined || n === null) ? 0 : n;
  return `<span class="seed-metric"><b>${label}:</b> <span class="seed-pill ${cls}">${v}</span></span>`;
}



function pill(label, value){
  const v = (value === undefined || value === null) ? "" : String(value);
  return `<span class="pill"><b>${label}:</b> ${v}</span>`;
}
function pillHtml(label, val, type="num"){
  const v = (val === undefined || val === null) ? "" : String(val);
  const cls = type === "bool" ? (String(val).toUpperCase()==="SI" ? "bad" : "ok") : "num";
  return `<span class="seed-metric"><b>${label}:</b> <span class="seed-pill ${cls}">${v}</span></span>`;
}



function formatMilesCL(val) {
  var n = Number(val);
  if (isFinite(n)) {
    try { return n.toLocaleString('es-CL'); }
    catch (e) { return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, "."); }
  }
  return (val !== undefined && val !== null) ? val : "";
}
function isMontoBraices(colName) {
  // normaliza (quita tildes, ignora may/min y espacios extremos)
  var up = String(colName).trim()
    .normalize ? colName.trim().normalize('NFD').replace(/[\u0300-\u036f]/g,'').toUpperCase()
               : String(colName).trim().toUpperCase();
  return up === "MONTO B. RAICES"; // nombre final que entrega la API
}

function buildStamp() {
  const now = new Date();
  const yyyy = String(now.getFullYear());
  const mm   = String(now.getMonth() + 1).padStart(2, '0');
  const dd   = String(now.getDate()).padStart(2, '0');
  const ss   = String(now.getSeconds()).padStart(2, '0');
  const ms2  = String(now.getMilliseconds()).padStart(3, '0').slice(0, 2); // 2 ms
  return `${yyyy}${mm}${dd}${ss}${ms2}`;
}

function normalizeRutForFile(rut) {
  return String(rut || '').replace(/\D+/g, ''); // solo números
}

function csvEscape(s) {
  var t = (s === null || s === undefined) ? "" : String(s);
  return '"' + t.replace(/"/g, '""') + '"';
}

function normalizeLabel(s) {
  // mayúsculas + sin tildes
  var t = String(s || '').trim();
  if (t.normalize) t = t.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
  return t.toUpperCase();
}


function formatMilesCL(val) {
  var n = Number(val);
  if (isFinite(n)) {
    try { return n.toLocaleString('es-CL'); }
    catch (e) {
      var s = Math.round(n).toString();
      return s.replace(/\B(?=(\d{3})+(?!\d))/g, ".");
    }
  }
  return (val !== undefined && val !== null) ? val : "";
}

function normalizeLabelSafe(s) {
  // MAYÚSCULAS, sin tildes, sin puntos extra
  var t = String(s || '').replace(/\./g, ' ').trim();
  // quitar tildes sin usar normalize (por compatibilidad)
  var map = {'Á':'A','É':'E','Í':'I','Ó':'O','Ú':'U','Ä':'A','Ë':'E','Ï':'I','Ö':'O','Ü':'U','á':'a','é':'e','í':'i','ó':'o','ú':'u','ä':'a','ë':'e','ï':'i','ö':'o','ü':'u'};
  t = t.replace(/[ÁÉÍÓÚÄËÏÖÜáéíóúäëïöü]/g, function(m){ return map[m] || m; });
  return t.toUpperCase();
}

function isMontoBraicesHeader(h) {
  var up = normalizeLabelSafe(h);
  return (up === 'MONTO B RAICES' || up === 'MONTO B  RAICES');
}

function csvEscape(s) {
  var t = (s === null || s === undefined) ? '' : String(s);
  return '"' + t.replace(/"/g, '""') + '"';
}

function rowsToCSVFormatted(rows) {
  if (!rows || !rows.length) return '';
  var headers = Object.keys(rows[0]);
  var lines = [];

  // encabezado
  lines.push(headers.map(function(h){ return csvEscape(h); }).join(','));

  // filas
  for (var i = 0; i < rows.length; i++) {
    var r = rows[i];
    var out = [];
    for (var j = 0; j < headers.length; j++) {
      var h = headers[j];
      var v = r[h];
      if (isMontoBraicesHeader(h)) v = formatMilesCL(v);   // ← formatea SOLO “Monto B. Raices”
      out.push(csvEscape(v));
    }
    lines.push(out.join(','));
  }
  return lines.join('\r\n');
}


function toCSV(rows){
  if (!rows?.length) return "";
  const cols = Object.keys(rows[0]);
  const esc = v => {
    const s = (v ?? "").toString().replace(/"/g,'""');
    return /[",\n]/.test(s) ? `"${s}"` : s;
  };
  return [cols.join(","), ...rows.map(r => cols.map(c => esc(r[c])).join(","))].join("\n");
}

async function saveViaPython(rows, fmt, defaultName){
  if (!rows?.length) return { ok:false, msg:"sin datos" };
  if (window.pywebview?.api?.guardar_tabla) {
    return await window.pywebview.api.guardar_tabla(rows, fmt, defaultName);
  }
  return { ok:false, msg:"api no disponible" };
}


function prepararFilasTabla(rowsIn){
  if (!Array.isArray(rowsIn)) return { exportRows: [], renderRows: [] };

  //alert(rowsIn.find(r=>String(r.RUT_REL).startsWith("E"))))
  const BOOL_COLS = ["ROE_ANOMALO", "FP", "ROS MP", "REQ. MP"];
  const BOOL_FROM_BOOL = ["Conc. Patrimonio", "Alto Riesgo"];
  const CNT_COLS = ["N° ROS REC.", "N° ROS ANT.", "N° CAUSAS REC.", "N° CAUSAS ANT."];

  const baseCols = rowsIn.length ? Object.keys(rowsIn[0]) : [];
  const order = [];
  for (const c of baseCols) {
    order.push(c);
    if (c === "RUT_PADRE") order.push("DV PADRE");
    if (c === "RUT_REL")   order.push("DV REL");
  }

  const exportRows = [];
  const renderRows = [];

  for (const r0 of rowsIn) {
    const exp = {};
    const ren = {};

    // RUTs tal cual (como string) para no perder letras
    const rutPadre = r0["RUT_PADRE"] ?? r0["RUT_ORIGEN_PADRE"];
    const rutRel   = r0["RUT_REL"]   ?? r0["RUT"];

    const bodyPadre = rutSoloCuerpo(rutPadre);
    const bodyRel   = rutSoloCuerpo(rutRel);

    // ← DV solo cuando el cuerpo es 100% numérico
    const dvPadre = /^\d+$/.test(bodyPadre) ? rutDv(bodyPadre) : "";
    const dvRel   = /^\d+$/.test(bodyRel)   ? rutDv(bodyRel)   : "";

    for (const c of order) {
      if (c === "DV PADRE"){ exp[c] = dvPadre; ren[c] = dvPadre; continue; }
      if (c === "DV REL"){   exp[c] = dvRel;   ren[c] = dvRel;   continue; }

      // Asegura que RUT_PADRE / RUT_REL se traten SIEMPRE como string (para “E...”)
      if (c === "RUT_PADRE"){ exp[c] = String(rutPadre ?? ""); ren[c] = String(rutPadre ?? ""); continue; }
      if (c === "RUT_REL"){   exp[c] = String(rutRel ?? "");   ren[c] = String(rutRel ?? "");   continue; }

      let v = r0[c];

      if (BOOL_FROM_BOOL.includes(c)){
        const siNo = v === true || String(v).toUpperCase() === "TRUE" ? "SI" : "NO";
        exp[c] = siNo;
        ren[c] = chipBoolTabla(siNo);
        continue;
      }
      if (BOOL_COLS.includes(c)){
        exp[c] = asSiNo(v);
        ren[c] = chipBoolTabla(v);
        continue;
      }
      if (CNT_COLS.includes(c)){
        exp[c] = Number(v) || 0;
        ren[c] = chipCountTabla(v);
        continue;
      }

      exp[c] = v;
      ren[c] = v;
    }

    exportRows.push(exp);
    renderRows.push(ren);
  }
  return { exportRows, renderRows };
}


// Reemplaza tu rutSoloCuerpo por esta versión
function rutSoloCuerpo(valor) {
  if (!valor) return "";
  const raw = String(valor).trim().toUpperCase();
  const s = raw.replace(/[.\s]/g, ""); // quita puntos/espacios, conserva letras

  // Si viene con guion → usar la parte antes del guion (solo dígitos)
  if (s.includes("-")) return s.split("-")[0].replace(/\D/g, "");

  // Sin guion:
  // - si contiene letras (pasaporte tipo 'E...') → devolver tal cual (para mostrarlo)
  if (/[A-Z]/.test(s)) return s;

  // - si son solo dígitos → NO cortar (asumimos que no trae DV separado)
  return s;
}



function setFechaHoy() {
  const hoy = new Date().toLocaleDateString("es-CL");
  document.getElementById("fecha").innerText = "Fecha: " + hoy;
}

// Espera pywebview listo; reintenta si hace falta
async function getUsuarioRobusto(intentos = 6, delayMs = 400) {
  const setLabel = (txt) => document.getElementById("usuario").innerText = "Usuario: " + txt;
  for (let i = 0; i < intentos; i++) {
    try {
      if (window.pywebview && window.pywebview.api && window.pywebview.api.get_usuario) {
        const user = await window.pywebview.api.get_usuario();
        setLabel(user || "Desconocido");
        return;
      }
    } catch (e) {
      console.error("get_usuario error:", e);
    }
    await new Promise(r => setTimeout(r, delayMs));
  }
  setLabel("Desconocido");
}

// ---- Pan/Zoom state ----

function applyTransform() {
  const img = document.getElementById("img");
  if (img) img.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
}

function resetView(useInitial = true) {
  scale = useInitial ? initialScale : 1;
  translateX = 0;
  translateY = 0;
  applyTransform();
}

// Calcula y aplica el zoom inicial según el modo configurado
function setInitialView(imgEl, wrapEl) {
  const nw = imgEl.naturalWidth || imgEl.width;
  const nh = imgEl.naturalHeight || imgEl.height;
  const cw = wrapEl.clientWidth;
  const ch = wrapEl.clientHeight;

  let s;
  if (typeof INITIAL_ZOOM_MODE === 'number') {
    s = INITIAL_ZOOM_MODE;
  } else if (INITIAL_ZOOM_MODE === 'width') {
    s = cw / nw;
  } else if (INITIAL_ZOOM_MODE === 'height') {
    s = ch / nh;
  } else { // 'fit' (por defecto)
    s = Math.min(cw / nw, ch / nh);
  }

  s *= INITIAL_ZOOM_MARGIN;
  s = Math.max(MIN_SCALE, Math.min(MAX_SCALE, s));

  initialScale = s;     // guardamos para el botón Reset
  resetView(true);      // centra y aplica initialScale
}

// ---- Consulta principal ----
async function consultar() {
    
    const rutInput = document.getElementById("rut");
    const rut = rutSoloCuerpo(rutInput.value);  // ← usar la nueva función
    
    if (!rut) { alert("Ingresa un RUT válido"); return; }
      // guarda el último PNG

    
   

  // Preparar UI
  document.querySelector(".viewer").style.display = "none";
  document.getElementById("network").style.display = "none";
  document.getElementById("imgWrap").style.display = "none";
  const img = document.getElementById("img");
  img.style.display = "none";
  document.getElementById("descargar").style.display = "none";
  document.getElementById("tabla").innerHTML = "";
  document.getElementById("loader").style.display = "flex";

  try {
    const res = await window.pywebview.api.calcular(
      rut, true, "cobertura", 0.80, 95, 2.0, 50, 0
    );

    if (!res || res.ok !== true) {
      const msg = (res && res.msg) ? res.msg : "No se encontraron relacionados para el RUT consultado.";
      // Mensaje amigable (puedes reemplazar alert por un banner en la página)
      alert(msg);

      // Limpiar UI
      document.getElementById("loader").style.display = "none";
      document.querySelector(".viewer").style.display = "none";
      document.getElementById("tabla").innerHTML = "";
      document.querySelector(".table-toolbar").style.display = "none";
      return;
    }
    
      last_png_b64 = res.png_b64;
      lastDrawState = res.draw_state;
      renderNetwork(lastDrawState);
      document.querySelector(".viewer").style.display = "block";
      document.getElementById("imgWrap").style.display = "none";
      document.getElementById("descargar").style.display = "inline-block";

    // --- Título + Resumen bajo el título (HTML) ---
    const titleEl = document.getElementById("chartTitle");
    const summaryEl = document.getElementById("netMetrics");
    
    
    const seed = res?.meta?.seed_info;
    const seedBox = document.getElementById("seedCard");
    if (seed) {
      const rutFull = rutConDv(seed.RUT ?? seedRut);
      const montoFmt = formatMilesCL(seed.MONTO_BRAICES || 0);
      seedBox.innerHTML = `
        <div class="seed-card__head">Información del RUT consultado</div>
        <div class="row">
          <span class="seed-metric"><b>RUT consultado:</b> <span class="seed-pill">${rutFull}</span></span>
          ${pillHtml("ROE anómalo", seed.ROE_ANOMALO, "bool")}
          ${pillHtml("FP",          seed.FP, "bool")}
          ${pillHtml("Inf. MP",     seed.TIENE_MP, "bool")}
          ${pillHtml("Req. MP",     seed.TIENE_REQ_MP, "bool")}
    
          ${pillCount("ROS rec.",    seed.ROS_RECIENTE)}
          ${pillCount("ROS ant.",    seed.ROS_ANTIGUO)}
          ${pillCount("Causas rec.", seed.CAUSAS_RECIENTES)}
          ${pillCount("Causas ant.", seed.CAUSAS_ANTIGUAS)}
    
          ${pillHtml("B. Raíces", seed.CANT_BRAICES)}
          ${pillHtml("Monto B. Raíces", montoFmt)}
          ${pillHtml("Vehículos (5 años)", seed.VEHICULOS_5AGNOS)}
        </div>`;
      seedBox.style.display = "block";
    } else {
      seedBox.style.display = "none";
    }
    
    const toolbar = document.getElementById("tableToolbar");
    
    toolbar.style.display = "block";
    // 1) Título con RUT investigado (agranda la fuente un poco)
    titleEl.style.fontSize = "18px";
    titleEl.innerText = `Red de relacionados`;

    // 2) Subtítulo con métricas
    const rr = res?.meta?.resumen_red || {};
    summaryEl.innerHTML = `
      <strong>Scoring RED: </strong> ${rr.scoring ?? 0}
      &nbsp;–&nbsp; <strong>Nº nodos:</strong> ${rr.n_nodos ?? 0}
      &nbsp;–&nbsp; <strong>Nº riesgosos:</strong> ${rr.n_riesgosos ?? 0}
      &nbsp;–&nbsp; <strong>Máx score:</strong> ${rr.max_score ?? 0}
    `;
   

// construir filtros con los tipos disponibles (todos marcados)
const tipos = res?.meta?.tipos_relacion || [];
buildRelFilter(tipos);
    
    const rows = JSON.parse(res.tabla || "[]");
    const { exportRows, renderRows } = prepararFilasTabla(rows);

    lastTableRowsExport  = exportRows;
    lastTableRowsRender  = renderRows;
    
    if (renderRows.length) {
      const cols  = Object.keys(renderRows[0]);        // ya trae DV PADRE / DV REL
      const thead = "<thead><tr>" + cols.map(c => `<th>${c}</th>`).join("") + "</tr></thead>";
      const tbody = "<tbody>" + renderRows.map(r => {
        return "<tr>" + cols.map(c => {
          let v = r[c];
          // formateo SOLO para mostrar “Monto B. Raices”
          if (isMontoBraicesHeader(c)) v = formatMilesCL(v);
          return `<td>${(v ?? "")}</td>`;
        }).join("") + "</tr>";
      }).join("") + "</tbody>";
      document.getElementById("tabla").innerHTML = "<table>" + thead + tbody + "</table>";
      document.querySelector(".table-toolbar").style.display = "flex";
    } else {
      document.getElementById("tabla").innerHTML = "<p>Sin registros.</p>";
      document.querySelector(".table-toolbar").style.display = "none";
    }


 } catch (e) {
  const msg = e?.stack
    ? `${e.name || 'Error'}: ${e.message || e}\n\n${e.stack}`
    : (typeof e === 'object' ? JSON.stringify(e, null, 2) : String(e));
  alert(`Ocurrió un error:\n\n${msg}`);
}finally {
    document.getElementById("loader").style.display = "none";
  }


  

    // Descargar
    const btnDescargar = document.getElementById("descargar");
    btnDescargar.style.display = "none";
    document.getElementById("descargar").onclick = async () => {
      const out = await window.pywebview.api.guardar_png(last_png_b64, "grafico_red.png");
      if (!out?.ok && out?.msg !== "cancelado") {
        alert("No se pudo guardar el archivo: " + (out.msg || "error desconocido"));
      }
    };

}


function buildRelFilter(tipos) {
  const cont  = document.getElementById("relFilter");
  const items = document.getElementById("relFilterItems");
  items.innerHTML = "";

  (tipos || []).forEach(t => {
    const id = "rel_" + String(t).replace(/\W+/g, "_");
    const label = document.createElement("label");
    label.style.display = "inline-flex";
    label.style.alignItems = "center";
    label.style.gap = "4px";

    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.id = id; cb.value = t; cb.checked = true;
    cb.addEventListener("change", debounceRedraw);

    label.appendChild(cb);
    label.appendChild(document.createTextNode(t));
    items.appendChild(label);
  });

  cont.style.display = (tipos && tipos.length) ? "flex" : "none";
  document.getElementById("relCheckAll").onclick = () => { setAll(true); debounceRedraw(); };
  document.getElementById("relUncheckAll").onclick = () => { setAll(false); debounceRedraw(); };

  function setAll(val) {
    items.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = val);
  }
}

let redrawTimer = null;
function debounceRedraw(){
  clearTimeout(redrawTimer);
  redrawTimer = setTimeout(() => {
    const checked = Array.from(document.querySelectorAll("#relFilterItems input[type='checkbox']:checked"))
                         .map(cb => cb.value);
    renderNetwork(lastDrawState, checked);
  }, 150);
}



// ---- Inicialización ----
window.addEventListener("DOMContentLoaded", () => {
  // Controles
  document.getElementById("btn").addEventListener("click", consultar);
  document.getElementById("rut").addEventListener("keyup", ev => { if (ev.key === "Enter") consultar(); });

  setFechaHoy();

  // Usuario: esperar pywebview
  if (window.pywebview) getUsuarioRobusto();
  window.addEventListener("pywebviewready", () => getUsuarioRobusto());

  document.getElementById("zoomIn").addEventListener("click", () => {
    if (network) {
      const s = network.getScale();
      network.moveTo({scale: s + ZOOM_STEP});
    }
  });
  document.getElementById("zoomOut").addEventListener("click", () => {
    if (network) {
      const s = network.getScale();
      network.moveTo({scale: Math.max(0.1, s - ZOOM_STEP)});
    }
  });
  document.getElementById("zoomReset").addEventListener("click", () => {
    if (network) network.fit();
  });
});

document.addEventListener("click", async (e) => {
  const id = e.target?.id;
  
  if (!id) return;

  // nombre dinámico
  const rutSeed  = document.getElementById("rut")?.value || "";
  const rutFile  = normalizeRutForFile(rutSeed);
  const stamp    = buildStamp();
  const csvName  = `datosRedes_${rutFile}_${stamp}.csv`;
  const xlsxName = `datosRedes_${rutFile}_${stamp}.xlsx`;

  // === CSV ===
  if (id === "saveCsv") {
      try {
        const out = await saveViaPython(lastTableRowsExport, "csv", csvName);
        if (out && out.ok) return;
        if (out && out.msg === "cancelado") return;
      } catch (_) {}
    
      // Fallback navegador
      var csv = rowsToCSVFormatted(lastTableRowsExport);
      var blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      var a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = csvName;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(a.href);
      return;
    }

  if (id === "saveXlsx") {
    // Intentamos XLSX; si tu entorno no tiene openpyxl, la API hace fallback a CSV
    const out = await saveViaPython(lastTableRows, "xlsx", "relacionados.xlsx");
    if (out.ok) {
      if (out.msg) alert(out.msg); // mensaje de fallback a CSV si aplica
      return;
    }
    if (out.msg !== "cancelado") {
      // Fallback navegador: generamos CSV compatible con Excel
      const csv = toCSV(lastTableRows);
      const blob = new Blob([csv], {type:"text/csv;charset=utf-8;"});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "relacionados.csv";
      a.click();
    }
  }
});
