# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 11:58:18 2025

@author: acespedes
"""

import pandas as pd
from collections import defaultdict, deque
import networkx as nx
import numpy as np
import unicodedata, re

APP_NAME = "AppRedes"        # identifica esta app en la tabla
APP_VERSION = "1.0.0"     # tu versión actual de la app
# DRIVER={SQL Server}
SERVER='10.0.2.21'
DATABASE ='AlmacenAIEADE'
USERNAME ='acespedes'
PASSWORD ='UaF_2023'
SQL_CONN_STR = f'DRIVER={{SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD};charset="utf8"'


from params_loader import load_scoring_params

def _load_scoring_hot():
    # Archivo: recarga si cambió (suficiente para la mayoría)    
    return load_scoring_params(APP_NAME, SQL_CONN_STR, 
                              prefer_file=True, reload_mode="always")
    # O, si quieres SIEMPRE leer el archivo en cada click:
    # return load_scoring_params(APP_NAME, SQL_CONN_STR, reload_mode="always")
    
    

SCORING = _load_scoring_hot()  # file > BD (por defecto)



##ALPHA, MODO_SCORING, , ,
##, , CFG_PROB, MAP_TIPO_REL

# ejemplo de uso:
ALPHA = float(SCORING["ALPHA"])
MODO_SCORING  = SCORING.get("MODO_SCORING", "prob")

CFG_PROB = SCORING.get("CFG_PROB", {})                  # ← w_bin, w_cnt, s_rel

MAP_TIPO_REL = SCORING.get("MAP_TIPO_REL", {})                          # p.ej. {"MADRE": "Familiar Directo", ...}
S_REL       = SCORING.get("CFG_PROB", {}).get("s_rel", {})              # pesos por categoría (prob)

gamma_tipo= SCORING["gamma_tipo"]
w_bin = SCORING["CFG_PROB"]["w_bin"]
w_cnt = SCORING["CFG_PROB"]["w_cnt"]
map_tipo_rel = SCORING.get("MAP_TIPO_REL", {})


import unicodedata, re

def get_maps_normalizados():
    SC = _load_scoring_hot()
    # lectura robusta por si la clave quedó rara o anidada
    def _norm_key(s):
        import unicodedata, re
        s = str(s or "").strip().upper()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return re.sub(r"\s+", " ", s)

    def _get_key_relaxed(d, target, default=None):
        nt = _norm_key(target)
        for k in d.keys():
            if _norm_key(k) == nt:
                return d[k]
        return default

    CFG_PROB     = _get_key_relaxed(SC, "CFG_PROB", {}) or {}
    S_REL        = _get_key_relaxed(CFG_PROB, "s_rel", {}) or {}
    MAP_TIPO_REL = (_get_key_relaxed(SC, "MAP_TIPO_REL", {}) or
                    _get_key_relaxed(CFG_PROB, "MAP_TIPO_REL", {}) or {})

    S_REL_N        = { _norm_key(k): float(v) for k, v in S_REL.items() }
    MAP_TIPO_REL_N = { _norm_key(k): v for k, v in MAP_TIPO_REL.items() }
    return SC, CFG_PROB, S_REL_N, MAP_TIPO_REL_N


def _has_scipy() -> bool:
    try:
        import scipy  # noqa
        return True
    except Exception:
        return False
    
def _get_key_relaxed(d: dict, target: str, default=None):
    nt = _norm_key(target)
    for k in d.keys():
        if _norm_key(k) == nt:
            return d[k]
    return default

def _norm_key(s: str) -> str:
    s = str(s or "").strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s

_S_REL_N        = { _norm_key(k): float(v) for k, v in CFG_PROB.get("s_rel", {}).items() }


CFG_PROB     = _get_key_relaxed(SCORING, "CFG_PROB", {}) or {}
_MAP_TIPO_REL_N = (_get_key_relaxed(SCORING, "MAP_TIPO_REL", {}) or
                _get_key_relaxed(CFG_PROB, "MAP_TIPO_REL", {}) or {})



def _canon_and_weight(t_raw: str, S_REL_N: dict, MAP_TIPO_REL_N: dict):
    k_raw = _norm_key(t_raw)
    w_raw = S_REL_N.get(k_raw)  # override específico (p.ej. 'MADRE')
    if k_raw in MAP_TIPO_REL_N:
        canon = MAP_TIPO_REL_N[k_raw]                 # 'Familiar Directo'
        w_cat = S_REL_N.get(_norm_key(canon), 1.0)    # peso de la categoría
    else:
        canon = _norm_key(t_raw).title()              # legible si no colapsa
        w_cat = S_REL_N.get(_norm_key(canon), 1.0)

    w_final = w_raw if w_raw is not None else w_cat
    return canon, float(w_final if w_final is not None else 1.0)

def _canon_tipo_rel(t_raw: str) -> str:
    k = _norm_key(t_raw)
    if k in _MAP_TIPO_REL_N:
        return _MAP_TIPO_REL_N[k]      # p.ej. "Familiar Directo"
    return _norm_key(t_raw).title()

def _peso_tipo_prob(tipo_canonico: str) -> float:
    """
    Busca el peso en s_rel (prob). Si no existe, neutro 1.0.
    """
    return _S_REL_N.get(_norm_key(tipo_canonico), 1.0)

def _layout_shell_by_bfs(G: nx.Graph, root=None):
    """Concentra por niveles (útil si tienes un nodo semilla)."""
    from collections import deque, defaultdict
    levels = defaultdict(list)
    if root in G:
        q = deque([(root, 0)])
        seen = {root}
        while q:
            u, d = q.popleft()
            levels[d].append(u)
            for v in G.neighbors(u):
                if v not in seen:
                    seen.add(v)
                    q.append((v, d + 1))
        nlist = [levels[k] for k in sorted(levels)]
        return nx.shell_layout(G, nlist=nlist)
    # fallback si no hay root
    return nx.shell_layout(G)

def compute_layout_safe(G: nx.Graph, seed=None, root=None):
    n = G.number_of_nodes()
    if _has_scipy():
        return nx.spring_layout(G, seed=seed, dim=2, iterations=50)
    if n <= 1000:
        return nx.spring_layout(G, seed=seed, dim=2, iterations=50)
    return _layout_shell_by_bfs(G, root=root)

# ============================================================================
# FUNCIÓN HELPER PARA NORMALIZAR RUTS
# ============================================================================
def _normalize_rut(rut):
    if pd.isna(rut):
        return ""
    rut_str = str(rut).strip().upper().replace(".", "").replace(" ", "")
    if "-" in rut_str:
        return rut_str.split("-")[0].replace("-", "").replace(".", "")
    if any(c.isalpha() for c in rut_str):
        return rut_str
    return rut_str

def _rut_to_int(rut):
    try:
        normalized = _normalize_rut(rut)
        return int(normalized) if normalized else 0
    except (ValueError, TypeError):
        return 0

def filtrar_tipo_sin_ros_y_podar_descendientes(
    df: pd.DataFrame,
    *,
    tipo_objetivo: str = "ROE2 en Común",
    nivel_minimo: int = 2,
    usar_arbol: bool = True,
    col_tipo: str = "TIPO_RELACION",
    col_nivel_dest: str = "NIVEL_DESTINO",
    col_origen: str = "RUT_ORIGEN",
    col_dest: str = "RUT_RELACIONADO",
    col_es_nodo_final: str = "ES_NODO_FINAL",
    col_ros_rec: str = "REL_ROS_RECIENTE",
    col_ros_ant: str = "REL_ROS_ANTIGUO",
):
    out = df.copy()
    out[col_origen] = out[col_origen].apply(_normalize_rut)
    out[col_dest] = out[col_dest].apply(_normalize_rut)

    if col_ros_rec not in out.columns:
        out[col_ros_rec] = 0
    if col_ros_ant not in out.columns:
        out[col_ros_ant] = 0
    out[col_ros_rec] = pd.to_numeric(out[col_ros_rec], errors="coerce").fillna(0).astype(int)
    out[col_ros_ant] = pd.to_numeric(out[col_ros_ant], errors="coerce").fillna(0).astype(int)

    if usar_arbol and col_es_nodo_final in out.columns:
        ref_edges = out[out[col_es_nodo_final] == True].copy()
    else:
        ref_edges = out.copy()

    ros_sum = (out.groupby(col_dest)[[col_ros_rec, col_ros_ant]]
                 .sum()
                 .sum(axis=1)
                 .to_dict())

    mask_tipo  = ref_edges[col_tipo].astype(str).str.strip().str.lower() == tipo_objetivo.lower()
    niveles    = pd.to_numeric(ref_edges[col_nivel_dest], errors="coerce").fillna(-1).astype(int)
    mask_nivel = niveles >= int(nivel_minimo)

    candidatos = set(
        ref_edges.loc[mask_tipo & mask_nivel, col_dest]
        .astype(str)
        .tolist()
    )
    candidatos = {rut for rut in candidatos if int(ros_sum.get(rut, 0)) == 0}

    if not candidatos:
        return out.reset_index(drop=True), set()

    hijos = defaultdict(list)
    for _, r in ref_edges.iterrows():
        try:
            p = str(r[col_origen])
            h = str(r[col_dest])
            hijos[p].append(h)
        except Exception:
            continue

    a_eliminar = set()
    for root in candidatos:
        if root in a_eliminar:
            continue
        q = deque([str(root)])
        while q:
            u = q.popleft()
            if u in a_eliminar:
                continue
            a_eliminar.add(u)
            for v in hijos.get(u, []):
                if v not in a_eliminar:
                    q.append(v)

    out = out[~out[col_origen].isin(a_eliminar) & ~out[col_dest].isin(a_eliminar)]
    return out.reset_index(drop=True), a_eliminar

# ============================================================================
# HELPERS ACTUALIZADOS PARA STRINGS
# ============================================================================
def _marcas_por_nodo(edges_df: pd.DataFrame, seed_rut) -> pd.DataFrame:
    df = edges_df.copy().rename(columns={"RUT_RELACIONADO": "RUT"})
    df["RUT"] = df["RUT"].apply(_normalize_rut)

    bool_cols = ["REL_ROE_ANOMALO", "REL_FP","REL_TIENE_MP","REL_TIENE_REQMP"]
    cnt_cols  = ["REL_ROS_RECIENTE", "REL_ROS_ANTIGUO", "REL_CAUSAS_RECIENTES", "REL_CAUSAS_ANTIGUAS",
                 "REL_CANT_BRAICES","REL_MONTO_BRAICES","REL_VEHICULOS_5AGNOS"]

    for c in bool_cols:
        if c not in df.columns:
            df[c] = "NO"
        df[c] = df[c].fillna("NO").astype(str).str.upper().str.strip()

    for c in cnt_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float64")

    agg_dict = {**{c: (lambda s: "SI" if (s == "SI").any() else "NO") for c in bool_cols},
                **{c: "max" for c in cnt_cols}}

    out = (df.groupby("RUT")[bool_cols + cnt_cols].agg(agg_dict))
    out.index = out.index.astype(str)
    return out

def _nivel_min_por_rut(edges_df: pd.DataFrame, seed_rut) -> dict:
    df = edges_df.copy()
    df["RUT_RELACIONADO"] = df["RUT_RELACIONADO"].apply(_normalize_rut)
    d = (df.groupby("RUT_RELACIONADO")["NIVEL_DESTINO"].min().dropna())
    d.index = d.index.astype(str)
    d = d.astype(int).to_dict()
    d[_normalize_rut(seed_rut)] = 0
    return d

def _tipo_max_por_nodo(
    edges_df: pd.DataFrame,
    *,
    usar_solo_arbol: bool = True,
    usar_min_nivel: bool = True,
    S_REL_N: dict,
    MAP_TIPO_REL_N: dict,
) -> tuple[dict, dict, dict]:
    df = edges_df.copy()
    df["RUT_RELACIONADO"] = df["RUT_RELACIONADO"].apply(_normalize_rut)

    if usar_solo_arbol and "ES_NODO_FINAL" in df.columns:
        df = df[df["ES_NODO_FINAL"] == True].copy()

    if usar_min_nivel and "NIVEL_DESTINO" in df.columns:
        minlv = df.groupby("RUT_RELACIONADO")["NIVEL_DESTINO"].min().rename("__MINLV__")
        df = df.merge(minlv, left_on="RUT_RELACIONADO", right_index=True, how="left")
        df = df[df["NIVEL_DESTINO"] == df["__MINLV__"]].copy()

    df[["__TIPO_NORM__", "__PESO__"]] = df["TIPO_RELACION"].apply(
        lambda t: pd.Series(_canon_and_weight(t, S_REL_N, MAP_TIPO_REL_N))
    )
    df["__PESO__"] = df["__PESO__"].astype(float)

    pesos, tipos_canon, tipos_raw = {}, {}, {}
    for rut, g in df.groupby("RUT_RELACIONADO"):
        if g.empty:
            pesos[str(rut)] = 1.0
            tipos_canon[str(rut)] = tipos_raw[str(rut)] = "OTRO"
            continue
        maxw = float(g["__PESO__"].max())
        rows_max = g[g["__PESO__"] == maxw]
        pesos[str(rut)] = maxw if np.isfinite(maxw) else 1.0
        tipos_canon[str(rut)] = " | ".join(sorted(set(rows_max["__TIPO_NORM__"].astype(str)))) or "OTRO"
        tipos_raw[str(rut)] = " | ".join(sorted(set(rows_max["TIPO_RELACION"].astype(str)))) or "OTRO"

    return pesos, tipos_canon, tipos_raw





def _padre_primer_salto(edges_df: pd.DataFrame) -> dict:
    df = edges_df.copy()
    df["RUT_ORIGEN"] = df["RUT_ORIGEN"].apply(_normalize_rut)
    df["RUT_RELACIONADO"] = df["RUT_RELACIONADO"].apply(_normalize_rut)
    df = df[df["ES_NODO_FINAL"] == True].copy()
    minlv = (df.groupby("RUT_RELACIONADO")["NIVEL_DESTINO"].min().rename("__MINLVL__"))
    df = df.merge(minlv, left_on="RUT_RELACIONADO", right_index=True, how="left")
    df = df[df["NIVEL_DESTINO"] == df["__MINLVL__"]].copy()
    df = (df.sort_values(["RUT_RELACIONADO","NIVEL_DESTINO"]).drop_duplicates(subset=["RUT_RELACIONADO"], keep="first"))
    out = df.set_index("RUT_RELACIONADO")["RUT_ORIGEN"].astype(str).to_dict()
    return out

# =======================
# NUEVO SCORING (prob)
# =======================
def _sat_count(x, s50):
    """Saturación suave: 1 - exp(-ln2 * x / s50)."""
    x = max(float(x or 0), 0.0)
    return 1.0 - np.exp(-np.log(2.0) * x / max(float(s50), 1e-6))

def _noisy_or(ps):
    q = 1.0
    for p in ps:
        p = min(max(float(p), 0.0), 1.0)
        q *= (1.0 - p)
    return 1.0 - q

def _amp_exp(p, exponent):
    p = min(max(float(p), 0.0), 1.0)
    exponent = max(float(exponent), 0.0)
    return 1.0 - (1.0 - p)**exponent

# ===== 1) Probabilidad base desde columnas REL_* (usa CFG_PROB del JSON) =====
def _p_base_from_rel_cols(row, cfg: dict | None = None) -> float:
    """
    Probabilidad base intrínseca del nodo a partir de REL_* (0..1).
    Usa:
      - cfg["w_bin"]: pesos para flags SI/NO
      - cfg["w_cnt"]: {clave: {"w": peso, "s50": escala}} con saturación suave
    """
    cfg = cfg or CFG_PROB
    ps = []

    # Binarias (SI/NO)
    for k, w in (cfg.get("w_bin", {}) or {}).items():
        if str(row.get(k, "NO")).strip().upper() == "SI":
            ps.append(float(w))

    # Cuantitativas con saturación
    for k, par in (cfg.get("w_cnt", {}) or {}).items():
        x = float(row.get(k, 0) or 0)
        if x > 0:
            w = float(par.get("w", 0))
            s50 = float(par.get("s50", 1))
            ps.append(w * _sat_count(x, s50))  # tu función existente

    return _noisy_or(ps) if ps else 0.0


# ===== 2) Probabilidad por tipo de relación a partir de edges =====
def _p_tipo_from_edges(edges_df: pd.DataFrame, cfg: dict | None = None) -> dict:
    """
    Para cada RUT_RELACIONADO combina severidades de TIPO_RELACION (noisy-OR) en el nivel mínimo.
    - Canoniza parentescos vía MAP_TIPO_REL -> 'Familiar Directo'
    - Pondera por s_rel (CFG_PROB['s_rel']); si no existe, usa 1.0
    Retorna: dict[str rut] -> prob (0..1)
    """
    cfg = cfg or CFG_PROB

    df = edges_df.copy()
    df["RUT_RELACIONADO"] = df["RUT_RELACIONADO"].apply(_normalize_rut)

    # Solo nodos finales si existe la columna
    if "ES_NODO_FINAL" in df.columns:
        df = df[df["ES_NODO_FINAL"] == True].copy()

    # Nivel mínimo por RUT_RELACIONADO si existe la columna
    if "NIVEL_DESTINO" in df.columns:
        minlv = df.groupby("RUT_RELACIONADO")["NIVEL_DESTINO"].min().rename("__MINLV__")
        df = df.merge(minlv, left_on="RUT_RELACIONADO", right_index=True, how="left")
        df = df[df["NIVEL_DESTINO"] == df["__MINLV__"]].copy()

    # Canoniza y obtiene peso (← aquí usa __s__)
    df[["__TIPO_NORM__", "__s__"]] = df["TIPO_RELACION"].apply(
        lambda t: pd.Series(_canon_and_weight(t))
    )
    df["__s__"] = df["__s__"].astype(float)

    out: dict[str, float] = {}
    for rut, g in df.groupby("RUT_RELACIONADO"):
        s_list = g["__s__"].tolist()
        out[str(rut)] = _noisy_or(s_list) if s_list else 0.0
    return out




# ============================================================================
# FUNCIÓN PRINCIPAL (gráfica intacta)
# ============================================================================
def calcular_riesgo_red_auto(
    edges_df: pd.DataFrame,
    *,
    seed_rut,  # Puede ser int o string
    normalizar: bool = True,
    estrategia: str = "umbral",
    cobertura: float = 0.50,
    percentil: float = 95.0,
    z: float = 2.0,
    umbral: float = 50,
    minimo: int = 0,
    maximo: int | None = None,
    dibujar: bool = True,
    usar_arbol_para_dibujo: bool = True,
    figsize: tuple = (12, 8)
) -> tuple[pd.DataFrame, dict]:
    """
    Calcula aportes/norm y riesgo total de la red; selecciona nodos riesgosos; dibuja.
    ***IMPORTANTE***: La parte gráfica no se toca.
    """
    import matplotlib.pyplot as plt
    

    SC, CFG_PROB, _S_REL_N, _MAP_TIPO_REL_N = get_maps_normalizados()    


    porc_con_patrimonio = 0.5  #CAMBIAR A PARAMETRO
    seed_rut_norm = _normalize_rut(seed_rut)

    if edges_df is None or edges_df.empty:
        meta = {
            "total_abs": 0.0, "total_norm_sum": 0.0, "total_norm_avg": 0.0,
            "n_nodos": 0,
            "alpha": ALPHA, 
            
            "seleccion": {"estrategia": estrategia, "marcados": 0, "motivo": "sin nodos"}
        }
        if dibujar:
            plt.figure(figsize=figsize)
            plt.title(f"Red seed {seed_rut} (sin nodos)")
            plt.axis("off")
            plt.show()
        return pd.DataFrame(), meta, None, None

    # 1) Normalizar
    edges_work = edges_df.copy()
    edges_work["RUT_ORIGEN"] = edges_work["RUT_ORIGEN"].apply(_normalize_rut)
    edges_work["RUT_RELACIONADO"] = edges_work["RUT_RELACIONADO"].apply(_normalize_rut)

    # 2) Agregaciones por nodo
    marcas_df = _marcas_por_nodo(edges_work, seed_rut_norm)
    dist_map = _nivel_min_por_rut(edges_work, seed_rut_norm)
    peso_por_nodo, tipo_sel_por_nodo, tipo_raw_por_nodo  = _tipo_max_por_nodo(
        edges_work, usar_solo_arbol=True, usar_min_nivel=True,
        S_REL_N=_S_REL_N, MAP_TIPO_REL_N=_MAP_TIPO_REL_N
    )
    padre_por_nodo = _padre_primer_salto(edges_work)

    # 3) Scoring por nodo (SOLO CAMBIAMOS ESTE BLOQUE cuando MODO_SCORING == "prob")
    registros = []
    if MODO_SCORING == "prob":
        # toma multiplicador por tipo (ej. 1.5 para Familiar Directo)
        peso_por_nodo, tipo_sel_por_nodo, tipo_raw_por_nodo  = _tipo_max_por_nodo(
            edges_work, usar_solo_arbol=True, usar_min_nivel=True,
            S_REL_N=_S_REL_N, MAP_TIPO_REL_N=_MAP_TIPO_REL_N
        )
    else:
        peso_por_nodo, tipo_sel_por_nodo = {}, {}


    # --- PRE: ya calculaste peso_por_nodo, tipo_sel_por_nodo con _tipo_max_por_nodo(...)

    for rut, row in marcas_df.iterrows():
        if rut == seed_rut_norm:
            continue
    
        nivel = int(dist_map.get(rut, 99))
        rut_origen_padre = str(padre_por_nodo.get(rut, seed_rut_norm))
        tipo_txt = tipo_raw_por_nodo.get(rut, "OTRO")
    
        # 3.a) Riesgo base (0..1)
        p_base = _p_base_from_rel_cols(row)
    
        # 3.b) Ajuste por tipo (multiplicador)
        factor_tipo = float(peso_por_nodo.get(rut, 1.0))   # ← aquí entra 1.5 para Familiar Directo
    
        # === COMPATIBILIDAD ===
        # Muchos lugares del código usan "p_tipo"; ahora lo definimos = multiplicador
        p_tipo = factor_tipo
    
        # 3.c) Riesgo ajustado (si ese es tu cálculo)
        p_ajustada = p_base * p_tipo
    
        # ... el resto de tu pipeline usa p_tipo / p_ajustada como antes
    
                # 3.c) Atenuación por nivel
            # donde calculas la atenuación:
        atenuacion = (ALPHA ** max(nivel - 1, 0))   # en vez de ALPHA ** nivel


        # 3.d) Aporte final
        p = p_base * p_tipo * atenuacion
        p = max(0.0, min(1.0, p))

        aporte = p
        aporte_norm = aporte  # luego puede renormalizarse por suma (más abajo)
        score_0_100 = 100.0 * aporte
        risk_base_0_100 = 100.0 * p_base

        registros.append({
            "RUT_ORIGEN_PADRE": rut_origen_padre,
            "RUT": str(rut),
            "NIVEL": nivel,
            "APORTE": float(aporte),
            "APORTE_NORM": round(float(aporte_norm),4),
            "SCORE_0_100": round(float(score_0_100),4),
            "RISK_BASE_0_100": round(float(risk_base_0_100),4),
            "PESO_TIPO_USADO": float(p_tipo),          # severidad efectiva [0,1]
            "TIPO_RELACION_SEL": tipo_txt,
        })
      

    df_aportes = pd.DataFrame(registros)

    # 4) Normalización (LEGADO)
    

    # 5) Merge con marcas
    if not df_aportes.empty:
        df_aportes = (df_aportes.merge(marcas_df.reset_index(), on="RUT", how="left"))

    if df_aportes.empty:
        meta = {
            "total_abs": 0.0, "total_norm_sum": 0.0, "total_norm_avg": 0.0,
            "n_nodos": 0,
            "alpha": ALPHA, 
            
            "scoring": {"modo": MODO_SCORING},
            "seleccion": {"estrategia": estrategia, "marcados": 0, "motivo": "sin nodos"}
        }
        if dibujar:
            import io, base64
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No se encontraron datos o relaciones.", ha="center", va="center", fontsize=14, color="gray")
            ax.set_title(f"Red de riesgo para RUT {seed_rut}", fontsize=16)
            ax.axis("off")
            plt.tight_layout()
            png_buf = io.BytesIO()
            fig.savefig(png_buf, format="png", dpi=160, bbox_inches="tight")
            png_b64 = base64.b64encode(png_buf.getvalue()).decode("ascii")
            plt.close(fig)
        return df_aportes, meta, None, None

    # === (desde aquí hacia abajo, toda TU LÓGICA GRÁFICA Y DE SELECCIÓN SE MANTIENE IGUAL) ===

    # Patrimonio
    PESOS_PATRIMONIO = {
        "REL_CANT_BRAICES": 0.2,
        "REL_MONTO_BRAICES": 0.5,
        "REL_VEHICULOS_5AGNOS": 0.3
    }
    for c in ["REL_CANT_BRAICES","REL_MONTO_BRAICES","REL_VEHICULOS_5AGNOS"]:
        if c not in df_aportes.columns:
            df_aportes[c] = 0
    df_aportes["REL_MONTO_BRAICES"] = df_aportes["REL_MONTO_BRAICES"].astype(str).astype("float64")

    df_aportes["SCORE_PATRIMONIO"] = (
        PESOS_PATRIMONIO["REL_CANT_BRAICES"]   *  np.log1p(df_aportes["REL_CANT_BRAICES"].fillna(0).astype("float64")) +
        PESOS_PATRIMONIO["REL_MONTO_BRAICES"]  * df_aportes["REL_MONTO_BRAICES"].fillna(0).astype("float64") +
        PESOS_PATRIMONIO["REL_VEHICULOS_5AGNOS"] * df_aportes["REL_VEHICULOS_5AGNOS"].fillna(0).astype("float64")
    ).astype("float64")

    total_patrimonio = df_aportes["SCORE_PATRIMONIO"].sum()
    df_aportes["PCT_PATRIMONIO"] = (df_aportes["SCORE_PATRIMONIO"] / total_patrimonio if total_patrimonio > 0 else 0)
    df_aportes["FLAG_PATRIMONIO"] = df_aportes["PCT_PATRIMONIO"] > 0.5

    # Totales y normalización por suma (si pides normalizar)
    if normalizar:
        total_red = float(df_aportes["APORTE"].sum())
        if total_red > 0:
            df_aportes["APORTE_NORM"] = (df_aportes["APORTE"] / total_red).clip(0.0, 1.0)
        else:
            df_aportes["APORTE_NORM"] = 0.0

    total_abs = float(df_aportes["SCORE_0_100"].sum())
    total_norm_sum = float(df_aportes["APORTE_NORM"].sum()) if normalizar else None
    total_norm_avg = float(df_aportes["APORTE_NORM"].mean()) if normalizar else None

    col = "APORTE_NORM" if (normalizar and "APORTE_NORM" in df_aportes.columns) else "APORTE"
    df_aportes = df_aportes.sort_values(col, ascending=False).reset_index(drop=True)

    # Selección (sin cambios)
    df_pos = df_aportes[df_aportes[col] > 0].copy()
    seleccion = np.zeros(len(df_aportes), dtype=bool)
    motivo = "sin valores positivos"

    if not df_pos.empty:
        vals = df_pos[col].to_numpy()
        if estrategia == "cobertura":
            total = vals.sum()
            acumulado = np.cumsum(vals) / total
            k = int(np.searchsorted(acumulado, cobertura, side="left")) + 1
            k = max(k, min(minimo, len(df_pos)))
            if maximo is not None:
                k = min(k, maximo)
            seleccion[df_pos.index[:k]] = True
            motivo = f"cobertura≥{cobertura:.0%} (min={minimo}, max={maximo})"
        elif estrategia == "percentil":
            thr = float(np.percentile(vals, percentil))
            idx = df_pos.index[df_pos[col] >= thr]
            if len(idx) < minimo:
                idx = df_pos.index[:min(minimo, len(df_pos))]
            if maximo is not None and len(idx) > maximo:
                idx = idx[:maximo]
            seleccion[idx] = True
            motivo = f"percentil≥P{int(percentil)} (thr={thr:.4f})"
        elif estrategia == "zscore":
            mu, sd = float(vals.mean()), float(vals.std(ddof=0))
            thr = mu + z * sd
            idx = df_pos.index[df_pos[col] >= thr]
            if len(idx) < minimo:
                idx = df_pos.index[:min(minimo, len(df_pos))]
            if maximo is not None and len(idx) > maximo:
                idx = idx[:maximo]
            seleccion[idx] = True
            motivo = f"zscore≥μ+{z}σ (thr={thr:.4f})"
        elif estrategia == "umbral":
            u = float(umbral)
            thr_score = u if u > 1.0 else (u * 100.0)
            serie = df_pos["SCORE_0_100"]
            idx = df_pos.index[serie >= thr_score]
        
            if len(idx) < minimo:
                idx = df_pos.index[:min(minimo, len(df_pos))]
            if maximo is not None and len(idx) > maximo:
                idx = idx[:maximo]
        
            seleccion[idx] = True
            motivo = f"umbral SCORE_0_100≥{thr_score:.2f}"
        
        else:
            raise ValueError("estrategia debe ser 'cobertura' | 'percentil' | 'zscore' | 'umbral'")

    df_aportes["FLAG_RIESGO"] = seleccion

    meta = {
            "totales_red": {
            "total_riesgo": round(total_abs, 4),
            #"total_score_sum": round(total_score_sum, 4),
            #"total_score_avg": round(total_score_avg, 4),
            "n_nodos": int(len(df_aportes)),
            "n_riesgosos": int(df_aportes["FLAG_RIESGO"].sum()),
            "pct_riesgosos": round(float(df_aportes["FLAG_RIESGO"].mean() * 100), 4) if len(df_aportes) else 0.0,
            "max_score": round(float(df_aportes["SCORE_0_100"].max() if not df_aportes.empty else 0.0), 4),
            "min_score": round(float(df_aportes["SCORE_0_100"].min() if not df_aportes.empty else 0.0), 4),
            "avg_score": round(float(df_aportes["SCORE_0_100"].mean() if not df_aportes.empty else 0.0), 4),
        },

        "parametros": {
            "alpha": ALPHA, "estrategia": estrategia, "umbral": umbral,
            "percentil": percentil, "z": z, "cobertura": cobertura,
            "minimo": minimo, "maximo": maximo,
            "pesos_binarios": CFG_PROB["w_bin"],
            "pesos_contadores": CFG_PROB["w_cnt"],
            "pesos_relaciones": CFG_PROB["s_rel"],
        },
        "seleccion": {"estrategia": estrategia, "marcados": int(df_aportes["FLAG_RIESGO"].sum()), "motivo": motivo},
        "resumen_red": {"seed_rut": seed_rut, "scoring": round(total_abs , 2), "n_nodos": int(len(df_aportes)),
                        "n_riesgosos": int(df_aportes["FLAG_RIESGO"].sum()),
                        "max_score": float(df_aportes["SCORE_0_100"].max() if "SCORE_0_100" in df_aportes else 0.0)}
    }

# ---- 5) Dibujo ----
    if dibujar:
        # --- construir el grafo como ya lo haces ---
        df_draw = edges_df[edges_df["ES_NODO_FINAL"] == True].copy() if usar_arbol_para_dibujo else edges_df.copy()
        G = nx.Graph()
        for _, r in df_draw.iterrows():
            G.add_edge(str(r["RUT_ORIGEN"]), str(r["RUT_RELACIONADO"]))
        
            # ======= PRUNING VISUAL: ocultar hojas verdes si la red es grande =======
            try:
                TH = HIDE_GREEN_LEAVES_THRESHOLD
            except NameError:
                TH = 60
            
            if G.number_of_nodes() > TH:
                seed = str(str(seed_rut).split("-")[0])
            
                col_metric = "APORTE_NORM" if (normalizar and "APORTE_NORM" in df_aportes.columns) else "APORTE"
                metric_map = df_aportes.set_index("RUT")[col_metric].to_dict()
            
                riesgo_set = set(df_aportes[df_aportes["FLAG_RIESGO"]]["RUT"].astype(str))
                #  construir patrimonio_set de forma explícita
                patrimonio_set = set()
                if "FLAG_PATRIMONIO" in df_aportes.columns:
                    patrimonio_set = set(df_aportes[df_aportes["FLAG_PATRIMONIO"] == True]["RUT"].astype(str))
            
                def es_verde(n):
                    return float(metric_map.get(str(n), 0.0)) <= 0.0
            
                cambiamos = True
                while cambiamos:
                    cambiamos = False
                    hojas = [n for n in list(G.nodes) if G.degree(n) == 1]
                    a_remover = []
                    for n in hojas:
                        if n == seed:            # no tocar seed
                            continue
                        if n in riesgo_set:      # conservar riesgosos
                            continue
                        if n in patrimonio_set:  #  conservar patrimonio
                            continue
                        if not es_verde(n):      # conservar si no es “verde”
                            continue
                        a_remover.append(n)
                    if a_remover:
                        G.remove_nodes_from(a_remover)
                        cambiamos = True
            
                # sincroniza df_draw con los nodos que quedaron
                nodos_restantes = set(G.nodes)
                df_draw = df_draw[
                    df_draw["RUT_ORIGEN"].isin(nodos_restantes) &
                    df_draw["RUT_RELACIONADO"].isin(nodos_restantes)
                ].copy()
            # ======= FIN PRUNING VISUAL =======
            
         # --- quedarnos solo con el componente conectado al seed ---
        seed = str(str(seed_rut).split("-")[0])
        if seed in G:
            try:
                comps = list(nx.connected_components(G))
                comp_seed = next((c for c in comps if seed in c), None)
                if comp_seed is not None:
                    G = G.subgraph(comp_seed).copy()
                    nodos_restantes = set(G.nodes)
                    # sincroniza df_draw
                    df_draw = df_draw[
                        df_draw["RUT_ORIGEN"].isin(nodos_restantes) &
                        df_draw["RUT_RELACIONADO"].isin(nodos_restantes)
                    ].copy()
            except Exception:
                pass         
                # ===== 1) capas por nivel (seed en centro) =====
        seed = str(str(seed_rut).split("-")[0])
        
        # nivel mínimo por nodo (usamos el mismo criterio que el modelo)
        niveles = (edges_df.groupby("RUT_RELACIONADO")["NIVEL_DESTINO"].min()
                   .dropna().astype(int).to_dict())
        niveles[seed] = 0
        
        # construimos shells: [ [seed], [nivel 1], [nivel 2], ... ]
        max_lvl = max(niveles.values()) if niveles else 0
        shells = [[seed]] + [
            [n for n, lv in niveles.items() if lv == L]
            for L in range(1, max_lvl + 1)
        ]
        # quitamos capas vacías (por si acaso)
        shells = [s for s in shells if len(s) > 0]
        
        # ===== 2) layout inicial concéntrico por nivel =====
        pos_shell = nx.shell_layout(G, nlist=shells, center=(0.0, 0.0), rotate=0)
        
        # ===== 3) refinamiento con spring (respetando el seed fijo) =====
        
        k = 1.2 / np.sqrt(max(len(G.nodes()), 2))  # más nodos -> más repulsión
        pos = compute_layout_safe(G, seed=42, root=str(seed_rut) if seed_rut in G else None)

        # ===== 4) pequeño jitter para separar coincidencias restantes =====
        rng = np.random.default_rng(42)
        def _jitter(p, escala=0.015):
            return (p[0] + float(rng.uniform(-escala, escala)),
                    p[1] + float(rng.uniform(-escala, escala)))
        
        pos = {n: _jitter(p, escala=0.012 if niveles.get(n, 1) <= 1 else 0.018)
               for n, p in pos.items()}


        # color/tamaño
        # calcular valores normalizados de riesgo (0..1)
        vmax = df_aportes[col].max() if not normalizar else 1.0
        metric_map = df_aportes.set_index("RUT")[col].to_dict()
        color_val = {n: (metric_map.get(n, 0.0) / (vmax or 1.0)) for n in G.nodes}
        size_map = {n: 300 + 1200 * color_val.get(n, 0.0) for n in G.nodes}
        
        # construir lista de colores
        node_colors = []
        riesgo_vals = []   # guardamos valores de riesgo > 0 para escalar la barra
        for n in G.nodes:
            v = color_val.get(n, 0.0)
            if v == 0:
                node_colors.append("honeydew")   # sin riesgo
            else:
                rgba = plt.cm.Reds(v)
                node_colors.append(rgba)
                riesgo_vals.append(v)
        
        # dibujar
        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_edges(G, pos, alpha=0.35)
        nodes = nx.draw_networkx_nodes(
            G, pos,
            nodelist=list(G.nodes),
            node_size=[size_map[n] for n in G.nodes],
            node_color=node_colors,   # usamos lista preparada
            linewidths=0.8, edgecolors="#444"
        )

        # quitamos capas vacías (por si acaso)
       
        labels = {n: n for n in G.nodes()}
            
        # Ajustamos las posiciones de los labels
        label_pos = {n: (x, y + 0.08) for n, (x, y) in pos.items()}
        
        nx.draw_networkx_labels(
            G,
            label_pos,  #
            labels=labels,
            font_size=8,
            font_color="black"
        )


        # =======================
        # RESALTAR CAMINOS DESDE EL SEED
        # =======================
        seed = seed_rut
        nodos_riesgo = df_aportes[df_aportes["FLAG_RIESGO"]]["RUT"].astype(str).tolist()
        nodos_concentracion = df_aportes[df_aportes["FLAG_PATRIMONIO"]]["RUT"].astype(str).tolist()
        
        caminos_riesgo = [nx.shortest_path(G, source=seed, target=n) for n in nodos_riesgo if n in G]
        caminos_conc = [nx.shortest_path(G, source=seed, target=n) for n in nodos_concentracion if n in G]
        
        # construir lista de aristas a resaltar
        aristas_riesgo = []
        for cam in caminos_riesgo:
            aristas_riesgo += list(zip(cam[:-1], cam[1:]))
        
        aristas_conc = []
        for cam in caminos_conc:
            aristas_conc += list(zip(cam[:-1], cam[1:]))
        
        # dibujar aristas resaltadas
        nx.draw_networkx_edges(G, pos, edgelist=aristas_riesgo, width=2.5, edge_color="red", style="solid")
        nx.draw_networkx_edges(G, pos, edgelist=aristas_conc, width=2.5, edge_color="gold", style="dashed")
        
        # =======================
        # DESTACAR LABELS DE DESTINOS
        # =======================
        ax = plt.gca()
        
        # riesgosos: fondo blanco + borde rojo
        for n in nodos_riesgo:
            if n in label_pos:
                x, y = label_pos[n]
                ax.text(x, y, str(n), fontsize=8, color="black",
                        ha="center", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.2", lw=0.8))
        
        # concentración: fondo blanco + borde dorado
        for n in nodos_concentracion:
            if n in label_pos:
                x, y = label_pos[n]
                ax.text(x, y, str(n), fontsize=8, color="black",
                        ha="center", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="gold", boxstyle="round,pad=0.2", lw=0.8))
        
        # =======================
        # MOSTRAR TIPO DE RELACION EN LOS ENLACES
        # =======================
        # Creamos un diccionario { (origen, destino) : tipo_relacion }
        rel_map = {}
        for _, r in edges_df.iterrows():
            u, v = str(r["RUT_ORIGEN"]), str(r["RUT_RELACIONADO"])
            tipo = str(r.get("TIPO_RELACION", ""))
            rel_map[(u, v)] = tipo
            rel_map[(v, u)] = tipo  # grafo no dirigido
        
        # Etiquetamos solo aristas resaltadas
        for (u, v) in set(aristas_riesgo + aristas_conc):
            if (u, v) in pos and (v, u) in pos:
                continue
            if (u, v) in rel_map:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                xm, ym = (x1 + x2) / 2, (y1 + y2) / 2  # punto medio
                ax.text(xm, ym, rel_map[(u, v)], fontsize=7, color="black",
                        ha="center", va="center",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2))

        

        # doble borde rojo para los riesgosos
        marcados = df_aportes[df_aportes["FLAG_RIESGO"]]["RUT"].astype(str).tolist()
        if marcados:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=marcados,
                node_size=[size_map[n]*1.35 for n in marcados],
                node_color="none", edgecolors="red", linewidths=3.0
            )
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=marcados,
                node_size=[size_map[n]*1.18 for n in marcados],
                node_color="none", edgecolors="red", linewidths=1.6
            )

       # --- Patrimonio: nodos con algún patrimonio (usar solo los que quedaron en G)
        nodes_present = set(G.nodes)
        
        nodos_patrimonio = []
        nodos_concentracion = []
        if "SCORE_PATRIMONIO" in df_aportes.columns:
            nodos_patrimonio = df_aportes[df_aportes["SCORE_PATRIMONIO"] > 0]["RUT"].astype(str).tolist()
        if "FLAG_PATRIMONIO" in df_aportes.columns:
            nodos_concentracion = df_aportes[df_aportes["FLAG_PATRIMONIO"] == True]["RUT"].astype(str).tolist()
        
        # filtrar para que solo queden los que están en el grafo actual
        nodos_patrimonio_plot   = [n for n in nodos_patrimonio   if n in nodes_present]
        nodos_concentracion_plot = [n for n in nodos_concentracion if n in nodes_present]
        
        # 1) Patrimonio bajo -> borde amarillo claro
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=list(set(nodos_patrimonio_plot) - set(nodos_concentracion_plot)),
            node_size=[(size_map.get(n, 600) * 1.4) for n in (set(nodos_patrimonio_plot) - set(nodos_concentracion_plot))],
            node_color="none", edgecolors="khaki", linewidths=2.5
        )
        
        # 2) Concentración -> halo diamante dorado
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodos_concentracion_plot,
            node_size=[(size_map.get(n, 600) * 2.0) for n in nodos_concentracion_plot],
            node_color="none", edgecolors="gold", linewidths=3.0, alpha=0.5, node_shape="D"
        )


        # destacar seed con doble borde azul
        seed_rut_int = str(str(seed_rut).split("-")[0])
        if seed_rut_int in G.nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[seed_rut_int],
                node_size=size_map.get(seed_rut_int, 600)*1.6,
                node_color="none", edgecolors="blue", linewidths=2.5
            )
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[seed_rut_int],
                node_size=size_map.get(seed_rut_int, 600)*1.3,
                node_color="none", edgecolors="blue", linewidths=1.0, alpha=0.7
            )

        ttl_norm = " (normalizado 0–1)" if normalizar else ""
        #plt.title(f"Red de relacionados rut: {seed_rut} – nodos riesgosos: {int(df_aportes['FLAG_RIESGO'].sum())} ", fontsize=11)

        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='RUT sin riesgo',  markerfacecolor='honeydew', markeredgecolor='black',markeredgewidth=0.5, markersize=9),
    
            Line2D([0], [0], marker='o', color='w', label='RUT con patrimonio',
                   markerfacecolor='white', markeredgecolor='khaki', markersize=9, linewidth=2),
        
            Line2D([0], [0], marker='o', color='w', label='RUT riesgoso',
                   markerfacecolor='white', markeredgecolor='red', markersize=9, linewidth=2),
        
            Line2D([0], [0], marker='D', color='w', label='Concentración patrimonio',
                   markerfacecolor='white', markeredgecolor='gold', markersize=9, linewidth=2),
        
            Line2D([0], [0], marker='o', color='w', label='RUT consultado',
                   markerfacecolor='white', markeredgecolor='blue', markersize=9, linewidth=2),
        
            Line2D([0], [0], marker='o', color='w', label='Escala de riesgo (rojo)',
                   markerfacecolor='lightcoral', markeredgecolor='darkred', markersize=9)
        ]
            
        
        #plt.legend(handles=legend_elements, loc='upper left', fontsize=9, frameon=True)
        """plt.legend(handles=legend_elements,
                   loc='upper left',
                   bbox_to_anchor=(1, 1),  # fuera del eje a la derecha
                   fontsize=8, frameon=True)"""


        # --- Métricas a mostrar ---
        puntaje_red = meta.get("total_riesgo", 0)   # ejemplo, o ajusta según tu lógica
        n_nodos = meta.get("n_nodos", len(df_aportes))
        n_riesgosos = int(df_aportes["FLAG_RIESGO"].sum())
        valor_max_riesgo =    df_aportes["RISK_BASE_0_100"].max() if "RISK_BASE_0_100" in df_aportes else df_aportes["APORTE"].max()
        
        # --- Texto resumen ---
        totales = meta.get("totales_red", {})
        seleccion = meta.get("seleccion", {})
    
        text_meta = (
            f"Scoring RED: {totales.get('total_riesgo', 0):.2f} "
            f"- N° nodos: {totales.get('n_nodos', 0)}\n"
            f"N° riesgosos: {totales.get('n_riesgosos', 0)} "
            f"- Máx score: {totales.get('max_score', 0):.2f}"            
            
        )
                
        from matplotlib.offsetbox import AnchoredText
    
        anchored = AnchoredText(
            text_meta,
            loc='lower center',    # 🔥 parte baja y centrado
            frameon=True,
            prop=dict(size=10, color="black")
        )
        anchored.patch.set_boxstyle("round,pad=0.8")
        anchored.patch.set_facecolor("white")
        anchored.patch.set_alpha(0.8)
        anchored.patch.set_edgecolor("grey")
        
        #plt.gca().add_artist(anchored)
    

        import io, base64

        plt.axis("off"); plt.tight_layout(); 
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=160, bbox_inches="tight")
        png_b64 = base64.b64encode(png_buf.getvalue()).decode("ascii")
        plt.close(fig)   # <- muy importante: liberar memoria
        # --- construir estado de dibujo para filtros posteriores ---
        edges_list = []
        tipos_set = set()
        for _, r in df_draw.iterrows():
            u, v = str(r["RUT_ORIGEN"]), str(r["RUT_RELACIONADO"])
            tipo_raw = str(r.get("TIPO_RELACION", "")).strip().upper()
            tipo = MAP_TIPO_REL.get(tipo_raw, tipo_raw)

            edges_list.append({"u": u, "v": v, "tipo": tipo})
            if tipo:
                tipos_set.add(tipo)
        
        nodos_riesgo = set(df_aportes[df_aportes["FLAG_RIESGO"]]["RUT"].astype(str).tolist())
        nodos_conc   = set(df_aportes[df_aportes.get("FLAG_PATRIMONIO", False) == True]["RUT"].astype(str).tolist())
        nodos_patr   = set(df_aportes[df_aportes.get("SCORE_PATRIMONIO", 0) > 0]["RUT"].astype(str).tolist())
        
        metric_map = df_aportes.set_index("RUT")[col].to_dict()
        
        draw_state = {
            "seed": seed_rut_int,
        
            # SIEMPRE el conjunto completo (nunca lo sobreescribas al filtrar)
            "nodes_full": [str(n) for n in G.nodes()],
            "edges_full": [{"u": str(u), "v": str(v), "tipo": str(rel_map.get((u, v), ""))}
                           for (u, v) in G.edges()],
            "pos_full": {str(n): (float(x), float(y)) for n, (x, y) in pos.items()},
        
            # métrica/sets que ya tienes
            "metric_map": df_aportes.set_index("RUT")[col].to_dict(),
            "riesgo_set": set(df_aportes[df_aportes["FLAG_RIESGO"]]["RUT"].astype(str)),
            "patrimonio_set": set(df_aportes[df_aportes.get("SCORE_PATRIMONIO", 0) > 0]["RUT"].astype(str))
                              if "SCORE_PATRIMONIO" in df_aportes.columns else set(),
            "concentracion_set": set(df_aportes[df_aportes.get("FLAG_PATRIMONIO", False) == True]["RUT"].astype(str))
                                 if "FLAG_PATRIMONIO" in df_aportes.columns else set(),
            "normalizar": bool(normalizar),
        }
        tipos_set = set()
        for e in draw_state["edges_full"]:
            t = str(e.get("tipo", "")).strip().upper()
            if t:
                t_norm = MAP_TIPO_REL.get(t, t)   # << aquí usas el diccionario
                tipos_set.add(t_norm)
        draw_state["tipos_disponibles"] = sorted(tipos_set)
        

   

    # ---- 10) Orden de inspección ----
    order_cols = ["APORTE_NORM","APORTE"] if (col == "APORTE_NORM") else ["APORTE"]
    df_aportes = df_aportes.sort_values(order_cols, ascending=False).reset_index(drop=True)

    # Añadimos las listas de tipos al meta para construir el UI en el front
    meta["tipos_relacion"] = draw_state["tipos_disponibles"]
    return df_aportes, meta, png_b64, draw_state


# === TU render_png_from_state ORIGINAL VA AQUÍ SIN CAMBIOS ===
# def render_png_from_state(...):  # <— deja tu implementación intacta
#     ...
def render_png_from_state(draw_state: dict,
                          allowed_types: set[str] | None = None,
                          figsize=(11, 8)) -> str:
    """
    Redibuja el grafo desde un snapshot, manteniendo ids como string
    (soporta RUT/pasaporte con letras) y estilo consistente con el render original.
    NO dibuja colorbar.
    """
    import io, base64, re, unicodedata
    import matplotlib.pyplot as plt
    import networkx as nx

    # --- normalizador robusto de nombres de tipo (para filtrar) ---
    def _norm_tipo(s):
        s = str(s or "")
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = re.sub(r"\s+", " ", s.strip())
        return s.upper()

    # ===== 0) Coerción de snapshot: TODO como string =====
    seed = str(draw_state.get("seed", ""))

    pos_all = dict(draw_state.get("pos_full") or draw_state.get("pos") or {})
    pos_all = {str(k): (tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in pos_all.items()}

    nodes_all = [str(n) for n in (draw_state.get("nodes_full") or draw_state.get("nodes") or [])]

    edges_all_raw = (draw_state.get("edges_full") or draw_state.get("edges") or [])
    edges_all = [{"u": str(e["u"]), "v": str(e["v"]), "tipo": e.get("tipo", "")} for e in edges_all_raw]

    metric_map = {str(k): float(v) for k, v in dict(draw_state.get("metric_map") or {}).items()}

    riesgo_set     = set(str(x) for x in (draw_state.get("riesgo_set") or []))
    patrimonio_set = set(str(x) for x in (draw_state.get("patrimonio_set") or []))
    conc_set       = set(str(x) for x in (draw_state.get("concentracion_set") or []))
    normalizar     = bool(draw_state.get("normalizar", True))

    if allowed_types is None:
        allowed_norm = None
    else:
        # allowed_types puede venir como list/tuple/set/iterable (o incluso string)
        if isinstance(allowed_types, (list, tuple, set)):
            allowed_norm = {_norm_tipo(t) for t in allowed_types if t is not None and str(t).strip() != ""}
        else:
            s = str(allowed_types or "").strip()
            allowed_norm = set([_norm_tipo(s)]) if s else set()

    # ===== 1) Grafo filtrado por tipo =====
    G = nx.Graph()
    G.add_nodes_from(nodes_all)

    rel_map = {}  # (u,v) -> tipo (para rótulos en aristas destacadas)
    for e in edges_all:
        t_raw = str(e.get("tipo", "")).strip().upper()
        t_norm = MAP_TIPO_REL.get(t_raw, t_raw)
        
        if (allowed_norm is None) or (_norm_tipo(t_norm) in allowed_norm):
            u, v = e["u"], e["v"]
            G.add_edge(u, v, tipo=t_norm)
            rel_map[(u, v)] = t_norm
            rel_map[(v, u)] = t_norm


    # ===== 2) Recortar al componente que contiene seed =====
    if seed and seed in G:
        try:
            comp = next(c for c in nx.connected_components(G) if seed in c)
            G = G.subgraph(comp).copy()
        except Exception:
            G = G.subgraph([seed]).copy()
    elif seed:
        G = nx.Graph(); G.add_node(seed)
    nodes = list(G.nodes) or ([seed] if seed else [])

    # ===== 3) Posiciones: conservar y completar sólo faltantes =====
    pos = {n: pos_all.get(n) for n in nodes if n in pos_all and pos_all.get(n) is not None}
    missing = [n for n in nodes if n not in pos]
    if missing:
        # ancla nodos con pos conocida; calcula el resto alrededor
        fixed = list(pos.keys())
        # k proporcional a tamaño del grafo para mantener densidad visual estable
        k = 1.2 / max(2, len(G.nodes()))**0.5
        pos2 = nx.spring_layout(G, seed=42, k=k, iterations=150,
                                pos=(pos if fixed else None), fixed=fixed)
        for n in missing:
            pos[n] = tuple(pos2[n])

    # ===== 4) Colores/Tamaños =====
    present_vals = [metric_map.get(n, 0.0) for n in nodes]
    if normalizar:
        # se asume [0,1]; saturamos para evitar errores
        norm = {n: max(0.0, min(1.0, float(metric_map.get(n, 0.0)))) for n in nodes}
    else:
        vmax = max(present_vals) if present_vals else 1.0
        vmax = vmax or 1.0
        norm = {n: max(0.0, min(1.0, float(metric_map.get(n, 0.0)) / vmax)) for n in nodes}

    size_map   = {n: 300 + 1200 * norm[n] for n in nodes}
    node_color = ["honeydew" if norm[n] <= 0.0 else plt.cm.Reds(norm[n]) for n in nodes]

    # ===== 5) Caminos a resaltar (riesgo / concentración) =====
    present = set(nodes)
    nodos_riesgo = list(riesgo_set & present)
    nodos_conc   = list(conc_set & present)

    ariesgo, aconc = set(), set()
    for n in nodos_riesgo:
        try:
            p = nx.shortest_path(G, seed, n)
            ariesgo.update(zip(p[:-1], p[1:]))
        except Exception:
            pass
    for n in nodos_conc:
        try:
            p = nx.shortest_path(G, seed, n)
            aconc.update(zip(p[:-1], p[1:]))
        except Exception:
            pass

    # ===== 6) Dibujo (orden idéntico al original) =====
    fig, ax = plt.subplots(figsize=figsize)

    # aristas
    nx.draw_networkx_edges(G, pos, alpha=0.35, ax=ax)
    if ariesgo:
        nx.draw_networkx_edges(G, pos, edgelist=list(ariesgo),
                               width=2.5, edge_color="red", style="solid", ax=ax)
    if aconc:
        nx.draw_networkx_edges(G, pos, edgelist=list(aconc),
                               width=2.5, edge_color="gold", style="dashed", ax=ax)

    # nodos
    nx.draw_networkx_nodes(G, pos,
        nodelist=nodes,
        node_size=[size_map[n] for n in nodes],
        node_color=node_color, linewidths=0.8, edgecolors="#444", ax=ax
    )

    # labels: primero los normales…
    labels      = {n: str(n) for n in nodes}
    label_pos   = {n: (xy[0], xy[1] + 0.07) for n, xy in pos.items()}
    normales    = [n for n in nodes if n not in nodos_riesgo]
    if normales:
        nx.draw_networkx_labels(G, {n: label_pos[n] for n in normales},
                                labels={n: labels[n] for n in normales},
                                font_size=8, font_color="black",
                                verticalalignment="bottom", ax=ax)
    # …y luego los riesgosos con borde rojo
    for n in nodos_riesgo:
        if n in label_pos:
            x, y = label_pos[n]
            ax.text(x, y, labels[n], fontsize=8, color="black", ha="center", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="red",
                              boxstyle="round,pad=0.2", lw=0.9),
                    zorder=6)

    # aros rojos dobles (riesgosos)
    if nodos_riesgo:
        nx.draw_networkx_nodes(G, pos, nodelist=nodos_riesgo,
                               node_size=[size_map[n]*1.35 for n in nodos_riesgo],
                               node_color="none", edgecolors="red", linewidths=3.0, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=nodos_riesgo,
                               node_size=[size_map[n]*1.18 for n in nodos_riesgo],
                               node_color="none", edgecolors="red", linewidths=1.6, ax=ax)

    # halos: patrimonio sin concentración
    only_patr = list((patrimonio_set - conc_set) & present)
    if only_patr:
        nx.draw_networkx_nodes(G, pos, nodelist=only_patr,
                               node_size=[size_map[n]*1.4 for n in only_patr],
                               node_color="none", edgecolors="khaki", linewidths=2.5, ax=ax)

    # halos: concentración (rombo dorado)
    if nodos_conc:
        nx.draw_networkx_nodes(G, pos, nodelist=nodos_conc,
                               node_size=[size_map[n]*2.0 for n in nodos_conc],
                               node_color="none", edgecolors="gold",
                               linewidths=3.0, alpha=0.5, node_shape="D", ax=ax)

    # seed con doble borde azul
    if seed and seed in present:
        nx.draw_networkx_nodes(G, pos, nodelist=[seed],
                               node_size=size_map.get(seed, 600)*1.6,
                               node_color="none", edgecolors="blue", linewidths=2.5, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[seed],
                               node_size=size_map.get(seed, 600)*1.3,
                               node_color="none", edgecolors="blue",
                               linewidths=1.0, alpha=0.7, ax=ax)

    # rótulo de tipo sobre aristas destacadas
    for (u, v) in set(list(ariesgo) + list(aconc)):
        tipo = rel_map.get((u, v), "")
        if tipo:
            x1, y1 = pos[u]; x2, y2 = pos[v]
            xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            ax.text(xm, ym, str(tipo), fontsize=7, color="black",
                    ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2))

    plt.axis("off"); plt.tight_layout()
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    out = base64.b64encode(buf.getvalue()).decode("ascii")
    plt.close(fig)
    return out


def draw_state_to_json(draw_state: dict) -> dict:
    """Convierte el snapshot de dibujo en listas JSON de nodos y aristas."""
    if not isinstance(draw_state, dict):
        return {"nodes": [], "edges": []}

    metric_map = {str(k): float(v) for k, v in (draw_state.get("metric_map") or {}).items()}
    riesgo_set = set(str(x) for x in (draw_state.get("riesgo_set") or []))
    patrimonio_set = set(str(x) for x in (draw_state.get("patrimonio_set") or []))
    conc_set = set(str(x) for x in (draw_state.get("concentracion_set") or []))
    pos_map = {str(k): v for k, v in (draw_state.get("pos_full") or {}).items()}

    nodes = []
    for nid in draw_state.get("nodes_full", []):
        nid = str(nid)
        pos = pos_map.get(nid, (None, None))
        x = pos[0] if isinstance(pos, (list, tuple)) else None
        y = pos[1] if isinstance(pos, (list, tuple)) and len(pos) > 1 else None
        nodes.append({
            "id": nid,
            "riesgo": metric_map.get(nid, 0.0),
            "es_riesgo": nid in riesgo_set,
            "es_patrimonio": nid in patrimonio_set,
            "es_concentracion": nid in conc_set,
            "x": x,
            "y": y,
        })

    edges = []
    for e in draw_state.get("edges_full", []):
        edges.append({
            "u": str(e.get("u")),
            "v": str(e.get("v")),
            "tipo": e.get("tipo", "")
        })

    return {"nodes": nodes, "edges": edges}
