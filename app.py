import base64
import io
import json
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import logging, webview

logging.basicConfig(level=logging.DEBUG)
from AIE_Dimensiones.business.relacionados import obtener_relacionados
from AIE_Dimensiones.business.relacionados_niveles import obtener_relacionados_niveles
from AIE_Dimensiones.business.relacionados_marcas import obtener_relacionados_marcas, obtener_marcas_semilla

import matplotlib
matplotlib.use("Agg")   # <- backend para generar imágenes sin GUI
import matplotlib.pyplot as plt
import os
import getpass
import socket


from modelo_red import calcular_riesgo_red_auto  # <- tu función tal cual
from modelo_red import filtrar_tipo_sin_ros_y_podar_descendientes 
# --- arriba del todo, variables globales ---


# DRIVER={SQL Server}
SERVER='10.0.2.21'
DATABASE ='AlmacenAIEADE'
USERNAME ='acespedes'
PASSWORD ='UaF_2023'
SQL_CONN_STR = f'DRIVER={{SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD};charset="utf8"'


import logging

DEBUG_MODE = False   # cámbialo a True cuando quieras logs

if DEBUG_MODE:
    logging.basicConfig(
        filename="debug.log",
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8"
    )
else:
    # No configuro archivo, solo desactivo todo
    logging.disable(logging.CRITICAL)

def normalize_rut(rut_str):
    if not isinstance(rut_str, str):
        return None
    
    # Intenta limpiar el RUT si tiene el formato típico
    rut_clean = rut_str.strip().replace('.', '').replace('-', '').upper()

    # Si el RUT normalizado no es puramente numérico (a excepción de la 'K'),
    # devolvemos el original para no perder información como la "E"
    if not rut_clean.replace('K', '').isdigit():
        return rut_str
    
    return rut_clean


def _extraer_n_nodos(meta: dict) -> int:
    """
    Devuelve n_nodos desde meta, sin importar si viene en la raíz
    o anidado en 'totales' / 'totales_red' / 'resumen', etc.
    """
    if not isinstance(meta, dict):
        return 0

    # Forma 1: campo directo
    if "n_nodos" in meta:
        try:
            return int(meta.get("n_nodos") or 0)
        except Exception:
            return 0

    # Forma 2: anidado (acepta varios posibles nombres)
    for key in ("totales", "totales_red", "resumen", "summary"):
        bloc = meta.get(key)
        if isinstance(bloc, dict) and "n_nodos" in bloc:
            try:
                return int(bloc.get("n_nodos") or 0)
            except Exception:
                return 0

    return 0


# logging de consultas: robusto y no rompe la app si falla
import pyodbc, json, time, socket, getpass, logging

APP_NAME = "AppRedes"        # identifica esta app en la tabla
APP_VERSION = "1.0.0"     # tu versión actual de la app


def _preflight_checks() -> tuple[bool, str]:
    # 1) versión
    ok_ver = enforce_version_or_exit()  # ya trazas fallos adentro
    if not ok_ver:
        return (False, "Error de versión: contacte a AIE o DTYS.")
    # 2) usuario
    usuario = _get_current_user_domain_slash()
    ok_user, err = _is_user_enabled(usuario)
    if not ok_user:
        # mensaje amigable, sin filtrar detalles internos
        return (False, f"El usuario '{usuario}' no está habilitado para usar la aplicación.")
    return (True, "OK")



def _get_current_user_domain_slash() -> str:
    try:
        import os, getpass
        user = os.environ.get("USERNAME") or getpass.getuser() or "Desconocido"
        domain = os.environ.get("USERDOMAIN")
        return f"{domain}\\{user}" if domain else user
    except Exception as e:
        return f"Error:{e}"

def _trace_auth_check(ok: bool, usuario: str, mensaje: str | None):
    try:
        params = {"evento": "auth_check", "usuario": usuario}
        log_consulta_sql(
            rut_seed=None,
            ok=ok,
            meta=None,
            params=params,
            mensaje=mensaje,
            duracion_ms=0
        )
    except Exception:
        pass

def _is_user_enabled(usuario: str) -> tuple[bool, str | None]:
    """
    Retorna (habilitado, error). Considera ventana de fechas.
    """
    try:
        import pyodbc
        with pyodbc.connect(SQL_CONN_STR, autocommit=True) as cn:
            row = cn.cursor().execute("""
                SELECT Habilitado
                      
                FROM dbo.AIE_APPREDES_USUARIOS WITH (NOLOCK)
                WHERE Usuario = ?
            """, usuario).fetchone()
        if not row:
            _trace_auth_check(False, usuario, "Usuario no encontrado en tabla de accesos")
            return (False, None)
        hab = int(row[0] or 0)
        ok = (hab == 1 )
        if not ok:
            _trace_auth_check(False, usuario, "Usuario deshabilitado o fuera de ventana")
        return (ok, None)
    except Exception as e:
        # si falla la consulta, considera denegar por seguridad
        _trace_auth_check(False, usuario, f"DB error: {e}")
        return (False, str(e))




def _get_db_version(app_name: str) -> tuple[str | None, str | None]:
    try:
        with pyodbc.connect(SQL_CONN_STR, autocommit=True) as cn:
            row = cn.cursor().execute(
                """
                SELECT TOP 1 Version
                FROM dbo.AIE_APPREDES_VERSION WITH (NOLOCK)
                WHERE NomApp = ?  and activa = 1               
                """, app_name
            ).fetchone()
        return (row[0].strip() if row and row[0] else None, None)
    except Exception as e:
        # si quieres mantener la app corriendo cuando no se puede leer, cambia el comportamiento abajo
        return (None, str(e))

def _trace_version_check(ok: bool, db_version: str | None, mensaje: str | None):
    # trazamos como “consulta especial” (sin rut) para no romper esquema
    params = {"evento": "version_check", "app_version": APP_VERSION, "db_version": db_version}
    try:
        log_consulta_sql(
            rut_seed=None,
            ok=ok,
            meta=None,
            params=params,
            mensaje=mensaje,
            duracion_ms=0
        )
    except Exception:
        pass  # nunca romper por la traza

def enforce_version_or_exit() -> bool:
    # Lee y normaliza la versión de BBDD
    db_ver, err = _get_db_version(APP_NAME) 

    logging.debug(f"db version columnas (raw): {db_ver!r} -> (norm): {db_ver!r}")

    # 1) Vacío/None/solo espacios => falla
    if db_ver == "":
        msg = "No se encontró versión activa de la app en la BBDD."
        _trace_version_check(False, db_ver, msg)  # <- Solo trazamos fallos
        return False

    # 2) Comparación estricta con la versión de la app (normalizada)
    app_ver = (str(APP_VERSION) or "").strip()
    if db_ver != app_ver:
        msg = f"Versión inválida. App={app_ver} vs BBDD={db_ver}"
        _trace_version_check(False, db_ver, msg)
        return False

    # ✅ OK (no trazamos éxito)
    return True


    

def _safe_json(d):
    try:
        return json.dumps(d, ensure_ascii=False)
    except Exception:
        return None

def _extraer_n_nodos(meta: dict) -> int:
    if isinstance(meta, dict):
        if "n_nodos" in meta:
            try: return int(meta["n_nodos"] or 0)
            except: return 0
        for k in ("totales", "totales_red", "resumen", "summary"):
            b = meta.get(k)
            if isinstance(b, dict) and "n_nodos" in b:
                try: return int(b["n_nodos"] or 0)
                except: return 0
    return 0

def log_consulta_sql(rut_seed, ok, meta=None, params=None, mensaje=None, duracion_ms=None):
    try:
        usuario = getpass.getuser()
        host = socket.gethostname()
        n_nodos = _extraer_n_nodos(meta) if meta else None
        params_json = _safe_json(params) if params else None

        with pyodbc.connect(SQL_CONN_STR, autocommit=True) as cn:
            cn.cursor().execute("""
                INSERT INTO dbo.AIE_APPREDES_TRAZABILIDAD
                (FechaUtc, Usuario, Host, RutConsultado, Parametros, Ok, Nodos, Mensaje, DuracionMs, VersionApp)
                VALUES (GETDATE(), ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            usuario, host, str(rut_seed), params_json,
            1 if ok else 0, n_nodos, mensaje, duracion_ms, APP_VERSION)

        return True
    except Exception as e:
        # No interrumpir la app si el log falla
        logging.warning("No se pudo insertar log de consulta: %s", e)
        return False




# TODO: reemplaza por tu carga real de datos
def cargar_edges_para_rut(rut):
    # Lee tu fuente local (CSV/Parquet/SQL offline). Debe devolver edges_df (como hoy).
    # Por ahora lanzo error para que lo implementes:
  
    df = obtener_relacionados_marcas(rut, max_niveles=2, return_meta=False)
    if df is None or df.empty:
        # Si no hay datos, retorna un DataFrame vacío para evitar errores posteriores.
        # Esto asegura que df.empty en la función 'calcular' funcione.
        return pd.DataFrame() 
    
    df, ruts_podados = filtrar_tipo_sin_ros_y_podar_descendientes(
    df,
    tipo_objetivo="ROE2 en Común",
    nivel_minimo=2,          # "cualquier nivel superior" => 2 o más
    usar_arbol=True          # usa ES_NODO_FINAL para definir padre/hijo
    )
    return df


class Api:
    def __init__(self):
        self._window = None  # se setea al crear la ventana        
        self._last_draw_state = None  # 👈 guardamos el último estado
        
    def _get_window(self):
        wnd = getattr(self, "_window", None)
        if wnd is None and getattr(webview, "windows", None):
            try:
                wnd = webview.windows[0]
            except Exception:
                wnd = None
        return wnd
        
    def get_usuario(self):
        import os, getpass
        try:
            user = os.environ.get("USERNAME") or getpass.getuser() or "Desconocido"
            domain = os.environ.get("USERDOMAIN")
            return f"{domain}\\{user}" if domain else user
        except Exception as e:
            return f"Error: {e}"
        
    def exit_app(self):
        try:
            import webview
            # Cierra todas las ventanas abiertas de pywebview
            for w in list(webview.windows):
                try:
                    w.destroy()
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback duro por si algo quedó vivo
        import os
        os._exit(0)
    def log_client_error(self, kind, message):
        import logging
        logging.debug(f"[JS {kind}] {message}")
        return True
    
    def log(self, level, msg):
        print(f"JS[{level}] {msg}")

    


    def check_version(self):
        try:
            db_ver, err = _get_db_version(APP_NAME)
            app_ver = str(APP_VERSION).strip()
            if err is not None or not db_ver or db_ver != app_ver:
                # ... (tu mismo manejo actual)
                return {"ok": False, "msg": "Versión inválida o error verificando versión. La aplicación se cerrará."}
    
            # ---- NUEVO: chequeo de usuario habilitado
            usuario = self.get_usuario()
            ok_user, err2 = _is_user_enabled(usuario)
            if not ok_user:
                return {"ok": False, "msg": f"El usuario '{usuario}' no está habilitado. La aplicación se cerrará."}
    
            return {"ok": True, "msg": "OK"}
    
        except Exception as e:
            # ... (tu manejo actual)
            return {"ok": False, "msg": "Error verificando requisitos. La aplicación se cerrará."}

    
    def redibujar_por_tipos(self, tipos_permitidos):
            """
            Redibuja el PNG usando el último estado y filtrando por tipos de relación.
            tipos_permitidos: lista de strings.
            """
            try:
                from modelo_red import render_png_from_state
                st = getattr(self, "_last_draw_state", None)
                if not st:
                    return {"ok": False, "msg": "No hay gráfico previo para filtrar."}
        
                tipos = set()
                if isinstance(tipos_permitidos, (list, tuple)):
                    tipos = {str(t) for t in tipos_permitidos}
                png_b64 = render_png_from_state(st, allowed_types=tipos, figsize=(11, 8))
                return {"ok": True, "png_b64": png_b64}
            except Exception as e:
                return {"ok": False, "msg": f"Error al redibujar: {e}"}


    # --- helper: diálogo "Guardar como" con tkinter ---
    def _save_dialog_tk(self, default_name, filetypes):
       try:
           import tkinter as tk
           from tkinter import filedialog
           root = tk.Tk()
           root.withdraw()
           root.attributes('-topmost', True)  # que aparezca al frente
           path = filedialog.asksaveasfilename(
               initialfile=default_name,
               filetypes=filetypes  # [('CSV (*.csv)','*.csv'), ...]
           )
           root.destroy()
           return path or None
       except Exception:
           return None
       
        
    def _save_dialog(self, default_name, file_types_text=None):
        """
        Abre SAVE dialog usando el formato que tu pywebview acepta en Windows:
        lista de strings tipo 'CSV (*.csv)'. Si falla, reintenta sin filtros.
        """
        wnd = self._get_window()
        if not (wnd and hasattr(wnd, "create_file_dialog")):
            return None

        # 1) Con filtros tipo texto
        try:
            if file_types_text:
                p = wnd.create_file_dialog(
                    webview.SAVE_DIALOG,
                    save_filename=default_name,
                    file_types=file_types_text  # ← OJO: lista de strings, NO tuplas
                )
                if p:
                    return p[0] if isinstance(p, (list, tuple)) else p
        except Exception:
            pass

        # 2) Sin filtros
        try:
            p = wnd.create_file_dialog(webview.SAVE_DIALOG, save_filename=default_name)
            if p:
                return p[0] if isinstance(p, (list, tuple)) else p
        except Exception:
            pass
        return None

    def guardar_png(self, png_b64, default_name: str = "grafico_red.png"):
        import base64, os, time
        try:
            # normalizar base64 (str/bytes/list/tuple; con o sin 'data:')
            if isinstance(png_b64, (list, tuple)):
                png_b64 = png_b64[0]
            if isinstance(png_b64, bytes):
                s = png_b64.decode("utf-8", errors="ignore")
            else:
                s = str(png_b64)
            if "," in s:
                s = s.split(",", 1)[1]
            data = base64.b64decode(s)

            # 1) pedir ruta con tkinter
            path = self._save_dialog_tk(default_name, [('PNG (*.png)','*.png'), ('Todos (*.*)','*.*')])

            # 2) si no hay diálogo (o error), guardamos en Descargas con timestamp
            if not path:
                downloads = os.path.join(os.path.expanduser("~"), "Downloads")
                os.makedirs(downloads, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(downloads, f"{ts}_{default_name}")

            with open(path, "wb") as f:
                f.write(data)
            return {"ok": True, "path": path}
        except Exception as e:
            return {"ok": False, "msg": f"{type(e).__name__}: {e}"}

    # --- Tabla CSV/XLSX ---
    def guardar_tabla(self, rows, fmt="csv", default_name="relacionados.csv"):
        import os, json, time
        import pandas as pd

        # normalizar rows
        if isinstance(rows, str):
            rows = json.loads(rows)
        if not isinstance(rows, (list, tuple)):
            return {"ok": False, "msg": "formato de datos inválido"}
        df = pd.DataFrame(rows)

        # tipos y nombre por defecto
        if fmt == "xlsx":
            filetypes = [('Excel (*.xlsx)','*.xlsx'), ('CSV (*.csv)','*.csv'), ('Todos (*.*)','*.*')]
            default_name = default_name if default_name.endswith(".xlsx") else "relacionados.xlsx"
        else:
            filetypes = [('CSV (*.csv)','*.csv'), ('Todos (*.*)','*.*')]
            default_name = default_name if default_name.endswith(".csv") else "relacionados.csv"

        # 1) diálogo con tkinter
        path = self._save_dialog_tk(default_name, filetypes)

        # 2) fallback a Descargas si no hay diálogo
        if not path:
            downloads = os.path.join(os.path.expanduser("~"), "Downloads")
            os.makedirs(downloads, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(downloads, f"{ts}_{default_name}")

        try:
            if fmt == "xlsx":
                try:
                    df.to_excel(path, index=False)   # necesita openpyxl/xlsxwriter si está disponible
                    return {"ok": True, "path": path}
                except Exception:
                    alt = os.path.splitext(path)[0] + ".csv"
                    df.to_csv(alt, index=False, sep=';', decimal=',', encoding='utf-8-sig')

                    return {"ok": True, "path": alt, "msg": "No se pudo crear XLSX. Guardado como CSV."}
            else:
                # CSV directo
                df.to_csv(path, index=False, sep=';', decimal=',', encoding='utf-8-sig')

                return {"ok": True, "path": path}
        except Exception as e:
            return {"ok": False, "msg": f"{type(e).__name__}: {e}"}


    
    def calcular(self, rut, normalizar=True, estrategia="umbral",
                 cobertura=0.80, percentil=95, z=2.0, umbral=50, minimo=3):
        # Ejecuta tu pipeline y devuelve PNG + tabla como JSON
        # 1) Datos
        t0 = time.perf_counter()
        params = {
            "normalizar": bool(normalizar),
            "estrategia": str(estrategia),
            "cobertura": float(cobertura),
            "percentil": float(percentil),
            "z": float(z),
            "umbral": float(umbral),
            "minimo": int(minimo)
        }

        try:
            rut = normalize_rut(rut)
            edges_df = cargar_edges_para_rut(rut)
      
            plt.close('all')
            df_aportes, meta, img_base64, draw_state = calcular_riesgo_red_auto(
                edges_df,
                seed_rut=rut,
                normalizar=normalizar,
                estrategia=estrategia,
                cobertura=cobertura,
                percentil=percentil,
                z=z,
                umbral=umbral,
                minimo=minimo,
                maximo=None,
                dibujar=True,                 # <- Matplotlib igual que hoy
                usar_arbol_para_dibujo=True,  # <- mantiene layout/hints actuales
                figsize=(11, 8)
            )
            self._last_draw_state = draw_state 
            
            # ↙️ Señal ‘sin red’ que ya entrega tu modelo (n_nodos=0).
            if df_aportes is None or len(df_aportes) == 0:
               log_consulta_sql(rut, False, meta, params, "No se encontraron relacionados", int((time.perf_counter()-t0)*1000))
               return {"ok": False, "msg": "No se encontraron relacionados para el RUT consultado."}

            if _extraer_n_nodos(meta) == 0:
                log_consulta_sql(rut, False, meta, params, "No se encontraron relacionados", int((time.perf_counter()-t0)*1000))
                return {"ok": False, "msg": "No se encontraron relacionados para el RUT consultado."}
            
            
            # ==== NUEVO: ficha de la semilla (RUT consultado) ====
            seed_info = None
            try:
                df_seed = obtener_marcas_semilla(rut)
                if df_seed is not None and not df_seed.empty:
                    r = df_seed.iloc[0].to_dict()
            
                    def _S(x):  # SI/NO string
                        s = str(r.get(x, "NO")).strip().upper()
                        return "SI" if s in ("SI", "1", "TRUE", "T") else "NO"
            
                    def _I(x):
                        try: return int(float(r.get(x, 0) or 0))
                        except: return 0
            
                    def _F(x):
                        try: return float(r.get(x, 0) or 0.0)
                        except: return 0.0
            
                    seed_info = {
                        "RUT": str(str(r.get("RUT", rut)).split("-")[0]),
                        "ROE_ANOMALO": _S("ROE_ANOMALO"),
                        "FP": _S("FP"),
                        "TIENE_MP": _S("TIENE_MP"),
                        "TIENE_REQ_MP": _S("TIENE_REQ_MP"),
                        "ROS_RECIENTE": _I("ROS_RECIENTE"),
                        "ROS_ANTIGUO": _I("ROS_ANTIGUO"),
                        "CAUSAS_RECIENTES": _I("CAUSAS_RECIENTES"),
                        "CAUSAS_ANTIGUAS": _I("CAUSAS_ANTIGUAS"),
                        "CANT_BRAICES": _I("CANT_BRAICES"),
                        "MONTO_BRAICES": _F("MONTO_BRAICES"),
                        "VEHICULOS_5AGNOS": _I("VEHICULOS_5AGNOS"),
                    }
            except Exception:
                seed_info = None  # no romper si falla
            
            # pega la ficha dentro de meta (si existe)
            if isinstance(meta, dict):
                meta.setdefault("resumen_red", {})
                meta["resumen_red"]["seed_rut"] = rut  # refuerza
                meta["seed_info"] = seed_info


            COLS_OUT = [
                "RUT_ORIGEN_PADRE","RUT","NIVEL","RISK_BASE_0_100",
                "SCORE_0_100",  "TIPO_RELACION_SEL","REL_ROE_ANOMALO","REL_FP","REL_TIENE_MP",
                "REL_TIENE_REQMP","REL_ROS_RECIENTE","REL_ROS_ANTIGUO", "REL_CAUSAS_RECIENTES","REL_CAUSAS_ANTIGUAS",
                "REL_CANT_BRAICES","REL_MONTO_BRAICES","REL_VEHICULOS_5AGNOS","FLAG_PATRIMONIO", "FLAG_RIESGO"
    
            ]
            
            # selección tolerante a columnas faltantes + orden
            COLS_SAFE = [c for c in COLS_OUT if c in df_aportes.columns]
            df_aportes = df_aportes.loc[:, COLS_SAFE].copy()
            
            # 2) Renombres deseados
            RENAME = {
                "RUT_ORIGEN_PADRE": "RUT_PADRE",
                "RUT": "RUT_REL",
                "NIVEL": "Nivel",            
                "RISK_BASE_0_100": "Riesgo Base",
                "SCORE_0_100": "Riesgo Ajustado",            
                "TIPO_RELACION_SEL": "Tipo_Relación",            
                "REL_ROE_ANOMALO": "ROE_ANOMALO",
                "REL_FP": "FP",            
                "REL_TIENE_MP": "ROS MP",
                "REL_TIENE_REQMP": "REQ. MP",
                "REL_ROS_RECIENTE": "N° ROS REC.",
                "REL_ROS_ANTIGUO": "N° ROS ANT.",
                "REL_CAUSAS_RECIENTES": "N° CAUSAS REC.",
                "REL_CAUSAS_ANTIGUAS": "N° CAUSAS ANT.",
                "REL_ROE_ANOMALO": "ROE_ANOMALO",
                "REL_CANT_BRAICES": "N° B. Raices",
                "REL_MONTO_BRAICES": "Monto B. Raices",
                "REL_VEHICULOS_5AGNOS": "N° Vehiculos",
                "FLAG_PATRIMONIO": "Conc. Patrimonio",
                "FLAG_RIESGO": "Alto Riesgo",
            }
            
            rename_safe = {k: v for k, v in RENAME.items() if k in df_aportes.columns}
            df_aportes.rename(columns=rename_safe, inplace=True)
    
            # 3) Serializar la figura a PNG base64      
            
            png_b64 = img_base64
    
            # 4) Serializar tabla (solo columnas clave)
            col_rank = "Riesgo Ajustado" if ("Riesgo Ajustado" in df_aportes.columns) else "Riesgo Ajustado"
           
            tabla = df_aportes.sort_values(col_rank, ascending=False).head(100)
            tabla_json = json.dumps(tabla.to_dict(orient="records"), ensure_ascii=False)
            dur = int((time.perf_counter() - t0) * 1000)
            log_consulta_sql(rut, True, meta, params, None, dur)
            return {
                "ok": True,
                "png_b64": png_b64,
                "tabla": tabla_json,
                "meta": meta
            }
        
        except Exception as e:
            # Falla controlada
         
            import traceback
            logging.error("Error en calcular:\n" + traceback.format_exc())
            return {"ok": False, "msg": f"Error interno: {str(e)}"}
        
def _show_error_and_exit(msg: str):
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        messagebox.showerror("Acceso denegado", msg)
        root.destroy()
    except Exception:
        pass
    import os
    os._exit(1)
            

def resource_path(relative_path):
    """Devuelve la ruta absoluta, compatible con PyInstaller."""
    if hasattr(sys, "_MEIPASS"):  # cuando corre como .exe
        base_path = sys._MEIPASS
    else:  # cuando corre como script normal
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


if __name__ == "__main__":
    ok, msg = _preflight_checks()
    if not ok:
        _show_error_and_exit(msg)   # 👈 ahora SÍ ve un cuadro de error
        
    api = Api()        
    window = webview.create_window(
        title="Red de riesgo - Área de Inteligencia Estrategica",
        url="web/index.html",
        js_api=api
    )
    
    def maximizar(w):
        try:
            w.maximize()
        except Exception as e:
            print("No se pudo maximizar:", e)
            
        
    api._window = window  # ← clave: pasar la ventana a la API
    webview.start(maximizar, window, gui='edgechromium')  # <- fuerza WebView2
    #webview.start(debug=True, http_server=True, gui='edgechromium')



import logging, os, sys

log_path = os.path.join(os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__), "app.log")

logging.basicConfig(
    filename=log_path,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)