# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 10:15:41 2025

@author: acespedes
"""

# params_loader.py
from __future__ import annotations
import json, os, sys, logging
from typing import Tuple, Any, Dict

import os, json, time, threading, logging

# cache global y lock (thread-safe para tu app web)
_CACHE = {
    "data": None,          # dict con los parámetros
    "source": None,        # "file" | "db"
    "file_path": None,
    "file_mtime": None,    # último mtime visto
    "db_last_fetch": 0.0,  # timestamp del último fetch a BD
}
_LOCK = threading.Lock()

def _app_dir():
    import sys, os
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(__file__)


def _local_file_path():
    return os.path.join(_app_dir(), "params_scoring.json")

# --- Validación mínima del JSON ---
def _validate_scoring_params(cfg: Dict[str, Any]) -> Tuple[bool, str | None]:
    required = ["ALPHA", "MODO_SCORING",  "CFG_PROB"]
    for k in required:
        if k not in cfg:
            return False, f"Falta clave requerida: {k}"
    try:
        float(cfg["ALPHA"])  # que sea numérico
    except Exception:
        return False, "ALPHA debe ser numérico"
    return True, None

# --- Lectura desde archivo local ---
def _load_from_file() -> Tuple[Dict[str, Any] | None, str | None]:
    path = _local_file_path()
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ok, err = _validate_scoring_params(data)
        if not ok:
            return None, f"Archivo de parámetros inválido: {err}"
        logging.info(f"Parametros scoring cargados desde archivo: {path}")
        return data, None
    except Exception as e:
        return None, f"Error leyendo archivo de parámetros: {e}"

# --- Lectura desde BD ---
def _load_from_db(app_name: str, cn_str: str) -> Tuple[Dict[str, Any] | None, str | None]:
    try:
        import pyodbc  # asumes que ya lo usas
        with pyodbc.connect(cn_str, autocommit=True) as cn:
            row = cn.cursor().execute("""
                SELECT TOP(1) JsonPayload
                FROM dbo.AIE_APPREDES_PARAMS WITH (NOLOCK)
                WHERE NomApp = ? AND ParamSet = 'scoring' AND IsActive =1
                """, app_name).fetchone()
        if not row:
            return None, "No hay parámetros activos en la BD"
        data = json.loads(row[0])
        ok, err = _validate_scoring_params(data)
        if not ok:
            return None, f"Parámetros en BD inválidos: {err}"
        logging.info("Parametros scoring cargados desde BD")
        return data, None
    except Exception as e:
        return None, f"Error leyendo parámetros desde BD: {e}"

# --- API pública ---
def load_scoring_params(app_name: str,
                        cn_str: str,
                        *,
                        prefer_file: bool | None = None,
                        reload_mode: str = "auto",
                        db_ttl_seconds: float = 30.0) -> dict:
    """
    reload_mode:
      - "always": si hay archivo -> leer SIEMPRE del disco en cada llamada.
                  si no hay archivo -> forzar fetch a BD en cada llamada (sin TTL).
      - "auto"  : si hay archivo -> recargar solo si cambió el mtime (hot-reload).
                  si no hay archivo -> usar cache con TTL (db_ttl_seconds).
      - "never" : no recargar; usar cache si existe.
    """
    with _LOCK:
        # Decidir prefer_file por defecto
        if prefer_file is None:
            env = (os.environ.get("AIE_PARAMS_SOURCE") or "").lower()
            if env == "file": prefer_file = True
            elif env == "db": prefer_file = False
        if prefer_file is None:
            prefer_file = os.path.exists(_local_file_path())

        # --- RAMA ARCHIVO ---
        if prefer_file:
            path = _local_file_path()
            if not os.path.exists(path):
                logging.warning("params_loader: no hay archivo local, intento BD.")
            else:
                if reload_mode == "always":
                    data, err = _load_from_file()
                    if data is None: raise RuntimeError(err or "Error leyendo archivo de parámetros")
                    _CACHE.update({"data": data, "source": "file",
                                   "file_path": path, "file_mtime": os.path.getmtime(path)})
                    return data

                if reload_mode == "auto":
                    mtime = os.path.getmtime(path)
                    if (_CACHE.get("source") != "file") or (_CACHE.get("file_path") != path) or (_CACHE.get("file_mtime") != mtime):
                        data, err = _load_from_file()
                        if data is None: raise RuntimeError(err or "Error leyendo archivo de parámetros")
                        _CACHE.update({"data": data, "source": "file",
                                       "file_path": path, "file_mtime": mtime})
                    return _CACHE["data"]

                # "never"
                if _CACHE.get("data") is not None and _CACHE.get("source") == "file":
                    return _CACHE["data"]
                data, err = _load_from_file()
                if data is None: raise RuntimeError(err or "Error leyendo archivo de parámetros")
                _CACHE.update({"data": data, "source": "file",
                               "file_path": path, "file_mtime": os.path.getmtime(path)})
                return data

        # --- RAMA BD ---
        now = time.time()
        if reload_mode == "always":
            data, err = _load_from_db(app_name, cn_str)
            if data is None: raise RuntimeError(err or "Error leyendo parámetros desde BD")
            _CACHE.update({"data": data, "source": "db", "db_last_fetch": now})
            return data

        if reload_mode == "auto":
            if (_CACHE.get("data") is None) or (_CACHE.get("source") != "db") or (now - _CACHE.get("db_last_fetch", 0) >= db_ttl_seconds):
                data, err = _load_from_db(app_name, cn_str)
                if data is None: raise RuntimeError(err or "Error leyendo parámetros desde BD")
                _CACHE.update({"data": data, "source": "db", "db_last_fetch": now})
            return _CACHE["data"]

        # "never"
        if _CACHE.get("data") is not None:
            return _CACHE["data"]
        data, err = _load_from_db(app_name, cn_str)
        if data is None: raise RuntimeError(err or "Error leyendo parámetros desde BD")
        _CACHE.update({"data": data, "source": "db", "db_last_fetch": now})
        return data

def invalidate_params_cache():
    """Si quieres limpiar la cache manualmente (p.ej. botón 'Recargar')."""
    with _LOCK:
        _CACHE.clear()