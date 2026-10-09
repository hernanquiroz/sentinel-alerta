# -*- coding: utf-8 -*-
"""
SENTINEL-ALERTA v0.2 · Detección temprana: deforestación, quemas e invasión
============================================================================
Piloto: municipio de Bello (Antioquia) y su anillo periurbano.

Los invasores de tierras suelen ejecutar "quemas controladas" para limpiar
el terreno ANTES de ocuparlo. La cicatriz de quema es visible en Sentinel-2
semanas antes de que aparezcan las primeras estructuras -> NBR/dNBR es el
mejor predictor temprano de invasión.

Fuente de datos (gratuita, sin credenciales):
  - Copernicus / ESA Sentinel-2 L2A (10-20 m, revisión ~5 días)
  - Vía Microsoft Planetary Computer STAC API

Instalación:
  pip install -r requirements.txt
Ejecución:
  streamlit run app.py
"""

import streamlit as st
import folium
from folium.plugins import MiniMap, MeasureControl
from streamlit_folium import st_folium
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import pystac_client
import odc.stac
import planetary_computer

st.set_page_config(page_title="Sentinel-Alerta", page_icon="🛰️", layout="wide")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLECCION = "sentinel-2-l2a"  # Copernicus/ESA, nivel 2A (reflectancia superficie)

# Planetary Computer distribuye Copernicus en blobs privados que exigen
# URLs firmadas (SAS tokens). Esto configura la firma automática.
odc.stac.configure_rio(
    client="planetary_computer",       # firma automática de cada asset
    aws={"aws_unsigned": True},        # datos en Azure, no AWS
)

# ----------------------------------------------------------------------------
# Zonas piloto. Prioridad 1: Bello (Antioquia) — anillo periurbano.
# Orden del bbox: (sur, oeste, norte, este)
# ----------------------------------------------------------------------------
ZONAS = {
    "Bello – anillo periurbano completo (piloto)": (6.27, -75.62, 6.45, -75.50),
    "Bello – zona alta occidental (veredas rurales)": (6.30, -75.60, 6.42, -75.52),
    "Bello – norte / vía San Pedro": (6.38, -75.58, 6.45, -75.51),
    "Bello – sur / límite Medellín (Granizal y alrededores)": (6.27, -75.58, 6.34, -75.50),
    "Sierra de La Macarena (Meta) – deforestación": (-3.05, -74.25, -2.75, -74.00),
    "Definir zona manualmente (bbox)": None,
}

MODOS = {
    "🔥 Quemas e invasión de tierras (NBR/dNBR)": "quemas",
    "🪓 Deforestación (NDVI)": "deforestacion",
    "🏠 Invasión consolidada (NDVI + estructuras)": "invasion",
}

# ----------------------------------------------------------------------------
# Índices espectrales
# ----------------------------------------------------------------------------
def ndvi(nir, red):
    nir, red = nir.astype("float32"), red.astype("float32")
    return (nir - red) / (nir + red + 1e-6)


def nbr(nir, swir):
    """Normalized Burn Ratio — estándar para cicatrices de quema.
    NIR=B08 (10m), SWIR=B12 (20m, re-muestreada)."""
    nir, swir = nir.astype("float32"), swir.astype("float32")
    return (nir - swir) / (nir + swir + 1e-6)


def mask_clouds(scl):
    """SCL: 3=sombra, 8/9=nube, 10=cirrus."""
    return ~scl.isin([3, 8, 9, 10])


def detectar_quemas(nbr_pre, nbr_post, umbral_nbr=0.27):
    """dNBR alto = área quemada entre fechas (escala USGS/EFFIS)."""
    return (nbr_pre - nbr_post) > umbral_nbr


def detectar_deforestacion(ndvi_pre, ndvi_post, umbral=0.18):
    return (ndvi_pre - ndvi_post) > umbral


def detectar_invasion(alerta_quema, ndvi_post, ndvi_pre):
    """Cicatriz de quema + NDVI que NO se recupera = ocupación en curso.
    (La vegetación natural rebrota; un terreno ocupado no.)"""
    sin_recuperacion = (ndvi_post - ndvi_pre) < 0.05
    return alerta_quema & sin_recuperacion


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🛰️ Sentinel-Alerta · Bello piloto")
st.caption("Copernicus Sentinel-2 · Quemas predictoras de invasión · Deforestación · Open source")

with st.sidebar:
    st.header("1️⃣ Zona")
    zona = st.selectbox("Región (piloto: Bello, Antioquia)", list(ZONAS.keys()))
    if ZONAS[zona] is None:
        txt = st.text_area("BBox (sur, oeste, norte, este)", "6.30, -75.60, 6.42, -75.52")
        try:
            roi = tuple(float(x) for x in txt.split(","))
        except ValueError:
            st.error("BBox inválido.")
            roi = ZONAS["Bello – anillo periurbano completo (piloto)"]
    else:
        roi = ZONAS[zona]

    st.header("2️⃣ Modo de análisis")
    modo = st.selectbox("Tipo de detección", list(MODOS.keys()))

    st.header("3️⃣ Fechas a comparar")
    fecha_pre = st.date_input("Imagen base (antes)", value=None,
                              help="Para quemas: idealmente inicio de temporada seca")
    fecha_post = st.date_input("Imagen reciente", value=None)

    st.header("4️⃣ Sensibilidad")
    if MODOS[modo] == "quemas":
        umbral = st.slider("dNBR mínimo (quema)", 0.10, 0.60, 0.27, 0.01,
                           help="0.27 = quemado moderado/severo (escala USGS/EFFIS)")
    else:
        umbral = st.slider("Caída mínima de NDVI", 0.05, 0.40, 0.18, 0.01)

    analizar = st.button("🔍 Analizar", type="primary", use_container_width=True)

col_map, col_info = st.columns([3, 2])
with col_map:
    m = folium.Map(location=[(roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2], zoom_start=12)
    folium.Rectangle(bounds=[[roi[0], roi[1]], [roi[2], roi[3]]],
                     color="orange", fill=True, fill_opacity=0.05,
                     tooltip=zona).add_to(m)
    MiniMap().add_to(m)
    MeasureControl().add_to(m)
    st_folium(m, height=420)

with col_info:
    st.subheader("Lógica anticipatoria de invasión")
    st.markdown(
        """
        **La secuencia típica de una invasión en la periferia:**

        1. 🌿 Terreno con vegetación o rastrojo
        2. 🔥 **Quema controlada** ← *aquí es visible en el satélite*
        3. 🏚️ Lotes marcados y primeras estructuras
        4. 🏘️ Ocupación consolidada (irreversible en la práctica)
