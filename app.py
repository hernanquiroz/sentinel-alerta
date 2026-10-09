# -*- coding: utf-8 -*-
"""
SENTINEL-ALERTA v0.3 · Detección temprana: deforestación, quemas e invasión
============================================================================
Piloto: municipio de Bello (Antioquia) y su anillo periurbano.

v0.3: sin odc-stac/rioxarray (incompatibles con Python 3.14 de Streamlit
Cloud). Carga las bandas de Copernicus Sentinel-2 L2A con rasterio + pystac
+ planetary_computer (firmas SAS), directamente en arrays NumPy.

Datos:
  - Copernicus / ESA Sentinel-2 L2A vía Microsoft Planetary Computer STAC
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
import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds

st.set_page_config(page_title="Sentinel-Alerta", page_icon="🛰️", layout="wide")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLECCION = "sentinel-2-l2a"  # Copernicus/ESA, L2A (reflectancia de superficie)

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
# Índices y detección (NumPy puro, sin xarray)
# ----------------------------------------------------------------------------
def ndvi_np(nir, red):
    nir = nir.astype("float32"); red = red.astype("float32")
    return (nir - red) / (nir + red + 1e-6)


def nbr_np(nir, swir):
    nir = nir.astype("float32"); swir = swir.astype("float32")
    return (nir - swir) / (nir + swir + 1e-6)


def leer_banda(href, bbox, escara=0.0001):
    """Lee la banda COG (URL firmada) recortada al bbox, en EPSG:4326."""
    with rasterio.open(href) as src:
        win = from_bounds(bbox[1], bbox[0], bbox[3], bbox[2], src.transform)
        banda = src.read(1, window=win).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            banda = np.where(banda == nodata, np.nan, banda)
        return banda * escara  # L2A viene escalado por 10000


def escoger_item(items, bbox):
    """Escoge la escena con mayor cobertura del bbox y menos nubes."""
    def puntaje(it):
        try:
            g = it.geometry
            cov = min(g.bounds[2], bbox[3]) - max(g.bounds[0], bbox[1])
            nub = it.properties.get("s2:mgrs_tile") and it.properties.get(
                "eo:cloud_cover", 100)
            return -float(nub)
        except Exception:
            return -100.0
    return sorted(items, key=puntaje)[0]


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
                              help="Para quemas: inicio de temporada seca")
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
        **Secuencia típica de una invasión en la periferia:**

        1. 🌿 Terreno con vegetación o rastrojo
        2. 🔥 **Quema controlada** ← *visible en el satélite*
        3. 🏚️ Lotes marcados y primeras estructuras
        4. 🏘️ Ocupación consolidada (irreversible en la práctica)

        El detector actúa en el **paso 2**: cicatriz de quema (dNBR) antes
        de que haya ocupación. Ventana de acción: **2–8 semanas**.

        *Confirmación:* quema + NDVI que no rebrota en los meses siguientes
        (el bosque se recupera; un lote ocupado, no).
        """
    )

# ----------------------------------------------------------------------------
# Procesamiento
# ----------------------------------------------------------------------------
if analizar:
    if fecha_pre is None or fecha_post is None or fecha_post <= fecha_pre:
        st.error("Selecciona dos fechas válidas (la reciente posterior a la base).")
        st.stop()

    st.info("Buscando imágenes Sentinel-2 (Copernicus)…")
    catalog = pystac_client.Client.open(STAC_URL,
                                        modifier=planetary_computer.sign_inplace)
    import datetime as _dt

    def buscar(fecha):
        q = catalog.search(collections=[COLECCION],
                           bbox=[roi[1], roi[0], roi[3], roi[2]],
                           datetime=(str(fecha), str(fecha)))
        return list(q.items())

    items_pre, items_post = [], []
    for off in range(7):  # tolerancia ±6 días por nubes
        f1 = fecha_pre + _dt.timedelta(days=off)
        f2 = fecha_post - _dt.timedelta(days=off)
        items_pre = items_pre or buscar(f1)
        items_post = items_post or buscar(f2)
        if items_pre and items_post:
            break
    if not items_pre or not items_post:
        st.error("Sin imágenes Sentinel-2 en esas fechas. Prueba otras fechas.")
        st.stop()

    item_pre, item_post = escoger_item(items_pre, roi), escoger_item(items_post, roi)

    st.info("Descargando bandas B04, B08, B12, SCL…")
    try:
        bandas = {}
        for clave, item in (("pre", item_pre), ("post", item_post)):
            item = planetary_computer.sign(item)
            bandas[clave] = {
                "B04": leer_banda(item.assets["B04"].href, roi),
                "B08": leer_banda(item.assets["B08"].href, roi),
                "B12": leer_banda(item.assets["B12"].href, roi),
                "SCL": leer_banda(item.assets["SCL"].href, roi, escara=1.0),
            }
    except Exception as e:
        st.error(f"Error descargando bandas: {e}")
        st.stop()

    # Verificar misma resolución entre fechas
    def forma_ok(b):
        return (b["B04"].shape == b["B08"].shape)

    if not (forma_ok(bandas["pre"]) and forma_ok(bandas["post"])):
        st.error("Las escenas tienen resoluciones distintas; recorta el bbox o cambia fechas.")
        st.stop()

    # Alinear formas por si difieren (recorte al mínimo común)
    h = min(bandas["pre"]["B04"].shape[0], bandas["post"]["B04"].shape[0])
    w = min(bandas["pre"]["B04"].shape[1], bandas["post"]["B04"].shape[1])
    for c in bandas:
        for b in bandas[c]:
            bandas[c][b] = bandas[c][b][:h, :w]

    def con_nubes(c):
        scl = bandas[c]["SCL"]
        nubes = np.isin(scl, [3, 8, 9, 10])  # sombra, nube media/alta, cirrus
        banda = bandas[c].copy()
        for b in ("B04", "B08", "B12"):
            banda[b] = np.where(nubes | np.isnan(banda[b]), np.nan, banda[b])
        return banda

    pre, post = con_nubes("pre"), con_nubes("post")

    nd_pre, nd_post = ndvi_np(pre["B08"], pre["B04"]), ndvi_np(post["B08"], post["B04"])
    nb_pre, nb_post = nbr_np(pre["B08"], pre["B12"]), nbr_np(post["B08"], post["B12"])

    if MODOS[modo] == "quemas":
        alerta = (nb_pre - nb_post) > umbral
        leyenda = "🔥 Cicatrices de quema (dNBR) — predictor temprano de invasión"
    elif MODOS[modo] == "deforestacion":
        alerta = (nd_pre - nd_post) > umbral
        leyenda = "🪓 Pérdida de cobertura vegetal (NDVI)"
    else:
        q = (nb_pre - nb_post) > 0.27
        sin_rebrote = (nd_post - nd_pre) < 0.05
        alerta = q & sin_rebrote
        leyenda = "🏠 Quema sin recuperación = posible ocupación en curso"

    alerta = np.nan_to_num(alerta.astype("float32"), nan=0.0).astype(bool)

    # Resolución: aproximar el tamaño de píxel desde el bbox y el ancho
    lado_m = ((roi[3] - roi[1]) * 111_320 / w) if w else 10.0
    ha_px = (max(lado_m, 5.0) ** 2) / 10_000
    n_alerta = int(alerta.sum())
    ha_alerta = n_alerta * ha_px

    st.success(f"✅ {ha_alerta:,.2f} ha detectadas · {n_alerta:,} píxeles")

    c1, c2, c3 = st.columns(3)
    c1.metric("Área detectada (aprox.)", f"{ha_alerta:,.2f} ha")
    c2.metric("Pixeles", f"{n_alerta:,}")
    if MODOS[modo] == "quemas" and ha_alerta > 5:
        c3.metric("Nivel de alerta", "🔴 ALTA", "quema extensa en periurbano")
    elif ha_alerta > 0:
        c3.metric("Nivel de alerta", "🟡 REVISAR", "verificar en campo")
    else:
        c3.metric("Nivel de alerta", "🟢 NORMAL")

    t1, t2, t3 = st.tabs(["🗺️ Detección", "🌿 NDVI antes/después", "🔥 NBR antes/después"])

    def a_img(arr, vmin=-1, vmax=1):
        x = (arr - vmin) / (vmax - vmin)
        x = np.clip(np.nan_to_num(x, nan=0.0), 0, 1)
        g = (x * 255).astype(np.uint8)
        return np.dstack([g, g, g])

    with t1:
        st.markdown(leyenda)
        rojo = np.zeros((*alerta.shape, 3), dtype=np.uint8)
        rojo[..., 0] = (alerta * 255).astype(np.uint8)
        st.image(rojo, use_container_width=True)
    with t2:
        st.image(a_img(nd_pre), caption="NDVI antes", use_container_width=True)
        st.image(a_img(nd_post), caption="NDVI después", use_container_width=True)
    with t3:
        st.image(a_img(nb_pre), caption="NBR antes", use_container_width=True)
        st.image(a_img(nb_post), caption="NBR después", use_container_width=True)

    # Exportar GeoJSON (sin geopandas: generación manual)
    if st.button("⬇️ Exportar detección (GeoJSON)"):
        ys, xs = np.where(alerta)
        if len(ys) == 0:
            st.info("Sin píxeles en alerta con este umbral.")
        else:
            paso = 10  # celdas ~10 píxeles para reducir tamaño
            lon0, lat1 = roi[1], roi[2]
            dlon = (roi[3] - roi[1]) / alerta.shape[1]
            dlat = (roi[2] - roi[0]) / alerta.shape[0]
            feats = []
            for y, x in set(zip(((ys // paso) * paso).tolist(),
                                ((xs // paso) * paso).tolist())):
                x0 = lon0 + x * dlon
                x1 = lon0 + min(x + paso, alerta.shape[1]) * dlon
                ya = lat1 - min(y + paso, alerta.shape[0]) * dlat
                yb = lat1 - y * dlat
                feats.append({
                    "type": "Feature",
                    "properties": {"modo": MODOS[modo]},
                    "geometry": {"type": "Polygon",
                                 "coordinates": [[[x0, ya], [x1, ya],
                                                   [x1, yb], [x0, yb], [x0, ya]]]},
                })
            gj = {"type": "FeatureCollection", "features": feats}
            import json
            st.download_button("Descargar GeoJSON", json.dumps(gj),
                               file_name=f"alerta_{MODOS[modo]}_bello.geojson",
                               mime="application/geo+json")

st.divider()
st.caption("Sentinel-Alerta v0.3 · Datos © Copernicus/ESA (Sentinel-2) vía Microsoft "
           "Planetary Computer · MIT · Siguiente: U-Net multiclase + alertas "
           "programadas para Planeación de Bello.")
