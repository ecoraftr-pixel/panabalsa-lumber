# -*- coding: utf-8 -*-
"""
App de Control de Calidad y Registro para Mesa de Corte de Madera Balsa
Industria maderera - Yaguachi, Ecuador

Autor: Generado con Claude (Anthropic)
Descripción:
    Aplicación web (Streamlit + OpenCV) que permite:
      1. Capturar/subir una imagen del listón de balsa (foto de cámara o archivo).
      2. Detectar automáticamente el contorno del listón y medir su grosor bruto
         en píxeles, convirtiéndolo a pulgadas mediante un factor de calibración.
      3. Clasificar el grosor bruto según la Tabla de Equivalencias de Motosierra
         (medida comercial exacta).
      4. Seleccionar la longitud según el tope físico de la mesa (simulado con botones).
      5. Registrar cada corte (fecha/hora, grosor rústico, medida comercial,
         longitud, estado) en una tabla que se puede descargar en CSV o Excel.
      6. Mostrar métricas de producción del día y una estimación de desperdicio.

Ejecutar con:
    streamlit run app.py
"""

import io
import math
from datetime import datetime, date

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Control de Calidad - Mesa de Corte Balsa | Yaguachi",
    page_icon="🪵",
    layout="wide",
)

# ------------------------------------------------------------------------------------
# CONFIGURACIÓN DE NEGOCIO (AJUSTABLE)
# ------------------------------------------------------------------------------------
# Longitudes reales de los topes físicos de la mesa de corte (en pulgadas).
# Formato: (etiqueta visible, longitud en pulgadas)
TOPES_LONGITUD = [
    ("Tope 1", 25.75),
    ("Tope 2", 24.75),
    ("Tope 3", 22.75),
    ("Tope 4", 20.75),
    ("Tope 5", 18.75),
    ("Tope 6", 16.75),
    ("Tope 7", 14.75),
    ("Tope 8", 12.75),
]

# Tabla de equivalencias de motosierra: (min_in, max_in, etiqueta comercial, valor nominal in)
# El orden importa: se evalúa secuencialmente y se retorna la primera coincidencia.
TABLA_EQUIVALENCIAS = [
    (1.500, 2.000, '1"', 1.0),
    (2.125, 2.500, '1 1/2"', 1.5),
    (2.625, 3.000, '2"', 2.0),
    (3.001, 3.500, '2 1/2"', 2.5),   # 3.001 para no solaparse con la regla anterior (3.000)
    (3.625, 4.000, '3"', 3.0),
    (4.125, 4.500, '3 1/2"', 3.5),
    (4.625, 5.000, '4"', 4.0),
]

CSV_COLUMNS = [
    "Fecha_Hora",
    "Grosor_Rustico_in",
    "Medida_Comercial",
    "Medida_Comercial_Nominal_in",
    "Longitud_Tope",
    "Longitud_in",
    "Desperdicio_in",
    "Estado",
]

# ------------------------------------------------------------------------------------
# ESTADO DE SESIÓN
# ------------------------------------------------------------------------------------
if "registros" not in st.session_state:
    st.session_state.registros = pd.DataFrame(columns=CSV_COLUMNS)

if "px_per_inch" not in st.session_state:
    st.session_state.px_per_inch = 40.0  # valor por defecto, se ajusta con calibración

if "ultima_medicion" not in st.session_state:
    st.session_state.ultima_medicion = None


# ------------------------------------------------------------------------------------
# FUNCIONES DE VISIÓN ARTIFICIAL
# ------------------------------------------------------------------------------------
def to_cv2_image(uploaded_file) -> np.ndarray:
    """Convierte un archivo subido por Streamlit (UploadedFile) a imagen BGR de OpenCV."""
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def detectar_listón(
    img_bgr: np.ndarray,
    umbral_min: int = 60,
    umbral_max: int = 255,
    invertir: bool = False,
    modo: str = "Otsu (automático)",
):
    """
    Detecta el contorno principal (el listón de madera) en la imagen y devuelve:
      - imagen anotada (BGR)
      - ancho_px, alto_px del rectángulo mínimo que envuelve el contorno
      - máscara binaria usada (para depuración visual)
    Estrategia:
      1. Escala de grises + desenfoque gaussiano.
      2. Umbralización (Otsu automático o manual con sliders).
      3. Operaciones morfológicas para limpiar ruido.
      4. Se toma el contorno de mayor área como el listón.
      5. Se calcula el rectángulo de área mínima (cv2.minAreaRect) -> (w, h) en píxeles.
    """
    if img_bgr is None:
        return None, 0, 0, None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    if modo == "Otsu (automático)":
        thresh_type = cv2.THRESH_BINARY_INV if invertir else cv2.THRESH_BINARY
        _, mask = cv2.threshold(blur, 0, 255, thresh_type + cv2.THRESH_OTSU)
    else:
        thresh_type = cv2.THRESH_BINARY_INV if invertir else cv2.THRESH_BINARY
        _, mask = cv2.threshold(blur, umbral_min, umbral_max, thresh_type)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    anotada = img_bgr.copy()

    if not contornos:
        return anotada, 0, 0, mask

    contorno_max = max(contornos, key=cv2.contourArea)

    # Filtrar contornos demasiado pequeños (ruido)
    area_img = img_bgr.shape[0] * img_bgr.shape[1]
    if cv2.contourArea(contorno_max) < 0.01 * area_img:
        return anotada, 0, 0, mask

    rect = cv2.minAreaRect(contorno_max)  # ((cx, cy), (w, h), angulo)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    (w_px, h_px) = rect[1]

    cv2.drawContours(anotada, [box], 0, (0, 255, 0), 3)
    cx, cy = int(rect[0][0]), int(rect[0][1])
    cv2.circle(anotada, (cx, cy), 6, (0, 0, 255), -1)
    cv2.putText(
        anotada,
        f"{w_px:.0f}px x {h_px:.0f}px",
        (cx - 80, cy - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
    )

    return anotada, w_px, h_px, mask


def px_a_pulgadas(valor_px: float, px_per_inch: float) -> float:
    if px_per_inch <= 0:
        return 0.0
    return valor_px / px_per_inch


def clasificar_grosor(grosor_in: float):
    """
    Clasifica el grosor bruto (en pulgadas decimales) según la Tabla de Equivalencias
    de Motosierra. Devuelve (etiqueta_comercial, valor_nominal_in, estado).
    """
    if grosor_in <= 0:
        return "N/A", 0.0, "Sin medición"

    for min_in, max_in, etiqueta, nominal in TABLA_EQUIVALENCIAS:
        if min_in <= grosor_in <= max_in:
            return etiqueta, nominal, "Aprobado"

    if grosor_in < TABLA_EQUIVALENCIAS[0][0]:
        return "Fuera de rango (delgado)", 0.0, "Rechazado"
    if grosor_in > TABLA_EQUIVALENCIAS[-1][1]:
        return "Fuera de rango (grueso)", 0.0, "Rechazado"

    return "Zona muerta entre rangos", 0.0, "Revisión manual"


def formatear_fraccion(valor_in: float) -> str:
    """Convierte un decimal de pulgadas a una fracción legible tipo 2 5/8"."""
    entero = int(valor_in)
    resto = valor_in - entero
    denom = 8
    numer = round(resto * denom)
    if numer == 0:
        return f'{entero}"' if entero > 0 else '0"'
    if numer == denom:
        return f'{entero + 1}"'
    g = math.gcd(numer, denom)
    numer, denom = numer // g, denom // g
    if entero > 0:
        return f'{entero} {numer}/{denom}"'
    return f'{numer}/{denom}"'


# ------------------------------------------------------------------------------------
# BARRA LATERAL
# ------------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    st.subheader("Calibración de cámara")
    st.caption(
        "Para convertir píxeles a pulgadas reales, indica cuántos píxeles equivalen "
        "a 1 pulgada en tu encuadre. Puedes calibrarlo colocando un objeto de ancho "
        "conocido (ej. una regla) frente a la cámara y ajustando el valor hasta que "
        "la medición coincida."
    )
    st.session_state.px_per_inch = st.number_input(
        "Píxeles por pulgada (px/in)",
        min_value=1.0,
        max_value=2000.0,
        value=float(st.session_state.px_per_inch),
        step=1.0,
        help="Factor de conversión: ancho_en_px / ancho_real_en_pulgadas",
    )

    st.divider()
    st.subheader("Umbral de detección (OpenCV)")
    modo_umbral = st.radio(
        "Modo de umbralización",
        ["Otsu (automático)", "Manual"],
        index=0,
    )
    invertir = st.checkbox(
        "Invertir máscara (usar si el listón sale oscuro sobre fondo claro)",
        value=False,
    )
    umbral_min, umbral_max = 60, 255
    if modo_umbral == "Manual":
        umbral_min = st.slider("Umbral mínimo", 0, 255, 60)
        umbral_max = st.slider("Umbral máximo", 0, 255, 255)

    st.divider()
    st.subheader("🗑️ Datos")
    if st.button("Borrar todos los registros", type="secondary"):
        st.session_state.registros = pd.DataFrame(columns=CSV_COLUMNS)
        st.success("Registros eliminados.")


# ------------------------------------------------------------------------------------
# ENCABEZADO
# ------------------------------------------------------------------------------------
st.title("🪵 Control de Calidad - Mesa de Corte de Balsa")
st.caption("Yaguachi, Ecuador · Sierra péndulo + topes de longitud · Registro automatizado")

tab_captura, tab_equivalencias, tab_registro, tab_dashboard = st.tabs(
    ["📷 Captura y Detección", "📐 Tabla de Equivalencias", "📝 Registro de Corte", "📊 Métricas del Día"]
)

# ------------------------------------------------------------------------------------
# TAB 1: CAPTURA Y DETECCIÓN
# ------------------------------------------------------------------------------------
with tab_captura:
    st.subheader("1. Obtener imagen del listón")

    modo_entrada = st.radio(
        "Fuente de imagen",
        ["📁 Subir imagen", "📷 Cámara web (foto en vivo)"],
        horizontal=True,
    )

    imagen_bgr = None

    if modo_entrada == "📁 Subir imagen":
        archivo = st.file_uploader(
            "Sube una foto del listón (extremo/corte transversal visible)",
            type=["jpg", "jpeg", "png"],
        )
        if archivo is not None:
            imagen_bgr = to_cv2_image(archivo)
    else:
        st.caption(
            "Nota: Streamlit Cloud captura una foto (snapshot) desde tu cámara web; "
            "para video continuo se requeriría el paquete adicional `streamlit-webrtc`."
        )
        foto = st.camera_input("Apunta la cámara al extremo del listón y toma la foto")
        if foto is not None:
            imagen_bgr = to_cv2_image(foto)

    if imagen_bgr is not None:
        col1, col2 = st.columns(2)

        anotada, w_px, h_px, mask = detectar_listón(
            imagen_bgr,
            umbral_min=umbral_min,
            umbral_max=umbral_max,
            invertir=invertir,
            modo=modo_umbral,
        )

        with col1:
            st.image(cv2.cvtColor(anotada, cv2.COLOR_BGR2RGB), caption="Detección de contorno", use_container_width=True)
        with col2:
            st.image(mask, caption="Máscara binaria (depuración)", use_container_width=True)

        if w_px == 0 and h_px == 0:
            st.warning(
                "⚠️ No se detectó un contorno válido. Ajusta el umbral en la barra lateral "
                "o verifica la iluminación/fondo de la imagen."
            )
        else:
            grosor_px = min(w_px, h_px)  # el lado corto del rectángulo = grosor transversal
            grosor_in_auto = px_a_pulgadas(grosor_px, st.session_state.px_per_inch)

            st.subheader("2. Grosor bruto medido")
            st.caption(
                "El sistema mide automáticamente el lado corto del contorno detectado. "
                "Si el listón está achaflanado, puedes ingresar manualmente el grosor "
                "del LADO MÁS GRUESO para forzar la clasificación por ese valor."
            )

            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Grosor detectado (px)", f"{grosor_px:.1f}")
            with colB:
                st.metric("Grosor detectado (in)", f"{grosor_in_auto:.3f}")
            with colC:
                usar_manual = st.checkbox("Sobrescribir manualmente", value=False)

            if usar_manual:
                grosor_final_in = st.number_input(
                    "Grosor bruto del lado más grueso (pulgadas, decimal)",
                    min_value=0.0,
                    max_value=10.0,
                    value=round(grosor_in_auto, 3),
                    step=0.01,
                )
            else:
                grosor_final_in = grosor_in_auto

            st.session_state.ultima_medicion = grosor_final_in

            etiqueta, nominal, estado = clasificar_grosor(grosor_final_in)
            color_estado = {
                "Aprobado": "🟢",
                "Rechazado": "🔴",
                "Revisión manual": "🟡",
                "Sin medición": "⚪",
            }.get(estado, "⚪")

            st.subheader("3. Resultado de clasificación")
            colD, colE, colF = st.columns(3)
            colD.metric("Grosor rústico (in)", f"{grosor_final_in:.3f}\"  ({formatear_fraccion(grosor_final_in)})")
            colE.metric("Medida comercial", etiqueta)
            colF.metric("Estado", f"{color_estado} {estado}")

            st.info(
                "Pasa a la pestaña **📝 Registro de Corte** para seleccionar la longitud "
                "según el tope físico y guardar este resultado."
            )


# ------------------------------------------------------------------------------------
# TAB 2: TABLA DE EQUIVALENCIAS
# ------------------------------------------------------------------------------------
with tab_equivalencias:
    st.subheader("📐 Tabla de Equivalencias de Motosierra")
    st.caption("Clasificación del grosor bruto (medido en el lado más grueso) a medida comercial exacta.")

    df_tabla = pd.DataFrame(
        [
            {
                "Rango Grosor Bruto": f"{formatear_fraccion(mi)} a {formatear_fraccion(ma)}",
                "Medida Comercial": et,
            }
            for mi, ma, et, _ in TABLA_EQUIVALENCIAS
        ]
    )
    st.table(df_tabla)

    st.divider()
    st.subheader("🔍 Probar clasificación manual")
    grosor_test = st.number_input(
        "Ingresa un grosor bruto (pulgadas decimales) para ver a qué medida comercial corresponde",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.0625,
        format="%.4f",
    )
    et_test, nom_test, estado_test = clasificar_grosor(grosor_test)
    st.write(
        f"**{grosor_test:.3f}\" ({formatear_fraccion(grosor_test)})** → "
        f"Medida comercial: **{et_test}** · Estado: **{estado_test}**"
    )


# ------------------------------------------------------------------------------------
# TAB 3: REGISTRO DE CORTE
# ------------------------------------------------------------------------------------
with tab_registro:
    st.subheader("📝 Registrar Corte")

    if st.session_state.ultima_medicion is None:
        st.warning(
            "Aún no hay ninguna medición activa. Ve a la pestaña "
            "**📷 Captura y Detección** y procesa una imagen primero."
        )
    else:
        grosor_actual = st.session_state.ultima_medicion
        etiqueta, nominal, estado = clasificar_grosor(grosor_actual)

        st.write(
            f"Grosor bruto activo: **{grosor_actual:.3f}\" ({formatear_fraccion(grosor_actual)})** → "
            f"Medida comercial: **{etiqueta}** · Estado: **{estado}**"
        )

        st.subheader("Selecciona la longitud según el tope físico de la mesa")
        etiquetas_topes = [
            f"{nombre} ({formatear_fraccion(pulg)})" for nombre, pulg in TOPES_LONGITUD
        ]
        seleccion = st.radio("Tope activo", etiquetas_topes, horizontal=True)
        idx_tope = etiquetas_topes.index(seleccion)
        nombre_tope, longitud_in = TOPES_LONGITUD[idx_tope]

        desperdicio_in = max(0.0, grosor_actual - nominal) if nominal > 0 else 0.0

        colX, colY = st.columns(2)
        with colX:
            st.metric("Longitud seleccionada", f"{formatear_fraccion(longitud_in)} ({nombre_tope})")
        with colY:
            st.metric("Desperdicio estimado (grosor)", f"{desperdicio_in:.3f} in")

        if st.button("✅ Registrar Corte", type="primary", use_container_width=True):
            nuevo = {
                "Fecha_Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Grosor_Rustico_in": round(grosor_actual, 3),
                "Medida_Comercial": etiqueta,
                "Medida_Comercial_Nominal_in": nominal,
                "Longitud_Tope": nombre_tope,
                "Longitud_in": longitud_in,
                "Desperdicio_in": round(desperdicio_in, 3),
                "Estado": estado,
            }
            st.session_state.registros = pd.concat(
                [st.session_state.registros, pd.DataFrame([nuevo])],
                ignore_index=True,
            )
            st.success("Corte registrado correctamente ✅")

    st.divider()
    st.subheader("📋 Historial de registros")

    if st.session_state.registros.empty:
        st.info("No hay registros todavía.")
    else:
        st.dataframe(st.session_state.registros, use_container_width=True)

        col_csv, col_xlsx = st.columns(2)

        with col_csv:
            csv_bytes = st.session_state.registros.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Descargar CSV",
                data=csv_bytes,
                file_name=f"registros_corte_balsa_{date.today().isoformat()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_xlsx:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                st.session_state.registros.to_excel(writer, index=False, sheet_name="Registros")
            st.download_button(
                "⬇️ Descargar Excel",
                data=buffer.getvalue(),
                file_name=f"registros_corte_balsa_{date.today().isoformat()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )


# ------------------------------------------------------------------------------------
# TAB 4: MÉTRICAS DEL DÍA
# ------------------------------------------------------------------------------------
with tab_dashboard:
    st.subheader("📊 Métricas de Producción")

    df = st.session_state.registros.copy()

    if df.empty:
        st.info("No hay datos registrados todavía. Registra un corte para ver métricas aquí.")
    else:
        df["Fecha"] = pd.to_datetime(df["Fecha_Hora"]).dt.date
        hoy = date.today()
        df_hoy = df[df["Fecha"] == hoy]

        piezas_hoy = len(df_hoy)
        aprobadas_hoy = (df_hoy["Estado"] == "Aprobado").sum()
        rechazadas_hoy = (df_hoy["Estado"] == "Rechazado").sum()
        desperdicio_total_hoy = df_hoy["Desperdicio_in"].sum()
        desperdicio_prom_hoy = df_hoy["Desperdicio_in"].mean() if piezas_hoy > 0 else 0.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Piezas cortadas hoy", piezas_hoy)
        col2.metric("Aprobadas", int(aprobadas_hoy))
        col3.metric("Rechazadas", int(rechazadas_hoy))
        col4.metric("Desperdicio acumulado (in)", f"{desperdicio_total_hoy:.2f}")

        st.caption(
            f"Desperdicio promedio por pieza hoy: **{desperdicio_prom_hoy:.3f} in** "
            "(diferencia entre grosor bruto medido y el nominal comercial asignado)."
        )

        st.divider()
        st.subheader("Distribución de medidas comerciales (histórico)")
        conteo = df["Medida_Comercial"].value_counts()
        st.bar_chart(conteo)

        st.subheader("Producción histórica por día")
        piezas_por_dia = df.groupby("Fecha").size()
        st.line_chart(piezas_por_dia)

        st.subheader("Desperdicio histórico por día (in)")
        desperdicio_por_dia = df.groupby("Fecha")["Desperdicio_in"].sum()
        st.line_chart(desperdicio_por_dia)
