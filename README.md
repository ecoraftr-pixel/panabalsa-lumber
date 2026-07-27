Control de Calidad - Mesa de Corte de Balsa (Yaguachi, Ecuador)
Aplicación Streamlit + OpenCV para automatizar el control de calidad y el registro
de cortes en una mesa con sierra péndulo y topes de longitud.
Ejecutar localmente
Bash
pip install -r requirements.txt
streamlit run app.py
Ejecutar en Streamlit Community Cloud
Sube app.py y requirements.txt a un repositorio de GitHub.
Ve a https://share.streamlit.io/ y conecta el repositorio.
Selecciona app.py como archivo principal y despliega.
Configuración pendiente (¡importante!)
TOPES_LONGITUD (dentro de app.py): actualmente contiene longitudes de
ejemplo (6 a 12 pies). Reemplázalas con las longitudes reales de cada tope
físico de tu mesa de corte.
Calibración de cámara (px/in): en la barra lateral, ajusta "Píxeles por
pulgada" colocando un objeto de ancho conocido frente a la cámara/imagen y
afinando el valor hasta que la medición coincida con la realidad. Idealmente,
usa siempre la misma distancia/zoom de cámara para que la calibración sea
consistente.
Cómo funciona la detección
La imagen se convierte a escala de grises y se aplica desenfoque gaussiano.
Se umbraliza (Otsu automático o manual) para separar el listón del fondo.
Se limpian ruidos con operaciones morfológicas (close + open).
Se toma el contorno de mayor área y se calcula su rectángulo de área mínima
(cv2.minAreaRect), obteniendo ancho y alto en píxeles.
El lado más corto del rectángulo se interpreta como el grosor transversal
del listón y se convierte a pulgadas usando el factor de calibración.
El usuario puede sobrescribir manualmente el valor si el listón está
achaflanado y el "lado más grueso" no coincide con el detectado
automáticamente.
Notas sobre la cámara en vivo
Streamlit Cloud no soporta video continuo de forma nativa: st.camera_input
toma una foto (snapshot). Para video en vivo con procesamiento cuadro a cuadro
se necesitaría el paquete streamlit-webrtc, que requiere configuración
adicional de servidores STUN/TURN y no siempre funciona igual de bien en la nube
gratuita. Por eso esta versión usa snapshots, que son suficientes para el
flujo de control de calidad (foto del extremo del listón antes de cada corte).
