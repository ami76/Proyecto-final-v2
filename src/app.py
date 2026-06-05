import streamlit as st
import pickle
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
 
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis de Sentimientos Amazon",
    page_icon="🛍️",
    layout="wide"
)
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
 
# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE MODELOS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    with open(os.path.join(MODELS_DIR, 'modelo_xgboost.pkl'), 'rb') as f:
        xgb = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    modelo_nn = load_model(os.path.join(MODELS_DIR, 'modelo_red_neuronal.keras'))
    return xgb, tfidf, le, tokenizer, modelo_nn
 
xgb, tfidf, le, tokenizer, modelo_nn = cargar_modelos()
 
# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE LIMPIEZA
# ─────────────────────────────────────────────────────────────────────────────
stop_words = set(stopwords.words('spanish'))
negaciones = {'no', 'nada', 'nunca', 'jamás', 'ningún',
              'ninguna', 'tampoco', 'sin', 'ni', 'pero'}
stop_words_filtradas = stop_words - negaciones
 
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    palabras = texto.split()
    palabras = [p for p in palabras if p not in stop_words_filtradas]
    return ' '.join(palabras)
 
# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────
def predecir(texto):
    texto_limpio = limpiar_texto(texto)
 
    # XGBoost - predicción y probabilidades
    texto_tfidf = tfidf.transform([texto_limpio])
    pred_xgb = xgb.predict(texto_tfidf)[0]
    # Se obtienen probabilidades del XGBoost 
    proba_xgb = xgb.predict_proba(texto_tfidf)[0]
    sentimiento_xgb = le.inverse_transform([pred_xgb])[0]
 
    # Red Neuronal - predicción y probabilidades
    seq = tokenizer.texts_to_sequences([texto_limpio])
    pad = pad_sequences(seq, maxlen=100, truncating='post', padding='post')
    # Se obtienen probabilidades de la red neuronal
    proba_nn = modelo_nn.predict(pad, verbose=0)[0]
    pred_nn = proba_nn.argmax()
    sentimiento_nn = le.inverse_transform([pred_nn])[0]
 
    # Se calcula el ensemble (promedio de probabilidades de ambos modelos)
    # Combinar ambos modelos reduce errores individuales y mejora la precisión
    proba_ensemble = (proba_xgb + proba_nn) / 2
    pred_ensemble = proba_ensemble.argmax()
    sentimiento_ensemble = le.inverse_transform([pred_ensemble])[0]
 
    return {
        'sentimiento_xgb':    sentimiento_xgb,
        'sentimiento_nn':     sentimiento_nn,
        'sentimiento_final':  sentimiento_ensemble,
        'proba_xgb':          proba_xgb,
        'proba_nn':           proba_nn,
        'proba_ensemble':     proba_ensemble,
        'coinciden':          sentimiento_xgb == sentimiento_nn
    }
 
# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.title('🛍️ Análisis de Sentimientos en Reseñas')
st.subheader('Amazon Reviews en Español')
 
tab1, tab2, tab3 = st.tabs(['📝 Análisis Individual', '📂 Análisis por Lote', '📊 Historial'])
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 - ANÁLISIS INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    texto = st.text_area('Escribe tu reseña aquí:', height=150)
 
    if st.button('Analizar', type='primary'):
        if texto.strip() == '':
            st.warning('Por favor escribe una reseña')
        else:
            resultado = predecir(texto)
            emojis = {'positivo': '😊', 'neutro': '😐', 'negativo': '😞'}
            colores = {'positivo': 'green', 'neutro': 'orange', 'negativo': 'red'}
 
            st.divider()
 
            # ── Resultado final (ensemble)
            sentimiento_final = resultado['sentimiento_final']
            st.markdown(f"### Resultado: {emojis[sentimiento_final]} :{colores[sentimiento_final]}[{sentimiento_final.upper()}]")
 
            if not resultado['coinciden']:
                st.warning(f"⚠️ Los modelos no coinciden — XGBoost: **{resultado['sentimiento_xgb']}** | Red Neuronal: **{resultado['sentimiento_nn']}** — se muestra el resultado combinado.")
            else:
                st.success(f"✅ Ambos modelos coinciden en: **{sentimiento_final}**")
 
            st.divider()
 
            # ── Resultados por modelo
            col1, col2 = st.columns(2)
 
            with col1:
                st.markdown("** XGBoost **")
                st.metric('Predicción', f"{emojis[resultado['sentimiento_xgb']]} {resultado['sentimiento_xgb'].upper()}")
                # Se muestran las probabilidades por clase del XGBoost
                # Se puede ver qué tan seguro está el modelo, no solo la clase predicha
                st.markdown("Probabilidades:")
                for i, clase in enumerate(le.classes_):
                    st.progress(float(resultado['proba_xgb'][i]),
                                text=f"{clase}: {resultado['proba_xgb'][i]*100:.1f}%")
 
            with col2:
                st.markdown("** Red Neuronal **")
                st.metric('Predicción', f"{emojis[resultado['sentimiento_nn']]} {resultado['sentimiento_nn'].upper()}")
                # Se muestran las probabilidades por clase de la red neuronal
                st.markdown("Probabilidades:")
                for i, clase in enumerate(le.classes_):
                    st.progress(float(resultado['proba_nn'][i]),
                                text=f"{clase}: {resultado['proba_nn'][i]*100:.1f}%")
 
            # Se agrega al historial de la sesión
            # Permite ver el historial de análisis realizados en la sesión actual
            if 'historial' not in st.session_state:
                st.session_state.historial = []
            st.session_state.historial.append({
                'Reseña':      texto[:80] + '...' if len(texto) > 80 else texto,
                'XGBoost':     resultado['sentimiento_xgb'],
                'Red Neuronal': resultado['sentimiento_nn'],
                'Resultado':   sentimiento_final,
                'Coinciden':   '✅' if resultado['coinciden'] else '⚠️'
            })
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 - ANÁLISIS POR LOTE (BATCH)
# Análisis masivo de reseñas desde un CSV.
# Permite analizar múltiples reseñas a la vez y descargar los resultados,
# lo cual es un caso de uso real en empresas con grandes volúmenes de reseñas.
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("Sube un archivo CSV con una columna llamada **`review`** para analizar múltiples reseñas a la vez.")
 
    archivo = st.file_uploader("Selecciona un archivo CSV", type=['csv'])
 
    if archivo is not None:
        df_batch = pd.read_csv(archivo)
 
        if 'review' not in df_batch.columns:
            st.error("El archivo debe tener una columna llamada 'review'")
        else:
            st.info(f"Se encontraron **{len(df_batch)}** reseñas. Procesando...")
 
            resultados = []
            barra = st.progress(0)
            for i, (idx, fila) in enumerate(df_batch.iterrows()):
                r = predecir(str(fila['review']))
                resultados.append({
                    'review':       fila['review'],
                    'xgboost':      r['sentimiento_xgb'],
                    'red_neuronal': r['sentimiento_nn'],
                    'resultado':    r['sentimiento_final'],
                    'coinciden':    '✅' if r['coinciden'] else '⚠️'
                })
                barra.progress((i + 1) / len(df_batch))
 
            df_resultados = pd.DataFrame(resultados)
            st.table(df_resultados)
 
            # Resumen
            st.markdown("### Resumen")
            col1, col2, col3 = st.columns(3)
            conteo = df_resultados['resultado'].value_counts()
            col1.metric("😊 Positivas", conteo.get('positivo', 0))
            col2.metric("😐 Neutras",   conteo.get('neutro', 0))
            col3.metric("😞 Negativas", conteo.get('negativo', 0))
 
            # Descarga de resultados
            csv = df_resultados.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar resultados CSV",
                data=csv,
                file_name='resultados_sentimientos.csv',
                mime='text/csv'
            )
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 - HISTORIAL DE LA SESIÓN
# Permite ver y comparar todos los análisis que ha hecho
# durante la sesión actual sin tener que recordarlos manualmente.
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if 'historial' not in st.session_state or len(st.session_state.historial) == 0:
        st.info("Aún no has analizado ninguna reseña en esta sesión. Ve a la pestaña 'Análisis Individual'.")
    else:
        st.markdown(f"**{len(st.session_state.historial)}** reseñas analizadas en esta sesión:")
        df_historial = pd.DataFrame(st.session_state.historial)
        st.dataframe(df_historial, use_container_width=True)
 
        if st.button("🗑️ Limpiar historial"):
            st.session_state.historial = []
            st.rerun()