import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import os
from tensorflow.keras.models import load_model

# --- Configuración y Carga de Modelo ---
MODEL_PATH = 'unet_multi_task_model.h5' 
IMG_SIZE = 256

# Definiciones de métricas necesarias para cargar el modelo
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32) 
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

@st.cache_resource
def load_dl_model(path):
    if not os.path.exists(path):
        st.error(f"El archivo del modelo no existe: '{path}'. Ejecute primero 'python train_model.py'.")
        return None
    
    custom_objects = {
        'dice_loss': dice_loss, 
        'dice_coef': dice_coef,
        'MeanIoU': tf.keras.metrics.MeanIoU(num_classes=2) 
    }
    try:
        model = load_model(path, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_dl_model(MODEL_PATH)

# --- Funciones de Preprocesamiento y Predicción ---
def preprocess_and_predict(image_file, model):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if img is None: return None, None, None
        
    original_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = original_img[np.newaxis, :, :, np.newaxis] / 255.0
    
    seg_pred, clf_pred = model.predict(img_normalized, verbose=0)
    
    mask_pred = (seg_pred.squeeze() > 0.5).astype(np.uint8) * 255 
    prob_tumor = clf_pred[0][0]
    
    return original_img, mask_pred, prob_tumor

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Deep Learning Image System", layout="wide")

st.title("🧠 Clasificación y Segmentación de Tumores MRI")
st.markdown("Sistema basado en **U-Net** para la detección y delimitación de tumores.")
st.write("---")

## 1. Reporte de Métricas (Valores de ejemplo)
st.header("📊 Reporte de Métricas de Prueba (Ejemplo)")
st.caption("Nota: Estos valores deben ser consistentes con la salida de `train_model.py`.")

metrics_data = {
    'Métrica': ['Accuracy (Clasificación)', 'F1-Score (Clasificación)', 'Mean IoU (Segmentación)'],
    'Valor (Aprox.)': ['94.5%', '0.928', '0.84']
}
df_metrics = pd.DataFrame(metrics_data).set_index('Métrica')

st.dataframe(df_metrics)

st.subheader("Interpretación de Métricas")
st.info("""
* **Accuracy:** Proporción de imágenes correctamente clasificadas.
* **F1-Score:** Equilibrio entre la precisión y la capacidad de detección.
* **Mean IoU:** Mide la superposición de la máscara predicha con la real.
""")
st.write("---")

## 2. Predicción en Tiempo Real
st.header("🚀 Predicción en Tiempo Real")

uploaded_file = st.file_uploader("Sube una imagen MRI (tif/png) para analizar", type=["tif", "png"])

if uploaded_file is not None and model is not None:
    with st.spinner('Analizando imagen...'):
        original_img, mask_pred, prob_tumor = preprocess_and_predict(uploaded_file, model)
    
    if original_img is not None:
        clase_detectada = "Tumor Detectado" if prob_tumor > 0.5 else "Sin Anomalía"
        
        st.subheader("Resultados del Análisis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_img, caption="Imagen de Entrada", use_container_width=True) # CORRECCIÓN Streamlit
            st.metric(label="Clasificación Final", value=clase_detectada, delta=f"Confianza: {prob_tumor*100:.2f}%")
            
        with col2:
            original_color = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
            color_mask = np.zeros_like(original_color, dtype=np.uint8)
            color_mask[mask_pred > 0] = [0, 0, 255] # Máscara roja
            
            combined_img = cv2.addWeighted(original_color, 0.7, color_mask, 0.3, 0)
            
            st.image(combined_img, caption="Máscara de Segmentación Superpuesta (Rojo)", use_container_width=True) # CORRECCIÓN Streamlit
            
        st.markdown("### Interpretación de la Predicción")
        if clase_detectada == "Tumor Detectado":
            st.error(f"El modelo clasifica la imagen como **Tumor Detectado** (Confianza: {prob_tumor*100:.2f}%). La zona roja muestra la segmentación de la anomalía.")
        else:
            st.success(f"El modelo no detectó ninguna anomalía significativa (Confianza: {prob_tumor*100:.2f}%).")
