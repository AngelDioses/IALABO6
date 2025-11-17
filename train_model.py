import os
import kagglehub
import glob
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

# --- Hiperparámetros ---
IMG_SIZE = 256
N_CLASSES = 1
MODEL_PATH = 'unet_multi_task_model.h5'

# --- 1. Métrica y Pérdida Personalizada (Dice Coefficient) ---
def dice_coef(y_true, y_pred, smooth=1e-6):
    # CORRECCIÓN DE ERROR: Usamos tf.reshape([-1]) para aplanar el tensor
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32) 
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# --- 2. Arquitectura U-Net Multi-Task ---
def build_unet(input_size=(IMG_SIZE, IMG_SIZE, 1)):
    inputs = Input(input_size)
    
    # CODIFICADOR
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs); c1 = Dropout(0.1)(c1); c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1); p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1); c2 = Dropout(0.1)(c2); c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2); p2 = MaxPooling2D((2, 2))(c2)
    
    # BRIDGE
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2); c5 = Dropout(0.3)(c5); c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)
    
    # CABEZA DE CLASIFICACIÓN
    flat = Flatten()(c5)
    d1 = Dense(128, activation='relu')(flat)
    clf_output = Dense(1, activation='sigmoid', name='classification_output')(d1) 
    
    # DECODIFICADOR
    u6 = UpSampling2D((2, 2))(c5); u6 = concatenate([u6, c2])
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u6); c6 = Dropout(0.1)(c6); c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(c6)
    u7 = UpSampling2D((2, 2))(c6); u7 = concatenate([u7, c1])
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(u7); c7 = Dropout(0.1)(c7); c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(c7)
    
    # SALIDA DE SEGMENTACIÓN
    seg_output = Conv2D(N_CLASSES, (1, 1), activation='sigmoid', name='segmentation_output')(c7)
    
    model = Model(inputs=[inputs], outputs=[seg_output, clf_output])
    
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss={'segmentation_output': dice_loss, 
                        'classification_output': 'binary_crossentropy'},
                  loss_weights={'segmentation_output': 0.8, 
                                'classification_output': 0.2},
                  metrics={'segmentation_output': [dice_coef, MeanIoU(num_classes=2, name='mean_iou')],
                           'classification_output': ['accuracy', 'Precision', 'Recall']})

    return model

# --- 3. Carga y Preprocesamiento de Datos ---
def load_data():
    print("Iniciando descarga del dataset...")
    try:
        dataset_root_path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
        IMAGE_DIR = os.path.join(dataset_root_path, 'kaggle_3m/')
        print(f"Dataset disponible en: {IMAGE_DIR}")
    except Exception as e:
        print(f"Error al descargar con kagglehub: {e}. Usando ruta local de respaldo.")
        IMAGE_DIR = 'lgg-mri-segmentation/kaggle_3m/'
        
    all_files = sorted(glob.glob(os.path.join(IMAGE_DIR, '*/*_mask.tif')))
    
    data = []
    for mask_path in all_files:
        img_path = mask_path.replace('_mask.tif', '.tif')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        label = 1 if np.sum(mask) > 0 else 0 
        data.append({'image_path': img_path, 'mask_path': mask_path, 'label': label})
        
    df = pd.DataFrame(data).dropna()
    return df

def load_and_preprocess_image(path, is_mask=False):
    if not os.path.exists(path): return None
        
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
        
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR)
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0
    return img

def create_datasets(df):
    # Solución de error: Se inicializan los arrays al tamaño máximo
    X = np.zeros((len(df), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    Y_seg = np.zeros((len(df), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    Y_clf = np.zeros((len(df), 1), dtype=np.float32)

    j = 0 # Contador corregido para el array de datos
    print(f"Iniciando carga de {len(df)} archivos...")
    
    for _, row in df.iterrows(): 
        img = load_and_preprocess_image(row['image_path'])
        mask = load_and_preprocess_image(row['mask_path'], is_mask=True)
        
        if img is not None and mask is not None:
            X[j] = img
            Y_seg[j] = mask
            Y_clf[j] = row['label']
            j += 1 
    
    # Se recortan los arrays al tamaño real de archivos cargados (j)
    X = X[:j]
    Y_seg = Y_seg[:j]
    Y_clf = Y_clf[:j]
    
    print(f"Carga finalizada. Se procesaron {j} archivos válidos de {len(df)}.")
    
    return X, Y_seg, Y_clf

# --- 4. Ejecución Principal ---
if __name__ == "__main__":
    df_full = load_data()
    
    # Particionamiento 80/20
    df_train, df_test = train_test_split(df_full, test_size=0.2, stratify=df_full['label'], random_state=42)

    print(f"\nPartición de Entrenamiento (80%): {len(df_train)} imágenes")
    print(f"Partición de Prueba (20%): {len(df_test)} imágenes")

    # Crear tensores de datos
    X_train, Y_seg_train, Y_clf_train = create_datasets(df_train)
    X_test, Y_seg_test, Y_clf_test = create_datasets(df_test)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("ERROR: No se pudieron cargar suficientes imágenes válidas.")
    else:
        model = build_unet()

        print("\n--- Iniciando Entrenamiento (5 Épocas Demostrativas) ---")
        history = model.fit(
            X_train, 
            {'segmentation_output': Y_seg_train, 'classification_output': Y_clf_train},
            validation_data=(X_test, {'segmentation_output': Y_seg_test, 'classification_output': Y_clf_test}),
            epochs=5, 
            batch_size=16 
        )

        # 5. Guardar el Modelo
        model.save(MODEL_PATH)
        print(f"\n✅ Modelo guardado exitosamente como {MODEL_PATH}")

        # 6. Mostrar métricas finales de prueba
        test_results = model.evaluate(X_test, {'segmentation_output': Y_seg_test, 'classification_output': Y_clf_test}, verbose=0)
        
        metrics_names = ['loss', 'seg_loss', 'clf_loss', 'seg_dice_coef', 'seg_mean_iou', 'clf_accuracy', 'clf_precision', 'clf_recall']
        metrics_dict = dict(zip(metrics_names, test_results))
        
        precision = metrics_dict['clf_precision']
        recall = metrics_dict['clf_recall']
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n--- Métricas Finales en Conjunto de Prueba (20%) ---")
        print(f"Accuracy (Clasificación): {metrics_dict['clf_accuracy']:.4f}")
        print(f"Precision (Clasificación): {precision:.4f}")
        print(f"F1-Score (Calculado): {f1_score:.4f}")
        print(f"Mean IoU (Segmentación): {metrics_dict['seg_mean_iou']:.4f}")
