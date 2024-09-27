import tensorflow as tf
from PIL import Image
import time
import matplotlib.pyplot as plt
from laser_segmentation_quantizer.quantizer import super_fast_prediction, get_quantized_model


# 1. Загружаем обученную модель Keras

model = tf.keras.models.load_model('../data/models/model_unet_Light6_t30.h5')
# квантуем модель
tflite_model =get_quantized_model(model)

# 2. Загрузка изображения с помощью PIL
image_path = '../data/images/test.bmp'
image = Image.open(image_path)

# Начало измерения времени
start_time = time.time()
# 3. Предсказание модели
mask = super_fast_prediction(tflite_model, image)
# Конец измерения времени
end_time = time.time()
# Расчет времени выполнения
execution_time = end_time - start_time
# Вывод времени предсказания
print(f"Время выполнения предсказания: {execution_time:.6f} секунд")


# 3. Извлечение маски из массива (удаление первой размерности)
mask = mask.squeeze(axis=0)  # Приведет к форме (64, 256)

# 4. Проверка формы маски
print(mask.shape)  # Должно вывести (64, 256)

# 5. Визуализация маски
plt.imshow(mask, cmap='gray')  # Отображение в оттенках серого
plt.axis('off')  # Отключение осей
plt.title('Бинарная маска')
plt.show()