import tensorflow as tf
import numpy as np

def super_fast_prediction(tflite_model, image):
    input_data = image_processing(image)
    prediction = quantized_model_predicton(tflite_model, input_data)
    return prediction

def image_processing(img):
    # 1. Преобразование изображения в RGB
    img = img.convert('RGB')  # Преобразование изображения в трехканальный формат RGB
    # 2. Изменение размера изображения до (64, 256)
    img_resized = img.resize((256, 64))  # Ширина (256) и высота (64)

    # 3. Преобразование изображения в numpy массив
    img_array = np.array(img_resized)  # Теперь img_array имеет форму (64, 256, 3)

    # 4. Нормализация изображения (если это требуется для модели)
    img_array = img_array.astype(np.float32)  # Нормализация значений пикселей в диапазоне [0, 1]

    # 5. Добавление размерности для батча (1, 64, 256, 3)
    input_data = np.expand_dims(img_array, axis=0)  # Теперь input_data имеет форму (1, 64, 256, 3)
    return input_data
def get_quantized_model(model):
    # Конвертация модели в TFLite с квантованием
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return tflite_model

def quantized_model_predicton(tflite_model, input_data):
  # Загрузка квантованной модели (TFLite) из байтового объекта
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()  # Инициализация интерпретатора

  # Получаем информацию о входных и выходных тензорах
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Подаем данные в модель
  interpreter.set_tensor(input_details[0]['index'], input_data)

  # Выполняем инференс
  interpreter.invoke()

  # Получаем результат
  prediction = interpreter.get_tensor(output_details[0]['index'])
  return prediction