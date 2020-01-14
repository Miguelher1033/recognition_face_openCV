Reconocimiento facial con OpenCV – Python

Requerimientos:
Python 3
Ejecutar pip install -r  requirements.txt

Funcionamiento:
1.	Ejecutar archivo dataset.py: almacena imágenes en la ruta especificada (dataset) con la siguiente estructura:
Etiqueta_1
2.	Ejecutar archivo train.py: recorre cada una de las carpetas almacenadas en imagenes (dataset), y crea archivo train.yml con la data del entrenamiento.

3.	Ejecutar Recognition_face.py: abre venta para realizar la captura de rostros en tiempo real, y compararla con la data del entrenamiento.


Codigo basado en: https://github.com/jorge190588/face_recognitionOpenCv2

Agradecimiento a jorge190588

