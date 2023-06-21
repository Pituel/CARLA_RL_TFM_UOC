# <p align="center"> Aplicación de técnicas de aprendizaje por refuerzo profundo para conducción autónoma en el simulador CARLA</p>

**Autor: Pablo López Ladrón de Guevara**

## Instalación y ejecución

Para poder entrenar/testear los modelos de forma correcta se debe tener instalada la versión 0.9.8 de [CARLA](http://carla.org/) y un entorno virtual con la versión 3.7 de python. Con el entorno virtual activado y situados en la carpeta principal del proyecto instalamos las librerías necesarias mediante el comando `pip install -r requirements.txt`. Una vez realizados estos pasos ya podemos ejecutar el proyecto mediante el comando `python driver.py`. Por defecto se ejecutará el modelo CNN+MLP en modo test. Esta configuración se puede modificar en el archivo settings.py. En este mismo archivo es posible configurar los hiperparámetros del modelo en el caso de que se quiera volver a entrenar de nuevo con una configuración diferente.


## Licencia

Este proyecto está licenciado bajo licencia MIT ([LICENSE.md](LICENSE.md)).



