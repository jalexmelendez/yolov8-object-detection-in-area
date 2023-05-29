import cv2
import supervision as sv

import json
import time
import uuid

import paho.mqtt.client as mqtt

from ultralytics import YOLO
from typing import Any, Union, Tuple, List
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, BaseSettings
#from cv.object_detection import main as cv_main

###########################################
# Configuracion de librerias y frameworks #
###########################################

class Settings(BaseSettings):
    # Configuracion MQTT
    MQTT_CHANNEL: str = "88e0f9a0/88e0f9a0-2248-41b5-bc8a-95f1484ce5ad"

    # Necesitamos un ID de cliente distinto en caso de tener diversas instancias del servidor.
    MQTT_CLIENT: str = f"TECNM_CHIH-{uuid.uuid4()}"
    MQTT_SERVER: str = "test.mosquitto.org"
    MQTT_PORT: int = 1883

# Instancia de FastAPI
app = FastAPI(
    title="API IOT SENSORES"
)
settings = Settings()

# Cliente de MQTT
client = mqtt.Client(settings.MQTT_CLIENT)

#################
# Base de Datos #
#################

"""
Almacena datos en la lista en memoria, usado solo
en demostrar uso de mqtt, en azure se debe reemplazar
por mongo.
"""
#MOCK_DATASTORE = []

def store_data(lecture: Any):
    """
    Esta es la funcion que nos ayuda a escribir en nuestra base de
    datos las lecturas recibidas por paho, convierte el json codificado en 
    una instancia de DataLecture.
    """
    casted_lecture = lecture
    print(f"Lectura recibida por MQTT: {casted_lecture}")
    # NOTA: Ejemplifica el proceso de almacenar datos, se debe de usar mongo para el proyecto.
    #MOCK_DATASTORE.append(casted_lecture)

#####################
# Cliente MQTT Paho #
#####################

def on_connect(client, userdata, flags, rc):
    """
    La devolución de llamada para cuando el cliente recibe una respuesta CONNACK del servidor.
    """
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(settings.MQTT_CHANNEL)

def on_disconnect(client, userdata,  rc):
    """
    El evento de desconexion.
    """
    print("El cliente se ha desconectado, iniciando proceso de intento de reconexion...")

def on_message(client, userdata, msg):
    """
    El callback cuando un PUBLISH message es recibido por el broker.
    """
    print(msg.topic+" "+str(msg.payload))
    try:
        bytes_to_json = json.loads(msg.payload)
        store_data(lecture=bytes_to_json)
    except Exception as e:
        print(e)

def on_publish(client, userdata, mid):
    """
    El Callback cuando un mensaje es publicado
    """
    print(f"Publicaste el mensaje: {mid}")

client.on_connect = on_connect
client.on_message = on_message
client.on_publish = on_publish
client.on_disconnect = on_disconnect

client.connect(host=settings.MQTT_SERVER, port=settings.MQTT_PORT, keepalive=60)

#
# Esto es parte de la interfaz de cliente con subprocesos. 
# Llame a esto una vez para iniciar un nuevo hilo para procesar el tráfico de red. 
# Esto proporciona una alternativa para llamar repetidamente a loop() usted mismo.
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# loop_start() reconecta automaticamente, no es necesario hacer una reconexion manual.
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# NOTA: si se desea correr solamente PAHO sin fastapi comentar la linea "loop_start"
# y descomentar "loop_forever()"
 
# client.loop_forever()
client.loop_start()

LAST_SENT_MESSAGE = time.time()

class ActionArea:
    """
    The action area rectangle
    """

    BLUE_RGB_TUPLE = (225, 0, 0)

    def __init__(self, start_point: Tuple = (5,5), end_point: Tuple = (220, 220), color: Tuple =  BLUE_RGB_TUPLE) -> None:
        """
        :start_point: The rectangle start point
        :end_point: The rectangle end point
        :color: The RGB color
        """
        self.start_point: Tuple = start_point
        self.end_point = end_point
        self.color = color

class ObjectCoordinate:
    """
    The coordinates of the YOLO bounding box
    """

    def __init__(self, x, y, w, h) -> None:
        self.x = x
        self.y = y
        self.width = w
        self.height = h

    def get_center_coordinates(self) -> Tuple:
        """
        Returns the center of the object coordinates in the form of
        Tuple(X, Y)
        """
        return (
            ((self.x + self.width) / 2),
            ((self.y + self.height) / 2)
        )

class ObjectInstance:
    """
    The dected object, it contains the label and the object coordinates
    """

    def __init__(self, label:str, coordinates: ObjectCoordinate) -> None:
        self.label = label
        self.coordinate = coordinates


def object_origin_in_action_area(action_area: ActionArea, object_instance: ObjectInstance):
    """
    Detects if the object origin (the blue dot) is in the action area (the blue rectangle)
    """
    global LAST_SENT_MESSAGE

    x_start, y_start = action_area.start_point
    x_end, y_end = action_area.end_point

    detection_x, detection_y = object_instance.coordinate.get_center_coordinates()

    if (x_start < detection_x < x_end) and (y_start < detection_y < y_end):
        """
        If the object in action area check the label
        """
        if object_instance.label != 'bottle':
            """
            If the label is not bottle send the message to the ESP32
            """
            current_time = time.time()

            print(current_time -LAST_SENT_MESSAGE)

            if (current_time - LAST_SENT_MESSAGE) > 1:
                client.publish(topic=settings.MQTT_CHANNEL, payload=json.dumps({'activate': 1}))
                LAST_SENT_MESSAGE = current_time

def main():
    capture = cv2.VideoCapture(0)

    action_area = ActionArea(start_point=((int(640/2 - 50), int(640/2 - 50))), end_point=(int(640/2 + 50), int(640/2 + 50)))

    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoxAnnotator()

    while True:
        ret, frame = capture.read()
        result = model(frame)[0]

        detections = sv.Detections.from_yolov8(result)

        try:
            # Extraemos los labels
            labels = [
                f"{model.model.names[class_id]}"
                for class_id in detections.class_id
            ]

            # Extraemos las coordenadas del objeto
            detection_coords: List[ObjectCoordinate] = [
                ObjectCoordinate(xywh[0], xywh[1], xywh[2], xywh[3]) 
                for xywh in detections.xyxy
            ]

            objects: Union[List[ObjectInstance], List[Any]] = []

            for i, label in enumerate(labels):
                objects.append(ObjectInstance(label=label, coordinates=detection_coords[i]))

            if labels.__len__() > 0:
                for i, obj in enumerate(objects):
                    
                    # Draw points on origin of objects
                    detection_x, detection_y = obj.coordinate.get_center_coordinates()

                    frame = cv2.circle(frame, (int(detection_x), int(detection_y)), 10, ActionArea.BLUE_RGB_TUPLE)

                    # Run the detection Logic
                    object_origin_in_action_area(action_area=action_area, object_instance=obj)

            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        except Exception as e:
            print(e)
            frame = box_annotator.annotate(scene=frame, detections=detections)

        frame = cv2.rectangle(img=frame, pt1=action_area.start_point, pt2=action_area.end_point, color=action_area.color, thickness=2)
        
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()
    #cv_main()
