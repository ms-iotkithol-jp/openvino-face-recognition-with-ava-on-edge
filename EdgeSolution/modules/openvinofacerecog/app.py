import threading
from PIL import Image
import os
import io
import json
import logging
import linecache
import sys
import datetime
import asyncio
from argparse import ArgumentParser

import cv2
import numpy as np

from openvino.inference_engine import IECore

from flask import Flask, request, jsonify, Response
from file_uploader import FileUploader
from object_detector import ObjectDetector
from face_recognizer import FaceRecognizer
from iot_client import IoTClient, IoTModuleClient, IoTDeviceClient

import requests
import tarfile
from six.moves import input
# from azure.iot.device import IoTHubModuleClient

intervalSec = 60
inferenceMark = False
sendDetection = False
modelLoaded = False

def update_reported_properties_of_loaded_model(client, od_model, fr_fd_model, fr_lm_model, fr_reid_model):
    current_status = {"loaded_models": {'od':od_model, 'fr_fd':fr_fd_model, 'fr_lm':fr_lm_model, 'fr_reid':fr_reid_model }}
    client.patch_twin_reported_properties(current_status)

def update_reported_properties_of_ipaddress(client, ipaddress):
    current_status = {'ipaddress': ipaddress}
    client.patch_twin_reported_properties(current_status)

def downloadModelsFile(modelUrl, modelFileName, modelFolderName):
    modelFolderPath = None
    response = requests.get(modelUrl)
    if response.status_code == 200:
        os.makedirs(modelFolderName, exist_ok=True)
        saveFileName = os.path.join(modelFolderName, modelFileName)
        with open(saveFileName, 'wb') as saveFile:
            saveFile.write(response.content)
            logging.info('Succeeded to download new model.')
            if modelFileName.endswith('.tgz'):
                tar = tarfile.open(saveFileName, 'r:gz')
                tar.extractall(modelFolderName)
                os.remove(saveFileName)
            modelFolderPath = os.path.join(os.getcwd(), modelFolderName)
    else:
        logging.info(f'Failed to download {modelUrl}')
    return modelFolderPath

def parse_desired_properties_request(client, configSpec, od, fr, configLock):
    sendKey = 'send-telemetry'
    global sendDetection, modelLoaded
    if sendKey in configSpec:
        configLock.acquire()
        sendDetection = bool(configSpec[sendKey])
        configLock.release()
        logging.info(f'new send detection telemetry = {sendDetection}')
    uploadImageKey = 'upload'
    global intervalSec, inferenceMark
    if uploadImageKey in configSpec:
        configLock.acquire()
        if 'interval-sec' in configSpec[uploadImageKey]:
            intervalSec = int(configSpec[uploadImageKey]['interval-sec'])
        if 'inference-mark' in configSpec[uploadImageKey]:
            inferenceMark = bool(configSpec[uploadImageKey]['inference-mark'])
        configLock.release()
        logging.info(f'new interval-sec = {intervalSec}')
        logging.info(f'new inference-mark = {inferenceMark}')

    # AI models
    if modelLoaded:
        logging.info('Model update will be done at next starting.')
        return

    odModelName = None
    fdModelName = None
    lmModelName = None
    reidModelName = None
    odModelLoaded = False
    frModelLoaded = False

    odModelFileKey = 'model-od'
    if odModelFileKey in configSpec:
        modelUrl = configSpec[odModelFileKey]['url']
        modelFileName = configSpec[odModelFileKey]['filename']
        odModelName = configSpec[odModelFileKey]['name']
        odLabelName = configSpec[odModelFileKey]['label']
        odArchType = configSpec[odModelFileKey]['architecture-type']
        modelFolderPath = os.path.join(os.getcwd(), odModelFileKey)
        modelPath = os.path.join(modelFolderPath, odModelName)
        labelPath = os.path.join(modelFolderPath, odLabelName)
        logging.info(f'Receive request of object detection model update - Model:{odModelName}, Label:{odLabelName}, ArchitectureType:{odArchType} - {modelUrl}')
        if os.path.exists(modelPath) and os.path.exists(labelPath):
            logging.info(f'Model:{odModelName} and Labels:{odLabelName} has been downloaded.')
        else:
            logging.info(f'Loading {modelFileName}...')
            modelFolderPath = downloadModelsFile(modelUrl,modelFileName,odModelFileKey)
            modelPath = os.path.join(modelFolderPath, odModelName)
            labelPath = os.path.join(modelFolderPath, odLabelName)
        configLock.acquire()
        res = od.LoadModel(modelPath, labelPath, odArchType)
        configLock.release()
        if res == 0:
            logging.info('object detection model load succeeded')
        odModelLoaded = True


    frModelFileKey = 'model-fr'
    if frModelFileKey in configSpec:
        modelUrl = configSpec[frModelFileKey]['url']
        modelFileName = configSpec[frModelFileKey]['filename']
        fdModelName = configSpec[frModelFileKey]['fd-name']
        lmModelName = configSpec[frModelFileKey]['lm-name']
        reidModelName = configSpec[frModelFileKey]['reid-name']
        logging.info(f'Receive request of face recognition model update - {modelFileName} - {modelUrl}')
        modelFolderPath = os.path.join(os.getcwd(), frModelFileKey)
        fdModelPath = os.path.join(modelFolderPath, fdModelName)
        lmModelPath = os.path.join(modelFolderPath, lmModelName)
        reidModelPath = os.path.join(modelFolderPath, reidModelName)
        if os.path.exists(fdModelPath) and os.path.exists(lmModelPath) and os.path.exists(reidModelPath):
            logging.info(f'FD Model:{fdModelName}, LM Model:{lmModelName}, REID Model:{reidModelName} have been downloaded.')
        else:
            logging.info(f'Loading {modelFileName}...')
            modelFolderPath = downloadModelsFile(modelUrl,modelFileName,frModelFileKey)
            fdModelPath = os.path.join(modelFolderPath, fdModelName)
            lmModelPath = os.path.join(modelFolderPath, lmModelName)
            reidModelPath = os.path.join(modelFolderPath, reidModelName)
        configLock.acquire()
        res = fr.LoadModel(fdModelPath, lmModelPath, reidModelPath)
        configLock.release()
        if res == 0:
            logging.info('face recognition model load succeeded')
        frModelLoaded = True
    modelLoaded = odModelLoaded and frModelLoaded

    if odModelName and fdModelName and lmModelName and reidModelName and client:
        update_reported_properties_of_loaded_model(client, odModelName, fdModelName, lmModelName, reidModelName)

def twin_patch_handler(patch):
    logging.info(f'Twin desired properties patch updated. - {patch}')
    parse_desired_properties_request(client, patch, None, None, configLock)

def twin_update_listener(client, configLock):
    while True:
        global intervalSec, inferenceMark
        patch = client.receive_twin_desired_properties_patch()  # blocking call
        logging.info(f'Twin desired properties patch updated. - {patch}')
        parse_desired_properties_request(client, patch, None, None, configLock)


def setup_iot(edgeDeviceId, od, fr, configLock, cs):
    if not edgeDeviceId and not cs:
        logging.info('This execution may be test.')
        return None
    try:
        global inferenceMark, intervalSec
        if not sys.version >= "3.5.3":
            raise Exception( "The sample requires python 3.5.3+. Current version of Python: %s" % sys.version )
        logging.info( "IoT Hub Client for Python" )

        iotClient = None
        if edgeDeviceId:
            # The client object is used to interact with your Azure IoT hub.
            iotClient = IoTModuleClient()
            logging.info('Initialized as IoT Edge Module.')
        elif cs:
            iotClient = IoTDeviceClient(cs)
            logging.info('Initialized as IoT Device App')

        if iotClient:
            # connect the client.
            iotClient.connect()
            logging.info("Connected to Edge Runtime.")
        
            currentTwin = iotClient.get_twin()
            configSpec = currentTwin['desired']
            parse_desired_properties_request(iotClient, configSpec, od, fr, configLock)

            iotClient.receive_twin_desired_properties_patch_handler(parse_desired_properties_request, configLock)
            # twin_update_listener_thread = threading.Thread(target=twin_update_listener, args=(iotClient, configLock))
            # twin_update_listener_thread.daemon = True
            # twin_update_listener_thread.start()
            logging.info('IoT Edge or Device settings done.')

        return iotClient

    except Exception as e:
        logging.warning( "Unexpected error %s " % e )
        raise

logging.basicConfig(level=logging.INFO)

def build_argparser():
    parser = ArgumentParser()
    general = parser.add_argument_group('General')
    general.add_argument('-pg', '--pg', required=False, help='Required. Folder path of Face DB.')
    general.add_argument('-d', '--device', required=False, help='Required. should be CPU | GPU | MYRIAD | HETERO | HDDL')
    general.add_argument('-m', '--inference_mask', default=False, required=False, action='store_true', help='Optional. Mark Inference raectangle on a frame.')

    iot = parser.add_argument_group('IoT Hub')
    iot.add_argument('-cs', '--connection_string', required=False)
    iot.add_argument('-opod', '--output_path_od', required=False)
    iot.add_argument('-opfr', '--output_path_fr', required=False)

    od = parser.add_argument_group('Object Detection')
    od.add_argument('-ou', '--od_url', required=False, help='Required. Specify url of object detection tgz file.')
    od.add_argument('-of', '--od_filename', required=False, help='Required. Specify file name of the tgz file.')
    od.add_argument('-on', '--od_name', required=False, help='Required. Specify file name of object detection model.')
    od.add_argument('-ol', '--od_label', required=False, help='Required. Specify file name of object detection labels.')
    od.add_argument('-oa', '--od_architecture_type', required=False, help='Required. Specify architecture type of the object detection model. should be ssd | ctpn | yolo | yolov4 | faceboxes | centernet | retinaface | ultra_lightweight_face_detection | retinaface-pytorch')

    fr = parser.add_argument_group('Face Recognition')
    fr.add_argument('-fu', '--fr_url', required=False, help='Required. Specify url of face recoginition tgz file.')
    fr.add_argument('-ff', '--fr_filename', required=False, help='Required. Specify file name of the tgz file.')
    fr.add_argument('-fd', '--fr_fd_name', required=False, help='Required. Specify file name of face detection model.')
    fr.add_argument('-fl', '--fr_lm_name', required=False, help='Required. Specify file name landmark detection model.')
    fr.add_argument('-fr', '--fr_reid_name', required=False, help='Required. Specify file name of face recognition model.')
    return parser

def pil2cv(image):
    cvimg = np.array(image, dtype=np.uint8)
    if cvimg.ndim == 2:  # Black and White
        pass
    elif cvimg.shape[2] == 3:   # Color
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)
    elif cvimg.shape[2] == 4:   # Transparent
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGBA2BGRA)
    return cvimg

def cv2pil(image):
    img = image.copy()
    if img.ndim == 2:
        pass
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif pilImage.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    pilImage = Image.fromarray(img)
    return pilImage

async def main():
    args = None
    iotClient = None

    IOTEDGE_DEVICEID = None
    if 'IOTEDGE_DEVICEID' in os.environ:
        IOTEDGE_DEVICEID=os.environ['IOTEDGE_DEVICEID']
    else:
        args = build_argparser().parse_args()

    fileUploader = None
    if 'BLOB_ON_EDGE_MODULE' in os.environ and 'BLOB_ON_EDGE_ACCOUNT_NAME' in os.environ and 'BLOB_ON_EDGE_ACCOUNT_KEY' in os.environ and 'BLOB_CONTAINER_NAME':
        BLOB_ON_EDGE_MODULE = os.environ['BLOB_ON_EDGE_MODULE']
        BLOB_ON_EDGE_ACCOUNT_NAME = os.environ['BLOB_ON_EDGE_ACCOUNT_NAME']
        BLOB_ON_EDGE_ACCOUNT_KEY = os.environ['BLOB_ON_EDGE_ACCOUNT_KEY']
        BLOB_CONTAINER_NAME = os.environ['BLOB_CONTAINER_NAME']
        logging.info(f'Blob Service specified. BLOB_ON_EDGE_MODULE={BLOB_ON_EDGE_MODULE},BLOB_ON_EDGE_ACCOUNT_NAME={BLOB_ON_EDGE_ACCOUNT_NAME},BLOB_ON_EDGE_ACCOUNT_KEY={BLOB_ON_EDGE_ACCOUNT_KEY},BLOB_CONTAINER_NAME={BLOB_CONTAINER_NAME}')
        fileUploader = FileUploader(BLOB_ON_EDGE_MODULE, BLOB_ON_EDGE_ACCOUNT_NAME, BLOB_ON_EDGE_ACCOUNT_KEY, BLOB_CONTAINER_NAME)
        fileUploader.initialize()
        logging.info('FileUploader initialized.')
    FACEDB_FOLDER_NAME = None
    if 'FACEDB_FOLDER_NAME' in os.environ:
        FACEDB_FOLDER_NAME = os.environ['FACEDB_FOLDER_NAME']
    else:
        if args:
            FACEDB_FOLDER_NAME = args.pg
    logging.info(f'Face DB Folder - {FACEDB_FOLDER_NAME}')
    OPENVINO_DEVICE = None
    if 'OPENVINO_DEVICE' in os.environ:
        OPENVINO_DEVICE = os.environ['OPENVINO_DEVICE']
    else:
        if args:
            OPENVINO_DEVICE = args.device
    logging.info((f'OpenVino Device - {OPENVINO_DEVICE}'))
    MSG_OUTPUT_PATH_OD = None
    if 'MSG_OUTPUT_PATH_OD' in os.environ:
        MSG_OUTPUT_PATH_OD = os.environ['MSG_OUTPUT_PATH_OD']
    else:
        if args:
            MSG_OUTPUT_PATH_OD = args.output_path_od
        else:
            if IOTEDGE_DEVICEID:
                MSG_OUTPUT_PATH_OD = 'output_od'
    MSG_OUTPUT_PATH_FR = None
    if 'MSG_OUTPUT_PATH_FR' in os.environ:
        MSG_OUTPUT_PATH_FR = os.environ['MSG_OUTPUT_PATH_FR']
    else:
        if args:
            MSG_OUTPUT_PATH_FR = args.output_path_fr
        else:
            if IOTEDGE_DEVICEID:
                MSG_OUTPUT_PATH_FR = 'output_fr'
    logging.info(f'IoT Edge Module Output Path - OD:{MSG_OUTPUT_PATH_OD},FR:{MSG_OUTPUT_PATH_FR}')
    iotDeviceCS = None
    if args:
        iotDeviceCS = args.connection_string
        logging.info(f'IoT Device App Mode : {iotDeviceCS}')
    else:
        if IOTEDGE_DEVICEID:
            logging.info(f'IoT Edge Model Mode : {IOTEDGE_DEVICEID}')

    app = Flask(__name__)

    logging.info('Initialize Infarence Engine...')
    ie = IECore()
    objectDetector = ObjectDetector(ie, OPENVINO_DEVICE)
    faceRecognizer = FaceRecognizer(ie, OPENVINO_DEVICE, FACEDB_FOLDER_NAME)


    # res = ovmv.LoadModel('/result/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml')
    configLock = threading.Lock()
    iotClient = setup_iot(IOTEDGE_DEVICEID, objectDetector, faceRecognizer, configLock, iotDeviceCS)
    if args:
        if args.od_url and args.od_filename and args.od_name and args.od_label and args.od_architecture_type and args.fr_url and args.fr_filename and args.fr_fd_name and args.fr_lm_name and args.fr_reid_name :
            configSpec = {
                'send-telemetry': False,
                'upload' : { 'interval-sec': 60, 'inference-mark': args.inference_mask },
                'model-od' : {
                    'url': args.od_url,
                    'filename': args.od_filename,
                    'name': args.od_name,
                    'label': args.od_label,
                    'architecture-type': args.od_architecture_type
                },
                'model-fr': {
                    'url': args.fr_url,
                    'filename': args.fr_filename,
                    'fd-name': args.fr_fd_name,
                    'lm-name': args.fr_lm_name,
                    'reid-name': args.fr_reid_name
                }
            }
            parse_desired_properties_request(iotClient, configSpec, objectDetector, faceRecognizer, configLock)
    # time_delta_sec = 60
    if fileUploader and iotClient:
        update_reported_properties_of_ipaddress(iotClient, fileUploader.gethostname())

    @app.route("/facerecognition", methods = ['POST'])
    async def facerecognition():
        try:
            global intervalSec, inferenceMark, sendDetection
            logging.info('received recognition request')
            # import pdb; pdb.set_trace()

            imageData = io.BytesIO(request.get_data())
            pilImage = Image.open((imageData))
            frame = pil2cv(pilImage)
            logging.info('Recognizing...')

            configLock.acquire()
            nowTime = datetime.datetime.now()
            timeDelta = nowTime - faceRecognizer.GetScoredTime()
            logging.info(f'process time - now={nowTime}, scored time={faceRecognizer.GetScoredTime()}')
            detectedObjects, inferencedImageFile = faceRecognizer.Process(frame, inferenceMark)
            isSendTelemetry = sendDetection
            configLock.release()

            logging.info('Recognized.')
            if isSendTelemetry:
                telemetry = {'timestamp':'{0:%Y-%m-%dT%H:%M:%S.%fZ}'.format(nowTime), 'face-recognition': detectedObjects }
                sendMsg = json.dumps(telemetry)
                # iotClient.send_message(sendMsg, 'face_recognition_monitor')
                iotClient.send_message(sendMsg, MSG_OUTPUT_PATH_FR)
                logging.info('Send detection message to face_recognition_monitor of IoT Edge Runtime')

            if len(detectedObjects) > 0 :
                logging.info(f'check time - timeDelta.seconds={timeDelta.seconds}')
                if timeDelta.seconds > intervalSec:
                    logging.info('scored long again.')
                    if fileUploader:
                        imageData = None
                        if inferenceMark:
                            imageData = open(inferencedImageFile, 'rb')
                            pilImage = Image.open(imageData)
                        else:
                            imageData = io.BytesIO(request.get_data())
                        fileUploader.upload(imageData, IOTEDGE_DEVICEID, '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now()), pilImage.format.lower())
                        if inferenceMark:
                           imageData.close()
                respBody = {
                    'inferences' : detectedObjects
                }
                respBody = json.dumps(respBody)
                logging.info(f'Sending response - {respBody}')
                return Response(respBody, status=200, mimetype='application/json')
            else :
                logging.info('Sending empty response')
                return Response(status=204)

        except Exception as e:
            if configLock.locked():
                configLock.release()
            logging.error(f'exception - {e}')
            return Response(response='Exception occured while processing face recognition of the image.', status=500)

    @app.route("/objectdetection", methods = ['POST'])
    async def objectdetection():
        try:
            global intervalSec, inferenceMark, sendDetection
            logging.info('received request')
            # import pdb; pdb.set_trace()

            imageData = io.BytesIO(request.get_data())
            pilImage = Image.open((imageData))
            frame = pil2cv(pilImage)
            logging.info('Scoring...')

            configLock.acquire()
            nowTime = datetime.datetime.now()
            timeDelta = nowTime - objectDetector.GetScoredTime()
            detectedObjects, output_frame = objectDetector.Score(frame, inferenceMark)
            isSendTelemetry = sendDetection
            configLock.release()

            logging.info('Scored.')
            if isSendTelemetry and len(detectedObjects)>0:
                totalDtected = 0
                detected = {}
                for d in detectedObjects:
                    label = d['entity']['tag']['value']
                    if label in detected:
                        detected[label] = detected[label] + 1
                    else:
                        detected[label] = 1
                    totalDtected = totalDtected + 1
                telemetry = {'timestamp':'{0:%Y-%m-%dT%H:%M:%S.%fZ}'.format(nowTime), 'totaldetection': totalDtected, 'detected':detected, 'detectedObjects':detectedObjects  }
                sendMsg = json.dumps(telemetry)
                iotClient.send_message(sendMsg, MSG_OUTPUT_PATH_OD)
                logging.info('Send detection message to detection_monitor of IoT Edge Runtime')

            if len(detectedObjects) > 0 :
                if timeDelta.seconds > intervalSec:
                    logging.info('scored long again.')
                    if fileUploader:
                        imageData = None
                        if inferenceMark:
                            pilImage = cv2pil(output_frame)
                            imageData = io.BytesIO()
                            pilImage.save(imageData,'JPEG')
                            imageData = imageData.getvalue()
                            # imageData = open(inferencedImageFile, 'rb')
                            # pilImage = Image.open(imageData)
                        else:
                            imageData = io.BytesIO(request.get_data())
                        fileUploader.upload(imageData, IOTEDGE_DEVICEID, '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now()), pilImage.format.lower())
                        if inferenceMark:
                           imageData.close()
                scored_time = nowTime
                respBody = {
                    'inferences' : detectedObjects
                }
                respBody = json.dumps(respBody)
                logging.info(f'Sending response - {respBody}')
                return Response(respBody, status=200, mimetype='application/json')
            else :
                logging.info('Sending empty response')
                return Response(status=204)
        except Exception as e:
            if configLock.locked():
                configLock.release()
            logging.error(f'exception - {e}')
            return Response(response='Exception occured while processing object detection of the image.', status=500)

    @app.route("/")
    def healty():
        return "Healthy"

    app.run(host='0.0.0.0', port=8888)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
