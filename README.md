# Openvino Face Recognition with Azure Video Analyzer on Edge

### Create Options  
```json
{
  "NetworkingConfig": {
    "EndpointsConfig": {
      "host": {}
    }
  },
  "HostConfig": {
    "Binds": [
      "/dev:/dev",
      "/facedb:/facedb"
    ],
    "Privileged": true,
    "PortBindings": {
      "8888/tcp": [
        {
          "HostPort": "8888"
        }
      ]
    },
    "NetworkMode": "host",
    "ExposedPorts": {
      "8888/tcp": {}
    }
  }
}
```

### Module Twins  
```json
{
    "send-telemetry": true,
    "upload": {
        "interval-sec": 70,
        "inference-mark": true
    },
    "model-od": {
        "url": "https://egsa20210715.blob.core.windows.net/models/od-model.tgz?sp=r&st=2021-08-26T08:51:05Z&se=2021-09-29T16:51:05Z&spr=https&sv=2020-08-04&sr=b&sig=b31zRHMJJkAUgVuxIaXkMKQsWTimSS4vuhyIdyd%2FAC8%3D",
        "filename": "od-model.tgz",
        "name": "yolo-v2-ava-0001.xml",
        "label": "voc_20cl.txt",
        "architecture-type": "yolo"
    },
    "model-fr": {
        "url": "https://egsa20210715.blob.core.windows.net/models/fr-model.tgz?sp=r&st=2021-08-26T08:49:14Z&se=2021-09-29T16:49:14Z&spr=https&sv=2020-08-04&sr=b&sig=F820D%2FQSVUyFDE6GTNYlJm0iWcYKt4SsuLRpIfmyddk%3D",
        "filename": "fr-model.tgz",
        "fd-name": "face-detection-retail-0004.xml",
        "lm-name": "landmarks-regression-retail-0009.xml",
        "reid-name": "face-reidentification-retail-0095.xml"
    }
}
```

### Environment Variable 
For the case of Raspberry Pi + INTEL Movidius
|Name|Value|
|-|-|
|BLOB_ON_EDGE_MODULE| name of the deployed 'Blob on Edge' module|
|BLOB_ON_EDGE_ACCOUNT_NAME|local account name specifyed in 'Blob on Edge' module|
|BLOB_ON_EDGE_ACCOUNT_KEY|local account key specifyed in 'Blob on Edge' module|
|BLOB_CONTAINER_NAME|container name specifyed in 'Blob on Edge' module|
|FACEDB_FOLDER_NAME|folder name of face database of face recognition. the folder should be same of create option specified|
|OPENVINO_DEVICE|MYRIAD|
|MSG_OUTPUT_PATH_OD|message output path for object detection. ex) output_od|
|MSG_OUTPUT_PATH_FR|message output path for face recognition. ex) output_fr|