#!/usr/bin/env python3
"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import colorsys
import logging
import random
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
import datetime

import cv2
import numpy as np
from openvino.inference_engine import IECore

# sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[0] / 'lib'))

import models
import monitors
from pipelines import get_user_config, AsyncPipeline
from images_capture import open_images_capture
from performance_metrics import PerformanceMetrics
from helpers import resolution

log = logging.getLogger()

class ObjectDetector:
    def __init__(self, ie, device):
        self.ie = ie
        self.architecture_type = None
        self.reverse_input_channels = False
        self.mean_values = None
        self.scale_values = None
        self.keep_aspect_ratio = False
        self.input_size = (600, 600)
        self.prob_threshold = 0.5
        self.keep_aspect_ratio = False
        # self.device = 'MYRIAD'
        self.device = device
        self.num_streams = ''
        self.num_threads = None
        self.num_infer_requests = 0
        self.output_resolution = None
        self.plugin_config = get_user_config(self.device, self.num_streams, self.num_threads)
        self.scored_time = datetime.datetime.strptime('2000-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')

    class ColorPalette:
        def __init__(self, n, rng=None):
            assert n > 0

            if rng is None:
                rng = random.Random(0xACE)

            candidates_num = 100
            hsv_colors = [(1.0, 1.0, 1.0)]
            for _ in range(1, n):
                colors_candidates = [(rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0))
                                     for _ in range(candidates_num)]
                min_distances = [self.min_distance(hsv_colors, c) for c in colors_candidates]
                arg_max = np.argmax(min_distances)
                hsv_colors.append(colors_candidates[arg_max])

            self.palette = [self.hsv2rgb(*hsv) for hsv in hsv_colors]

        @staticmethod
        def dist(c1, c2):
            dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
            ds = abs(c1[1] - c2[1])
            dv = abs(c1[2] - c2[2])
            return dh * dh + ds * ds + dv * dv

        @classmethod
        def min_distance(cls, colors_set, color_candidate):
            distances = [cls.dist(o, color_candidate) for o in colors_set]
            return np.min(distances)

        @staticmethod
        def hsv2rgb(h, s, v):
            return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

        def __getitem__(self, n):
            return self.palette[n % len(self.palette)]

        def __len__(self):
            return len(self.palette)

    def LoadModel(self, modelFile, labels, archtype, num_infer_requests=0):
        self.architecture_type = archtype # 'ssd', 'yolo', 'yolov4', 'faceboxes', 'centernet', 'ctpn', 'retinaface', 'ultra_lightweight_face_detection', 'retinaface-pytorch'
        self.model = self.get_model(modelFile, labels)
        self.detector_pipeline = AsyncPipeline(self.ie, self.model, self.plugin_config,
                                      device=self.device, max_num_requests=self.num_infer_requests)
        self.palette = ObjectDetector.ColorPalette(len(self.model.labels) if self.model.labels else 100)

        return 0

    def get_model(self, model, labels):
        input_transform = models.InputTransform(self.reverse_input_channels, self.mean_values, self.scale_values)
        common_args = (self.ie, model, input_transform)
        if self.architecture_type in ('ctpn', 'yolo', 'yolov4', 'retinaface',
                                      'retinaface-pytorch') and not input_transform.is_trivial:
            raise ValueError("{} model doesn't support input transforms.".format(self.architecture_type))

        if self.architecture_type == 'ssd':
            return models.SSD(*common_args, labels=labels, keep_aspect_ratio_resize=self.keep_aspect_ratio)
        elif self.architecture_type == 'ctpn':
            return models.CTPN(self.ie, model, input_size=self.input_size, threshold=self.prob_threshold)
        elif self.architecture_type == 'yolo':
            return models.YOLO(self.ie, model, labels=labels,
                               threshold=self.prob_threshold, keep_aspect_ratio=self.keep_aspect_ratio)
        elif self.architecture_type == 'yolov4':
            return models.YoloV4(self.ie, model, labels=labels,
                                 threshold=self.prob_threshold, keep_aspect_ratio=self.keep_aspect_ratio)
        elif self.architecture_type == 'faceboxes':
            return models.FaceBoxes(*common_args, threshold=self.prob_threshold)
        elif self.architecture_type == 'centernet':
            return models.CenterNet(*common_args, labels=labels, threshold=self.prob_threshold)
        elif self.architecture_type == 'retinaface':
            return models.RetinaFace(self.ie, args.model, threshold=self.prob_threshold)
        elif self.architecture_type == 'ultra_lightweight_face_detection':
            return models.UltraLightweightFaceDetection(*common_args, threshold=self.prob_threshold)
        elif self.architecture_type == 'retinaface-pytorch':
            return models.RetinaFacePyTorch(self.ie, model, threshold=self.prob_threshold)
        else:
            raise RuntimeError('No model type or invalid model type (-at) provided: {}'.format(self.architecture_type))

    def Score(self, frame, inferenceMark=False):
        frame_id = 0
        output_resolution = None
        if self.detector_pipeline.callback_exceptions:
            raise self.detector_pipeline.callback_exceptions[0]
        if self.detector_pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            output_transform = models.OutputTransform(frame.shape[:2], self.output_resolution)
            output_resolution = (frame.shape[1], frame.shape[0])
            # Submit for inference
            self.detector_pipeline.submit_data(frame, frame_id, {'frame': frame, 'start_time': start_time})
        else:
            # Wait for empty request
            detector_pipeline.await_any()
        detectedObjects = []
        results = self.detector_pipeline.get_result(frame_id)
        if results:
            objects, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']
            frame = self.draw_detections(frame, objects, self.prob_threshold, output_transform, detectedObjects, inferenceMark)

        if len(detectedObjects) > 0:
            self.scored_time = datetime.datetime.now()
            logging.info(f'object detection scored_time={self.scored_time}')

        # output_filename = 'od-result.bmp'
        # if inferenceMark:
        #    cv2.imwrite(output_filename, frame)
        #    logging.info(f'Saved - {output_filename}')

        return detectedObjects, frame

    def draw_detections(self, frame, detections, threshold, output_transform, detectedObjects, inferenceMark):
        size = frame.shape[:2]
        labels = self.model.labels
        frame = output_transform.resize(frame)
        for detection in detections:
            if detection.score > threshold:
                class_id = int(detection.id)
                color = self.palette[class_id]
                det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
                xmin = max(int(detection.xmin), 0)
                ymin = max(int(detection.ymin), 0)
                xmax = min(int(detection.xmax), size[1])
                ymax = min(int(detection.ymax), size[0])
                xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
                detected = { 'type':'entity', 'entity': { 'tag': { 'value': det_label, 'confidence' : float(detection.score) }, 'box': { 'l': xmin, 't':ymin, 'w':xmax-xmin, 'h':ymax-ymin } } }
                detectedObjects.append(detected)
                if inferenceMark:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                                (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
                    if isinstance(detection, models.DetectionWithLandmarks):
                        for landmark in detection.landmarks:
                            landmark = output_transform.scale(landmark)
                            cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 255), 2)
        return frame

    def print_raw_results(self, size, detections, threshold):
        labels = self.model.labels
        log.info(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')
        for detection in detections:
            if detection.score > threshold:
                xmin = max(int(detection.xmin), 0)
                ymin = max(int(detection.ymin), 0)
                xmax = min(int(detection.xmax), size[1])
                ymax = min(int(detection.ymax), size[0])
                class_id = int(detection.id)
                det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
                log.info('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '
                         .format(det_label, detection.score, xmin, ymin, xmax, ymax))

    def GetScoredTime(self):
        return self.scored_time