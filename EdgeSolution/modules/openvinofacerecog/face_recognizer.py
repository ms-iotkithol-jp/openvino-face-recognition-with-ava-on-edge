import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import datetime

import cv2
import numpy as np
from openvino.inference_engine import IECore

tmp = str(Path(__file__).resolve().parents[0] / 'lib')
sys.path.append(str(Path(__file__).resolve().parents[0] / 'lib'))

from utils import crop
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
from age_gender_detector import AgeGenderDetector

import monitors
from helpers import resolution
from images_capture import open_images_capture
from models import OutputTransform
from performance_metrics import PerformanceMetrics

log = logging.getLogger()

class FaceRecognizer:
    QUEUE_SIZE = 16

    def __init__(self, ie, device, fg):
        self.ie = ie
        self.gpu_ext = ''
        self.perf_count = False
        # self.allow_grow = args.allow_grow and not args.no_show
        self.allow_grow = False
        # self.d_fd = 'MYRIAD'
        # self.d_lm = 'MYRIAD'
        # self.d_reid = 'MYRIAD'
        self.d_fd = device
        self.d_lm = device
        self.d_reid = device
        self.d_ag = device
        self.fd_input_size = (0,0)
        self.t_fd = 0.6
        self.t_id = 0.3
        self.match_algo = 'HUNGARIAN'
        self.exp_r_fd = 1.15
        self.run_detector = False
        self.fg = fg
        self.scored_time = datetime.datetime.strptime('2000-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')

        self.face_detector = None
        self.landmarks_detector = None
        self.face_identifier = None
        self.faces_database = None
        self.age_gender_detector = None
   
    def LoadModel(self, m_fd, m_lm, m_reid, m_ag=None):
        # m_ag: model for age gender. this feature is optional
        log.info('Loading face recognition networks...')
        self.model_fd = Path(m_fd)
        self.model_lm = Path(m_lm)
        self.model_reid = Path(m_reid)
        self.model_ag = None
        if m_ag:
            self.model_ag = Path(m_ag)
            log.info(' with age gender model...')
        self.face_detector = FaceDetector(self.ie, self.model_fd,
                                          self.fd_input_size,
                                          confidence_threshold=self.t_fd,
                                          roi_scale_factor=self.exp_r_fd)
        self.landmarks_detector = LandmarksDetector(self.ie, self.model_lm)
        self.face_identifier = FaceIdentifier(self.ie, self.model_reid,
                                              match_threshold=self.t_id,
                                              match_algo=self.match_algo)
        self.face_detector.deploy(self.d_fd, self.get_config(self.d_fd))
        self.landmarks_detector.deploy(self.d_lm, self.get_config(self.d_lm), self.QUEUE_SIZE)
        self.face_identifier.deploy(self.d_reid, self.get_config(self.d_reid), self.QUEUE_SIZE)
        self.age_gender_detector = None
        if self.model_ag:
            self.age_gender_detector = AgeGenderDetector(self.ie, self.model_ag)
            self.age_gender_detector.deploy(self.d_ag, self.get_config(self.d_ag), self.QUEUE_SIZE)

        log.info('Building faces database using images from "{}"'.format(self.fg))
        self.faces_database = FacesDatabase(self.fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if self.run_detector else None, True)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered {} identities'.format(len(self.faces_database)))

        return 0

    def get_config(self, device):
        config = {
            "PERF_COUNT": "YES" if self.perf_count else "NO",
        }
        if device == 'GPU' and self.gpu_ext:
            config['CONFIG_FILE'] = self.gpu_ext
        return config

    def Process(self, frame, inferenceMark=False):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. Will be processed only {} of {}'
                        .format(self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        ageGenders = []
        if self.age_gender_detector:
            ageGenders = self.age_gender_detector.infer((frame, rois))
        else:
            for r in rois:
                ageGenders.append(None)

        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                    (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                    (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop_image = crop(orig_image, rois[i])
                name = self.faces_database.ask_to_save(crop_image)
                if name:
                    id = self.faces_database.dump_faces(crop_image, face_identities[i].descriptor, name)
                    face_identities[i].id = id
        
        output_transform = OutputTransform(frame.shape[:2], None)
        results, frame = self.draw_detections(frame, [rois, landmarks, face_identities, ageGenders], output_transform, inferenceMark)

        # output_filename = 'fr-result.bmp'
        #if inferenceMark:
        #    cv2.imwrite(output_filename, frame)
        #    logging.info(f'Saved - {output_filename}')

        return results, frame

    def draw_detections(self, frame, detections, output_transform, inferenceMark):
        size = frame.shape[:2]
        frame = output_transform.resize(frame)
        results = []
        for roi, landmarks, identity, ageGender in zip(*detections):
            text = self.face_identifier.get_identity_label(identity.id)
            identified_label = text
            if identity.id != FaceIdentifier.UNKNOWN_ID:
                text += ' %.2f%%' % (100.0 * (1 - identity.distance))
            if ageGender:
                text += f' {ageGender.gender[0]}'
                text += ' {:.1f}'.format(ageGender.age)

            xmin = max(int(roi.position[0]), 0)
            ymin = max(int(roi.position[1]), 0)
            xmax = min(int(roi.position[0] + roi.size[0]), size[1])
            ymax = min(int(roi.position[1] + roi.size[1]), size[0])
            xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
            if inferenceMark:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)

                for point in landmarks:
                    x = xmin + output_transform.scale(roi.size[0] * point[0])
                    y = ymin + output_transform.scale(roi.size[1] * point[1])
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)
                textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
                cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            identified = {'id':identified_label, 'distance':identity.distance, 'box': { 'l': xmin, 't':ymin, 'w':xmax-xmin, 'h':ymax-ymin } }
            if ageGender:
                identified['gender'] = {'gender':ageGender.gender, 'male_score':float(ageGender.male_score), 'female_score':float(ageGender.female_score)}
                identified['age'] = float(ageGender.age)
            results.append(identified)

        if len(results) > 0:
            self.scored_time = datetime.datetime.now()
            logging.info(f'face recognition scored_time={self.scored_time}')

        return results, frame

    def get_performance_stats(self):
        stats = {
            'face_detector': self.face_detector.get_performance_stats(),
            'landmarks': self.landmarks_detector.get_performance_stats(),
            'face_identifier': self.face_identifier.get_performance_stats(),
        }
        return stats

    def GetScoredTime(self):
        return self.scored_time