import textwrap
from detector import Detector
#from generation_process import GenerationProcess
#from generation_process.resnet_ensamble import AttDataset
from generation_process_new.model_config import ATT_KEYS, CHECKPOINT_PATH
from generation_process_new import GenerationProcess

import numpy as np
from dataclasses import dataclass, field

from typing import Dict, Any
import traceback
from logging import Logger

import time
from PIL import Image
import random
import cv2


class __UniqueDict__(dict):
    def __eq__(self, other):
        if isinstance(other, __UniqueDict__):
            return self is other
        return False
    def __ne__(self, other):
        return not self.__eq__(other)

MARK_UNPREPARED = __UniqueDict__({k:v for k,v in zip(ATT_KEYS, ['не определен' for _ in range(9)])})  #AttDataset.ATT_KEYS in case of using old generation_process
MARK_FILTERED = __UniqueDict__({k:v for k,v in zip(ATT_KEYS, ['не определен' for _ in range(9)])})
MARK_IN_PROGRESS = __UniqueDict__({k:v for k,v in zip(ATT_KEYS, ['не определен' for _ in range(9)])})

@dataclass
class TrackData:
    scores: Dict[int, float] = field(default_factory=dict)
    colors: Dict[int, tuple] = field(default_factory=dict)
    crops: Dict[int, Image.Image] = field(default_factory=dict)
    attributes: Dict[int, Dict[str, str]] = field(default_factory=dict)
    heartbeat_dict: Dict[int, float] = field(default_factory=dict)
    first_appearance_times: Dict[int, float] = field(default_factory=dict)

    def update_tracks_appearence(self, tracks: list[int]):
        for track_key in tracks:
            if not track_key in self.first_appearance_times:
                self.first_appearance_times[track_key] = time.time()
                self.colors[int(track_key)] = (random.randrange(255),random.randrange(255),random.randrange(255))
                self.attributes[track_key] = MARK_UNPREPARED
            self.heartbeat_dict[track_key] = time.time()

    def update_filtered_dets(self, dets: list, img:np.ndarray, logger: Logger):
            for det in dets:
                track_key = det[4]
                x0, y0, x1, y1 = det[0], det[1], det[2], det[3]
                score = det[6]

                self.scores[track_key] = score
                try:
                    self.crops[track_key] = Image.fromarray(img[y0:y1, x0:x1])
                except Exception as e:
                    logger.warning(f"Error while croping image with x0:{x0}, y0:{y0}, x1:{x1}, y1:{y1}: {e}")
                    self.crops[track_key] = Image.new(mode = "RGB", size = (200, 200), color = (0, 0, 0))
                if MARK_UNPREPARED == self.attributes[track_key]:
                    self.attributes[track_key] = MARK_FILTERED

class SilhouetteDetector(Detector):
    max_age_seconds_td = 60
    TDD: Dict[Any, TrackData] = {}
    TD_heartbeat = {}
    GenPD: Dict[Any, GenerationProcess] = {}
    GenP_heartbeat = {}

    max_age_seconds = 5
    score_th_capture = 0.2
    min_live_seconds = 0.3#2

    def __init__(self, model_path: str, classifier_path: str):
        super().__init__(model_path)
        self.classifier_path = classifier_path

    @staticmethod
    def dets_filter_non_person(dets: list) -> list:
        return list(filter(lambda x: x[5] == 0, dets))

    @staticmethod
    def dets_get_track_indices(dets: list) -> list:
        return [det[4] for det in dets]

    def capture_score(self, score, track_key, task_id) -> bool:
        return not track_key in self.TDD[task_id].scores or score > self.TDD[task_id].scores[track_key] + self.score_th_capture

    def capture_enough_live(self, track_key, task_id) -> bool:
        return track_key in self.TDD[task_id].heartbeat_dict and\
        track_key in self.TDD[task_id].first_appearance_times and\
        self.TDD[task_id].heartbeat_dict[track_key] - self.TDD[task_id].first_appearance_times[track_key] > self.min_live_seconds

    def filter_dets(self, dets: list, task_id: int):
        dets = list(filter(lambda det: self.capture_enough_live(det[4], task_id),
                            dets))
        dets = list(filter(lambda det: self.capture_score(det[6], det[4], task_id), #or self.capture_first_appearence(track_v[1], task_id),
                             dets))
        return dets

    def remove_stale_tracks(self, task_id):
        now = time.time()
        for track_key, last_update_time in list(self.TDD[task_id].heartbeat_dict.items()):
            if now - last_update_time > self.max_age_seconds:
                for data_dict in [self.TDD[task_id].scores,
                                   self.TDD[task_id].heartbeat_dict,
                                   self.TDD[task_id].first_appearance_times,
                                   self.TDD[task_id].colors,
                                   self.TDD[task_id].crops,
                                   self.TDD[task_id].attributes]:
                    try:
                        del data_dict[track_key]
                    except KeyError:
                        pass

    def remove_stale_TD(self):
        now = time.time()
        for td_key, last_update_time in list(self.TD_heartbeat.items()):
            if now - last_update_time > self.max_age_seconds_td:
                try:
                    del self.TDD[td_key]
                    del self.TD_heartbeat[td_key]
                    self.logger.info(f"Task data of task {td_key} is removed")
                except KeyError:
                    self.logger.warning(f"There's no task data of task {td_key}")

    def remove_stale_GenP(self):
        now = time.time()
        for td_key, last_update_time in list(self.GenP_heartbeat.items()):
            if now - last_update_time > self.max_age_seconds_td:
                try:
                    self.GenPD[td_key].stop()
                    self.GenPD[td_key].terminate()
                    del self.GenPD[td_key]
                    del self.GenP_heartbeat[td_key]
                    self.logger.info(f"Generation process of task {td_key} is removed")
                except KeyError:
                    self.logger.warning(f"There's no generation process of task {td_key}")
                except Exception as e:
                    self.logger.error("Error while terminating and removing stale generation process")

    def remove_GenP(self, task_id):
        try:
            self.GenPD[task_id].stop()
            self.GenPD[task_id].terminate()
            del self.GenPD[task_id]
            del self.GenP_heartbeat[task_id]
            self.logger.info(f"Generation process of task {task_id} is removed")
        except KeyError:
            self.logger.warning(f"There's no generation process of task {task_id}")
        except Exception as e:
            self.logger.error("Error while terminating and removing stale generation process")

    def remove_TD(self, task_id):
        try:
            del self.TDD[task_id]
            del self.TD_heartbeat[task_id]
            self.logger.info(f"Task data of task {task_id} is removed")
        except KeyError:
            self.logger.warning(f"There's no task data of task {task_id}")

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if name == "_perform_inference_async" and callable(attr):
            def wrapped(*args, **kwargs):
                result = attr(*args, **kwargs)

                task_id = None
                try:
                    task_id = kwargs['task_id']
                except KeyError:
                    try:
                        task_id = args[1]
                        if not isinstance(task_id, int):
                            raise AssertionError("Task id should be int")
                    except Exception as e:
                        self.logger.error("Can't access to task_id at wrapper of _perform_inference_async\nargs: {args}, kwargs: {kwargs}\n{e}")
                if task_id:
                    self.remove_GenP(task_id)
                    self.remove_TD(task_id)
                else:
                    for task_id in self.GenPD:
                        self.remove_GenP(task_id)
                        self.remove_TD(task_id)

                return result
            return wrapped
        return attr

    def run(
        self, img: np.ndarray, task_id: int
    ) -> list:
        self.TD_heartbeat[task_id] = time.time()
        self.GenP_heartbeat[task_id] = time.time()
        try:
            self.remove_stale_TD()
            self.remove_stale_GenP()
        except Exception as e:
            self.logger.warning("%s:\n%s", e, traceback.format_exc())

        if not task_id in self.TDD:
            self.TDD[task_id] = TrackData()

        if not task_id in self.GenPD:
            self.GenPD[task_id] = GenerationProcess("Silhouette Attributes", self.logger, self.classifier_path)
            self.GenPD[task_id].start(True)

        result_dets = []
        dets = self.dets_filter_non_person(
            Detector.run(self, img, task_id)
            )

        if len(dets) > 0:

            self.TDD[task_id].update_tracks_appearence(
                self.dets_get_track_indices(dets))

            dets_filtered = self.filter_dets(dets, task_id)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.TDD[task_id].update_filtered_dets(dets_filtered, img, self.logger)

            track_keys_filtered = self.dets_get_track_indices(dets_filtered)

            processed_track_keys = []
            try:
                for track_key in track_keys_filtered:
                    crop = self.TDD[task_id].crops[track_key].copy()
                    if self.GenPD[task_id].send(data={'id': track_key, 'img': crop}, timeout=0.01):
                        self.TDD[task_id].attributes[track_key] = MARK_IN_PROGRESS
                    continue
                recv_data, _ = self.GenPD[task_id].receive(timeout=0.01)
                if recv_data:
                    track_key, attributes = recv_data['id'], recv_data['desc']
                    if track_key in self.TDD[task_id].attributes:
                        self.logger.debug(f"Got {track_key}:{attributes}")
                        self.TDD[task_id].attributes[track_key] = attributes
                        processed_track_keys.append(track_key)
            except Exception as e:
                self.logger.error(f"Error while generating person's attributes")
                pass

            for det in dets:
                x0, y0, x1, y1 = det[0], det[1], det[2], det[3]
                track_key = det[4]
                class_id = det[5]
                conf = det[6]
                try:
                    attributes = dict(self.TDD[task_id].attributes[track_key])
                    color = self.TDD[task_id].colors[track_key]
                #не должно происходить
                except Exception as e:
                    attributes = dict(MARK_UNPREPARED)
                    color = (0, 0, 0)
                    self.logger.error(f"Error while acsessing track data by {task_id}")

                result_dets.append([x0, y0, x1, y1, track_key, class_id, conf, attributes, color])
        try:
            self.remove_stale_tracks(task_id)
        except Exception as e:
            self.logger.warning("%s:\n%s", e, traceback.format_exc())

        return result_dets

    @staticmethod
    def attributes_dict2text(attributes):
        res = ""
        for idx, att in enumerate(ATT_KEYS):
            res+=f"{idx+1}. {attributes[att]}\n"
        return res

    @staticmethod
    def draw_results(img: np.ndarray, dets: list) -> np.ndarray:
        inf_img = img.copy()

        #        0   1   2   3   4          5         6     7           8
        #dets: [[x0, y0, x1, y1, track_key, class_id, conf, attributes, color]]
        for det in dets:
            cv2.rectangle(
                img=inf_img,
                pt1=(det[0] + 10, det[1] + 10),
                pt2=(det[2] - 0, det[3] - 10),
                color=det[8],
                thickness=3,
            )
            cv2.putText(
                img=inf_img,
                text=f"ID ({det[4]}) - {round(det[6], 2)}",
                org=(int(det[0]) + 13, int(det[1]) - 13),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1.5,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            height, width, _ = inf_img.shape
            overlay = np.ones_like(inf_img, np.uint8)
            font_size = width/2133#0.9
            font = cv2.FONT_HERSHEY_COMPLEX
            font_thickness = 1
            lines_info = []
            x0, y0, x1, y1, = det[0], det[1], det[2], det[3]
            text = SilhouetteDetector.attributes_dict2text(det[7])
            wrapped_text = textwrap.wrap(text, width=13) #width is chars long //8/21
            wtext_height = (cv2.getTextSize('H', font, font_size, font_thickness)[0][1] + int(width//2/192)) * len(wrapped_text)
            rightStick = True
            if x1+(width//8) <= width:
                overlay = cv2.rectangle(overlay, (x1, y0), (int(x1+(width//8)), max(y1, y0 + wtext_height)), (0,0,0), cv2.FILLED)
            else:
                rightStick = False
                overlay = cv2.rectangle(overlay, (int(x0-(width//8)), y0), (x0, max(y1, y0 + wtext_height)), (0,0,0), cv2.FILLED)
            #x, y = x1 + int(width//2/192), y0 #10
            for i_line, line in enumerate(wrapped_text):
                textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
                gap = textsize[1] + int(width//2/192)#10
                y0+=gap
                y = y0#int((frame2.shape[0] + textsize[1]) / 2) + i_line * gap
                if rightStick:
                    x = x1#+ textsize[0] // 2
                else:
                    x = int(x0-(width//8))
                lines_info.append((line, x, y))
            alpha = 0.5
            mask = ~overlay.astype(bool)
            inf_img[mask] = cv2.addWeighted(inf_img, alpha, overlay, 1 - alpha, 0)[mask]
            for line, x, y in lines_info:
                cv2.putText(inf_img, line, (x, y), font,
                            font_size,
                            (255,255,255),
                            font_thickness,
                            lineType = cv2.LINE_AA)
        return inf_img


if __name__ == "silhouette_detector":
    detector = SilhouetteDetector("./models/yolov8s_576x1024_v2.onnx", CHECKPOINT_PATH)
