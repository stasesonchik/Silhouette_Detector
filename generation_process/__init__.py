from torch.multiprocessing import Process, Queue, Event
from queue import Full, Empty
from multiprocessing.synchronize import Event as EventClass
import torch
from logging import Logger
import logging
from .resnet_ensamble import Classifier
from config import DETECTOR_DEVICE


def main(name: str,
         classifier_path: str,
         stop_event: EventClass,
         queue_images: Queue,
         queue_attributes: Queue):
    cls = Classifier()
    cls.load_state_dict(torch.load(classifier_path, weights_only=True))
    cls.eval()
    device = DETECTOR_DEVICE
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.DEBUG)
    try:
        cls = cls.to(device)
    except Exception:
        logger.warning(f"No such device: {device}! Classifier model is running on CPU")
        device = "cpu"

    logger.info(f"Classifier model is running on device: {device}")

    while not stop_event.is_set():
        try:
            tracklet = queue_images.get(timeout=1)
            img = tracklet['img']
            logger.debug(f"{name} got img={img}; id={tracklet['id']}")

            attributes = cls.to_json(cls.pred(img, device))
            queue_attributes.put({'id': tracklet['id'], 'desc': attributes})
            logger.debug(f"{name} processed img={img}; id={tracklet['id']}")

        except Empty:
            logger.warning(f"Queue of images is empty")

        except Full:
            logger.warning(f"Queue of texts is full")

        except Exception as e:
            if stop_event.is_set():
                break
            logger.warning(f"Error while processing tracklet: {e}")

    logger.info(f"Finishing {name}")
    exit(0)


class GenerationProcess:
    def __init__(self, name: str, logger: Logger, classifier_path: str):
        self.name = name
        self.logger = logger
        self.classifier_path = classifier_path

        self.__stop_event = Event()
        self.__queue_images = Queue(maxsize=5)
        self.__queue_attributes = Queue(maxsize=5)

        self.__process = None

    def start(self, a):
        self.logger.info(f"Starting {self.name}")
        self.__process = Process(target=main, args=(
            self.name,
            self.classifier_path,
            self.__stop_event,
            self.__queue_images,
            self.__queue_attributes))
        self.__process.daemon = True
        self.__process.start()
        self.logger.info(f"{self.name} process started")

    def send(self, data, timeout: float) -> bool:
        """Отправка данных в очередь изображений."""
        try:
            self.__queue_images.put(data, block=True, timeout=timeout)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to send data to {self.name}: {e}")
            return False

    def receive(self, timeout: float):
        """Получение данных из очереди атрибутов."""
        try:
            data = self.__queue_attributes.get(block=True, timeout=timeout)
            return data, True
        except Exception:
            return None, False

    def stop(self):
        """Останавливает процесс обработки."""
        self.logger.info(f"Stopping {self.name}")
        self.__stop_event.set()

    def terminate(self):
        """Принудительно завершает процесс."""
        self.logger.info(f"Terminating {self.name}")
        if self.__process is not None:
            self.__process.terminate()
            self.__process.join()
            self.logger.info(f"{self.name} is terminated")