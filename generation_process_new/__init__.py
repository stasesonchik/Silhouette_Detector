from torch.multiprocessing import Process, Queue, Event
from queue import Full, Empty
from multiprocessing.synchronize import Event as EventClass
import logging

from .predict import AttributePredictor


class GenerationProcess:
    def __init__(self, name: str, logger: logging.Logger, checkpoint_path: str):
        self.name = name
        self.logger = logger
        self.checkpoint_path = checkpoint_path

        self.__stop_event = Event()
        self.__model_ready = Event()
        self.__queue_images = Queue(maxsize=5)
        self.__queue_attributes = Queue(maxsize=5)

        self.__process = None

    def start(self, detach: bool):
        self.logger.info(f"Starting {self.name}")
        self.__process = Process(
            target=self._main,
            args=(
            self.name,
            self.checkpoint_path,
            self.__stop_event,
            self.__model_ready,
            self.__queue_images,
            self.__queue_attributes
            ),
        )
        self.__process.daemon = True
        self.__process.start()
        self.logger.info(f"{self.name} process started, waiting for model to be ready...")
        self.__model_ready.wait()
        self.logger.info(f"{self.name} model is ready.")

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

    @staticmethod
    def _main(name: str,
              checkpoint_path:str,
              stop_event: EventClass,
              model_ready_event: EventClass,
              queue_images: Queue,
              queue_attributes: Queue):
        """Главная функция процесса: инициализируем предиктор один раз, затем обрабатываем кадры"""
        logger = logging.getLogger(name)
        logging.basicConfig(level=logging.INFO)

        # Инициализируем предиктор единожды
        try:
            predictor = AttributePredictor(checkpoint_path)
            logger.info(f"{name}: AttributePredictor initialized")
            model_ready_event.set()
        except Exception as e:
            logger.error(f"Failed to init predictor {e}")
            return

        #цикл обработки: получаем кадры и предсказываем атрибуты
        while not stop_event.is_set():
            try:
                tracklet = queue_images.get(timeout=1)
                img = tracklet['img']
                track_id = tracklet['id']

                attrs = predictor.predict(img)
                queue_attributes.put({'id': track_id, 'desc': attrs})
            except Empty:
                logger.warning(f"Queue of images is empty for {name}")
                continue
            except Full:
                logger.warning(f"Attributes queue full for {name}")
            except Exception as e:
                logger.error(f"Error in {name} processing: {e}")
                if stop_event.is_set():
                    break

        logger.info(f"Finishing {name}")



