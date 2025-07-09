from abc import ABCMeta, abstractmethod
from bridgic.automa.worker import Worker


class Automa(Worker, metaclass=ABCMeta):
    pass

class GoalOrientedAutoma(Automa):
    @abstractmethod
    def remove_worker(self, worker_name: str) -> Worker:
        pass
