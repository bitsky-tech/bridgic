from abc import ABCMeta, abstractmethod
from typing import List
from bridgic.automa.worker import Worker

class Automa(Worker, metaclass=ABCMeta):
    def all_workers(self) -> List[str]:
        """
        Gets a list containing the names of all workers registered in this Automa.

        Returns
        -------
        List[str]
            A list of worker names.
        """
        ...
        # TODO: Implement this method here.

class GoalOrientedAutoma(Automa):
    @abstractmethod
    def remove_worker(self, name: str) -> None:
        """
        Remove a worker from the Automa. In GoalOrientedAutoma and its subclasses, this method can be called at any time to remove a worker from the Automa.

        Parameters
        ----------
        name : str
            The name of the worker to be removed.

        Returns
        -------
        None

        Raises
        ------
        AutomaDeclarationError
            If the worker specified by worker_name does not exist in the Automa, this exception will be raised.
        """
        ...
