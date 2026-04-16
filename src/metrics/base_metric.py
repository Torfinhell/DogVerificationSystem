from abc import abstractmethod


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def update(self, **batch):
        """
        Updates internal state of metric
        """
        raise NotImplementedError()
    @abstractmethod
    def compute(self, **batch):
        """
        compute the aggreageted metric from collected interanal states
        """
        raise NotImplementedError()
    @abstractmethod
    def reset(self, **batch):
        """
        Reset internal states to initial
        """
        raise NotImplementedError()
    
