import numpy as np

class EarlyStopping:
    def __init__(self, moniter = 'val_loss', min_delta: float = 0, patience: int = 0) -> None:
        self.__moniter_metric = moniter
        self.__min_delta = min_delta
        self.__patience = patience
        self.__curr_epoch = 0
        self.__metric_list = []
    def check(self, loss):
        self.__metric_list.append(loss)
        self.__curr_epoch += 1
        if len(self.__metric_list) > 1:
            diff = np.absolute(self.__metric_list[-1] - self.__metric_list[0])
            if diff <= self.__min_delta:
                if self.__curr_epoch >= self.__patience + 1:
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            return 0
