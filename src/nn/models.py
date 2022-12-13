import numpy as np
from . import layers
from . import losses
from . import optimizers
import os
import copy

class Sequential:
    def __init__(self, layers: list = [], loss: losses.Loss = losses.CategoricalCrossentropy) -> None:
        self.layers = layers
        self.loss_function = loss
        self.__loss = 0
        self.__least_loss = 99999
        self.__output = []

    @property
    def model_loss(self):
        return self.__least_loss

    def add(self, layer: layers.Layer):
        self.layers.append(layer)
        return len(self.layers)

    def fit(self, X, y, epoch: int, optimizer = optimizers.Adam(learning_rate=0.05, decay=5e-7), smooth_output: bool = False,  verbose: bool = False, iteration: int = 10000) -> list:
        """Function for training neural network

        Args:
            X (list | arrayLike): data for training
            y (list | arrayLike): prediction for training
            epoch (int): epochs fof training
            smooth_output (bool, optional): If True, outputs the training progressbar smoothly. Defaults to False.
            verbose (bool, optional): If True, outputs the training progressbar. Defaults to False.
            iteration (int, optional): Iteration to find best weights in a single epoch. Defaults to 5000.

        Returns:
            list(float): The list of least loss at epoch
        """
        self.iteration = iteration
        self.history = []
        whole_string = ""
        self.__best_layers = []
        for epoch_number in range(epoch):
            my_str = ""
            optimizer = optimizer
            for i in range(self.iteration):
                self.__current_layers = []
                __layers = []
                for count, layer in enumerate(self.layers):
                    if count == 0:
                        layer.forward(X)
                        activation = layer.activation
                        self.__output = activation.forward(activation, layer.output)
                    elif (count == (len(self.layers)-1)): # last layer
                        layer.forward(self.__output)
                        loss_activation = layer.activation
                        self.__loss = loss_activation.forward(layer.output, y)
                        predictions = np.argmax(loss_activation.output, axis=1)
                        if len(y.shape) == 2:
                            y = np.argmax(y, axis=1)
                        accuracy = np.mean(predictions==y)
                        self.__accuracy = accuracy
                        self.__output_after_forward_pass = loss_activation.output
                    else:
                        layer.forward(self.__output)
                        activation = layer.activation
                        self.__output = activation.forward(activation, layer.output)
                        self.__output = activation.output
                    __layers.append(copy.deepcopy(layer))
                self.__current_layers.append(__layers)

                # if not i % 100:
                #     print(f'epoch: {i}, ' +
                #         f'acc: {self.__accuracy:.3f}, ' +
                #         f'loss: {self.__loss:.3f}, ' +
                #         f'lr: {optimizer.current_learning_rate}')

                for count, layer in enumerate(list(reversed(self.layers))):
                    if count == 0:
                        loss_activation = layer.activation
                        loss_activation.backward(self.__output_after_forward_pass, y)
                        self.__dinputs = loss_activation.dinputs
                        layer.backward(self.__dinputs)
                        self.first__dinputs = layer.dinputs
                    else:
                        activation = layer.activation
                        activation.backward(activation, self.first__dinputs)
                        layer.backward(activation.dinputs)
                        self.__dinputs = layer.dinputs

                _layers = optimizer.optimize(self.layers)
                del self.layers
                self.layers = _layers
                
                if self.__loss < self.__least_loss:
                    self.__least_loss = self.__loss
                    self.__best_layers = copy.deepcopy(self.__current_layers)

                if verbose:
                    if smooth_output:
                        perc_done = i / self.iteration
                        eq_count = int(perc_done * 50)
                        eq_left = 49 - eq_count
                        my_str = f"Epoch {epoch_number+1}/{epoch}" + "[" + "="*eq_count + ">" + " "*eq_left + "] " + f"loss = {self.__loss :.4f}, {self.__accuracy = :.4f}"
                        os.system('cls')
                        whole_string += my_str
                        print(whole_string)
                        whole_string = whole_string[0:-len(my_str)]
                    else:
                        if not i % 100:
                            perc_done = i / self.iteration
                            eq_count = int(perc_done * 50)
                            eq_left = 49 - eq_count
                            my_str = f"Epoch {epoch_number+1}/{epoch}" + "[" + "="*eq_count + ">" + " "*eq_left + "] " + f"loss = {self.__loss :.4f}, {self.__accuracy = :.4f}"
                            os.system('cls')
                            whole_string += my_str
                            print(whole_string)
                            whole_string = whole_string[0:-len(my_str)]

            self.history.append(self.__loss)
            print(f"Epoch {epoch_number+1}: loss = {self.__least_loss}")
        return self.history

    def predict(self, X):
        """Function for predicions.

        Args:
            X (list | arrayLike): Input data for prediction

        Returns:
            ndarray: predictions array with probability dictionaries.
        """
        predictions = []
        for x_point in X:
            for count, layer in enumerate(self.__best_layers[0]):
                if count == 0:
                    layer.forward(x_point)
                    activation = layer.activation
                    self.__output = activation.forward(activation, layer.output)
                elif (count == (len(self.layers)-1)): # last layer
                    layer.forward(self.__output)
                    loss_activation = layer.activation
                    self.__loss = loss_activation.activation.forward(layer.output)
                    prediction = np.argmax(self.__output, axis=1)
                else:
                    layer.forward(self.__output)
                    activation = layer.activation
                    self.__output = activation.forward(activation, layer.output)
            predictions.append(prediction)
        return np.array(predictions)

    def save(self, name: str):
        """Saves the model

        Args:
            name (str, optional): path to folder where model is to be saved.

        Returns:
            bool: If model is saved, return True. Else False.
        """
        if os.path.exists(f"./saved_models"):
            if os.path.exists(f"./saved_models/{name}"):
                import pickle
                with open(f"./saved_models/{name}/model.nn", "wb") as f: 
                    pickle.dump(self, f)
                return True
            elif not os.path.exists(f"./saved_models/{name}"):
                os.mkdir(f"./saved_models/{name}")
                import pickle
                with open(f"./saved_models/{name}/model.nn", "wb") as f: 
                    pickle.dump(self, f)
                return True
        elif not os.path.exists(f"./saved_models"):
            os.mkdir("saved_models")
            os.mkdir(f"./saved_models/{name}")
            import pickle
            with open(f"./saved_models/{name}/model.nn", "wb") as f: 
                pickle.dump(self, f)
            return True
        return False
    
def load_model(model_name: str):
    """Loades model.

    Args:
        model_name (str): you know what it means.

    Raises:
        FileNotFoundError: If folder not found, raises. 

    Returns:
        Sequential: model.
    """
    if os.path.exists(f"./saved_models/{model_name}"):
        import pickle
        with open(f"./saved_models/{model_name}/model.nn", "rb") as f: 
            model: Sequential = pickle.load(f)
        return model
    else:
        raise FileNotFoundError(f"Model {model_name} not found.")
