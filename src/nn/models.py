import numpy as np
from .layers import Layer
from . import losses
import os

class Sequential:
    def __init__(self, layers: list, loss: losses.Loss = losses.CategoricalCrossentropy) -> None:
        self.layers = layers
        self.loss_function = loss
        self.__loss = 0
        self.__least_loss = 99999
        self.__output = []
        self.__best_weights = []
        self.__best_biases = []

    @property
    def model_loss(self):
        return self.__least_loss

    def fit(self, X, y, epoch: int, smooth_output: bool = False,  verbose: bool = False, iteration: int = 5000) -> list[float]:
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
        self.__best_weights = [layer.weights for layer in self.layers]
        self.__best_biases = [layer.biases for layer in self.layers]
        history = []
        whole_string = ""
        for epoch_number in range(epoch):
            my_str = ""
            for i in range(self.iteration):
                self.__current_weights = []
                self.__current_biases  = []
                weights_list = []
                bias_list = []
                for count, layer in enumerate(self.layers):
                    if count == 0:
                        layer.forward(X)
                        activation = layer.activation
                        self.__output = activation.forward(activation, layer.output)
                    else:
                        layer.weights = layer.weights + 0.05 * np.random.randn(layer.shape[0], layer.shape[1])
                        layer.biases = layer.biases + 0.05 * np.random.randn(1, layer.shape[1])
                        layer.forward(self.__output)
                        activation = layer.activation
                        self.__output = activation.forward(activation, layer.output)
                    weights_list.append(layer.weights)
                    bias_list.append(layer.biases)
                self.__current_weights.append(weights_list)
                self.__current_biases.append(bias_list)
                loss_function = self.loss_function
                self.__loss = loss_function.calculate(loss_function, self.__output, y)
                predictions = np.argmax(self.__output, axis=1)
                accuracy = np.mean(predictions==y)
                self.__accuracy = accuracy
                if self.__loss < self.__least_loss:
                    if verbose:
                        perc_done = i / self.iteration
                        eq_count = int(perc_done * 50)
                        eq_left = 49 - eq_count
                        my_str = f"Epoch {epoch_number+1}/{epoch}" + "[" + "="*eq_count + ">" + " "*eq_left + "] " + f"loss = {self.__loss :.4f}, {accuracy = :.4f}"
                        os.system('cls')
                        whole_string += my_str
                        print(whole_string)
                        whole_string = whole_string[0:-len(my_str)]
                    self.__best_weights = self.__current_weights.copy()
                    self.__best_biases = self.__current_biases.copy()
                    self.__least_loss = self.__loss
                else:
                    if (smooth_output and verbose):
                        perc_done = i / self.iteration
                        eq_count = int(perc_done * 50)
                        eq_left = 49 - eq_count
                        my_str = f"Epoch {epoch_number+1}/{epoch}" + "[" + "="*eq_count + ">" + " "*eq_left + "] " + f"loss = {self.__least_loss :.4f}, {accuracy = :.4f}"
                        os.system('cls')
                        whole_string += my_str
                        print(whole_string)
                        whole_string = whole_string[0:-len(my_str)]
                    for count, layer in enumerate(self.layers):
                        layer.weights = self.__best_weights[0][count]
                        layer.biases = self.__best_biases[0][count]
            whole_string = whole_string + f"Epoch {epoch_number+1}/{epoch}" + "[" + "="*50 + "] " + f"loss = {self.__least_loss :.4f}, accuracy = {self.__accuracy :.4f}" + "\n"
            history.append(self.__least_loss)
            print(f"Epoch {epoch_number+1}: loss = {self.__least_loss}")
        os.system('cls')
        print(whole_string)
        return history

    def predict(self, X):
        """Function for predicions.

        Args:
            X (list | arrayLike): Input data for prediction

        Returns:
            ndarray: predictions array with probability dictionaries.
        """
        predictions = []
        for x_point in X:
            for count, layer in enumerate(self.layers):
                if count == 0:
                    layer.forward(x_point)
                    activation = layer.activation
                    self.__output = activation.forward(activation, layer.output)
                else:
                    layer.forward(self.__output)
                    activation = layer.activation
                    self.__output = activation.forward(activation, layer.output)
            predictions.append(self.__output)
        return predictions

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
