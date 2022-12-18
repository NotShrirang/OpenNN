import numpy as np
from . import layers
from . import losses
from . import optimizers
import os

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

    def fit(self, X, y, epoch: int, optimizer = optimizers.Adam(learning_rate=0.05, decay=5e-7), smooth_output: bool = False,  verbose: bool = False, iteration: int = 10000, validation: tuple = (), callbacks: list = []) -> dict:
        """Method for training the model

        Args:
            X (arrayLike | numpy array): featureset
            y (arrayLike | numpy array): classes
            epoch (int): number of epochs
            optimizer (Optimizer, optional): Optimizer for optimization. Defaults to optimizers.Adam(learning_rate=0.05, decay=5e-7).
            smooth_output (bool, optional): Prints smoother output. Defaults to False.
            verbose (bool, optional): Prints output. Defaults to False.
            iteration (int, optional): number of iterations per epoch. Defaults to 10000.
            validation (tuple, optional): validation testing. Defaults to ().
            callbacks (list, optional): callbacks as EarlyStopping. Defaults to [].

        Returns:
            dict: dictionary of history of metrices
        """
        self.iteration = iteration
        self.history = {'loss' : [], 'accuracy' : [], 'val_loss' : [], 'val_accuracy' : []}
        whole_string = ""
        _optimizer = optimizer
        for epoch_number in range(epoch):
            my_str = ""
            for i in range(self.iteration):
                for count, layer in enumerate(self.layers):
                    if count == 0:
                        layer.forward(X)
                        layer.activation.forward(layer.output)
                        self.__output = layer.activation.output
                    elif (count == (len(self.layers)-1)): # last layer
                        layer.forward(self.__output)
                        self.__loss = layer.activation.forward(layer.output, y)
                        self.__output = layer.activation.output
                    else:
                        layer.forward(self.__output)
                        layer.activation.forward(layer.output)
                        self.__output = layer.activation.output
                
                predictions = np.argmax(self.__output, axis=1)
                if len(y.shape) == 2:
                    y = np.argmax(y, axis=1)
                accuracy = np.mean(predictions==y)
                self.__accuracy = accuracy

                # if not i % 100:
                #     print(f'epoch: {i}, ' +
                #         f'acc: {self.__accuracy:.3f}, ' +
                #         f'loss: {self.__loss:.3f}, ' +
                #         f'lr: {optimizer.current_learning_rate}')

                for count, layer in enumerate(list(reversed(self.layers))):
                    if count == 0:
                        layer.activation.backward(self.__output, y)
                        self.__dinputs = layer.activation.dinputs
                        layer.backward(self.__dinputs)
                        self.__dinputs = layer.dinputs
                    else:
                        layer.activation.backward(self.__dinputs)
                        self.__dinputs = layer.activation.dinputs
                        layer.backward(self.__dinputs)
                        self.__dinputs = layer.dinputs

                _optimizer.pre_update_params()
                for layer in self.layers:
                    _optimizer.update_params(layer)
                _optimizer.post_update_params()
                
                if self.__loss < self.__least_loss:
                    self.__least_loss = self.__loss

                if verbose:
                    if smooth_output:
                        perc_done = i / self.iteration
                        eq_count = int(perc_done * 50)
                        eq_left = 49 - eq_count
                        my_str = str("Epoch " + str(epoch_number+1) + " / " + str(epoch) + " [" + "="*eq_count + ">" + " "*eq_left + "] " +  "loss = " +  "{:.4f}".format(self.__loss) + " accuracy :" + "{:.4f}".format(self.__accuracy))
                        if os.name == 'nt':
                            os.system('cls')
                        else:
                            os.system('clear')
                        whole_string += my_str
                        print(whole_string)
                        whole_string = whole_string[0:-len(my_str)]
                    else:
                        if not i % 100:
                            perc_done = i / self.iteration
                            eq_count = int(perc_done * 50)
                            eq_left = 49 - eq_count
                            my_str = str("Epoch " + str(epoch_number+1) + " / " + str(epoch) + " [" + "="*eq_count + ">" + " "*eq_left + "] " +  "loss = " +  "{:.4f}".format(self.__loss) + " accuracy :" + "{:.4f}".format(self.__accuracy))
                            if os.name == 'nt':
                                os.system('cls')
                            else:
                                os.system('clear')
                            whole_string += my_str
                            print(whole_string)
                            whole_string = whole_string[0:-len(my_str)]
            if validation != ():
                whole_string = whole_string + str("Epoch " + str(epoch_number+1) + " / " + str(epoch) + " [" + "="*50 + "] " +  "loss = " +  "{:.4f}".format(self.__loss) + " accuracy :" + "{:.4f}".format(self.__accuracy))
                ls, acc = self.__validate(validation[0], validation[1])
                whole_string = whole_string + str(" val_loss = ") + "{:.4f}".format(ls) + " val_accuracy = " + "{:.4f}".format(acc) + "\n"
                if os.name == 'nt':
                    os.system('cls')
                else:
                    os.system('clear')
                print(whole_string)

                self.history['loss'].append(self.__loss)
                self.history['accuracy'].append(self.__accuracy)
                self.history['val_loss'].append(ls)
                self.history['val_accuracy'].append(acc)
            else:
                whole_string = whole_string + str("Epoch " + str(epoch_number+1) + " / " + str(epoch) + " [" + "="*50 + "] " +  "loss = " +  "{:.4f}".format(self.__loss) + " accuracy :" + "{:.4f}".format(self.__accuracy) + "\n")
                self.history['loss'].append(self.__loss)
                self.history['accuracy'].append(self.__accuracy)
            
            if callbacks != []:
                call_back = callbacks[0]
                if call_back.check(self.__loss) == 1:
                    break
            # print(str("Epoch " + str(epoch_number+1) + ": loss = " + str(self.__least_loss) + " accuracy = " + str(self.__accuracy)))
        return self.history

    def __validate(self, X_test, y_test) -> tuple():
        __layers = self.layers
        for count, layer in enumerate(__layers):
            if count == 0:
                layer.forward(X_test)
                layer.activation.forward(layer.output)
                __output = layer.activation.output
            elif (count == (len(__layers)-1)): # last layer
                layer.forward(__output)
                __loss = layer.activation.forward(layer.output, y_test)
                __output = layer.activation.output
            else:
                layer.forward(__output)
                layer.activation.forward(layer.output)
                __output = layer.activation.output
        
        predictions = np.argmax(__output, axis=1)
        if len(y_test.shape) == 2:
            y_test = np.argmax(y_test, axis=1)
        __accuracy = np.mean(predictions==y_test)
        return __loss, __accuracy

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
                    layer.forward(X)
                    _output = layer.output.copy()
                    self.__output = layer.activation.forward(_output)
                    self.__output = layer.activation.output
                elif (count == (len(self.layers)-1)): # last layer
                    layer.forward(self.__output)
                    _output = layer.output.copy()
                    layer.activation.activation.forward(_output)
                    __output = layer.activation.activation.output
                    # prediction = np.argmax(self.__output, axis=1)
                else:
                    layer.forward(self.__output)
                    _output = layer.output.copy()
                    activation = layer.activation
                    self.__output = activation.forward(_output)
                    self.__output = activation.output
            predictions.append(__output)
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
