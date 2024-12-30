class HyperparameterTuner:
    def __init__(self, base_model, X_train, y_train, X_val, y_val):
        self.base_model = base_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def tune(self, lstm_units_list, dropout_rates, epochs_list):
        """Optimiert die Hyperparameter."""
        best_model = None
        best_mse = float('inf')

        for lstm_units in lstm_units_list:
            for dropout_rate in dropout_rates:
                for epochs in epochs_list:
                    model = self.base_model(
                        input_shape=self.X_train.shape[1:],
                        lstm_units=lstm_units,
                        dropout_rate=dropout_rate
                    )
                    history = model.train(
                        self.X_train, self.y_train, self.X_val, self.y_val,
                        epochs=epochs
                    )
                    evaluator = ModelEvaluator(model.model)
                    _, y_pred = evaluator.evaluate(self.X_val, self.y_val)
                    mse = mean_squared_error(self.y_val, y_pred)

                    if mse < best_mse:
                        best_mse = mse
                        best_model = model

        print(f"Bestes Modell: MSE = {best_mse}")
        return best_model
