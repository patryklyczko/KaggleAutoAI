import tensorflow as tf
from sklearn.metrics import log_loss
import optuna.visualization as vis
from ...classification.metrics import Metrics, accuracy_score
from transformers import TFAutoModel
import optuna
import numpy as np
optuna.logging.disable_default_handler()


class TransformerModel():
    def __init__(self, num_classes, seq_len=128, bert_model_name='bert-base-uncased'):
        self.classes = num_classes
        self.bert_model = TFAutoModel.from_pretrained(bert_model_name)
        self.layers = []
        self.seq_len = seq_len
        self.model = None

    def add_dense(self, dimensions=128, activation="relu"):
        self.layers.append(tf.keras.layers.Dense(dimensions, activation=activation))

    def build(self):
        input_ids = tf.keras.Input(shape=(self.seq_len, ),name="input_ids", dtype=tf.int32)
        attention_mask = tf.keras.Input(shape=(self.seq_len, ),name="attention_mask", dtype=tf.int32)

        bert = self.bert_model([input_ids, attention_mask])
        x = bert[1]
        for layer in self.layers:
            x = layer(x)
        if self.classes < 2:
            y = tf.keras.layers.Dense(self.classes, activation='sigmoid')(x)
        else:
            y = tf.keras.layers.Dense(self.classes, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=y)

    def compile(self, loss, optimizer, metrics):
        self.build()
        self.model.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=metrics)
    
    def fit(self, X_data,X_label, val_data, val_label, epochs=1, batch=32, layer_untrained=-1):
        if layer_untrained >= 0:
            self.model.layers[layer_untrained].trainable = False
        self.model.fit(X_data, X_label,
                       validation_data=(val_data, val_label),
                       epochs=epochs)
        
    def fit_optuna(self, X_data, X_label, val_data, val_label, batch=32, n_trials=10,
                   layer_untrained=-1, params=None, show_plot=False, show_features=False):
        params_columns = ["layers","layers_dimensions",]
        params_basic = {
            "layers": [2, 10],
            "layers_dimensions": [32, 1024],
        }
        if params is None:
            params = params_basic
        else:
            for parameter in params_columns:
                if parameter not in params.keys():
                    params[parameter] = params_basic[parameter]
        
        def objective(trial):
            self.reset()
            layers = trial.suggest_int("Layers", params["layers"][0], params["layers"][1])
            for i in range(layers):
                self.add_dense(trial.suggest_int(f'Layer_{i}', params["layers_dimensions"][0], params["layers_dimensions"][1]))
            self.fit(X_data, X_label, val_data, val_label,batch,layer_untrained)
            preds = self.predict(val_data)
            acc = accuracy_score(val_label, preds)
            return acc
        
        study = optuna.create_study(direction="maximize",pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        if show_plot:
            optimization_history_plot = vis.plot_optimization_history(study)
            optimization_history_plot.show()
        if show_features:
            param_importance_plot = vis.plot_param_importances(study)
            param_importance_plot.show()
            
        best_params = study.best_trial.params
        self.reset()
        for i in range(best_params["Layers"]):
            self.add_dense(best_params[f"Layer_{i}"])
        self.fit(X_data, X_label, val_data, val_label,batch,layer_untrained)
                
    def predict(self, data):
        return self.model.predict(data)
    
    def summary(self):
        self.model.summary()
        
    def reset(self):
        self.layers = []





    