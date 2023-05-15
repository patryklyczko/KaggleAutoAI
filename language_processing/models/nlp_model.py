import tensorflow as tf
from transformers import TFAutoModel

class NLPModel():
    def __init__(self,seq_len=128) -> None:
        self.embed= None
        self.layers = []
        self.output_layer = None

        self.input_ids = tf.keras.layers.Input(shape=(seq_len,), name='input_ids', dtype='int64')
        self.mask = tf.keras.layers.Input(shape=(seq_len,), name='attention_mask', dtype='int64')

        self.loss = None
        self.optimizer = None
        self.metrics = None

        self.model = None

    def add_layer(self, dimensions=128, activation=None):
        self.layers.append(tf.keras.layers.Dense(dimensions, activation=activation))

    def embedding(self, model_name, layer_extract=0,seq_len=128):        
        model_extract = TFAutoModel.from_pretrained(model_name)
        
        if 'gpt' in model_name or 'roberta' in model_name:
            outputs = model_extract(self.input_ids)
            embeddings = outputs.last_hidden_state
        else:
            outputs = model_extract(self.input_ids, attention_mask=self.mask)
            embeddings = outputs.last_hidden_state
        self.embed = embeddings

    def output(self, dimension=1, activation="sigmoid"):
        self.output_layer = tf.keras.layers.Dense(dimension, activation=activation)
    
    def compile(self, loss, optimizer, metrics):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def fit(self, train, val, epochs=10, layer_untrained=0):
        embed = self.embed
        x = self.layers[0](embed)
        for i in range(1, len(self.layers)):
            x = self.layers[i](x)
        y = self.output_layer(x)
        if self.mask is not None:
            model = tf.keras.Model(inputs=[self.input_ids, self.mask], outputs=y)
        else:
            model = tf.keras.Model(inputs=self.input_ids, outputs=y)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        model.layers[layer_untrained].trainable = False
        model.fit(train, validation_data=val, epochs=epochs)
        self.model = model
    
    def show_compile(self):
        print(f"Loss: {self.loss}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Metrics: {self.metrics}")

    def show_layer(self):
        print(f"Embed: {self.embed}")
        for i, layer in enumerate(self.layers):
            print(f"{i} layer: {layer}")
        print(f"Output: {self.output}")





    