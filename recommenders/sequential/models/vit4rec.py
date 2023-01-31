from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialRecsysModelBuilder
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense
import tensorflow as tf
from transformers import TFViTModel
from tensorflow.keras import activations


class Vit4Rec(SequentialRecsysModelBuilder):
    VIT_SIZE = 224
    def __init__(self, output_layer_activation='linear', embedding_size=None, max_history_len=VIT_SIZE):    
        super().__init__(output_layer_activation, embedding_size, max_history_len)

    def get_model(self):
        return Vit4RecModel(self.num_items, self.output_layer_activation)



class Vit4RecModel(Model):
    def __init__(self, n_items, activation):
        super().__init__()
        vit_image_size = Vit4Rec.VIT_SIZE
        VIT_EMBEDDING_SIZE=768
        self.embeddings_r = Embedding(n_items+1, vit_image_size)
        self.embeddings_g = Embedding(n_items+1, vit_image_size)
        self.embeddings_b = Embedding(n_items+1, vit_image_size)
        self.projection = Dense(VIT_EMBEDDING_SIZE, input_shape=(3 * vit_image_size,))
        self.n_items = n_items 
        self.all_items = tf.range(0, self.n_items)
        self.vit = TFViTModel.from_pretrained("google/vit-base-patch16-224")
        self.output_activation = activations.get(activation)
    
    def call(self, inputs):
        seqs = inputs[0]
        red = self.embeddings_r(seqs) 
        green = self.embeddings_g(seqs)
        blue = self.embeddings_b(seqs)
        images = tf.tanh(tf.stack([red, green, blue], axis=1))
        encoded = self.vit(images).pooler_output
        all_items_red = self.embeddings_r(self.all_items)
        all_items_green = self.embeddings_r(self.all_items)
        all_items_blue = self.embeddings_r(self.all_items)
        all_embs = self.projection(tf.concat([all_items_red, all_items_green, all_items_blue], axis=1))
        result = self.output_activation(tf.einsum("be, ie -> bi", encoded, all_embs))
        return result


        