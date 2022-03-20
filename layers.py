import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from metrics import MaskedAccuracy

class TagVectorization(tf.keras.layers.Layer):
    def __init__(self):
        super(TagVectorization, self).__init__()
        self.tags = ['UH','WP','PRP',')','DT','JJR','VBG','CD','RBR','JJ','RBS','PRP$', 'RB', 'CC','EX',',','VBN','POS','VBD','NN','``',
                     'PDT','NNP',"''",'MD','RP','FW', 'NNS','SYM','VB','NNPS',
                     'WRB',':','#','TO','$','(','JJS','WP$','WDT','.','VBP','VBZ','IN']
        self.labels = tf.range(0,len(self.tags))
        self.table_tag = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(self.tags, self.labels),
                                              default_value=-1)
        self.table_label = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(self.labels, self.tags),
                                              default_value='UNK')
    def tag2vec(self, tags):
        labels = self.table_tag.lookup(tags)
        
        labels = labels.to_tensor(shape=(labels.shape[0],109))
        labels = tf.pad(labels,tf.constant([[0,0],[1,0]]))
        return labels
    def vec2tag(self, labels, mask):
        labels = tf.cast(labels,dtype=tf.dtypes.int32)
        labels = self.table_label.lookup(labels)
        no_pad = tf.ragged.boolean_mask(labels, mask)[:,1:-1]
        return no_pad
    def call(self,inputs):
        pass


class BertEmbeddings(tf.keras.layers.Layer):
    def __init__(self,bert_preproc, bert_model, seq_length):
        super(BertEmbeddings, self).__init__()
        self.preproc =  hub.load(bert_preproc)
        self.bert = hub.KerasLayer(bert_model,trainable= True)
        self.seq_length = seq_length
        self.tokenize = hub.KerasLayer(self.preproc.tokenize)
        self.bert_pack = bert_pack_inputs = hub.KerasLayer(self.preproc.bert_pack_inputs,
                                                           arguments=dict(seq_length=self.seq_length))
    def call(self, text):
        
        print('boom')
        tokens = self.tokenize(text)
        tokens = tokens[:,:,:1]
        shape = self.bert_out_shape(text)
        tokens = self.make_final_logits(tokens,shape)[:,:,:1]

        encoder_inputs = self.bert_pack([tokens])
        mask = encoder_inputs['input_mask']
        outputs = self.bert(encoder_inputs)['sequence_output']
        
        mask = tf.cast(mask,dtype=tf.bool)

        return outputs, mask
    def bert_out_shape(self,inputs):
        words = tf.strings.split(inputs)
        words_spaced = tf.strings.regex_replace(words, "([^a-zA-Z0-9 ])", r' \1 ')
        bert_input = tf.strings.split(words_spaced)

        return bert_input.nested_row_lengths()
    def make_final_logits(self,logits,shape):
        logits = tf.nest.flatten(logits,expand_composites=True)[0]
        logits = tf.RaggedTensor.from_nested_row_lengths(logits,shape)
        return logits

class Tagger_Dense(tf.keras.layers.Layer):
    def __init__(self,n_tags):
        super(Tagger_Dense, self).__init__()
        self.n_tags = n_tags
        self.dense = tf.keras.layers.Dense(n_tags)
    def call(self,inputs):
        logits = self.dense(inputs)
        return logits
    def get_config(self):
        return {"a": self.n_tags}

class Tagger_Attention(tf.keras.layers.Layer):
    def __init__(self,n_tags):
        super(Tagger_Attention, self).__init__()
        self.n_tags = n_tags
        self.attention = tf.keras.layers.Attention()
    def build(self,input_shape):
        self.W1 = self.add_weight(
            shape=(input_shape[-1], self.n_tags),
            initializer="glorot_uniform",
            trainable=True, name = 'W1')
    def call(self,inputs,mask):
        context = self.attention(inputs = [tf.matmul(inputs,self.W1),tf.matmul(inputs,self.W1)],
                                mask = [mask,mask])
        return context
    def get_config(self):
        return {"a": self.n_tags}

class PosTagger(tf.keras.Model):
    def __init__(self, seq_length, bert_urls, attention= None):
        super(PosTagger, self).__init__()
        self.seq_length = seq_length
        
        self.tagvec = TagVectorization()
        self.n_tags = len(self.tagvec.tags)
        self.bert = BertEmbeddings(bert_urls['preproc'],bert_urls['encoder'],seq_length)
        if attention is None:
            self.tagger = Tagger_Dense(self.n_tags)
        else:
            self.tagger = Tagger_Attention(self.n_tags)
        self.acc = MaskedAccuracy()
    def train_step(self,inputs):
        text, tags = inputs
        targets = self.tagvec.tag2vec(tags)

        with tf.GradientTape() as tape:
            embeddings, mask = self.bert(text)
            if isinstance(self.tagger, Tagger_Attention):
                logits = self.tagger(embeddings,mask)
            else:
                logits = self.tagger(embeddings)

            loss = self.loss(targets,logits)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        labels = tf.argmax(logits,axis=-1, output_type=tf.dtypes.int32)
        self.acc.update_state(targets, labels, mask)
        trained_metrics = {'loss': loss, self.acc.name: self.acc.result()}
    
        return trained_metrics
    def call(self,inputs):
        embeddings, mask = self.bert(inputs)

        
        if isinstance(self.tagger, Tagger_Attention):
            logits = self.tagger(embeddings,mask)
        else:
            logits = self.tagger(embeddings)
        labels = tf.argmax(logits,axis=-1, output_type=tf.dtypes.int32)

        tags = self.tagvec.vec2tag(labels, mask)
        return tags
    @property
    def metrics(self):
        return [self.acc]