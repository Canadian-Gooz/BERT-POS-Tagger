import tensorflow as tf

class MaskedAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='accuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp',initializer='zeros', dtype = tf.dtypes.int32)
        self.all = self.add_weight(name='all',initializer='zeros',dtype = tf.dtypes.int32)
    def update_state(self, y_true, y_pred, mask, sample_weight= None):
        
        matches = tf.math.equal(y_true,y_pred)
        matches = tf.cast(matches, tf.dtypes.int32) 
        mask = y_pred = tf.cast(mask, tf.dtypes.int32)
        matches *= mask
        self.all.assign_add(tf.reduce_sum(mask))
        self.true_positives.assign_add(tf.reduce_sum(matches))

    def result(self):
        return self.true_positives/self.all

class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.name = 'masked_loss'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        self.reduction = None
    def __call__(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(y_true != 0, tf.float32)
        loss *= mask

        return tf.reduce_sum(loss)