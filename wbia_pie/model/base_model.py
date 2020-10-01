# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from tensorflow.keras.callbacks import Callback

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from .backend import (
    DummyNetFeature,
    InceptionV3Feature,
    VGG16Feature,
    ResNet50Feature,
    MobileNetV2Feature,
    DenseNet121Feature,
    DenseNet201Feature,
    # EfficientNetB2Feature,
)
from backend import InceptionResNetV2Feature
from utils import make_batches
from top_models import glob_pool_norm, glob_pool, glob_softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import keras.backend as K


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())



class BaseModel(object):
    def __init__(
        self,
        input_shape,
        backend,
        frontend,
        embedding_size,
        connect_layer=-1,
        train_from_layer=0,
        distance='l2',
        weights='imagenet',
        optimizer='adam',
        use_dropout=False,
    ):
        """Base model consists of backend feature extractor (pretrained model) and a front-end model.

        Input:
        backend: string: one of predefined features extractors. Name matches model name from keras.applications
        input_shape: 3D tuple of integers, shape of input tuple
        frontend: string, name of a function to define top model from top_models.py
        embedding_size: ingeter, size of produced embedding, eg. 256
        connect_layer: integer (positive or negative) or a string: either index of a layer or name of a layer
                        that is used to connect base model with top model
        train_from_layer: integer (positive or negative) or a string: either index of a layer or name of a layer
                           to train the model from.
        distance: string, distance function to calculate distance between embeddings. TODO: implement

        """
        self.input_shape = input_shape
        self.embedding_size = embedding_size
        self.weights = weights
        self.optimizer = optimizer
        self.use_dropout = use_dropout
        self.backend = backend
        self.frontend = frontend
        self.feature_extractor()
        self.connect_layer = self.get_connect_layer(connect_layer)
        self.backend_model()
        self.features_shape()
        self.train_from_layer = self.get_train_from_layer(train_from_layer)
        self.top_model()
        self.distance = distance

    def feature_extractor(self):
        """ Base feature extractor """
        if self.backend == 'InceptionV3':
            self.backend_class = InceptionV3Feature(self.input_shape, self.weights)
        elif self.backend == 'VGG16':
            self.backend_class = VGG16Feature(self.input_shape, self.weights)
        elif self.backend == 'ResNet50':
            self.backend_class = ResNet50Feature(self.input_shape, self.weights)
        elif self.backend == 'InceptionResNetV2':
            self.backend_class = InceptionResNetV2Feature(self.input_shape, self.weights)
        elif self.backend == 'DummyNet':
            self.backend_class = DummyNetFeature(self.input_shape, self.weights)
        elif self.backend == 'MobileNetV2':
            self.backend_class = MobileNetV2Feature(self.input_shape, self.weights)
        elif self.backend == 'DenseNet121':
            self.backend_class = DenseNet121Feature(self.input_shape, self.weights)
        elif self.backend == 'DenseNet201':
            self.backend_class = DenseNet201Feature(self.input_shape, self.weights)
        # elif self.backend == 'EfficientNetB2':
        #     self.backend_class = EfficientNetB2Feature(self.input_shape, self.weights)
        else:
            raise Exception(
                'Architecture is not supported! Use only MobileNet, VGG16, ResNet50, DenseNet201, and Inception3.'
            )

        self.feature_extractor = self.backend_class.feature_extractor

    def normalize_input(self, image):
        """Normalise input to a CNN depending on a backend"""
        return self.backend_class.normalize(image)

    def backend_model(self):
        """ Model to obtain features from a specific layer of feature extractor."""
        self.backend_model = Model(
            inputs=self.feature_extractor.get_input_at(0),
            outputs=self.feature_extractor.layers[self.connect_layer].get_output_at(0),
            name='features_model',
        )

    def features_shape(self):
        self.features_shape = self.backend_model.get_output_shape_at(0)[1:]
        print('Shape of base features: {}'.format(self.features_shape))

    def preproc_predict(self, imgs, batch_size=32, augmentation_seed=None):
        """Preprocess images and predict with the model (no batch processing for first step)
        Input:
        imgs: 4D float or int array of images
        batch_size: integer, size of the batch
        Returns:
        predictions: numpy array with predictions (num_images, len_model_output)
        """
        print('base_model preproc_predict!')
        # import utool as ut
        # ut.embed()
        batch_idx = make_batches(imgs.shape[0], batch_size)
        imgs_preds = np.zeros((imgs.shape[0],) + self.model.get_output_shape_at(0)[1:])
        print('Computing predictions with the shape {}'.format(imgs_preds.shape))

        # do some augmentation here
        use_augmentation = augmentation_seed is not None
        print('use_augmentation = %s and augmentation_seed = %s' % (use_augmentation, augmentation_seed))
        if use_augmentation:
            gen_args = dict(
                rotation_range=30,
                width_shift_range=0.15,
                height_shift_range=0.15,
                shear_range=0.1,
                zoom_range=0.15,
                channel_shift_range=0.15,
                data_format=K.image_data_format(),
                fill_mode='reflect',
                preprocessing_function=self.backend_class.normalize,
            )
            aug_gen = ImageDataGenerator(**gen_args)

        for sid, eid in batch_idx:
            if use_augmentation:
                # [0] found experimentally
                preproc = aug_gen.flow(imgs[sid:eid], batch_size=batch_size, seed=augmentation_seed)
                assert len(preproc) == 1
                assert len(preproc[0]) <= batch_size
                preproc = preproc[0]
            else:
                preproc = self.backend_class.normalize(imgs[sid:eid])
            imgs_preds[sid:eid] = self.model.predict_on_batch(preproc)

        print('imgs_preds = %s' % imgs_preds)

        return imgs_preds

    def top_model(self, verbose=1):
        """Model on top of features."""
        if self.frontend == 'glob_pool_norm':
            self.top_model = glob_pool_norm(
                embedding_size=self.embedding_size, backend_model=self.backend_model
            )
        elif self.frontend == 'glob_pool':
            self.top_model = glob_pool(
                embedding_size=self.embedding_size, backend_model=self.backend_model, use_dropout=self.use_dropout
            )
        elif self.frontend == 'glob_softmax':
            self.top_model = glob_softmax(
                embedding_size=self.embedding_size, backend_model=self.backend_model
            )
        else:
            raise Exception('{} is not supported'.format(self.frontend))

        # Freeze layers as per config
        self.set_trainable()

    def get_connect_layer(self, connect_layer):
        """If connect_layer is a string (layer name), return layer index.
        If connect layer is a negative integer, return positive layer index."""
        index = None
        if isinstance(connect_layer, str):
            for idx, layer in enumerate(self.feature_extractor.layers):
                if layer.name == connect_layer:
                    index = idx
                    break
        elif isinstance(connect_layer, int):
            if connect_layer >= 0:
                index = connect_layer
            else:
                index = connect_layer + len(self.feature_extractor.layers)
        else:
            raise ValueError
            print('Check type of connect_layer')
        print(
            'Connecting layer {} - {}'.format(
                index, self.feature_extractor.layers[index].name
            )
        )
        return index

    def get_train_from_layer(self, train_from_layer):
        """If train_from_layer is a string (layer name), return layer index.
        If train_from_layer layer is a negative integer, return positive layer index."""
        index = None
        if isinstance(train_from_layer, str):
            for idx, layer in enumerate(self.feature_extractor.layers):
                if layer.name == train_from_layer:
                    index = idx
                    break
        if isinstance(train_from_layer, int):
            if train_from_layer >= 0:
                index = train_from_layer
            else:
                index = train_from_layer + len(self.feature_extractor.layers)
        print(
            'Train network from layer {} - {}'.format(
                index, self.feature_extractor.layers[index].name
            )
        )
        return index

    def load_weights(self, weight_path, by_name=False):
        self.model.load_weights(weight_path, by_name)

    def set_all_layers_trainable(self):
        for i in range(len(self.top_model.layers)):
            self.top_model.layers[i].trainable = True

    def set_trainable(self):
        self.set_all_layers_trainable()
        for i in range(self.train_from_layer):
            self.top_model.layers[i].trainable = False
        print(
            'Layers are frozen as per config. Non-trainable layers are till layer {} - {}'.format(
                self.train_from_layer, self.top_model.layers[self.train_from_layer].name
            )
        )

    def warm_up_train(
        self,
        train_gen,
        valid_gen,
        nb_epochs,
        batch_size,
        learning_rate,
        steps_per_epoch,
        distance='l2',
        saved_weights_name='best_weights.h5',
        logs_file='history.csv',
        plot_file='plot.png',
        debug=False,
    ):
        """Train only randomly initialised layers of top model"""
        # Freeze base model
        self.set_all_layers_trainable()

        backend_model_len = len(self.backend_model.layers)
        print('Freezeing layers before warm-up training')
        for i in range(backend_model_len):
            self.top_model.layers[i].trainable = False
        # for layer in self.top_model.layers:
            # print(layer.name, layer.trainable)

        # Compile the model
        self.compile_model(learning_rate)

        # Warm-up training
        csv_logger = CSVLogger(logs_file, append=True)
        callbacks = [csv_logger]

        if self.optimizer == 'sgd':
            clr = CyclicLR(base_lr=learning_rate, max_lr=0.0001, step_size=2000., mode='triangular2')
            callbacks.append(clr)

        self.model.fit_generator(
            generator=train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epochs,
            verbose=2 if debug else 1,
            validation_data=valid_gen,
            validation_steps=steps_per_epoch // 5 + 1,
            callbacks=callbacks,
            max_queue_size=32,
            workers=21,
            use_multiprocessing=False,
        )

        self.top_model.save_weights(saved_weights_name)

        # Freeze layers as per config
        self.set_trainable()

    def train(
        self,
        train_gen,
        valid_gen,
        nb_epochs,
        batch_size,
        learning_rate,
        steps_per_epoch,
        distance='l2',
        saved_weights_name='best_weights.h5',
        logs_file='history.csv',
        debug=False,
        weights=None,
    ):

        # Compile the model
        if weights is None:
            self.compile_model(learning_rate)
        else:
            self.compile_model(learning_rate, weights=weights)

        # Make a few callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,  # changed from 5 h/t JP  # changed from 3
            min_delta=0.001,
            mode='min',
            verbose=1,
        )
        checkpoint = ModelCheckpoint(
            saved_weights_name,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            period=1,
        )
        csv_logger = CSVLogger(logs_file, append=True)
        callbacks = [early_stop, checkpoint, csv_logger]

        if self.optimizer == 'sgd':
            clr = CyclicLR(base_lr=learning_rate, max_lr=0.006, step_size=2000., mode='triangular2')
            callbacks.append(clr)

        ############################################
        # Start the training process
        ############################################
        self.model.fit_generator(
            generator=train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epochs,
            verbose=1 if debug else 2,
            validation_data=valid_gen,
            validation_steps=steps_per_epoch // 5 + 1,
            callbacks=callbacks,
            max_queue_size=32,
            workers=21,
            use_multiprocessing=False,
        )

    def precompute_features(self, imgs, batch_size):
        imgs = self.backend_class.preprocess_imgs(imgs)
        features = self.backend_model.predict(imgs, batch_size)
        return features
