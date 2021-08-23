from src.dependencies import *


class NET:
    def __init__(self):
        self.tensorboard_callback = self.cp_callback = None
        self.save_dir = f'models/new_model'
        self.cp_path = self.save_dir + '/Checkpoints/cp.ckpt'
        self.log_dir = f'logs/fit/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.last_finished_epoch = 0

        self.create_callbacks()
        self.model = NET.make_model()
        self.load_data()

        print(self.model.summary())
        print(f'Network successfully built...\n\n')

    @staticmethod
    def make_model():
        from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, \
            LeakyReLU, Reshape, Input
        model = tf.keras.Sequential()
        model.compile(
            loss='mse',
            optimizer='adam',
            metrics=['accuracy']
        )

        return model

    def create_callbacks(self):
        self.cp_dir = os.path.dirname(self.cp_path)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.cp_path,
                                                              save_weights_only=True)
        print('Set up checkpoint path for: {}'.format(self.cp_dir))

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        print('Set up tensorboard log for: {}'.format(self.log_dir))

    def load_data(self):
        dataset = tf.keras.datasets.fashion_mnist.load_data()
        (self.train_x, self.train_labels), (self.test_x, self.test_labels) = self.train_data, self.test_data = dataset
        self.train_x = self.train_x / 255
        self.test_x = self.test_x / 255

        return dataset

    def train(self):

        hist = self.model.fit(
            x=self.train_x,
            y=self.train_x,
            epochs=30,
            validation_data=(self.test_x, self.test_x),
            initial_epoch=self.last_finished_epoch,
            callbacks=[
                self.cp_callback,
                self.tensorboard_callback
            ]
        )

        self.model.save(self.save_dir+'/saved_model')


    def reload_model(self):   # Returns bool True if weights reloaded
        try:
            self.model = tf.keras.models.load_model(self.save_dir+'/saved_model')
            return True

        except Exception as e:
            print(e)
            print('Loading saved weights')
            latest_cp = tf.train.latest_checkpoint(checkpoint_dir=self.cp_dir)
            if latest_cp == None:
                print('No saved weights found, using initialized weights...')
                return False
            else:
                print('Latest saved weights found in:', latest_cp)
                self.model.load_weights(latest_cp)
                return True


