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

        print(f'Network successfully built...\n\n')

    def submit_latest_model(self):
        self.reload_model()
        x = self.load_data(SUB=True)
        preds = np.argmax(self.model(x), axis=-1)

        with open('data/submission.csv','w') as fhandle:
            fhandle.write('ImageId,Label\n')
            for i, pred in enumerate(preds):
                fhandle.write(str(i+1)+','+str(pred)+'\n')

        self.model.save(self.save_dir+'/latest_subbed_model')
        print('Submission file updated and model saved')


    @staticmethod
    def make_model():
        from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, \
            LeakyReLU, Reshape, Input
        model = tf.keras.Sequential()
        leak = 0.1
        [model.add(layer) for layer in [
            InputLayer(input_shape=(28,28,1)),
            Conv2D(18, kernel_size=(3,3), padding='same'),
            LeakyReLU(leak),
            Conv2D(32, kernel_size=(3,3), padding='same'),
            MaxPooling2D(padding='same'),
            Dropout(0.25),
            Conv2D(32, kernel_size=(3,3), padding='same'),
            MaxPooling2D(padding='same'),
            Dropout(0.25),
            LeakyReLU(leak),
            Conv2D(64, kernel_size=(3,3), padding='same'),
            LeakyReLU(leak),

            Flatten(),
            Dense(256),
            Dropout(0.1),
            LeakyReLU(leak),
            Dense(28),
            Dropout(0.1),
            LeakyReLU(leak),
            Dense(10),
            LeakyReLU(leak),
            Activation('softmax')

        ]]

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )

        return model

    def create_callbacks(self):
        self.cp_dir = os.path.dirname(self.cp_path)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.cp_path,
            save_weights_only=True
        )
        print('Set up checkpoint path for: {}'.format(self.cp_dir))

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1
        )
        print('Set up tensorboard log for: {}'.format(self.log_dir))
        # tensorboard --logdir logs/fit

    def load_data(self, SUB=False):
        print('Loading data..')

        def load_file(SUB=False):
            if SUB: loc = 'data/test.csv'
            else: loc = 'data/train.csv'
            with open(loc) as fhandle:
                csv_reader = csv.reader(fhandle, delimiter=',')
                X = []
                _labels = []
                for i, line in enumerate(csv_reader):
                    if i > 0:
                        if SUB:
                            label = None
                            array = line
                        else:
                            label = line[0]
                            array = line[1:]

                        entry = np.reshape(array, (28,28))
                        X.append(entry)
                        _labels.append(label)

                X = np.array(X).astype('float32')
                if not SUB:
                    _labels = np.array(_labels).astype(int)
                    _labels = tf.keras.utils.to_categorical(_labels)
                else:
                    _labels = np.array(_labels)
                X = X / 255 - 0.5  # Normalize input
                X = np.expand_dims(X, -1)  # Expand dims for convolutions
                return X, _labels

        x, labels = load_file(SUB)
        m = x.shape[0] * (x.shape[0] == labels.shape[0])
        if SUB:
            print('Submission data successfully loaded')
            return x
        else:
            # shuffle
            ind_list = list(range(m))
            random.shuffle(ind_list)
            x = x[ind_list][:,:]
            labels = labels[ind_list]

            # Split the training file 90/10:
            n = int(m * 0.8) # 37800, how many elements of the set go to training
            train_x = x[:n]
            test_x = x[n:]
            train_labels = labels[:n]
            test_labels = labels[n:]
            assert(test_labels.shape[0] == test_x.shape[0])
            assert(train_labels.shape[0] == train_x.shape[0])

            self.train_x = train_x
            self.train_labels = train_labels
            self.test_x = test_x
            self.test_labels = test_labels

            print('Training data successfully loaded')
            return (self.train_x, self.train_labels), (self.test_x, self.test_labels)

    def train(self):
        print(self.test_labels.shape, '\n\n',self.test_labels)
        hist = self.model.fit(
            x=self.train_x,
            y=self.train_labels,
            epochs=30,
            batch_size=64,
            validation_data=(self.test_x, self.test_labels),
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


