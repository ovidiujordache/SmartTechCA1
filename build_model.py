


from  __init__  import *

from data_loading_exploration import *

class BuildModel:
    def __init__(self):
            self.dex= DataExploration()
            (self.X_train,self.y_train),(self.X_test,self.y_test)=self.dex.X_y_test_train()
            
            self.label_names=self.dex.labels()
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)
            self.shuffle_data()
    def build_model(self):
        shuffle_data()
        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(100, activation='softmax'))  #

    # Compile the model
        model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
        history = model.fit(self.X_train, self.y_train, epochs=10, validation_data=(self.X_valid, self.y_valid))

    # Evaluate the model on the test set
        test_loss, test_acc = model.evaluate(self.X_test, self.y_test)
        print(f'Test accuracy: {test_acc * 100:.2f}%')




    def shuffle_data(self):
        self.X_train = self.X_train.reshape(self.X_train.shape[0], -1)

        # Combine X_train and y_train into a single array with axis=1
        combined_data = np.column_stack((self.X_train, self.y_train))

        # Shuffle the combined array along axis=0
        np.random.shuffle(combined_data)

        # Split the shuffled array back into X_train and y_train
        self.X_train = combined_data[:, :-1].reshape(-1, 32, 32, 3)  # Adjust the reshape according to your actual data dimensions
        self.y_train = combined_data[:, -1]

# bm=BuildModel()
# bm.build_model()
