


from  __init__  import *

from data_processing import *

class BuildModel:
    def __init__(self):
            # self.dex= DataExploration()
            # self.label_names=self.dex.labels()
            self.dp=DataProcessing()
            (self.X_train,self.y_train),(self.X_test,self.y_test)=self.dp.apply_preprocess()
            
            
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)
            self.shuffle_data(randomstate=42)
    def to_categorical(self):
        self.y_train = to_categorical(self.y_train, 99)
        self.y_test = to_categorical(self.y_test, 99)
        print("shape", y_train.shape)
    def build_model(self):
      
        model = models.Sequential()



        model.add(layers.Conv2D(32, (4, 4), activation='relu', input_shape=(32, 32, 1)))
        model.add(layers.MaxPooling2D((2, 2)))


        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = "same"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding = "same"))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(250, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(100, activation='softmax'))

    # Compile the model
        model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Compile the model
        model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

        history = model.fit(self.X_train, self.y_train, epochs=25, validation_data=(self.X_valid, self.y_valid), batch_size=400, verbose=1, shuffle=1)
        test_loss, test_acc = model.evaluate(self.X_test, self.y_test)
        print(f'Test accuracy: {test_acc * 100:.2f}%')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(["Trianing", "Validation"])
        plt.title("Loss")
        plt.xlabel("Epoch")



    def shuffle_data(self,randomstate):
        #shuffle from sklearn.util
        #looks like when entering data from other set cifar10 ,it starts overfitting
        self.X_train, self.y_train=shuffle(self.X_train,self.y_train,random_state=randomstate)


# bm=BuildModel()
# bm.build_model()
