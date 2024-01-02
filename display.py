
from data_loading_exploration import * 


class Display:
  def __init__(self):
    self.dex= DataExploration()
    (self.X_train,self.y_train),(self.X_test,self.y_test)=self.dex.X_y_test_train()

    self.label_names=self.dex.labels()

  def plot_images_per_label(self):
        unique_labels, counts = np.unique(self.y_train, return_counts=True)

        label_names = [self.dex.reverse_dict_lookup(self.dex.them_lot_labels, label) for label in unique_labels]

        plt.bar(label_names, counts)
        plt.xlabel('Label')
        plt.ylabel('Number of Images')
        plt.title('Number of Images per Label')
        plt.xticks(rotation=45, ha="right")
        plt.show()


  def plot_images(self):
        num_of_samples = 5
        unique_labels = np.unique(self.y_train)
        self.X_train =self.X_train / 255.0

        for label in unique_labels:
            X_selected = np.where(self.y_train == label)[0]

            selected_indices = np.random.choice(X_selected, size=num_of_samples)

            plt.figure(figsize=(10, 2))
            for i, index in enumerate(selected_indices, 1):
                plt.subplot(1, num_of_samples, i)
                image = self.X_train[index]
                label_name = self.dex.reverse_dict_lookup(self.dex.them_lot_labels, label)

                plt.imshow(image)
                plt.title(f"Label: {label_name}")
                plt.axis('off')

            plt.show()