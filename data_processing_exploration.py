from __init__ import *


##LABEL 2
cifar10 = tf.keras.datasets.cifar10

# Distribute it to train and test set
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#LABEL 3
cifar100 = tf.keras.datasets.cifar100

# Distribute it to train and test set
#xtrain1 is ciphra100 dataset
(X_train1, y_train1), (X_test1, y_test1) = cifar100.load_data()
def test_data_loaded():
  print(X_train1.shape, y_train1.shape, X_test1.shape, y_test1.shape)
  print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand*hsv[:, :, 2]
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img
def test_image_brightness():
  img = random_brightness(X_train1[420])
  plt.imshow(img)
  plt.show()
  plt.axis("off")

def zoom(image):
    zoom_pix = random.randint(0, 10)
    zoom_factor = 1 + (2*zoom_pix)/32
    image = cv2.resize(image, None, fx=zoom_factor,
                       fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    top_crop = (image.shape[0] - 32)//2
    left_crop = (image.shape[1] - 32)//2
    image = image[top_crop: top_crop+32,
                  left_crop: left_crop+32]
    return image

def test_image_zoom():
  img = zoom(X_train1[400])
  plt.imshow(X_train[1])
  plt.axis("off")
  plt.show()

def flip_vertically(image):
  flip_prob = random.uniform(0, 1)
  flip_new_image = image
  if flip_prob > 0.5:
    flip_new_image = cv2.flip(image, 0)
  return flip_new_image

def test_image_flipV():
  img = flip_vertically(X_train1[400])
  plt.imshow(img)
  plt.axis("off")
  plt.show()


def flip_right_image(image):
  flip_prob = random.uniform(0, 1)
  flip_right_image = image
  if flip_prob > 0.5:
    flip_right_image = cv2.flip(image, 1)
  return flip_right_image

def test_image_flip_right():
  img = flip_right_image(X_train1[400])
  plt.imshow(img)
  plt.axis("off")
  plt.show()


def x_train1_preprocess(img):
  img = random_brightness(img)
  img = zoom(img)
  img = flip_vertically(img)
  img = flip_right_image(img)
  img/255
  return img




X_train1 = np.array(list(map(x_train1_preprocess, X_train1)))
X_test1 = np.array(list(map(x_train1_preprocess, X_test1)))
X_train1 = X_train1.reshape(X_train1.shape[0], 32, 32, 3)
X_test1 = X_test1.reshape(X_test1.shape[0], 32, 32, 3)


augmented_images = [x_train1_preprocess(image) for image in X_train1]

# Convert the list of augmented images to a NumPy array
X_train_augmented = np.array(augmented_images)

X_train1 = np.concatenate((X_train1, X_train_augmented))
y_train1 = np.concatenate((y_train1, y_train1))

#print(X_train1.shape)

x_train_combined = np.concatenate([X_train, X_train1], axis=0)
y_train_combined = np.concatenate([y_train, y_train1 + 10], axis=0)  # Add 10 to CIFAR-100 labels

x_test_combined = np.concatenate([X_test, X_test1], axis=0)
y_test_combined = np.concatenate([y_train, y_train1 + 10], axis=0)

def  test_combined_data():
  print("Combined Training Data Shape:", x_train_combined.shape)
  print("Combined Training Labels Shape:", y_train_combined.shape)
  print("Combined Testing Data Shape:", x_test_combined.shape)
  print("Combined Testing Labels Shape:", y_test_combined.shape)

def test_xtrain1_images():
  fig, ax = plt.subplots(5, 5)
  k = 0

  for i in range(5):
    for j in range(5):
        ax[i][j].imshow(X_train1[k], aspect='auto')
        k += 1

  plt.show()

def display_random_images():
  class_range = range(90, 110)

# Display random images from the specified class range
  cols = 5
  rows = len(class_range) // cols + 1
  fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 2 * rows))
  fig.tight_layout()

  for i, class_label in enumerate(class_range):
    class_indices = np.where(y_train_combined.flatten() == class_label)[0]
    random_index = np.random.choice(class_indices)
    selected_image = x_train_combined[random_index]

    axs[i // cols, i % cols].imshow(selected_image)
    axs[i // cols, i % cols].axis("off")
    axs[i // cols, i % cols].set_title(f"Class {class_label}")

  plt.show()


classes_to_drop = [0, 6, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 46, 47, 48, 49, 50, 52, 53, 54, 55, 59, 60, 61, 63, 64, 65, 67, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 101, 102, 103, 104, 105, 109]
#classes_to_drop = [0, 6, 8]

# Create a mask for training data
mask_train = np.isin(y_train_combined.flatten(), classes_to_drop, invert=True)
#test_mask = np.isin(y_test_combined.flatten(), classes_to_drop, invert=True)

X_new_train = x_train_combined[mask_train]
y_new_train = y_train_combined[mask_train]



#print("Original shapes:", X_train.shape, y_train.shape)
#print("Filtered shapes:", X_new_train.shape, y_new_train.shape)
#class_range = range(10, 20)

# Display random images from the specified class range


# cols = 5
# rows = len(class_range) // cols + 1
# fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 2 * rows))
# fig.tight_layout()

# for i, class_label in enumerate(class_range):
#     x_selected = X_new_train[y_new_train.flatten() == j]
#     if len(x_selected) > 0:
#       axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)- 1), :, :], cmap=plt.get_cmap('gray'))
#       selected_image = X_new_train[random_index]

#     axs[i // cols, i % cols].imshow(selected_image)
#     axs[i // cols, i % cols].axis("off")
#     axs[i // cols, i % cols].set_title(f"Class {class_label}")

# plt.show()

#data = [0,1, 2, 3, 4, 5,6, 7,8, 9]
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'truck', 'cattle', 'fox', 'baby', 'boy', 'girl', 'man', 'woman', 'rabbit', 'squirrel', 'trees', 'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 'lawn-mower', 'tractor']
data = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck',10, 11,'cattle',13, 14, 15, 16, 17, 19, 20, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 46, 47, 48, 49, 50, 52, 53, 54, 55, 59, 60, 61, 63, 64, 65, 67, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 101, 102, 103, 104, 105, 109]


# num_of_samples = []
# cols = 5
# num_classes = len(data)

# fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 7))

# for i in range(cols):
#   for j in range(len(data)):
#     x_selected = X_new_train[y_new_train.flatten() == j]
#     if len(x_selected) > 0:
#       axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)- 1), :, :], cmap=plt.get_cmap('gray'))
#       axs[j][i].axis("off")
#       if i == 2:
#         num_of_samples.append(len(x_selected))
#         axs[j][i].set_title(data[j])

# plt.show

# print(num_of_samples)
# plt.figure(figsize=(12,4))
# plt.bar(range(0, 23), num_of_samples)
# plt.title("Distribution of the training set")
# plt.xlabel("Class number")
# plt.ylabel("Number of images")

# plt.show()

# plt.imshow(X_new_train[40000])
# plt.axis("off")
# plt.show()

# print(X_new_train[40000].shape)
# print(y_new_train[40000])

def grayscale(img):


  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  return img

def test_grescale_image():
  img = grayscale(X_new_train[40000])
  plt.imshow(img)
  plt.axis("off")
  plt.show()

def equalize(img):
  img = cv2.equalizeHist(img)
  return img

# img = equalize(img)
# plt.imshow(img)

def preprocess(img):
  img = grayscale(img)
  img = equalize(img)
  img = img/255
  #same image stacked for 3 channels
 # img = np.stack((img, img, img), axis=-1)
  return img

# plt.imshow(X_new_train[random.randint(0, (len(X_new_train)-1))])
# plt.show()

X_new_train = np.array(list(map(preprocess, X_new_train)))
X_test = np.array(list(map(preprocess, X_test)))

X_new_train = X_new_train.reshape(X_new_train.shape[0], 32, 32, 1)

X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)

# print(X_new_train[:5])

def test_final_data():
  for i in range(15):
    plt.subplot(1, 15, i+1)
    plt.imshow(X_new_train[i])
    plt.axis('off')
    plt.title(f"Sample {i+1}")

  plt.show()

def data_x_train():
  return X_new_train

def data_y_train():
  return y_new_train
