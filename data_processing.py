from __init__ import *

from  data_loading_exploration import *







class DataProcessing:

	


	def __init__(self):
		#getting the data for two
		
		self.dex= DataExploration()
		(self.X_train,self.y_train),(self.X_test,self.y_test)=self.dex.X_y_test_train()
		
		self.label_names=self.dex.labels()


	def random_brightness(image):
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		rand = random.uniform(0.3, 1.0)
		hsv[:, :, 2] = rand*hsv[:, :, 2]
		new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	def zoom(image):
		zoom_pix = random.randint(0, 10)
		zoom_factor = 1 + (2*zoom_pix)/32
		image = cv2.resize(image, None, fx=zoom_factor,
                       fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
		top_crop = (image.shape[0] - 32)//2
		left_crop = (image.shape[1] - 32)//2
		image = image[top_crop: top_crop+32,
                  left_crop: left_crop+32]
	def flip_vertically(image):
		flip_prob = random.uniform(0, 1)
		flip_new_image = image
		if flip_prob > 0.5:
			flip_new_image = cv2.flip(image, 0)
		return flip_new_image

	def grayscale(img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		return img
	def equalize(img):
		img = cv2.equalizeHist(img)
		return img

