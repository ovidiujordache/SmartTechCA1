from __init__ import *

from  data_loading_exploration import *







class DataProcessing:

	


	
	def __init__(self):
		#getting the data for two
		
		self.dex= DataExploration()
		(self.X_train,self.y_train),(self.X_test,self.y_test)=self.dex.X_y_test_train()
		
		# self.cipar_100_train=self.dex.cipar_100_train()
		# self.cipar_100_test=self.dex.cipar_100_test()

		# self.cipar_10_train=self.dex.cipar_10_train()
		# sef.cipar_10_test=self.dex.cipar_10_test()
		

		self.label_names=self.dex.labels()
		

	
	def random_brightness(self,image):
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		rand = random.uniform(0.3, 1.0)
		hsv[:, :, 2] = rand*hsv[:, :, 2]
		new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	


	def zoom(self,image):
		zoom_pix = random.randint(0, 10)
		zoom_factor = 1 + (2*zoom_pix)/32
		image = cv2.resize(image, None, fx=zoom_factor,
                       fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
		top_crop = (image.shape[0] - 32)//2
		left_crop = (image.shape[1] - 32)//2
		image = image[top_crop: top_crop+32,
                  left_crop: left_crop+32]
	
	def flip_vertically(self,image):
		flip_prob = random.uniform(0, 1)
		flip_new_image = image
		if flip_prob > 0.5:
			flip_new_image = cv2.flip(image, 0)
		return flip_new_image

	def grayscale(self,img):
		img = (img * 255).astype(np.uint8)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		return img
	def equalize(self,img):
		img = cv2.equalizeHist(img)
		return img


	def gaussian(self,img):
		img = cv2.GaussianBlur(img, (3,3), 0)
		return img

	def preprocess(self,img):
  		img = self.grayscale(img)
  		img = self.equalize(img)
  		img = img/255
  		#same image stacked for 3 channels
 		# img = np.stack((img, img, img), axis=-1)
  		return img




	def apply_preprocess(self):
		self.X_train = np.array(list(map(self.preprocess, self.X_train)))
		self.X_test = np.array(list(map(self.preprocess, self.X_test)))
		self.X_train = self.X_train.reshape(self.X_train.shape[0], 32, 32, 1)
		self.X_test = self.X_test.reshape(self.X_test.shape[0], 32, 32, 1)


		return (self.X_train,self.y_train),(self.X_test,self.y_test)
	def shape(self):
		print("ytrain shape", self.y_train.shape)