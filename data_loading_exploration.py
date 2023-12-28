from __init__ import *

#this 3 globals lists/dictionary are from this github repo https://gist.github.com/adam-dziedzic/4322df7fc26a1e75bee3b355b10e30bc


fine_labels = [
    'apple',  # id 0
    'aquarium_fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'computer_keyboard',
    'lamp',
    'lawn_mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple_tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak_tree',
    'orange',
    'orchid',
    'otter',
    'palm_tree',
    'pear',
    'pickup_truck',
    'pine_tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet_pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow_tree',
    'wolf',
    'woman',
    'worm',
]

coarse_id_fine_id ={0: [4, 30, 55, 72, 95],
				1: [1, 32, 67, 73, 91],
				2: [54, 62, 70, 82, 92],
				3: [9, 10, 16, 28, 61],
				4: [0, 51, 53, 57, 83],
				5: [22, 39, 40, 86, 87],
				6: [5, 20, 25, 84, 94],
				7: [6, 7, 14, 18, 24],
				8: [3, 42, 43, 88, 97],
				9: [12, 17, 37, 68, 76],
				10: [23, 33, 49, 60, 71],
				11: [15, 19, 21, 31, 38],
				12: [34, 63, 64, 66, 75],
				13: [26, 45, 77, 79, 99],
				14: [2, 11, 35, 46, 98],
				15: [27, 29, 44, 78, 93],
				16: [36, 50, 65, 74, 80],
				17: [47, 52, 56, 59, 96],
				18: [8, 13, 48, 58, 90],
				19: [41, 69, 81, 85, 89]}



coarse_name_fine_name={'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
			'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
			'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
			'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
			'fruit and vegetables': ['apple',
			'mushroom',
			'orange',
			'pear',
			'sweet_pepper'],
			'household electrical device': ['clock',
			'computer_keyboard',
			'lamp',
			'telephone',
			'television'],
			'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
			'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
			'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
			'large man-made outdoor things': ['bridge',
			'castle',
			'house',
			'road',
			'skyscraper'],
			'large natural outdoor scenes': ['cloud',
											'forest',
											'mountain',
											'plain',
										'sea'],
				'large omnivores and herbivores': ['camel',
		'cattle',
		'chimpanzee',
		'elephant',
			'kangaroo'],
			'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
			'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
			'people': ['baby', 'boy', 'girl', 'man', 'woman'],
			'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
			'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
			'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
			'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
			'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']}
					
#what labels are required by CA requirements from cipar100
labels_100=["baby", "boy", "girl", "man", "woman", "rabbit", "squirrel",
 				"trees", "bicycle", "bus", "motorcycle", "pickup_truck","train","lawn_mower","tractor"]

#and what labels are required from cipar 10


cipar10_id_list = [1, 2, 3, 4, 5, 7, 9]



#end of global variables



class DataExploration:
	def __init__(self):
		#cipra100 X,y, test data
		self.X_train_100=None
		self.y_train_100=None
		self.X_test_100=None
		self.y_test_100=None

		#cipra10 X,y, test data
		self.Xtrain_10=None
		self.y_train_10=None
		self.X_test_100=None
		self.y_test_100=None

		#all data in cipra100
		self.cipar_100_train = {}

		#all test data in cipra100
		self.cipar_100_test={}

		#meta data in cipra100
		self.cipar_100_meta={}
		

		#all data in cipra10
		self.cipar_10_train={}
		
		#test data in cipra10
		self.cipar_10_test={}

		#meta data in cipra10
		self.cipar_10_meta={}
		

		
		#Fine labels of cipar100 no-duplicates 
		self.unique_labels_cipar_100=None
		
		#Course labels in cipar100
		self.coarse_labels_cipar_100=None

		#labels in meta cipar 100
		self.unique_labels_meta_cipar_100=None
		
		self.unique_coarse_labels_meta_100=None
		
		#labels in cipar10
		self.unique_labels_cipar_10=None
		
		#labels in meta cipar10
		self.unique_labels_meta_cipar_10=None
		
		#loading data from cipar100
		self._load_cipar_100_data()

		#loading cipar10 batches data from unzipped archive
		#data not in the repo
		self._load_cipar_10_data()
		
		#sorted labels from both sets 100 and 10

		self.them_lot_labels


	def _load_cipar_10_data(self):
		#all these  5 batches  of data are inserted into a dictionary #update(data)
		#data batch 1 cipar 10
		with open('./data/cifar-10-batches-py/data_batch_1', 'rb') as fo:
			batch_1 = pickle.load(fo, encoding='bytes')
			self.cipar_10_train.update(batch_1)

			#data batch 2 cipar 10

		with open('./data/cifar-10-batches-py/data_batch_2', 'rb') as fo:
			batch_2 = pickle.load(fo, encoding='bytes')
			self.cipar_10_train.update(batch_2)
	  

			#data batch3 cipar 10
		with open('./data/cifar-10-batches-py/data_batch_3', 'rb') as fo:
			batch_3 = pickle.load(fo, encoding='bytes')
			self.cipar_10_train.update(batch_3)
			#data batch 4 cipar 10
		with open('./data/cifar-10-batches-py/data_batch_4', 'rb') as fo:
			batch_4 = pickle.load(fo, encoding='bytes')
			self.cipar_10_train.update(batch_4)

			#data batch 5 cipar10
		with open('./data/cifar-10-batches-py/data_batch_5', 'rb') as fo:
			batch_5 = pickle.load(fo, encoding='bytes')
			self.cipar_10_train.update(batch_5)


			#test data  cipar 10
		with open('./data/cifar-10-batches-py/test_batch', 'rb') as fo:
			self.cipar_10_test = pickle.load(fo, encoding='bytes')
			# Meta data for cipar 10
		with open('./data/cifar-10-batches-py/batches.meta', 'rb')as fo:
			self.cipar_10_meta=pickle.load(fo,encoding='bytes')
			
	 	# loading X_train y_train from cipar10 using tensorflow datasets
		(self.X_train_10,self.y_train_10),(self.X_test_10,self.y_test_10)=datasets.cifar10.load_data()

		#ensure labels and other data is initialized
		#keys and labels are initialized in this 
		self.display_data_keys_and_labels()
		self.keep_data_for_labels()

	def _load_cipar_100_data(self):
		# Training data  cipar 100
		with open('./data/cifar-100-python/train', 'rb') as fo:
			self.cipar_100_train = pickle.load(fo, encoding='bytes')
			
			#Test data  cipar 100

		with open('./data/cifar-100-python/test', 'rb') as fo:
			 self.cipar_100_test= pickle.load(fo, encoding='bytes')

			 #meta data  cipar 100

		with open('./data/cifar-100-python/meta', 'rb') as fo:
			self.cipar_100_meta=pickle.load(fo,encoding='bytes')
			#loading X_train y_train, test from cipar100 using tensorflow datasets
		(self.X_train_100,self.y_train_100),(self.X_test_100,self.y_test_100)=datasets.cifar100.load_data()


	# Checking data types
	def display_data_type(self):
	    print("Data type for cipar_100_train: {}".format(type(selfcipar_100_train)))
	    print("Data type for cipar_10_train: {}".format(type(self.cipar_10_train)))
	    for dictItem in self.cipar_100_train:
	    	print("Dictionary Item in cipar_100_train".format(dictItem),type(self.cipar_100_train[dictItem]))
	    for dictItem in self.cipar_10_train:
	    	print("Dictionary Item in cipar_10_train".format(dictItem),type(self.cipar_10_train[dictItem]))


	def display_data_length(self):
		 print("Data length for cipar_100: {}".format(len(self.cipar_100_train)))
		 print("Data length for cipar_10: {}".format(len(self.cipar_10_train)))



	#extracting unique labels of datasets


	def display_data_keys_and_labels(self):

		self.unique_labels_cipar_100=self.cipar_100_train[b'fine_labels']
		# print("XYYYZZZs",self.unique_labels_cipar_100)

		self.coarse_labels_cipar_100=np.unique(self.cipar_100_train[b'coarse_labels'])
		
		self.unique_labels_cipar_10=np.unique(self.cipar_10_train[b'labels'])
		
		self.unique_labels_meta_cipar_100=np.unique(self.cipar_100_meta[b'fine_label_names'])


		
		self.unique_coarse_labels_meta_100=np.unique(self.cipar_100_meta[b'coarse_label_names'])
		
		keys_meta_cipar_10=self.cipar_10_meta.keys()
		keys_meta_cipar_100=self.cipar_100_meta.keys()
		
		self.unique_labels_meta_cipar_10=np.unique(self.cipar_10_meta[b'label_names'])
		print("KEYS FOR TRAINING / META DATA *****************************************************")

		print("Meta:cipar 10 Keys:",keys_meta_cipar_10)
		print("Meta cipar100 keys",keys_meta_cipar_100)

		print("Data keys for cipar_100_train: {}".format(self.cipar_100_train.keys()))
		
		print("Data keys for cipar_10_train: {}".format(self.cipar_10_train.keys()))



		print("************************************************************************************")
		
		print("keys for X_train_100",self.y_train_10[0])
		# print("keys for _y_train_100")

		print("\nLABELS TRAINING DATA/ META DATA********************************")
		fine_labels_length=len(self.unique_labels_meta_cipar_100)

		print("Meta :cipar 100: Label Names length",fine_labels_length)
		
		print("Data COARSE LABELS UNIQUE cipar_100_train",self.coarse_labels_cipar_100)
		
		print("Data LABELS cipar_10_train:",self.unique_labels_cipar_10)

		print("Meta :cipar 100:Fine Label Names",self.unique_labels_meta_cipar_100)
		print("Meta :cipar 100:Coarse Label Names",self.unique_coarse_labels_meta_100)

		print("Meta:cipar 10 :Label Names:",self.unique_labels_meta_cipar_10)
		print("******** X_TRAIN /Y_TRAIN X_TEST/Y_TEST   OBJECT TYPE************")
		print("X_train_100 ",type(self.X_train_100))
		print("y_train_100 ",type(self.y_train_100))
		print("X_test_100 ",type(self.X_test_100))
		print("y_test_100 ",type(self.y_test_100))


		print("X_train_10 ",type(self.X_train_10))
		print("y_train_10 ",type(self.y_train_10))
		print("X_test_10 ",type(self.X_test_10))
		print("y_test_10 ",type(self.y_test_10))

		print("*******************************************************************************")
		print("*********X_train/y_train X_Test/y_test SHAPE")

		print("X_train_100 SHAPE ",self.X_train_100.shape)
		print("y_train_100 SHAPE ",self.y_train_100.shape)
		print("X_test_100 SHAPE ",self.X_test_100.shape)
		print("y_test_100 SHAPE ",self.y_test_100.shape)
		




		print("X_train_10 SHAPE ",self.X_train_10.shape)
		print("y_train_10 SHAPE ",self.y_train_10.shape)
		print("X_test_10 SHAPE ",self.X_test_10.shape)
		print("y_test_10 SHAPE ",self.y_test_10.shape)
		print("batch_label",self.cipar_100_train[b'batch_label'])

	#display data key and labels



	# def display_data_shape(self):
	# 	cipar_100_train_shape=self.cipar_100_train[b'data'].shape
	# 	print("Data shape cipar_100_train :".format())
	# 	print("Data shape cipar_10_train :".format(self.cipar_10_train[b'data'].shape))


	def display_data_shape(self):

		data_sample_cipar_100=self.cipar_100_train[b'data']
		
		print("Data sample cipar_100_train ",data_sample_cipar_100)
		
		data_sample_cipar_10 = self.cipar_10_train[b'data']
		
		print("Data sample cipar_10_train",data_sample_cipar_10)
		
		
		print("Data SHAPE cipar_100_train ",data_sample_cipar_100.shape)
		
		
		print("Data  SHAPE  cipar_10_train",data_sample_cipar_10.shape)

	def display_image_label(self):
		
		index = np.random.randint(0, len(self.X_train_10))
		image = self.X_train_10[index] 
		label = self.y_train_10[index][0]
		
		
		
		label_name=self.unique_labels_meta_cipar_10[label]
		print("keys for X_train_100",self.y_train_10[index])
		 # [b'airplane' b'automobile' b'bird' b'cat' b'deer' b'dog' b'frog' b'horse'
		plt.figure(figsize=(3,3)) 
		plt.imshow(image)
		plt.title(label_name)
		plt.axis('off')
		plt.show()
	
	

	def keep_data_for_labels(self):
		#labels from cipar100

		matched_labels_mapping_100 = {label: index for index, label in enumerate(fine_labels) if label in labels_100}
		print("Matched labels 100",matched_labels_mapping_100)
		#from requirements


		labels_10= self.unique_labels_10()
		matched_labels_mapping_10= {label.decode('utf-8'):index for index, label in enumerate(labels_10) if index in cipar10_id_list}

		# print(matched_labels_mapping_10)
		#now we have unique labels from both sets
		#label is key because is unique. There is an Id wich overlaps in both sets that is NO 2. no 2 exists in cipar_100 labels and in cipar10 labels
		#therefore the labels are the keys, and th value is their id.
		#concatenating both dictionary of cipar100 and cipar10 labels intoa  single one
		# ** unpacking
		# my_array = np.array(list(my_dict.values()))

		self.them_lot_labels={**matched_labels_mapping_100,**matched_labels_mapping_10}
		#convert the labels into an nparry so we can use it to index and filter
		# self.them_lot_labels=np.array(list(self.them_lot_labels.values()))
		
		# self.them_lot_labels_sorted=(dict(sorted(them_lot_labels.items())))

		# print(self.them_lot_labels_sorted)



# baby, boy, girl, man, woman, rabbit, squirrel, trees
# (superclass), bicycle, bus, motorcycle, pickup truck, train, lawn-mower



	def filter_data(self):
		print("them lot labels:",self.them_lot_labels)
		
		#iterating thorugh the array to find the indices of the labels
		# filtered_indices_100 = np.isin(self.unique_labels_meta_cipar_100, list(self.them_lot_labels))
		# filtered_indices_10 = np.isin(self.unique_labels_meta_cipar_10, list(self.them_lot_labels))
		# print(type(filtered_indices_100))
		#from requirements
		# print(labels_100)
		# print(filtered_indices_100)
		# print(cipar10_id_list)
		# print(filtered_indices_10)
		# image = self.X_train_10[index] 
		# label = self.y_train_10[index][0]


		filter_indices = np.array([((self.unique_labels_meta_cipar_100[label[0]]).decode('utf-8')) in self.them_lot_labels for label in self.y_train_100])
		print("New number of samples:", len(self.X_train_100))
# Remove elements from the original arrays in-place
		self.X_train_100 = self.X_train_100[filter_indices]
		self.y_train_100 = self.y_train_100[filter_indices]

		j=0
		# filtered_indices_100=[]
		for index in range(len(self.X_train_100)):
		# 			index = np.random.randint(0, len(self.X_train_10))
			# print("iterate :",j)
			j+=1
			
			
			label=self.y_train_100[index][0]
			label_name=(self.unique_labels_meta_cipar_100[label]).decode('utf-8')
		
			label = self.y_train_100[index][0]
			image = self.X_train_100[index]
		
			plt.figure(figsize=(2,2)) 
			plt.imshow(image)
			plt.title(label_name)
			plt.axis('off')
			plt.show()
	
		print("New number of samples:", len(self.X_train_100))
		
		# label_name=self.unique_labels_meta_cipar_10[label]
		# print("keys for X_train_100",self.y_train_10[index])
		#  # [b'airplane' b'automobile' b'bird' b'cat' b'deer' b'dog' b'frog' b'horse'
		# plt.figure(figsize=(3,3)) 
		# plt.imshow(image)
		# plt.title(label_name)
		# plt.axis('off')
		# plt.show()
			
			
		
		# filtered_indices_100=np.array(filtered_indices_100)		
		# print(len(filtered_indices_100))

		# X_train_100X= self.X_train_100[filtered_indices_100]
		# self.y_train_100=self.y_train_100[filtered_indices_100]
		# self.X_test_100=self.X_test_100[filtered_indices_100]
		# self.y_test_100=self.y_train_100[filtered_indices_100]
	def cipar_100_train(self):
		return self.cipar_100_train

	def cipar_10_train():
		return self.cipar_10_train

	def cipar_100_test():

		return self.cipar_100_test

	def cipar_10_test():
		return self.cipar_10_test


	def get_cipar_100_train(self):
		return self.cipar_100_train
	#unique labels
	def unique_labels(self):
		return self.unique_labels_cipar_100,self.unique_labels_cipar_10

	def unique_labels_10(self):
		return self.unique_labels_meta_cipar_10
	#returning cipra_100 cipra_10 x,y and test
	#already too much code in this file
	#
	def X_y_test_train_100(self):
		return (self.X_train_100,self.y_train_100),(self.X_test_100,self.y_test_100)

	def X_y_test_train_10(self):
		return (self.X_train_10,self.y_train_10),(self.X_test_10,self.y_test_10)

