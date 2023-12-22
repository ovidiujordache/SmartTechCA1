from __init__ import *



class DataExploration:
	def __init__(self):
		self.cipar_100_train = {}

		
		self.cipar_100_test={}
		
		self.cipar_10_train={}
		
	

		self.cipar_10_test={}
		
		self.cipar_100_meta={}

		#file name is batches.meta
		self.cipar_10_meta={}
		
		self.unique_labels_cipar_100=None
		
		self.coarse_labels_cipar_100=None
		
		self.unique_labels_cipar_10=None
		
		self.unique_labels_meta_cipar_100=None
		
		self.unique_labels_meta_cipar_10=None
		
		self._load_cipar_10_data()
		
		self._load_cipar_100_data()


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
		self.unique_labels_cipar_100=np.unique(self.cipar_100_train[b'fine_labels'])
		
		self.coarse_labels_cipar_100=np.unique(self.cipar_100_train[b'coarse_labels'])
		
		self.unique_labels_cipar_10=np.unique(self.cipar_10_train[b'labels'])
		
		self.unique_labels_meta_cipar_100=np.unique(self.cipar_100_meta[b'fine_label_names'])
		
		keys_meta_cipar_10=self.cipar_10_meta.keys()
		
		self.unique_labels_meta_cipar_10=np.unique(self.cipar_10_meta[b'label_names'])
		print("KEYS FOR TRAINING / META DATA *****************************************************")

		print("Meta:cipar 10 Keys:",keys_meta_cipar_10)


		print("Data keys for cipar_100_train: {}".format(self.cipar_100_train.keys()))
		
		print("Data keys for cipar_10_train: {}".format(self.cipar_10_train.keys()))

		print("\n****************************************************************************************************")
		

		print("\nLABELS TRAINING DATA/ META DATA********************************\n")
		print("Data FINE LABELS UNIQUE cipar_100_train",self.unique_labels_cipar_100)
		
		print("Data COARSE LABELS UNIQUE cipar_100_train",self.coarse_labels_cipar_100)
		
		print("Data LABELS cipar_10_train:",self.unique_labels_cipar_10)

		print("Meta :cipar 100: Label Names",self.unique_labels_meta_cipar_100)

		print("Meta:cipar 10 :Label Names:",self.unique_labels_meta_cipar_10)

		print("\n************************************************************************************************")

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


	def cipar_100_train(self):
		return self.cipar_100_train

	def cipar_10_train():
		return self.cipar_10_train

	def cipar_100_test():

		return self.cipar_100_test

	def cipar_10_test():
		return self.cipar_10_test

	#unique labels
	def unique_labels(self):
		return self.unique_labels_cipar_100,self.unique_labels_cipar_10


	# display_data_keys_and_labels()

	# display_data_shape()
	# display_data_length()
