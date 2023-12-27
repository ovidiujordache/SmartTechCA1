from __init__ import *

from  data_loading_exploration import *

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
					




class DataProcessing:

			
		#this coars_id_fine_id is from github  https://gist.github.com/adam-dziedzic/4322df7fc26a1e75bee3b355b10e30bc







	def __init__(self):
		#getting the data for two
		
		self.dex= DataExploration()
		(self.X_train_100,self.y_train_100),(self.X_test_100,self.y_test_100)=self.dex.X_y_test_train_100()
		(self.X_train_10,self.y_train_10),(self.X_test_10,self.y_test_10)=self.dex.X_y_test_train_10()
		#this is combined data
		(self.X_train,self.y_train),(self.X_test,self.y_test)=(0,0),(0,0)

	def summary(self):
		self.dex.display_image_label()
		

	def x_train(self):
		# Data keys for cipar_100_train: dict_keys([b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data'])	
		# print(coarse_id_fine_id)
		# print(coarse_name_fine_name)
		pass


	def keep_data_for_labels(self):
		#labels from cipar100
		labels=["baby", "boy", "girl", "man", "woman", "rabbit", "squirrel",
 				"trees", "bicycle", "bus", "motorcycle", "pickup_truck","train","lawn_mower","tractor"]
		matched_labels_mapping = {index: label for index, label in enumerate(fine_labels) if label in labels}
		print(matched_labels_mapping)

# baby, boy, girl, man, woman, rabbit, squirrel, trees
# (superclass), bicycle, bus, motorcycle, pickup truck, train, lawn-mower