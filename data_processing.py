from __init__ import *

from  data_loading_exploration import *







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
		# self.dex.display_image_label()
	
		self.dex.filter_data()

	def x_train(self):
		# Data keys for cipar_100_train: dict_keys([b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data'])	
		# print(coarse_id_fine_id)
		# print(coarse_name_fine_name)
		pass


			