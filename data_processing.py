from __init__ import *

from  data_loading_exploration import *







class DataProcessing:

			
		#this coars_id_fine_id is from github  https://gist.github.com/adam-dziedzic/4322df7fc26a1e75bee3b355b10e30bc







	def __init__(self):
		#getting the data for two
		
		self.dex= DataExploration()
		(self.X_train,self.y_train),(self.X_test,self.y_test)=self.dex.X_y_test_train()
		
		self.label_names=self.dex.labels()
	def summary(self):
		print(self.label_names)
		# self.dex.display_image_label()
		# #we have data here
	