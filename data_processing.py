from __init__ import *

from  data_loading_exploration import *


class DataProcessing:





	def __init__(self):
		#getting the data for two
		self.dex= DataExploration()
		(self.X_train_100,self.y_train_100),(self.X_test_100,self.y_test_100)=self.dex.X_y_test_train_100()
		(self.X_train_10,self.y_train_10),(self.X_test_10,self.y_test_10)=self.dex.X_y_test_train_10()
		#this is combined data
		(self.X_train,self.y_train),(self.X_test,self.y_test)=(0,0),(0,0)

	def set_data(self):
		self.dex.display_image_label()
		self.dex.map_labels()
		# self.X_train
		
# automobile, bird, cat, deer, dog, horse, truck cipar10

#cipar100
# cattle, fox, baby, boy, girl, man, woman, rabbit, squirrel, !!!! trees!!!!!
# , bicycle, bus, motorcycle, pickup_truck, train, lawn_mower and tractor 





 # 1 .bird, 2.cat, 3.deer, 4.dog, 5.horse, 6.truck, 7.cattle, 8.fox, 9.baby, 10,boy, 11.girl, 12.man, 13.woman, 14.rabbit, 15.squirrel,
 #16.trees, 17.bicycle, 18.bus, 19.motorcycle, 20.pickup truck, 
 # 21.train, 23.lawn-mower , 24.tractor
 #24 classes. 



# print(X_train)	
		
		



	# def intersection_cipar10_cipar_100():
	# 	return 

	# 	while cipar_100_labels =true:
	# 		cipra1.labe cipar_100_labels
	# 		X_train.update(cipra10[index])		
