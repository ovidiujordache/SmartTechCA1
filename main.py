from data_processing import *
import matplotlib.image as mpimg

def main():
	dex = DataExploration()

	# dex.display_data_keys_and_labels()

	# cipar_100_labels, cipar_10_labels = dex.unique_labels()

	# dex.display_data_keys_and_labels()

	# dex.display_data_shape()


	

	img = mpimg.imread('st.png')
	imgplot = plt.imshow(img)
	plt.show()



if  __name__ =="__main__":
	main()



	