#############################################################
#		Machine Learning COMP30027 Assignment 1 Code		|
#					 Name: Karim Khariat					|
# 				Predict whether Abalone is					|
# 			old (>10 rings) or young(<11 rings)				|
#############################################################
import math
import random
import csv

#ALTER THE FOLLOWING LINE SUIT PATH/TO/FILE
PATH_TO_File = "abalone.data"
#Adds headers to the data at filename and returns a 2-tuple dataset
#with tuple 1 being headers and tuple 2 being the original data
#filename must be path/to/file!
def preprocess_data(filename):
	#Open file
	data = []
	#open file
	fp= open(filename,"r")
	#append the data to a list
	data.append(fp.readlines())
	#perform split, will return tuple where dataset[0] = testset and dataset[1]= training set
	dataset=My_Holdout_Strategy(tuple(data))
	return dataset

#used to strip commas and create a list to be used to compare instances
def strip_commas(a):
	#start at i = 2 and numstart =2 to avoid the char used for sex and first comma
	i = 2
	a_stripped=[]
	numstart=2
	if a[0] == "I":
		a_stripped.append(3)
	elif a[0] == "M":
		a_stripped.append(1)
	else:
		a_stripped.append(2)
	#iterate over and remove commas, add value as a float to the list and increment
	while i <len(a):
		if a[i] == ",":
			a_stripped.append(float(a[numstart:i]))
			numstart= i+1
		i+=1
	#add last value in the row as it is not followed by a comma.
	a_stripped.append(int(a[numstart:i]))
	return a_stripped

#iterates over the given dataset and applies strip_commas to each instance, returns the result 
def strip_commas_set(dataset):
	for i in range(len(dataset)):
		dataset[i]= strip_commas(dataset[i])
	return  dataset

#Euclidean Distance function
def my_euclidean_dist(a,b):
	#calculates the sum of the square of ai-bi then returns the sqroot of the sum 
	return(math.sqrt(sum(pow(a[i]-b[i],2)for i in range(len(a)))))


# Method can be either Euclidean distance denoted by "euclids", "euclidean" or "euclidean distance" 
#Method is NOT case sensitive
def compare_instance(a, b, method):
	#make method lowercase to make it case insenstive
	method = method.lower()
	# goes to -1 to avoid including ring number as that it includes the class we are guessing!
	a = a[:-1]
	b= b[:-1]
	#raises exception if sizes aren't equal!
	if len(a) != len(b):
		raise Exception("Instances aren't of equal sizes!")
	#raises exception if instances are empty!
	elif (len(a) == 0 or len(b) ==0):
		raise Exception("One or both instances have a length of 0")
	#return euclids if method is euclids, otherwise raise exception
	if method == "euclidean distance" or method == "euclids" or method == "euclidean":
		return(my_euclidean_dist(a,b))
	else:
		raise Exception("Invalid method type in compare_instances!")


#Function to perform the Holdout Strategy
def My_Holdout_Strategy(dataset):
	#Keep only the data
	dataset= dataset[0]
	#seed to make it reproducable
	random.seed(55)
	#shuffle dataset incase of pre-existing ordering
	random.shuffle(dataset)
	#split the dataset into 20train:80test split
	split = int(len(dataset)/5)
	train = dataset[:split]
	test = dataset[split:]
	#strip commas from sets allowing them to later be iterated over and easily used.
	train = strip_commas_set(train)
	test = strip_commas_set(test)
	# return tuple with test and training data
	return (test, train)



#find the K- nearest neighbours. Method can only accept Euclidean distance as "euclids", "euclidean" or "euclidean distance" 
def get_neighbours(instance, training_data_set, k, method):
	dists=[]
	neighbours= []
	#iterate and calculate the distance between the instance and the training instance
	for i in training_data_set:
		distance= compare_instance(instance, i, method)
		#append tuple containing training and distance to list
		if int(i[-1]) >= 11:
			dists.append(("Old",distance))
		else:
			dists.append(("Young",distance))
	#sort the list of tuples by distance value :)
	dists.sort(key=lambda x: x[1])

	for i in range(k):
		neighbours.append(dists[i])
	return dists[:k]

#counts number of old and young abalone in neighbours
#returns Old if old_count> young_count, otherwise returns Young
def my_majority_class(neighbours):
	old_count = 0
	young_count = 0
	for i in neighbours:
		if i[0] == "Old":
			old_count+=1
		else:
			young_count+=1
	if old_count> young_count:
		return "Old"
	return "Young"

#calculates the inverse linear distance.
def my_inverse_linear_distance(neighbours):
	old_count=0
	young_count=0
	furthest_neighbour=float(neighbours[-1][-1])
	nearest_neighbour = float(neighbours[0][-1])
	for i in neighbours:
		increment= ((furthest_neighbour - float(i[-1]))/(furthest_neighbour-nearest_neighbour))
		#if first variable in neighbours tuple is old increment old count, otherwise increment youngcount
		if i[0] == 'Old':
			old_count += increment
		else:
			young_count+= increment
	#if number of old> number ofyoung return Old otherwise Young
	if old_count> young_count:
		return "Old"
	return "Young"

#returns the predicted class, takes in methods: 
#Inverse Linear Distance denoted with "ild" and "inverse linear distance" 
#and Majority class denoted by "majority class" or "mc"
#method input is NOT case sensitive
def predict_class(neighbours,method):
	#make method all lowercase
	method= method.lower()
	# if ILD, return class using ILD
	if method =="ild" or method == "inverse linear distance":
		return(my_inverse_linear_distance(neighbours))
	#if Majority Class return class using it
	elif method == "majority class" or method == "mc":
		return(my_majority_class(neighbours))
	#otherwise Raise exception
	else:
		raise Exception("Invalid Method type in predict_class!")

#returns accuracy of the predictions.
def my_Accuracy(test, predicted):
	right=0
	#iterate through test and check to see if it was correctly predicted, if it was increment right
	for i in range(len(test)):
		if test[i][-1] >= 11 and "Old"== predicted[i]:
			right+=1
		elif test[i][-1] < 11 and "Young"== predicted[i]:
			right+=1
	#divide right by the length of test_set to get accuracy
	accuracy = right/float(len(test))
	return accuracy

#assumes that young is negative and old is positive
#returns the accuracy with respect to the negative cases 
def my_specificity(test, predicted):
	trueNeg= 0
	falsePos= 0
	for i in range(len(test)):
		#if trueNegative increment trueneg
		if test[i][-1] <=10  and "Young" == predicted[i]:
			trueNeg+=1
		#if false positive increment it
		elif test[i][-1] >= 11 and "Old" != predicted[i]:
			falsePos+=1
	#return TN/(TN+FP)
	specificity= trueNeg/float(trueNeg+falsePos)
	return specificity


#call evaluate(dataset,metric) to run code end to end. 
#metric can take Accuracy OR specificity
def evaluate(dataset, metric):
	metric= metric.lower()
	#test data will be the first item in tuple, and training data will be second
	test = dataset[0]
	train = dataset[1]
	predicted= []
	#K reasoning in report
	k= 17
	#get neighbours for each instance in test data, predict the class and append to the predicted list
	for i in test:
		#get_neighbours can only take euclidean distance
		neighbours=get_neighbours(i,train, k, "euclids")
		#predict can take Inverse linear distance or Majority class
		#"ild" or "inverse linear distance" for ILD, "mc" or "majority class" for majority class 
		predict = predict_class(neighbours,"ild")
		predicted.append(predict)
	#if metric is accuracy return accuracy 
	if metric == "accuracy":
		return(my_Accuracy(test,predicted))
	#if metric is specificity return specificity
	elif metric == "specificity":
		return(my_specificity(test,predicted))
	#otherwise raise exception 
	else:
		raise Exception("Invalid metric type in Evaluate!")


#run the program end to end. 
#change Accuracy to specificity to find specificity.
evaluate(preprocess_data(PATH_TO_File),"accuracy")

