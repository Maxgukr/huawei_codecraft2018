import math
import copy
import random
import pickle
import time
import re
from preprocess import slice_data, roll_mean, normalize_data, recover_normalized_data, week_prob


def sigmoid(X):
	"""
	:param x: a scale or list
	:return: sigmoid(x)
	"""
	if type(X[0]) is list:
		s = []
		s_col = []
		for i in range(len(X)):
			for j in range(len(X[0])):
				s_col.append(1 / float((1+math.exp(-X[i][j]))))
			s.append(s_col)
			s_col = []
	else:
		s = []
		for i in range(len(X)):
			s.append(1 / float(1+math.exp(-X[i])))

	cache = s

	return s, cache


def relu(x):
	'''
	:param x: a scale or list
	:return:
	'''
	if type(x[0]) is list:
		s = []
		s_col = []
		for i in range(len(x)):
			for j in range(len(x[0])):
				if x[i][j] >= 0:
					s_col.append(x[i][j])
				else:
					s_col.append(0)
			s.append(s_col)
			s_col = []
	else:
		s = []
		for i in range(len(x)):
			if x[i] >= 0:
				s.append(x[i])
			else:
				s.append(0)
	cache = s

	return s, cache


def relu_backward(dA, cache):
	'''
	implement the backward propagation in single relu unit
	:argument dA:post activation gradient
	:argument cache: 'Z' where we store for computing backward propagation efficiently
	:return: dZ gradient of the cost with respect to Z
	'''
	Z = cache
	dZ = []
	dz_col = []
	#can't do this, it's not a array
	if type(dA[0]) is list:
		for i in range(len(dA)):
			for j in range(len(dA[0])):
				if dA[i][j] <= 0:
					dz_col.append(0)
				else:
					dz_col.append(dA[i][j])
			dZ.append(dz_col)
			dz_col = []
	else:
		for i in range(len(dA)):
			if dA[i] <= 0:
				dZ.append(0)
			else:
				dZ.append(dA[i])

	return dZ


def sigmoid_backward(dA, cache):
	'''
	implement the backward propagation in single relu unit
	:argument dA:post activation gradient
	:argument cache: 'Z' where we store for computing backward propagation efficiently
	:return: dZ gradient of the cost with respect to Z
	'''
	Z = cache
	# must note the dimension!!!
	(s,x1) = sigmoid(Z)

	if type(Z[0]) is list:
		dZ = []
		dZ_col = []
		for i in range(len(Z)):
			for j in range(len(Z[0])):
				dZ_col.append(dA[i][j] * s[i][j] * (1 - s[i][j]))
			dZ.append(dZ_col)
			dZ_col = []
	else:
		dZ = []
		for i in range(len(Z)):
			dZ.append(dA[i] * s[i] * (1 - s[i]))

	return dZ


def dot(w, A):
	'''
	implement matrix multiply with broadcast function
	:param w: matrix with shape (n1,n2)
	:param AT: matrix with shape (n2,n3)
	:return: s matrix with shape (n1,n2)
	'''
	if len(w)!=0:
		for i in range(len(w)):
			if type(w[i]) is not list:
				print 'input parameter type error!'
				exit(1)
	else:
		print 'input parameters type error!'
		exit(1)
	if type(A[0]) is list:
		if len(w[0]) != len(A[0]):
			A=transpose(A)

	#before implementing multiply, transpose the second matrix
	#AT = transpose(A)

	if type(A[0]) is list: #y with dimension 2
		s = [] # x's row number
		s_col = []
		for i in range(len(w)):
			for j in range(len(A)):
				s_col.append(sum(map(lambda (a,b):a*b, zip(w[i], A[j]))))
			s.append(s_col)
			s_col = []
	else: #y with dimension 1
		s = []
		for i in range(len(w)):
			#for j in range(len(y)):
			s.append(sum(map(lambda (a,b):a*b, zip(w[i], A))))

	return s


def add(WA, B):
	'''
	:param WA:
	:param B:
	:return:
	'''
	'''
	implement broadcast function
	'''
	z = []
	b = copy.deepcopy(B)
	if type(WA[0]) is list:
		for j in range(len(b)):
			#for i in range(len(WA[0])):
			b[j] = [b[j]]*len(WA[0])

		assert len(b) == len(WA)

		for i in range(len(WA)):
			z.append(map(lambda (a,b):a+b, zip(WA[i], b[i])))
	else:
		z = map(lambda (a,b):a+b,zip(WA, b))
	return z


def transpose(m):
	'''
	transpose a 2 dim list
	:param m: list:2 dim list input
	:return:
	'''
	if type(m[0]) is  not list:
		print 'type error!'
		exit(1)

	mt = [0]*len(m[0]) #change row and column between m and mt
	for j in range(len(m[0])):
		#for i in range(len(m)):
		mt[j] = [m[i][j] for i in range(len(m))]

	return mt


def initial_parameters(n_x, n_h, n_y):
	'''
	:param n_x: int: size of the input layer
	:param n_h: int: size of the hide layer
	:param n_y: int: size of the output layer
	:return: parameters:dict w1:weight matrix with shape (n_h,n_x)
							 b1:bias vector of shape (n_h,1)
							 w2:weight matrix with shape (n_y, n_h)
							 b2:bias vector of shape (n_y,1)
	'''
	random.seed(1)
	#generate a random float number range (0,1)
	w1 = [] # initial a w1 matrix
	w2 = []
	b1 = [] # need translate to column vector
	b2 = []
	w1_col= []
	w2_col = []
	for i in range(n_h):
		b1.append(0)
		for j in range(n_x):
			w1_col.append(random.random())
		w1.append(w1_col)
		w1_col = []

	for i in range(n_y):
		b2.append(0)
		for j in range(n_h):
			w2_col.append(random.random())
		w2.append(w2_col)
		w2_col = []

	parameters = {'w1':w1,
				  'b1':b1,
				  'w2':w2,
				  'b2':b2}

	return parameters


def linear_forward(A, W, b):
	'''
	implement a linear part of layer's forward propagation
	:param A: activation from previous layer (or input data):(size of previous layer, number of example)
	:param W: weight matrix
	:param b: bias vector
	:return: the input of activation function
	'''
	WA = dot(W,A)
	Z = add(WA,b)

	assert len(Z)==len(W)

	cache = (A, W, b)

	return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
	'''

	:param A_prev:
	:param W:
	:param b:
	:param activation:
	:return:
	'''
	if activation == 'sigmoid':
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache =sigmoid(Z)
	elif activation == 'relu':
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)

	cache = (linear_cache, activation_cache)

	return A, cache


def L_model_forward(X, parameters):
	'''

	:param X:
	:param parameters:
	:return:
	'''
	caches = []
	A = X
	L = len(parameters) // 2

	for l in range(1,L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters['w'+str(l)], parameters['b'+str(l)], activation='sigmoid')
		caches.append(cache)

	AL, cache = linear_activation_forward(A, parameters['w' + str(L)], parameters['b' + str(L)], activation='relu')

	return AL, caches


def compute_cost(AL, Y, parameters, lambd):
	'''
	 cost function define:
	:param AL:
	:param Y:
	:return:
	'''
	m = len(Y)
	#assert len(AL) == len(Y)
	A = transpose(AL)
	assert len(A) == len(Y)
	cost = [0]*len(Y)

	w1 = parameters['w1']
	w2 = parameters['w2']
	l2_regularization_cost = (1/float(m)) * (lambd/2) * (L2_regularization(w1) + L2_regularization(w2))

	for i_c in range(len(A)):
		cost[i_c] = (1 / float(m)) * sum(map(lambda (a,b):0.5*(a-b)**2, zip(A[i_c], Y[i_c])))

	cost_total = sum(cost) + l2_regularization_cost

	return cost_total


def L2_regularization(w):
	'''
	compute the L2 regularization item
	:param w: layer l weight matrix
	:return: matrix's square sum
	'''
	w_sum = []
	for i in range(len(w)):
		w_sum.append(sum(map(lambda a:a**2, w[i])))

	l2_cost = sum(w_sum)

	return l2_cost


def linear_backward(dZ, cache, lambd):
	'''
	implement the linear portion backward for a single layer
	:param dZ: gradient of the cost with respect to the linear output (of current layer)
	:param cache: tuple of (A_prev, w,b) coming from the forward propagation in the current layer
	:param lambd: regularization item
	:return:dA_prev:gradient of teh cost with respect to the activation
			dw:gradient of the cost with current layer l
			db:gradient of the cost with current layer l
	'''
	A_prev, w, b = cache
	m = len(A_prev)
	matrix = dot(dZ, A_prev)
	w_l2 = [0]*len(w)#copy.deepcopy(w)
	dw = [0]*len(w)

	for i in range(len(w)):
		dw[i] = map(lambda (a,b):a*b,zip(matrix[i],[1/float(m) for k in range(len(w[0]))]))
	#for i in range(len(w)):
		w_l2[i] = map(lambda a:a*lambd*(1/float(m)), w[i])
		dw[i] = map(lambda (a,b):a+b, zip(dw[i], w_l2[i]))

	dA_prev = dot(transpose(w),dZ)

	db = [0]*len(dZ)
	for i in range(len(dZ)):
		db[i] = dZ[i][0]

	return dA_prev, dw, db


def linear_activation_backward(dA, cache, activation, lambd):
	'''

	:param dA:post-activation gradient for current layer l
	:param cache:tuple of value (linear_cache, activation_cache) we store for computing backward
	:param activation :the activation to be used in this layer
	:param lambd :regularization item
	:return:dA_prev:gradient of the cost with respect to activation (of the previous layer l-1)
			dw:
			db:
	'''
	linear_cache, activation_cache = cache

	if activation == 'relu':
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dw, db = linear_backward(dZ, linear_cache, lambd)
	elif activation == 'sigmoid':
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dw, db = linear_backward(dZ, linear_cache, lambd)

	return dA_prev, dw, db


def L_model_backward(AL, Y, caches):
	'''
	implement the back propagation
	:param AL: digital vector, output of the L_model_forward()
	:param Y: true numerical vector
	:param caches: list of caches containing:
					(linear_cache, activation_cache)
					linear_cache = (W,A,b)
					activation_cache = Z
	:return: grads:dict
			grads{'dA':
				  'dw':
				  'db':}
	'''
	grads = {}
	L = len(caches)

	dAL = map(lambda (a,b):a-b, zip(AL, Y))

	current_cache = caches[L-1]
	grads['dA'+str(L)], grads['dw'+str(L)], grads['db'+str(L)] = linear_activation_forward(dAL, current_cache, activation='relu')

	for l in reversed(range(L-1)):
		current_cache = caches[l]
		grads['dA' + str(l+1)], grads['dw' + str(l+1)], grads['db' + str(l+1)] = linear_activation_forward(dAL, current_cache,
																									 activation='sigmoid')
	return grads


def update_parameters(parameters, grads, learning_rate):
	'''
	updating parameters using gradient descent
	:param parameters:
	:param grads:
	:param learning_rate:
	:return:
	'''
	L = len(parameters)//2

	for l in range(L):

		for i in range(len(parameters['w'+str(l+1)])):
			for j in range(len(parameters['w'+str(l+1)][0])):
				parameters['w'+str(l+1)][i][j] -= learning_rate * grads['dw'+str(l+1)][i][j]

		for i in range(len(parameters['b'+str(l+1)])):
			#for j in range(len(parameters['b'+str(l+1)][0])):
			parameters['b'+str(l+1)][i] -= learning_rate * grads['db'+str(l+1)][i]

	return parameters


def predict(X, parameters):
	output, caches = L_model_forward(X, parameters)
	return output


def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, lambd = 0.3, iterations = 3000):
	'''
	implement a two layer neural network: LINEAR->SIGMOID->LINEAR->RELU.
	:param X: input data, shape of (n_x, number of example)
	:param Y: true value vector
	:param layers_dims: dimensions of layer (n_x,n_h,n_y)
	:param learning_rate: learning rate of gradient descent update loop
	:param iterations: number of loop for optimize parameters
	:param print_cost: true or false to print cost every 100 times
	:return: parameters with w and b
	'''
	random.seed(1)
	grads = {}
	costs = []
	(n_x, n_h, n_y) = layers_dims

	parameters = initial_parameters(n_x, n_h, n_y)

	w1 = parameters['w1']
	b1 = parameters['b1']
	w2 = parameters['w2']
	b2 = parameters['b2']

	for i in range(0, iterations):

		A1, cache1 = linear_activation_forward(X, w1, b1, 'sigmoid')
		A2, cache2 = linear_activation_forward(A1, w2, b2, 'relu')

		cost = compute_cost(A2, Y, parameters, lambd)

		AL = transpose(A2)
		dA2 = []
		m = len(Y)
		for k in range(len(Y)):
			dA2.append(map(lambda (a, b): (1/float(m))*(a - b), zip(AL[k], Y[k])))
		dA2 = transpose(dA2)

		dA1, dw2, db2 = linear_activation_backward(dA2, cache2, 'relu', lambd)
		dA0, dw1, db1 = linear_activation_backward(dA1, cache1, 'sigmoid', lambd)

		grads['dw1'] = dw1
		grads['db1'] = db1
		grads['dw2'] = dw2
		grads['db2'] = db2

		parameters = update_parameters(parameters, grads, learning_rate)

		w1 = parameters['w1']
		b1 = parameters['b1']
		w2 = parameters['w2']
		b2 = parameters['b2']

		if i % 10 == 0:
			print (i, cost)
		#if print_cost and i % 100 == 0:
		costs.append(cost)

	return parameters


def nn_app_utils(flavor_sum, flavor, predict_date):
	'''

	:param flavor_sum:
	:param flavor:
	:return:
	'''
	flavor_normal, data_mean, data_sigma = normalize_data(flavor_sum)
	# flavor_sum1 = roll_mean(7, flavor_sum)
	day_len = (predict_date[-1] - predict_date[0]).days
	train_data, supervise_data = slice_data(flavor_normal, day_len)
	#train_prob, supervise_prob, week_prob_last = week_prob(flavor, flavor_sum, day_len)
	'''
	start2 = time.clock()
	# train prob of the each vm
	parameters2 = two_layer_model(train_prob, supervise_prob, (len(flavor.keys()), 9, len(flavor.keys())), learning_rate=0.25, lambd=0.08, iterations=60)
	elapsed2 = time.clock() - start2
	print 'prob train time:'
	print elapsed2

	flavor_prob = predict(week_prob_last, parameters2)
	for i in range(len(flavor_prob)):
		flavor_prob[i] = flavor_prob[i] / float(sum(flavor_prob))
	'''
	start1 = time.clock()
	# train total vm number in every predicted date
	parameters1 = two_layer_model(train_data, supervise_data, (day_len, 6, day_len), learning_rate=0.18, lambd=0.1, iterations=60)
	elapsed1 = time.clock() - start1
	print 'data train time:'
	print elapsed1
	'''
	start2 = time.clock()
	# train prob of the each vm
	parameters2 = two_layer_model(train_prob, supervise_prob, (len(flavor.keys()), 7, len(flavor.keys())), learning_rate=0.1, lambd=0.1, iterations=60)
	elapsed2 = time.clock() - start2
	print 'prob train time:'
	print elapsed2
	'''
	#predict flavor total number in each day
	test = flavor_normal[len(flavor_normal) - day_len:len(flavor_normal)]
	flavor_predict_normal = predict(test, parameters1)
	flavor_data = recover_normalized_data(flavor_predict_normal, data_mean, data_sigma)
	flavor_sum_predict = math.floor(sum(flavor_data))

	#predict prob in each vm
	#flavor_prob = predict(week_prob_last, parameters2)


	flavor_prob = {}
	flavor_total = {}
	total = sum(flavor_sum)

	for key in flavor.keys():
		flavor_total[key] = sum(flavor[key])
		flavor_prob[key] = flavor_total[key] / float(total)
		#if flavor_prob[key] < 0.025:
		#	flavor_prob[key] = 0.0

	test_total = {}
	test_prob = {}
	#print flavor_prob
	for key in flavor.keys():
		test_total[key] = sum(flavor[key][len(flavor_sum) - day_len:len(flavor_sum)])
		test_prob[key] = test_total[key] / float(sum(flavor_sum[len(flavor_sum) - day_len:len(flavor_sum)]))
		#if flavor_prob[key] < 0.025:
		#	flavor_prob[key] = 0.0

	test_prob_sort = sorted(test_prob.items(), key=lambda item:item[1])
	flavor_prob_sort = sorted(flavor_prob.items(), key=lambda item:item[1])

	flavor_resort_prob = {}
	for k in range(len(flavor_prob)):
		flavor_resort_prob[test_prob_sort[k][0]] = flavor_prob_sort[k][1]


	specific_flavor_num = {}
	flavor_total_indeed = 0

	for i in flavor_prob.keys():
		specific_flavor_num[i] = int(round(flavor_sum_predict * flavor_resort_prob[i]))
		flavor_total_indeed += specific_flavor_num[i]

	'''
	specific_flavor_num = {}
	flavor_total_indeed = 0
	key_list = flavor.keys()
	key_list.sort(key=lambda i: int(re.findall(r"\d+", i)[0]), reverse=False)

	for i in range(len(flavor_prob)):
		specific_flavor_num[key_list[i]] = int(round(flavor_sum_predict * flavor_prob[i]))
		flavor_total_indeed += specific_flavor_num[key_list[i]]

	#flavor_test = recover_normalized_data(test, data_max, data_min)

	'''

	return	flavor_total_indeed, specific_flavor_num


def main():
	f = open('flavor_sum.txt', 'rb')
	flavor_sum = pickle.load(f)
	flavor_normal, data_max, data_min = normalize_data(flavor_sum)
	# flavor_sum1 = roll_mean(7, flavor_sum)
	train_data, supervise_data = slice_data(flavor_normal, 7)

	start = time.clock()
	parameters = two_layer_model(train_data, supervise_data, (7,6,7), learning_rate=0.6, lambd=0.02, iterations=50)
	elapsed = time.clock() - start
	print elapsed

	test = flavor_normal[len(flavor_normal) - 7:len(flavor_normal)]
	#flavor_predict_normal = []

	#for i in range(7):
	flavor_predict_normal = predict(test, parameters)
		#.append(result[0])
		#test.append(result[0])
		#test.pop(0)

	flavor_data = recover_normalized_data(flavor_predict_normal, data_max, data_min)
	flavor_sum_predict = sum(flavor_data)
	print flavor_sum_predict

	f1 = open('flavor_dict_01-05.txt','rb')
	flavor = pickle.load(f1)
	flavor_prob = {}
	flavor_total = {}
	total = sum(flavor_sum)

	for key in flavor.keys():
		flavor_total[key] = sum(flavor[key])
		flavor_prob[key] = flavor_total[key] / float(total)
		if flavor_prob[key] < 0.025:
			flavor_prob[key] = 0.0

	print flavor_prob

	specific_flavor_num = {}
	for key in flavor_prob.keys():
		specific_flavor_num[key] = flavor_sum_predict * flavor_prob[key]

	print specific_flavor_num


if __name__ == '__main__':
	main()





