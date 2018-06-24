import math
import re
from collections import defaultdict, OrderedDict
import random


def weighted_mean(predict_date, flavor, input_date):
    """
    :param predict_date_range: the date interval need to predict,list
    :param train_date: the train date, type of list
    :param flavor:Type of flavor which need predict, has been chosen from train data, and transformed to dict
    :return:
    """
    if flavor is None:
        print 'weighted_mean() input parameter error! '
        return 0

    total_vm = 0
    specific_flavor_num = {}
    date_len = (predict_date[-1] - predict_date[0]).days
    #print 'train_date length:'
    #print len(input_date)  # sum(flavor[key])
    for key in flavor.keys():
        # caculate the sum of flavor type of key
        specific_flavor_num[key] = int(math.ceil((sum(flavor[key])) * (date_len / float(len(input_date)))))#up int
        total_vm = total_vm + specific_flavor_num[key]

    return total_vm, specific_flavor_num


def weighted_mean2(predict_date, flavor):
    """
    :param predict_date: list:date range, need predict
    :param flavor: dict:vm flavor history data, need to predict
    :param input_date: list:history date
    :return: total_vm:int:predicted total vm number
             specific_flavor_num:dict:predicted specific flavor number
    """
    total_vm = 0
    specific_flavor_num = {}
    random.seed(1)

    def randn(n_rand):
		a = []
		for i in range(n_rand):
			a.append(random.random() * 0.2)

		return a

    for key in flavor:
        w = randn(len(flavor[key]))
        w.sort()
        assert len(w) == len(flavor[key])
        specific_flavor_num[key] = int(math.ceil(sum(map(lambda (a,b):a*b, zip(w, flavor[key]))))) #*\
								   #(predict_date[-1] - predict_date[0]).days# f * (predict_date[-1] - predict_date[0]).days
        total_vm = total_vm + specific_flavor_num[key]

    return total_vm, specific_flavor_num


def weighted_mean3(predict_date, flavor, input_date):
	'''
	:param predict_date: list :predicted date range
	:param flavor: dict:need predict vm flavor
	:return: total_vm:int:predicted total vm number
             specific_flavor_num:dict:predicted specific flavor number
	'''
	total_vm = 0
	specific_flavor_num = {}
	#weight value arithmetic seq
	w = []
	# sample window length
	day_len = (predict_date[-1] - predict_date[0]).days
	win_len = 6
	k = (win_len) / float(2)
	w_step = 1 / float(k * win_len)
	# w first item
	a = (2 * k - win_len -1) / float(2 * k * win_len)
	for i in range(1,win_len+1):
		w.append((a+i*w_step)+0.02)

	print sum(w)

	flavor_day = []
	for key in flavor:
		for d in range(day_len):
			flavor_sample_list = flavor[key][len(flavor[key])-day_len-win_len+d:len(flavor[key])-day_len+d]
			#flavor_sample_list12 = flavor[key][len(flavor[key])-win_len:len(flavor[key])]
			#flavor_diff_list = map(lambda (a,b):b-a, zip(flavor_sample_list11,flavor_sample_list12))
			flavor_day.append(sum(map(lambda (a,b):a*b, zip(w, flavor_sample_list))))

		specific_flavor_num[key] = sum(flavor[key][len(flavor[key])-day_len:len(flavor[key])]) + int(round(sum(flavor_day)))
		total_vm = total_vm + specific_flavor_num[key]
		flavor_day = []

	return total_vm, specific_flavor_num


def exp_smooth(flavor, date, predict_date):

	total_flavor = 0
	specific_flavor_num = {}
	flavor_inc = defaultdict(list) # every flavor's increment during i th time slice
	smooth1 = defaultdict(list) # once smooth value
	smooth2 = defaultdict(list) # twice smooth
	average = {} #mean value of last week
	a = {}
	b = {}
	alpha = 0.18 # smooth constant

	day_len = (predict_date[-1] - predict_date[0]).days
	I = len(date) / day_len

	#caculate every flavor vm's increment in I time slice
	for key in flavor.keys():
		for i in range(1,I): # [0,I-1]
			delta_i = sum(flavor[key][day_len*i:day_len*(i+1)]) - sum(flavor[key][day_len*(i-1):day_len*i])
			if  delta_i > 0:
				flavor_inc[key].append(delta_i)
			else:
				flavor_inc[key].append(0)

	assert len(flavor_inc[key])==I-1

	#cnt_0 = 0 # statistic zero number in last week
	cnt_v = 0 # statistic having value number in last week
	bias = defaultdict(list)
	'''
	for key in flavor.keys():
		average[key] = sum(flavor[key][len(flavor[key])-2*day_len:-1]) / float(2*day_len) # last two weeks average value
		#statistic last week data
		for i in range(len(flavor[key])-day_len,len(flavor[key])): # last week
			if average[key] != 0 and flavor[key][i] / float(average[key]) > 1.5:
				flavor[key][i] = int(average[key]) # delete exceptional value
			if flavor[key][i] > 0:
				cnt_v = cnt_v + 1
			#else:
			#	cnt_0 = cnt_0 +1

		if cnt_v >= 3:
			bias[key] = sum(flavor[key][len(flavor[key])-day_len:-1]) #/ float(day_len)
		else:
			bias[key] = -(sum(flavor[key][len(flavor[key])-day_len:-1])) #/ float(day_len))

		#cnt_0 = 0
		cnt_v = 0
	'''
	#caculate the smooth value
	for key in flavor.keys():
		smooth1[key].append(flavor_inc[key][0]) # initial value
		smooth2[key].append(flavor_inc[key][0]) # initial value
		for i in range(1,I-1):
			smooth1[key].append(alpha * flavor_inc[key][i] - (1-alpha) * smooth1[key][i-1])
			smooth2[key].append(alpha * smooth1[key][i] - (1-alpha) * smooth2[key][i-1])
		a[key] = 2*smooth1[key][-1] - smooth2[key][-1]
		b[key] = (alpha /  (1 - float(alpha))) * (smooth1[key][-1] - smooth2[key][-1])
		#predict flavor number
		flavor_inc[key].append(a[key]+b[key])

		average[key] = sum(flavor[key][len(flavor[key]) - 2 * day_len:-1]) / float(2 * day_len)  # last two weeks average value
		# statistic last two weeks data
		for i in range(len(flavor[key]) - 2*day_len, len(flavor[key])):
			diff = flavor[key][i] - flavor[key][i-1] # last week
			if diff >= 4:#average[key] != 0 and flavor[key][i] / float(average[key]) > 1.5:
				flavor[key][i] = int(average[key])  # delete exceptional value
			if flavor[key][i] > 0:
				cnt_v = cnt_v + 1
			# else:
			#	cnt_0 = cnt_0 +1

		if cnt_v >= 3:
			bias[key] = sum(flavor[key][len(flavor[key]) - (day_len):-1])  # / float(day_len)
		else:
			bias[key] = -(sum(flavor[key][len(flavor[key]) - day_len:-1]))  # / float(day_len))

		# cnt_0 = 0
		cnt_v = 0

		specific_flavor_num[key] = int(math.ceil(sum(flavor[key][len(flavor[key])-day_len:-1]) + flavor_inc[key][-1]) + int(math.ceil(bias[key])))
		if specific_flavor_num[key] < 0:
			specific_flavor_num[key] = 0
		total_flavor = total_flavor + specific_flavor_num[key]


	print total_flavor

	return total_flavor, specific_flavor_num
