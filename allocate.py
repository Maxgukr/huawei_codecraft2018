import math
import re
from collections import defaultdict, OrderedDict
import random


def allocate_resource_first_fit(total_flavor, specific_flavor_num, input_file_dict):
    """
    first_fit is a simple method to solve the boxing problem
    :param specific_flavor_num: predicted specific flavor number
    :param input_file_dict:
    input_file_dict = {'cpu_mem_disk':['56','128','1200'],
                                'flavor_num':[''],
                               'flavor':{'flavor1':[1, 2024],...'flavor5':[2, 4096]},
                               'resource_name':[''],
                               'start_end_time':['','']}
    input file information,physical server volume core_num and mem_volume,optimize resource cpu/mem
    :return:total server number, allocate method
    """

    if input_file_dict is None:
        print 'allocate_resource_first_fit() func input parameter error!'
        return 0

    server_total = 1
    allocate_method = defaultdict(list)
    server_flavor = map(lambda x:int(x), input_file_dict['cpu_mem_disk']) #initial server flavor, 56 cpu,128G mem
    vm_set = []  # the set of virtual machine need allocate
    for key, value in specific_flavor_num.items():
        for i in range(value):
            vm_set.append(key)

    assert len(vm_set) == total_flavor
    # sort as number of flavor,like flavor15,flavor14,... up-down

    vm_set.sort(key=lambda i: int(re.findall(r"\d+", i)[0]), reverse=True)

    #first-fit-descend
    #note:when optimize cpu, mem resource can't be over divided
    for vm in vm_set:
        if input_file_dict['flavor'][vm][0] <= server_flavor[0] and \
                        (input_file_dict['flavor'][vm][1] / 1024) <= server_flavor[1]:
            allocate_method[str(server_total)].append(vm)
            server_flavor[0] = server_flavor[0] - input_file_dict['flavor'][vm][0]
            server_flavor[1] = server_flavor[1] - (input_file_dict['flavor'][vm][1] / 1024)
        else:
            server_total = server_total + 1 #start a new server
            # update server volume
            server_flavor  = map(lambda x:int(x), input_file_dict['cpu_mem_disk'])
            #allocate vm on new server
            allocate_method[str(server_total)].append(vm)
            server_flavor[0] = server_flavor[0] - input_file_dict['flavor'][vm][0]
            server_flavor[1] = server_flavor[1] - (input_file_dict['flavor'][vm][1] / 1024)

    assert server_flavor[0] >=0 and server_flavor[1] >= 0

    print 'cpu utilization ratio:'
    util_cpu = sum(input_file_dict['flavor'][vm][0] for vm in vm_set) / \
		  float(int(input_file_dict['cpu_mem_disk'][0]) * server_total)
    print util_cpu
    print 'mem utilization ratio:'
    util_mem = sum((input_file_dict['flavor'][vm][1]/1024) for vm in vm_set) / \
          float(int(input_file_dict['cpu_mem_disk'][1]) * server_total)
    print util_mem

    if input_file_dict['resource_name'] == 'CPU':
        return server_total, allocate_method, util_cpu
    else:
		return server_total, allocate_method, util_mem


def allocate_resource_best_fit(total_flavor, specific_flavor_num, input_file_dict):
    """
    first_fit is a simple method to solve the boxing problem
    :param specific_flavor_num: predicted specific flavor number
    :param input_file_dict:
    input_file_dict = {'cpu_mem_disk':['56','128','1200'],
                                'flavor_num':[''],
                               'flavor':{'flavor1':[1, 2024],...'flavor5':[2, 4096]},
                               'resource_name':[''],
                               'start_end_time':['','']}
    input file information,physical server volume core_num and mem_volume,optimize resource cpu/mem
    :return:total server number, allocate method
    """
    resource_cpu = 0
    resource_mem = 0
    flavor = copy.deepcopy(input_file_dict['flavor'])
    for key, value in specific_flavor_num.items():
        resource_cpu = resource_cpu + value * input_file_dict['flavor'][key][0]
        resource_mem = resource_mem + value * input_file_dict['flavor'][key][1]

	m1 = int(math.ceil(resource_cpu / float(input_file_dict['cpu_mem_disk'][0])))
    m2 = int(math.ceil(resource_mem / float(input_file_dict['cpu_mem_disk'][1])))
    # requirement server number
    ser_n = max(m1, m2)

    if input_file_dict is None:
        print 'allocate_resource_first_fit() func input parameter error!'
        return 0

    server_total = 1
    allocate_method = defaultdict(list)
    server_usage_log = defaultdict(list)
    server_sel_list = []
    sub_list = {}
    vm_set = []  # the set of virtual machine need allocate
    for key, value in specific_flavor_num.items():
        for i in range(value):
            vm_set.append(key)

    assert len(vm_set) == total_flavor
    # sort as number of flavor,like flavor15,flavor14,... up-down

    #vm_set.sort(key=lambda i: int(re.findall(r"\d+", i)[0]), reverse=True)

    #first-fit-descend
    #note:when optimize cpu, mem resource can't be over divided
    server_usage_log[str(server_total)] = map(lambda x:int(x), input_file_dict['cpu_mem_disk'][0:2])# initial server utilization
    '''
    allocate_method[str(server_total)].append(vm_set[0])
    server_usage_log[str(server_total)][0] = server_usage_log[str(server_total)][0] - input_file_dict['flavor'][vm_set[0]][0]
    server_usage_log[str(server_total)][1] = server_usage_log[str(server_total)][1] - (input_file_dict['flavor'][vm_set[0]][1] / 1024)
	'''
    for vm in vm_set:
		# server list which has residual space to allocate vm in this iterate
		for n in server_usage_log.keys():
			if (server_usage_log[n][0] >= input_file_dict['flavor'][vm][0]) and (input_file_dict['flavor'][vm][1]  <= server_usage_log[n][1]):
				server_sel_list.append(n)

		if len(server_sel_list) != 0:
			for key in list(set(server_sel_list)):
				#record diff between vm and server residual volum
				if input_file_dict['resource_name'] == 'CPU':
					sub_list[key] = server_usage_log[key][0] - input_file_dict['flavor'][vm][0]
				else:
					sub_list[key] = server_usage_log[key][1] - input_file_dict['flavor'][vm][1]

			# select minimum residual server to allocate
			key_sel = sorted(sub_list.items(), key=lambda sub_list:sub_list[1])[0][0]
			allocate_method[key_sel].append(vm)
			server_usage_log[key_sel][0] = server_usage_log[key_sel][0] - input_file_dict['flavor'][vm][0]
			server_usage_log[key_sel][1] = server_usage_log[key_sel][1] - input_file_dict['flavor'][vm][1]
			server_sel_list = []
			sub_list = {}
		else:
			server_total = server_total + 1 #start a new server
			server_usage_log[str(server_total)] = map(lambda x:int(x), input_file_dict['cpu_mem_disk'][0:2])
			# update server volume
			# allocate vm on new server
			allocate_method[str(server_total)].append(vm)
			server_usage_log[str(server_total)][0] = server_usage_log[str(server_total)][0] - input_file_dict['flavor'][vm][0]
			server_usage_log[str(server_total)][1] = server_usage_log[str(server_total)][1] - input_file_dict['flavor'][vm][1]

    for key_log, value_log in server_usage_log.items():
        assert value_log[0] >=0 and value_log[1] >= 0

    print server_usage_log
    print 'cpu utilization ratio:'
    util_cpu = 1 - (sum(server_usage_log[key][0] for key in server_usage_log.keys()) / \
			  float(int(input_file_dict['cpu_mem_disk'][0]) * server_total))
    print util_cpu
    print 'mem utilization ratio:'
    util_mem = 1 - (sum(server_usage_log[key][1] for key in server_usage_log.keys()) / \
			  float(int(input_file_dict['cpu_mem_disk'][1]) * server_total))
    print util_mem

    #util = 0
    #if input_file_dict['resource_name'] == 'MEM':
    	#return server_total, allocate_method, util_mem#utilization = util_mem
	#if input_file_dict['resource_name'] == 'CPU':
    return server_total, allocate_method, util_cpu

import copy
'''
there are some problem in method, when allocation is not good, server number will be more than initial number
'''
def least_distance_match(total_flavor, specific_flavor_num, input_file_dict):
	'''
	:param total_flavor:int
	:param specific_flavor_num: dict:predicted vm flavor number need allocate in physical server
	:param input_file_dict: dict:input file information
	:return: total server number, each server allocate vm flavor and number
	'''
	#calculate total server resource of total predicted vm
	resource_cpu = 0
	resource_mem = 0
	flavor = copy.deepcopy(input_file_dict['flavor'])
	for key, value in specific_flavor_num.items():
		resource_cpu = resource_cpu + value * input_file_dict['flavor'][key][0]
		resource_mem = resource_mem + value * (input_file_dict['flavor'][key][1])

	m1 = int(math.ceil(resource_cpu / float(input_file_dict['cpu_mem_disk'][0])))
	m2 = int(math.ceil(resource_mem / float(input_file_dict['cpu_mem_disk'][1])))
	# requirement server number
	ser_n = max(m1,m2)

	#vm sequence
	vm_set = []  # the set of virtual machine need allocate
	for key, value in specific_flavor_num.items():
		for i in range(value):
			vm_set.append(key)

	assert len(vm_set) == total_flavor

	#server performance feature vector
	ser_p_vector = {}
	#server residual
	ser_residual = {}
	#initial residual
	for i in range(1,ser_n+1):
		ser_residual[str(i)] = input_file_dict['cpu_mem_disk'][0:2]
	#initial server performance feature vector
	for k in ser_residual.keys():
		ser_p_vector[k] = map(lambda (a,b):a/float(b), zip(ser_residual[k],input_file_dict['cpu_mem_disk'][0:2]))

	#caculate each vm performance prefer vector
	vm_pp_vector = defaultdict(dict)
	for key in input_file_dict['flavor'].keys():
		for key2 in ser_residual.keys():
			vm_pp_vector[key][key2] = map(lambda (a,b):a/float(b), zip(input_file_dict['flavor'][key], ser_residual[key2]))
			#normalization
			vm_pp_vector[key][key2] = [vm_pp_vector[key][key2][0] / float(sum(vm_pp_vector[key][key2])),\
			 vm_pp_vector[key][key2][1] / float(sum(vm_pp_vector[key][key2]))]

	def update_ser_p_vector(ser_p_vector, ser_residual):
		for ser_key in ser_residual.keys():
			ser_p_vector[ser_key] = map(lambda (a, b): a / float(b),\
									zip(ser_residual[ser_key], input_file_dict['cpu_mem_disk'][0:2]))

		return ser_p_vector

	def update_vm_pp_vector(ser_residual):
		vm_pp_vector = defaultdict(dict)
		for key1 in input_file_dict['flavor'].keys():
			for key2,value in ser_residual.items():
				if value[0] > 0 and value[1] > 0:
					vm_pp_vector[key1][key2] = map(lambda (a, b): a / float(b),
											zip(input_file_dict['flavor'][key1], ser_residual[key2]))
					# normalization
					vm_pp_vector[key1][key2] = [vm_pp_vector[key1][key2][0] / float(sum(vm_pp_vector[key1][key2])),\
					vm_pp_vector[key1][key2][1] / float(sum(vm_pp_vector[key1][key2]))]
				else:
					vm_pp_vector[key1][key2] = [1.0,1.0]

		return vm_pp_vector

	#least match distance1:performance prefer match
	def Match_distance(vm_pp_v, ser_p_v):
	#param vm_pp_v:single vm prefer vector
	#param ser_p_V:single server performance vector
		return math.sqrt(sum(map(lambda (a,b):(a-b)**2, zip(vm_pp_v,ser_p_v))))

	#least match distance2:residual performance match
	#normalization performance
	def normalization(perform_vector):
    #param:dict:vm or server(residual) performance
		vector = perform_vector
		p_min = [vector[min(vector, key=lambda k:vector[k][i])][i] for i in range(0, 2)]
		p_max = [vector[max(vector, key=lambda k:vector[k][i])][i] for i in range(0, 2)]
		denominator = map(lambda (a,b):a-b, zip(p_max, p_min))

		for key in vector.keys():
			numerator = map(lambda (a,b):a-b, zip(vector[key], p_min))
			if denominator[0] != 0 and denominator[1] != 0:
				vector[key] = map(lambda (a,b): a / float(b), zip(numerator, denominator))
			else:
				vector[key] = [1.0, 1.0]

		return vector

	#main cycle
	match_distance1 = []
	match_distance2 = []
	vm_normal_v = normalization(flavor)
	r1 = 0.5
	r2 = 0.5
	allocate_method = defaultdict(list)
	server_list = list([])
	match_list = {}
	for vm in vm_set:
	#while len(vm_set) != 0:
		#vm = random.choice(vm_set)
		#index = vm_set.index(vm)

		for ser in ser_residual.keys():
			if ser_residual[ser][0] - input_file_dict['flavor'][vm][0] >= 0 and \
			ser_residual[ser][1] - input_file_dict['flavor'][vm][1] >= 0:
				server_list.append(ser)

		if len(server_list) == 0:
			ser_n = ser_n + 1
			ser_residual[str(ser_n)] = input_file_dict['cpu_mem_disk'][0:2]
			ser_p_vector[str(ser_n)] = map(lambda (a, b): a / float(b), \
									zip(ser_residual[str(ser_n)], input_file_dict['cpu_mem_disk'][0:2]))
			for key in input_file_dict['flavor'].keys():
				vm_pp_vector[key][str(ser_n)] = map(lambda (a, b): a / float(b),
											  zip(input_file_dict['flavor'][key], ser_residual[str(ser_n)]))
				# normalization
				vm_pp_vector[key][str(ser_n)] = [vm_pp_vector[key][str(ser_n)][0] / float(sum(vm_pp_vector[key][str(ser_n)])), \
										   vm_pp_vector[key][str(ser_n)][1] / float(sum(vm_pp_vector[key][str(ser_n)]))]
			server_list.append(str(ser_n))

		for ser in server_list:
			#caculate vm and server j match distance 1
			match_distance1.append({ser:Match_distance(vm_pp_vector[vm][ser], ser_p_vector[ser])})
			#caculate vm and server j match distance
			ser_residual_temp = copy.deepcopy(ser_residual)
			ser_normal_v = normalization(ser_residual_temp)
			match_distance2.append({ser:Match_distance(vm_normal_v[vm], ser_normal_v[ser])})
		# synthesize two match distance, find server with least distance, which is the server to placement this vm
		#match_d = map(lambda (a,b):{k:a[k]+b[k]} , zip(match_distance1, match_distance2))
		for i in range(len(server_list)):
			match_list.setdefault(server_list[i],match_distance1[i][server_list[i]] + match_distance2[i][server_list[i]])
		# set this vm in minimum match distance server
		ser_sel = min(match_list)
		allocate_method[ser_sel].append(vm)
		# update residual resource of server
		ser_residual[ser_sel] = map(lambda (a,b):a-b, zip(ser_residual[ser_sel], input_file_dict['flavor'][vm]))
		#update vm set
		#vm_set.pop(index)
		# update server performance vector
		ser_p_vector = update_ser_p_vector(ser_p_vector, ser_residual)
		vm_pp_vector = update_vm_pp_vector(ser_residual)
		# update match_d
		match_list = {}
		match_distance1 = []
		match_distance2 = []
		server_list = []

	assert sum(len(allocate_method[key]) for key in allocate_method.keys()) == total_flavor
	print ser_residual
	print ser_n
	print 'cpu utilization ratio:'
	util_cpu = 1 - sum(ser_residual[key][0] for key in ser_residual.keys()) / float(ser_n * input_file_dict['cpu_mem_disk'][0])
	print util_cpu

	print 'mem utilization ratio:'
	util_mem = 1 - sum(ser_residual[key][1] for key in ser_residual.keys()) / float(ser_n * input_file_dict['cpu_mem_disk'][1])
	print util_mem

	if input_file_dict['resource_name'] == 'CPU':
		return ser_n, allocate_method, util_cpu
	else:
		return ser_n, allocate_method, util_mem


def divide_group_fit(total_flavor, specific_flavor_num, input_file_dict):


	# calculate total server resource of total predicted vm
	resource_cpu = 0
	resource_mem = 0
	flavor = copy.deepcopy(input_file_dict['flavor'])
	cpu = copy.deepcopy(input_file_dict['cpu_mem_disk'][0])
	mem = copy.deepcopy(input_file_dict['cpu_mem_disk'][1])

	for key, value in specific_flavor_num.items():
		resource_cpu = resource_cpu + value * input_file_dict['flavor'][key][0]
		resource_mem = resource_mem + value * (input_file_dict['flavor'][key][1])

	m1 = int(math.ceil(resource_cpu / float(input_file_dict['cpu_mem_disk'][0])))
	m2 = int(math.ceil(resource_mem / float(input_file_dict['cpu_mem_disk'][1])))
	# requirement server number
	ser_n = max(m1, m2)

	vm_set = []  # the set of virtual machine need allocate
	for key, value in specific_flavor_num.items():
		for i in range(value):
			vm_set.append(key)

	assert len(vm_set) == total_flavor

	allocate_method = defaultdict(list)

	# vm sequence
	#vm_type = defaultdict(list)  # the set of virtual machine need allocate

	vm_type = {1:['flavor1','flavor4','flavor7','flavor10','flavor13'],
			   2:['flavor2','flavor5','flavor8','flavor11','flavor14'],
	           3:['flavor3','flavor6','flavor9','flavor12','flavor15']}

	'''
	for key, value in specific_flavor_num.items():
		if int(re.findall(r"\d+", key)[0]) % 3 == 1:
			for i in range(value):
				vm_type[1].append(key)
		elif int(re.findall(r"\d+", key)[0]) % 3 == 2:
			for i in range(value):
				vm_type[2].append(key)
		else:
			for i in range(value):
				vm_type[3].append(key)

	assert sum(len(vm_type[k]) for k in vm_type.keys()) == total_flavor
	
	#sort every vm type in descend
	for key in vm_type.keys():
		vm_type[key].sort(key=lambda i: int(re.findall(r"\d+", i)[0]), reverse=True)
	'''
	#list add function
	def add(list1, list2):
		return map(lambda (a,b):a+b, zip(list1, list2))

	def sub(list1,list2):
		return map(lambda (a,b):a-b, zip(list1, list2))

	#vm_list = list([])
	ser = 1
	server = {ser:[cpu, mem]}
	vm_set.sort(key=lambda i: int(re.findall(r"\d+", i)[0]), reverse=True)
	vm_num = 0
	#for ser in range(1,ser_n+1):
	#for i in range(len(vm_type[3])):
	#vm = [0,0]
	'''
	while len(vm_type[3])!= 0:
		#vm = add(vm, flavor[vm_type[3][0]])
		vm = flavor[vm_type[3][0]]
		#r = vm[0] / float(vm[1])
		#k =
		vm_list.append(vm_type[3][0])
		while server[ser][0] >= 0 and server[ser][1] >= 0:
			#vm_list.append(vm_type[3][0])
			if len(vm_type[1]) != 0:
				#for k1 in range(vm_type[1]):
					#vm = add(vm, flavor[vm_type[1][k]])
				vm = add(vm, flavor[vm_type[1][0]])
				if vm[0] > server[ser][0] or vm[1] > server[ser][1]:
					break
				else:
					vm_list.append(vm_type[1][0])
					vm_type[1].pop(0)
			elif len(vm_type[2]) != 0:
				#for k2 in range(len(vm_type[2])):
				vm = add(vm, flavor[vm_type[2][0]])
				if vm[0] > server[ser][0] or vm[1] > server[ser][1]:
					break
				else:
					vm_list.append(vm_type[2][0])
					vm_type[2].pop(0)
			else:
				break
			server[ser] = sub(server[ser], vm)
		vm_type[3].pop(0)
		allocate_method[str(ser)] = vm_list
		#if server[0]
		vm_list = []
		vm = []
		ser += 1
		server[ser]=[cpu,mem]
	'''

	while vm_num <= total_flavor:
		vm = random.choice(vm_set)
		index = vm_set.index(vm)
		while server[ser][0] / float(cpu) <= 0.5 and server[ser][1] / float(mem) <= 0.5:
			server = sub(server[ser], flavor[vm])
			allocate_method[str(ser)].append(vm)
			vm_set.pop(index)








	return total_server, allocate_method