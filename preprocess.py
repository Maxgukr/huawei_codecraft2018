import datetime as dt
import re
from collections import Counter, defaultdict, OrderedDict
import math
import pickle
import copy


def parse_datetime(time_string):
    """
    :param time_string:
    :return: like 2018-3-11,type of datetime class
    """
    date_str = time_string.split(' ')[0].split('-') # miss the hour min and sec
    date_time = dt.date(int(date_str[0]), int(date_str[1]), int(date_str[2]))
    return date_time


def find_num_in_str(string):
    """
    :param string:
    :return: the digit in the str,type of string,'1'
    """
    return re.findall(r"\d+", string)[0]


def process_train_data(data_lines):
    """
    :param list data_lines:
    :return:
    """

    flavor_dict = {'flavor1': [], 'flavor2': [], 'flavor3': [], 'flavor4': [], 'flavor5': [],
                   'flavor6': [], 'flavor7': [], 'flavor8': [], 'flavor9': [], 'flavor10': [],
                   'flavor11': [], 'flavor12': [], 'flavor13': [], 'flavor14': [], 'flavor15': []}

    if data_lines is None:
        print 'ecs information is none'
        return flavor_dict

    # pick out the flavor and date, as two lists
    flavor = list([])
    date_time = list([])
    for i in range(len(data_lines)):
        line = data_lines[i].rstrip('\r\n').split('\t')  # every line will be divided into 3 parts.ignore the hour,min,sec
        flavor.append(line[1])  # save as type as list
        date_time.append(parse_datetime(line[2]))

    """
    the following code traversal all dates in file date_lines
    """
    date_t = list(set(date_time))
    date_t.sort(key=date_time.index)
    date_comp = date_t[0]  # assign the initial value
    while date_comp <= date_t[-1]:
        # find the repeated times and its name in the specific date, and convert to dict
        # this statement may have a little problem, spend too much running time
        # the index for flavor with date = date_comp
        index = [i for i, x in enumerate(date_time) if x == date_comp]  # this place can be faster, need improve
        flavor_date_dict = dict(Counter(flavor[k] for k in index))  # the list of flavor under certain date
        flavor_date_list = list(flavor_date_dict.keys())
        flavor_list = list(flavor_dict.keys())

        # add the repeated times in this date to flavor_dict value
        for key1 in flavor_date_list:
            if key1 in flavor_list:
                flavor_dict[key1].append(flavor_date_dict[key1])

        # add 0 to the flavor_dict which flavor don't appear in this date
        for key2 in list(set(flavor_list) - set(flavor_date_list)):
            flavor_dict[key2].append(0)

        # date add one day
        date_comp = date_comp + dt.timedelta(days=1)

    # remove duplicated items in date list
    flavor_sum = []
    for i in range(len(flavor_dict['flavor1'])):
         flavor_sum.append(sum(flavor_dict[key][i] for key in flavor_dict.keys()))

    #f1 = open('./flavor_sum.txt','wb')
    #pickle.dump(flavor_sum, f1)

    #f = open('./flavor_dict_01-05.txt','wb')
    #pickle.dump(flavor_dict,f)

    return flavor_dict, date_t, flavor_sum


def process_input_file(input_array, flavor_dict):
    """
    :param list input_array: read from txt
    :param dict flavor_dict: flavor's number in every date, save as dict
    :return: input_file_dict = {'cpu_mem_disk':['','',''],
                                'flavor_num':[''],
                               'flavor':{'flavor1':[1, 2024],...'flavor5':[2, 4096]},
                               'resource_name':[''],
                               'start_end_time':['','']}
    """
    if input_array is None:
        print 'input file information is none'
        return 0
    if flavor_dict is None:
        print 'input parameter type is wrong'
        return 0
    assert type(flavor_dict) == dict

    while '\r\n' in input_array:
        input_array.remove('\r\n')  # remove the \n

    input_file_dict = {}
    flavor = {}
    # save cpu_core, mem_num, disk_volume
    # strip()delete the \n in string end
    input_file_dict['cpu_mem_disk'] =map(lambda x:int(x), input_array[0].rstrip('\r\n').split())
    input_file_dict['flavor_num'] = input_array[1].rstrip('\r\n')

    for i in range(1, int(input_file_dict['flavor_num'])+1):
        line = input_array[1 + i].rstrip('\r\n').split()
        flavor[line[0]] = [int(line[1]), int(line[2])/1024]

    input_file_dict['flavor'] = flavor
    input_file_dict['resource_name'] = input_array[1+int(input_file_dict['flavor_num'])+1].rstrip('\r\n')
    # start_end_time need transform to datetime type
    input_file_dict['start_end_time'] = [parse_datetime(input_array[len(input_array)-2].rstrip('\r\n'))]
    input_file_dict['start_end_time'].append(parse_datetime(input_array[len(input_array)-1]))

    # abstract info from flavor_dict as input file indicating
    flavor_sel = {key: value for key, value in flavor_dict.items() if key in input_file_dict['flavor'].keys()}

    return input_file_dict, flavor_sel


def generate_result_file(total_vm, specific_flavor_num, server_total, allocate_method):
    """
    :param int total_vm: int type; total virtual machine number
    :param dict specific_flavor_num: dict type; every specific flavor's number of vm
    :param int server_total: int type; total server number
    :param dict allocate_method: dict type; dict type like :
        {'1':flavor1,flavor1
         '2':flavor2,flavor2,flavor3}
         need refactor
    :return:
    """
    """
    example:
    6 #total flavor numbers
    flavor5  3 #specific flavor numbers
    flavor10  2
    flavor15  1

    4 #total servers number
    1  flavor5  2
    2  flavor5  1  flavor10  1
    3  flavor15  1
    4  flavor10  1
    """
    if allocate_method is None:
        print 'generate_result_file() parameter error!'

    assert type(allocate_method) == defaultdict

    result_array = list([])
    result_array.append(str(total_vm))

    specific_flavor_num_list = list(specific_flavor_num.keys())
    specific_flavor_num_list.sort(key=lambda i: int(re.findall(r"\d+", i)[0]), reverse=False)
    for key1 in specific_flavor_num_list:
        result_array.append(str(key1)+' '+str(specific_flavor_num[key1]))

    result_array.append('')
    result_array.append(str(server_total))

    sub_dict = OrderedDict()
    allocate_list = list(allocate_method.keys())
    allocate_list.sort(key=lambda i: int(re.findall(r"\d+", i)[0]), reverse=False)

    for key2 in allocate_list:
        sub_dict[key2] = dict(Counter(allocate_method[key2]))

    allocate_str = str('')

    for key3, value3 in sub_dict.items():
        allocate_str = str(key3)
        for key4, value4 in value3.items():
            allocate_str = allocate_str + ' ' + str(key4) + ' ' + str(value4)
        result_array.append(allocate_str)

    return result_array


def test(input_file_dict, predict_flavor_num, test_array, util_ratio):
    test_flavor, test_date, test_sum = process_train_data(test_array)
    for key in predict_flavor_num.keys():
        if key in test_flavor.keys():
            test_flavor[key] = sum(test_flavor[key])
        else:
            test_flavor[key] = 0

    assert len(test_flavor.keys())==len(predict_flavor_num.keys())

    s_real = math.sqrt((1/float(int(input_file_dict['flavor_num'])))*sum(test_flavor[key]**2 for key in test_flavor.keys()))
    s_predict = math.sqrt((1/float(int(input_file_dict['flavor_num'])))*sum(math.pow(predict_flavor_num[key],2) for key in predict_flavor_num.keys()))
    s_diff = math.sqrt((1/float(int(input_file_dict['flavor_num'])))*sum(math.pow(test_flavor[key]-predict_flavor_num[key],2) for key in test_flavor.keys()))

    score = (1.0 - (s_diff / float(s_predict + s_real))) * util_ratio
    print 'predict score:'
    print score / float(util_ratio)

    return score


def slice_data(input_data, supervise_len):
    '''
    :param input_data: input data
    :param train_len: every training example data length
    :param supervise_len: every supervise data length, same as predict date length
    training length longer than supervise length
    :return:
    '''
    '''
    slice data for training by neural network
    previous week data as train data, next week data as supervise data, divide by predict date length
    '''

    example_num = int(math.floor((len(input_data)) / float(supervise_len)))
    train_example = []
    supervise_example = []

    for i in range(example_num-1):
        #temp1 = input_data[i * train_len:(i + 1) * train_len]
        train_example.append(input_data[i * supervise_len:(i+1) * supervise_len])#temp1
        #index1 = i * supervise_len + train_len
        supervise_example.append(input_data[(i+1) * supervise_len:(i+2) * supervise_len])
        #index2 = i * supervise_len + train_len + supervise_len
        #supervise_example[i] = temp2

    #train_example[example_num-1] = input_data[len(input_data) - train_len - supervise_len:len(input_data) - supervise_len]
    #supervise_example[example_num-1] = input_data[len(input_data) - supervise_len:len(input_data)]

    return train_example, supervise_example


def roll_mean(win_len, flavor_sum):
    flavor_sum1 = copy.deepcopy(flavor_sum)
    for i in range(len(flavor_sum1)):
        flavor_temp = flavor_sum1[i * win_len:(i + 1) * win_len]
        flavor_mean = sum(flavor_temp) / win_len
        if len(flavor_temp) == win_len:
            for j in range(i * win_len,(i + 1) * win_len):
                if flavor_sum1[j] > 2*flavor_mean:
                    flavor_sum1[j] = copy.deepcopy(flavor_mean)
        else:
            for j in range(len(flavor_sum) - len(flavor_sum) % win_len, len(flavor_sum)):
                if flavor_sum1[j] > 2 * flavor_mean:
                    flavor_sum1[j] = copy.deepcopy(flavor_mean)
    return flavor_sum1


def normalize_data(input_data):
    '''
    normalize the input data to (0,1)
    :param input_data:
    :return: normalized_data
    '''
    m = len(input_data)
    data_mean = sum(input_data) / float(m)
    data_sigma = sum(map(lambda (a,b):(a-b)**2, zip(input_data, [data_mean for i in range(m)]))) / float(m)

    normalized_data = []

    for i in range(len(input_data)):
        data = (input_data[i] - data_mean) / float(data_sigma)
        normalized_data.append(data)

    return normalized_data, data_mean, data_sigma


def recover_normalized_data(normal_data, data_mean, data_sigma):
    '''

    :param normal_data: normalized data need recover
    :param data_max:
    :param data_min:
    :return:
    '''
    rec_data = []
    for i in range(len(normal_data)):
        rec_data.append(normal_data[i] * data_sigma + data_mean)
    return rec_data


def week_prob(flavor, flavor_sum, predict_len):
    '''

    :param flavor:
    :param flavor_sum:
    :return:
    '''
    #flavor_week_sum = 0
    vm_week_sum = {}
    week_prob_list = []
    num = int(math.floor(len(flavor_sum) / predict_len))
    prob = []

    for i in range(num):
        flavor_week_sum = sum(flavor_sum[i*predict_len:(i+1)*predict_len])

        for key, value in flavor.items():
            # each vm's prob in each week
            vm_week_sum[key] = sum(value[i*predict_len:(i+1)*predict_len]) / float(flavor_week_sum)

        key_list = vm_week_sum.keys()
        key_list.sort(key=lambda i: int(re.findall(r"\d+", i)[0]), reverse=False)

        for i in key_list:
            week_prob_list.append(vm_week_sum[i])

        prob.append(week_prob_list)
        week_prob_list = []

    train_prob = prob[0:len(prob)-1]
    supervise_prob = prob[1:len(prob)]

    #caculate last predict_len days
    vm_last_week_sum = {}
    week_prob_last = []
    flavor_last_week_sum = sum(flavor_sum[len(flavor_sum) - predict_len:len(flavor_sum)])

    for key_last, value_last in flavor.items():
        vm_last_week_sum[key_last] = sum(value_last[len(flavor_sum) - predict_len:len(flavor_sum)]) / float(flavor_last_week_sum)

    for i in key_list:
        week_prob_last.append(vm_last_week_sum[i])

    #key_last_list = vm_last_week_sum.keys()
    #key_last_list.sort(key=lambda i: int(re.findall(r"\d+", i)[0]), reverse=True)

    return train_prob, supervise_prob, week_prob_last


def main():
    f = open('flavor_sum.txt', 'rb')
    flavor_sum = pickle.load(f)
    flavor_normal, data_max, data_min = normalize_data(flavor_sum)
    #flavor_sum1 = roll_mean(7, flavor_sum)
    train_data, supervise_data = slice_data(flavor_normal, 7)

    f1 = open('./train_data.txt','wb')
    pickle.dump(train_data, f1)

    f2 = open('./supervised_data.txt', 'wb')
    pickle.dump(supervise_data, f2)

    f3 = open('./flavor_normal.txt','wb')
    pickle.dump(flavor_normal, f3)

if __name__ == '__main__':
    main()