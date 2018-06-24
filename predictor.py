from preprocess import process_train_data, process_input_file, generate_result_file, test
from predict_methods import weighted_mean, weighted_mean2, weighted_mean3, exp_smooth
from allocate import allocate_resource_first_fit, allocate_resource_best_fit, least_distance_match, divide_group_fit
from bp_neural_network import nn_app_utils


#def predict_vm(ecs_lines, input_lines,test_array):
def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    flavor_dict, date, flavor_sum = process_train_data(ecs_lines)
    input_file_dict, flavor_sel = process_input_file(input_lines, flavor_dict)

    # predict vm total number and specific flavor number #
    #vm_num, vm_flavor_num = weighted_mean2(input_file_dict['start_end_time'], flavor_sel)
    #vm_num, vm_flavor_num = weighted_mean(input_file_dict['start_end_time'], flavor_sel, date)
    #vm_num, vm_flavor_num = weighted_mean3(input_file_dict['start_end_time'], flavor_sel, date)
    if len(date) < 60:
    　　　　　vm_num, vm_flavor_num = exp_smooth(flavor_sel, date, input_file_dict['start_end_time'])
    else:
    	　vm_num, vm_flavor_num = nn_app_utils(flavor_sum, flavor_sel, input_file_dict['start_end_time'])
    # allocate vm in physical server #
    #server_num, alloc_way, util= allocate_resource_first_fit(vm_num, vm_flavor_num, input_file_dict)
    server_num, alloc_way, util= allocate_resource_best_fit(vm_num, vm_flavor_num, input_file_dict)
    #server_num, alloc_way, util = least_distance_match(vm_num, vm_flavor_num, input_file_dict)
    #server_num, alloc_way = divide_group_fit(vm_num, vm_flavor_num, input_file_dict)

    '''
    score = test(input_file_dict, vm_flavor_num, test_array, util)
    print 'final score:'
    print score*100
    '''
    result = list([])
    result = generate_result_file(vm_num, vm_flavor_num, server_num, alloc_way)

    return result
