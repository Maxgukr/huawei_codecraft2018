# coding=utf-8
import sys
import os
import predictor
import time



def main():
    print 'main function begin.'
    start = time.clock()

    if len(sys.argv) != 4:
        print 'parameter is incorrect!'
        print 'Usage: python esc.py ecsDataPath inputFilePath resultFilePath'
        exit(1)

    # Read the input files

    ecsDataPath = sys.argv[1]#'./TrainData_2015.1.1_2015.2.19.txt' #
    inputFilePath = sys.argv[2]#'./input_5flavors_cpu_7days.txt' #
    resultFilePath = sys.argv[3]#'./output_file.txt' #
    '''
    ecsDataPath = './data_2015_1-5.txt'#'./TrainData_2015.1.1_2015.2.19.txt' #
    #inputFilePath = './input_5flavors_cpu_7days.txt'#'./data_2015_1-5.txt'#'./TrainData_2015.1.1_2015.2.19.txt'#'./data_2015_1-5.txt' #
    inputFilePath = './input_5flavors_cpu_7days.txt'
    resultFilePath = './output_file.txt'
    testFilePath ='./test5.23-5.29.txt'#'./TestData_2015.2.20_2015.2.27.txt'#
    '''
    ecs_info_array = read_lines(ecsDataPath)
    input_file_array = read_lines(inputFilePath)
    #test_array = read_lines(testFilePath)

    # implementation the function predictVm
    #predict_result = predictor.predict_vm(ecs_info_array, input_file_array, test_array)
    predict_result = predictor.predict_vm(ecs_info_array, input_file_array)
    # write the result to output file
    if len(predict_result) != 0:
        write_result(predict_result, resultFilePath)
    else:
        predict_result.append("NA")
        write_result(predict_result, resultFilePath)

    print 'main function end.'
    elapsed = (time.clock() - start)
    print 'time used :'
    print elapsed


def write_result(array, outputFilePath):
    with open(outputFilePath, 'w') as output_file:
        for item in array:
            output_file.write("%s\n" % item)


def read_lines(file_path):
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                array.append(line)
        return array
    else:
        print 'file not exist: ' + file_path
        return None


if __name__ == "__main__":
    main()

