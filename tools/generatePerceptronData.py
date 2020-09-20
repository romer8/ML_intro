import numpy as np
import random

def _generateRandomData(linearSeparableBool,dataSize,rangeBetweenVals):
    randomArray = []

    targets_array = []

    for indx in range(dataSize[0]):
        if indx < (dataSize[0])/2:
            targets_array.append(1)
        else:
            targets_array.append(0)

    inputs_1= np.random.uniform(rangeBetweenVals[0],rangeBetweenVals[1],dataSize).tolist()
    inputs_2= np.random.uniform(rangeBetweenVals[0],rangeBetweenVals[1],dataSize).tolist()
    for input_1, input_2,targetx in zip(inputs_1,inputs_2,targets_array):
        if linearSeparableBool:
            patternInstance = [round(input_1,1),round(input_2,1),int(targetx)]
            randomArray.append(patternInstance)
        else:
            # patternInstance = [input_1, input_2,input_1*input_2,random.choice([0, 1])]
            input_3 = float(input_2)*float(input_1)
            patternInstance = [round(input_1,1),round(input_3,1),round(input_2,1),int(targetx)]

            randomArray.append(patternInstance)
    return randomArray

def _generateFileDict(nameFile,linearSeparableBool,dataSize,rangeBetweenVals):
    return_dict ={}
    return_dict[f'{nameFile}']=_generateRandomData(linearSeparableBool,dataSize,rangeBetweenVals)

    return return_dict

def generateDataSet(nameFile,linearSeparableBool,dataSize,rangeBetweenVals):
    d = _generateFileDict(nameFile,linearSeparableBool,dataSize,rangeBetweenVals)
    for original_filename in d.keys():
        m = original_filename
        output_filename = m +'.arff'
        with open(output_filename,"w") as fp:
            if linearSeparableBool:
                fp.write('''@relation linSeparableDataSet
@attribute a1 real
@attribute a2 real
@attribute class {0,1}
@data
''')
            else:
                fp.write('''@relation nonLinSeparableDataSet
@attribute a1 real
@attribute a2 real
@attribute a3 real
@attribute class {0,1}
@data
''')
            for datas in d[original_filename]:
                if linearSeparableBool:
                    data = (str(datas[0]),str(datas[1]),str(datas[2]))
                    fp.write("%s,%s,%s\n" % data)
                else:
                    data = (str(datas[0]),str(datas[1]),str(datas[2]),str(datas[3]))
                    fp.write("%s,%s,%s,%s\n" % data)



# generateDataSet("data1",True,(8,),[-1,1])
# generateDataSet("data2",False,(8,),[-1,1])
