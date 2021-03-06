import csv


###AUXILIARY###
def get_weights_list_format(weights):
    weight_list = []
    for layer in weights:
        weight_layer=[]
        for node in layer:
            w = node['weights']
            weight_layer.append(w)
        weight_list.append(weight_layer)
    return weight_list

def write_to_csv(weights,name_file):
    network = get_weights_list_format(weights)
    with open(name_file, "w", newline="") as f:
        for indxLayer in range(len(network)-1 , -1, -1):
            for indxNode in range(len(network[indxLayer])-1 , -1, -1):
                for weight in network[indxLayer][indxNode]:
                    f.write(str(weight))
                    f.write('\n')
    pass
