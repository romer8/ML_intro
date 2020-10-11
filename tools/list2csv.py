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

def write_to_csv(weights):
    network = get_weights_list_format(weights)
    with open("weights.csv", "w", newline="") as f:
        for layer in network:
            for node in layer:
                for weight in node:
                    f.write(str(weight))
                    f.write('\n')
    pass
