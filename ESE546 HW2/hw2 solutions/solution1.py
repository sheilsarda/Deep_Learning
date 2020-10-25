from resnet import resnet18
from torchsummary import summary
from pprint import pprint


# Part (c)
resnet = resnet18().to('cpu')
summary(resnet, (3, 224, 224))


# Part(c): Alternate method
def add_wt(num_wts, key, value):
    if key not in num_wts:
        num_wts[key] = 0
    num_wts[key] += value


layer_map = {}
wts_maps = {}
for name, obj in resnet.named_parameters():
    if name == "conv1.weight" or name == "bn1.weight" or name == "bn1.bias":
        add_wt(wts_maps, "conv1", obj.numel())
    elif name == "fc.weight" or name == "fc.bias":
        add_wt(wts_maps, "fc", obj.numel())
    else:
        name_s = name.split(".")
        layer_name = name_s[0] + "-" + name_s[1]
        add_wt(layer_map, name_s[0], obj.numel())
        if ("conv1.weight" in name) or ("bn1.weight" in name) or \
                ("bn1.bias" in name):
            add_wt(wts_maps, layer_name + "-conv1", obj.numel())
        elif ("conv2.weight" in name) or ("bn2.weight" in name) or \
                ("bn2.bias" in name):
            add_wt(wts_maps, layer_name + "-conv2", obj.numel())
        elif ("downsample" in name):
            add_wt(wts_maps, layer_name + "-downsample", obj.numel())

pprint(wts_maps)
print("---------")
pprint(layer_map)

# Part(e): Batch-norm split
batchNormGroup, biasGroup, restGroup = [], [], []
for name, param in resnet.named_parameters():
    numparam = 1
    print(name)
    for x in param.shape:
        numparam = numparam * x
    if "bn" in name or "downsample.1" in name:
        batchNormGroup.append((name, numparam))
    elif "bias" in name:
        biasGroup.append((name, numparam))
    else:
        restGroup.append((name, numparam))

# Batch norm weights/bias
tot_params = [0, 0, 0]
for name, numparam in batchNormGroup:
    tot_params[0] = tot_params[0] + numparam

# From Finaly FC and downsample layers
for name, numparam in biasGroup:
    tot_params[1] = tot_params[1] + numparam

# All conv wts/FC wts
for name, numparam in restGroup:
    tot_params[2] = tot_params[2] + numparam

print("--------------")
print("BatchNorm params: " + str((tot_params[0])))
print("Biases params: " + str((tot_params[1])))
print("Rest params: " + str((tot_params[2])))
print("Total params: " + str(sum(tot_params)))
