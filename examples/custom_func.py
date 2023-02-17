import torch
extra_outputs = 1
extra_features = -1

def custom_func(output, custom_input):
    temperatures = torch.tensor(custom_input).to(
                                                    device=output.device
                                                    , dtype=output.dtype)
    
    output = (output[:, 0] - (output[:, 1] / (temperatures / 500) )
                      ).reshape(-1, 1)

    return output

def modify_features(features_batch):
    custom_input = [features[0] for features in features_batch]
    features_batch = [features[1:] for features in features_batch]
    return custom_input, features_batch