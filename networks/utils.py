import torch

def load_pretrained_net(net, path):
    '''allow partial loading
    '''

    device = next(net.parameters()).device

    # load from checkpoint or state_dict
    print(f'trying to load pretrained from {path}')
    try:
        state_dict = torch.load(path, map_location=device)['state_dict']
    except:
        state_dict = torch.load(path, map_location=device)

    all_okay = True

    new_weights = net.state_dict()

    # partial loading. check key and shape
    for k in new_weights.keys():
        if not k in state_dict.keys():
            print(f'{k} is missing in pretrained')
            all_okay = False
        else:
            if new_weights[k].shape != state_dict[k].shape:
                print(f'skip {k}, required shape: {new_weights[k].shape}, pretrained shape: {state_dict[k].shape}')
                all_okay = False
            else:       
                new_weights[k] = state_dict[k]
    
    try:
        net.load_state_dict(new_weights)
        if all_okay:
            print('<All weights loaded successfully>')
    except:
        print(f'cannot load {path}. using intial net.')
        pass
    
    return net