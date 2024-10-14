import torch

def load_model_weights(saved_path, param_keys):
    device = torch.device("cpu")
    state_dict_ = torch.load(saved_path, map_location=device)
    state_dict = {}
    for k in state_dict_:
        if any(param in k for param in param_keys):   # 只加载指定的参数
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
    
    return state_dict

def compare_model_params(state_dict1, state_dict2, param_keys):
    # 要忽略的参数
    ignore_params = [ 'num_batches_tracked']
    
    for param in param_keys:
        # 筛选出指定模块的参数
        params1 = {k: v for k, v in state_dict1.items() if param in k}
        params2 = {k: v for k, v in state_dict2.items() if param in k}
        
        for k in params1:
            # 跳过忽略的参数
            if any(ignored in k for ignored in ignore_params):
                continue
            
            if k in params2:
                if torch.equal(params1[k], params2[k]):
                    pass
                    # print(f"Parameter '{k}' is the same in both models.")
                else:
                    print(f"Parameter '{k}' is different between the two models.")
            else:
                print(f"Parameter '{k}' not found in second model.")
        
        for k in params2:
            # 跳过忽略的参数
            if any(ignored in k for ignored in ignore_params):
                continue
            
            if k not in params1:
                print(f"Parameter '{k}' not found in first model.")


# 加载两个模型的权重
param_keys = ['model_infra', 'model_vehicle', 'reg_head', 'cls_head']
file1_path = 'opencood/logs/exp/flowPre_dair_where2comm_max_multiscale_resnet_2024_10_13_23_53_27/net_epoch7.pth'
file2_path = 'opencood/logs/exp/flowPre_dair_where2comm_max_multiscale_resnet_2024_10_13_23_53_27/net_epoch11.pth'

state_dict1 = load_model_weights(file1_path, param_keys)
state_dict2 = load_model_weights(file2_path, param_keys)

# 比较权重
compare_model_params(state_dict1, state_dict2, param_keys)
