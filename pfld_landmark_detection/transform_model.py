
import torch
import torch.functional as F
import models
import sys
import os
import argparse
from collections import OrderedDict
work_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(work_root)


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean) / var_sqrt * beta + gamma
    fused_conv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        groups=conv.groups,
        bias=True)
    fused_conv.weight = torch.nn.Parameter(w)
    fused_conv.bias = torch.nn.Parameter(b)
    return fused_conv


def fuse_module(m):
    children = list(m.named_children())
    c = None
    cn = None

    for name, child in children:
        if isinstance(child, torch.nn.BatchNorm2d) and c is not None:
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = torch.nn.Identity()
            c = None
        elif isinstance(child, torch.nn.Conv2d):
            c = child
            cn = name
        else:
            c = None
            fuse_module(child)


def pytorch_version_to_0_3_1():
    """
    pytorch0.3.1导入0.4.1以上版本模型时加入以下代码块,可对比查看_utils.py文件修正相似错误,
   错误类型为(AttributeError: Can't get attribute '_rebuild_tensor_v2' on
   <module 'torch._utils' from '<pytorch0.3.1>\lib\site-packages\torch\_utils.py'>)
    Returns
    -------
    """
    #  使用以下函数代替torch._utils中的函数(0.3.1中可能不存在或者接口不同导致的报错)
    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    try:
        torch._utils._rebuild_parameter
    except AttributeError:
        def _rebuild_parameter(data, requires_grad, backward_hooks):
            param = torch.nn.Parameter(data, requires_grad)
            param._backward_hooks = backward_hooks
            return param
        torch._utils._rebuild_parameter = _rebuild_parameter


def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = True
    elif isinstance(module, torch.nn.Upsample):
        module.align_corners = False
    else:
        for name, module1 in module._modules.items():
            module1 = recursion_change_bn(module1)


def recursion_change_bn1(module):
    print(type(module))
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = True
    elif isinstance(module, torch.nn.Upsample):
        del module.align_corners
    else:
        for name, module1 in module._modules.items():
            module1 = recursion_change_bn1(module1)


def pth_to_jit(model, save_path, device="cuda:0"):
    model.eval()
    input_x = torch.randn(1, 3, 144, 144).to(device)
    new_model = torch.jit.trace(model, input_x)
    torch.jit.save(new_model, save_path)


def jit_to_onnx(jit_model_path, onnx_model_path):
    model = torch.jit.load(jit_model_path, map_location=torch.device('cuda:0'))
    model.eval()
    example_input = torch.randn(1, 3, 144, 160).to("cuda:0")
    example_output = torch.rand(1, 735, 1).to("cuda:0"), torch.randn(1, 735, 4).to("cuda:0")
    torch.onnx._export(model, example_input, onnx_model_path, example_outputs=example_output, verbose=True)


def export_model_0_3_1(checkpoint_path, export_model_name, inputsize=[1, 3, 144, 144]):

    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    pytorch_version_to_0_3_1()
    check_point = torch.load(checkpoint_path, map_location=device)
    # model = check_point['net']
    # state_dict = check_point['net'].state_dict()
    # model = models.init_model(name="resnet18mid", pretrained=False, num_classes=11196)
    model = models.init_model(name="mobilenetv3_small", pretrained=False, num_classes=1000)
    state_dict = check_point
    model = model.cuda() if device == f"cuda:0" else model
    mapped_state_dict = OrderedDict()
    for name, module in model._modules.items():
        recursion_change_bn1(module)
    for key, value in state_dict.items():
        # print(key)
        mapped_key = key
        mapped_state_dict[mapped_key] = value
        if 'num_batches_tracked' in key:
            del mapped_state_dict[key]
    model.load_state_dict(mapped_state_dict)
    model.eval()
    # for key, value in model.state_dict().items():
    #     print(key)
    dummy_input = torch.autograd.Variable(torch.randn(inputsize))
    dummy_input = dummy_input.cuda() if device == f"cuda:0" else dummy_input
    torch.onnx.export(model=model, args=dummy_input, f=export_model_name, verbose=True)


def export_model(checkpoint_path, export_model_name, inputsize=[1, 3, 112, 112], combine_conv_bn=False):
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    check_point = torch.load(checkpoint_path, map_location=device)
    model = models.init_model(name="ghost_pfld")
    if combine_conv_bn:
        fuse_module(model)
    state_dict = check_point['net'].state_dict()
    model = model.to(device) if device == f"cuda:0" else model
    model.load_state_dict(state_dict)
    model.eval()
    for key, value in state_dict.items():
        print(key)
    dummy_input = torch.randn(inputsize).to(device)
    # torch.onnx.export(model=model, args=dummy_input, f=export_model_name, verbose=True)
    torch.onnx.export(model=model, args=dummy_input, f=export_model_name, verbose=True, input_names=['input'],
                      output_names=['angle', 'landmark'])  # 0.4.0以上支持更改输入输出层名称


def export_jit(checkpoint_path, export_model_name, combine_conv_bn=False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    check_point = torch.load(checkpoint_path, map_location=device)
    # model = check_point['net']
    # state_dict = check_point['net'].state_dict()
    model = models.init_model(name="osnet_x0_5", pretrained=False, num_classes=7361)
    if combine_conv_bn:
        fuse_module(model)
    state_dict = check_point['net']
    model = model.cuda() if device == "cuda:0" else model
    model.load_state_dict(state_dict)
    pth_to_jit(model, export_model_name, device)


def export_feature(checkpoint_path, export_model_name, combine_conv_bn=False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    check_point = torch.load(checkpoint_path, map_location=device)
    # model = check_point['net']
    # state_dict = check_point['net'].state_dict()
    # model = models.init_model(name="resnet18mid", pretrained=False, num_classes=9991)
    # state_dict = check_point['net']
    model = models.init_model(name="osnet_x0_5", pretrained=False, num_classes=11196)
    if combine_conv_bn:
        fuse_module(model)
    state_dict = check_point['net']
    # model = model.cuda() if device == "cuda:0" else model
    # model.load_state_dict(state_dict)
    torch.save(state_dict, export_model_name)



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        "--model_path",
        type=str,
        default=os.path.join(
            work_root, "models/ghost_pfld_374_1.0200.pth"),
        help="Path of the pytorch model file",
    )
    parser.add_argument(
        '-t',
        '--transform_type',
        type=int,
        default=2,
        help='inference times',
    )
    parser.add_argument(
        '-c',
        '--combine_conv_bn',
        action='store_true',
        help='use combine conv and bn, default is false!',
    )
    return parser.parse_args()


def main():
    args = parser_args()
    print(args)
    model_path = args.model_path
    export_model_name = model_path.replace(".pth", ".onnx")
    print("模型路径为：", os.path.realpath(model_path))
    transform_type = args.transform_type
    combine_conv_bn = args.combine_conv_bn
    if transform_type == 0:
        export_feature(model_path, export_model_name, combine_conv_bn=combine_conv_bn)
    if transform_type == 1:
        if torch.__version__ >= "0.4.0":
            print(torch.__version__)
            print("pytorch version must <  0.3.1, please check it!")
            exit(-1)
        export_model_0_3_1(checkpoint_path=model_path, export_model_name=export_model_name)
    if transform_type == 2:
        if torch.__version__ < "0.4.0":
            print("pytorch version must >  0.4.0, please check it!")
            exit(-1)
        export_model(checkpoint_path=model_path, export_model_name=export_model_name, combine_conv_bn=combine_conv_bn)
    if transform_type == 3:
        if torch.__version__ < "1.0.0":
            print("pytorch version must >  1.0.0, please check it!")
            exit(-1)
        export_jit(model_path, export_model_name, combine_conv_bn=combine_conv_bn)


if __name__ == '__main__':
    main()

