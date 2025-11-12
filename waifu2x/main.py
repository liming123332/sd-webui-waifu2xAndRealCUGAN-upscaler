import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from modules import devices, shared
from tqdm import tqdm

from .yu45020.utils.prepare_images import ImageSplitter
from .yu45020.Models import UpConv_7, CARN_V2, network_to_half
# 导入RealCUGAN模型
import sys
ROOT_PATH = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_PATH / 'models' / 'RealCUGAN'))
from upcunet_v3 import RealWaifuUpScaler


FILE_PATH = str(Path(__file__).parent.absolute())

def tensorToPil(img):
    normalized_out = img.squeeze(0).permute(1, 2, 0) * 255
    numpy_image = np.clip(normalized_out.numpy(), 0, 255).astype(np.uint8)
    result = Image.fromarray(numpy_image)
    del normalized_out, numpy_image
    return result


def processImageWithSplitter(model, img: Image.Image):
    img = img.convert('RGB')
    # overlapping split
    # if input image is too large, then split it into overlapped patches
    # details can be found at [here](https://github.com/nagadomi/waifu2x/issues/238)
    img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
    img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
    with torch.no_grad():
        out = []
        for i in tqdm(img_patches):
            if shared.state.interrupted: return img
            i = i.to(devices.device)
            out.append(model(i))
            del i
    img_upscale = img_splitter.merge_img_tensor(out)
    result = tensorToPil(img_upscale)
    del img_upscale, out
    return result



_models_cache = {}

def getModel(noise: int, style: str):
    global _models_cache
    fileName = os.path.join(style.lower(), f'noise{noise}_scale2.0x_model.json')

    if fileName not in _models_cache:
        model = UpConv_7()
        modelDir = os.path.join(FILE_PATH, 'yu45020', 'model_check_points', 'Upconv_7')
        weightsPath = os.path.join(modelDir, fileName)
        model.load_pre_train_weights(weightsPath)
        model = model.to(devices.device)
        _models_cache[fileName] = model

    return _models_cache[fileName]



_model_carnV2 = None

def getCarnV2Model():
    global _model_carnV2
    if _model_carnV2 is None:
        model = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                                single_conv_size=3, single_conv_group=1,
                                scale=2, activation=nn.LeakyReLU(0.1),
                                SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))

        model = network_to_half(model)
        modelDir = os.path.join(FILE_PATH, 'yu45020', 'model_check_points', 'CARN_V2')
        weightsPath = os.path.join(modelDir, 'CARN_model_checkpoint.pt')
        model.load_state_dict(torch.load(weightsPath, map_location='cpu'))
        model = model.to(devices.device)
        _model_carnV2 = model

    return _model_carnV2

# RealCUGAN模型支持
_real_cugan_models_cache = {}

def getRealCUGANModel(scale: int, noise_level: int, model_type: str = 'standard'):
    """
    获取RealCUGAN模型
    scale: 缩放比例 (2, 3, 4)
    noise_level: 降噪级别 (0=no_denoise, 1=denoise1x, 2=denoise2x, 3=denoise3x, -1=conservative)
    model_type: 模型类型 ('standard' 或 'pro')
    """
    global _real_cugan_models_cache
    
    # 确保scale是整数，避免文件名中出现小数点
    scale = int(scale)
    
    # 确定模型文件路径
    if model_type == 'pro':
        model_dir = ROOT_PATH / 'models' / 'RealCUGAN' / 'weights_pro'
        if noise_level == 0:
            weight_name = f'pro-no-denoise-up{scale}x.pth'
        elif noise_level == 3:
            weight_name = f'pro-denoise3x-up{scale}x.pth'
        elif noise_level == -1:
            weight_name = f'pro-conservative-up{scale}x.pth'
        else:
            # 对于pro模型，只支持特定的降噪级别
            weight_name = f'pro-denoise3x-up{scale}x.pth'
    else:
        model_dir = ROOT_PATH / 'models' / 'RealCUGAN' / 'updated_weights'
        if noise_level == 0:
            weight_name = f'up{scale}x-latest-no-denoise.pth'
        elif noise_level == 1:
            weight_name = f'up{scale}x-latest-denoise1x.pth'
        elif noise_level == 2:
            weight_name = f'up{scale}x-latest-denoise2x.pth'
        elif noise_level == 3:
            weight_name = f'up{scale}x-latest-denoise3x.pth'
        elif noise_level == -1:
            weight_name = f'up{scale}x-latest-conservative.pth'
        else:
            weight_name = f'up{scale}x-latest-no-denoise.pth'
    
    weight_path = model_dir / weight_name
    cache_key = f'{scale}_{noise_level}_{model_type}'
    
    if cache_key not in _real_cugan_models_cache:
        try:
            # 使用half精度以节省显存
            use_half = devices.device.type == 'cuda'
            upscaler = RealWaifuUpScaler(scale, str(weight_path), half=use_half, device=str(devices.device))
            _real_cugan_models_cache[cache_key] = upscaler
        except Exception as e:
            print(f"Failed to load RealCUGAN model {weight_name}: {e}")
            return None
    
    return _real_cugan_models_cache[cache_key]

def processImageWithRealCUGAN(img: Image.Image, scale: int, noise_level: int, model_type: str = 'standard', tile_mode: int = 4, cache_mode: int = 0, alpha: float = 1.0):
    """
    使用RealCUGAN模型处理图像
    img: 输入图像
    scale: 缩放比例 (2, 3, 4)
    noise_level: 降噪级别 (0=no_denoise, 1=denoise1x, 2=denoise2x, 3=denoise3x, -1=conservative)
    model_type: 模型类型 ('standard' 或 'pro')
    tile_mode: 分块模式 (0=不分块, 1=长边减半, 2+=分块数量)
    cache_mode: 缓存模式 (0=使用缓存, 1=8bit量化缓存, 2=不使用缓存)
    alpha: 锐化程度
    """
    # 获取模型
    upscaler = getRealCUGANModel(scale, noise_level, model_type)
    if upscaler is None:
        print("RealCUGAN model not available, returning original image")
        return img
    
    # 转换PIL图像为numpy数组
    img_np = np.array(img.convert('RGB'))
    
    # 调用模型进行超分辨率处理
    with torch.no_grad():
        if shared.state.interrupted:
            return img
        # RealWaifuUpScaler的__call__方法接受numpy数组并返回numpy数组
        result_np = upscaler(img_np, tile_mode=tile_mode, cache_mode=cache_mode, alpha=alpha)
    
    # 转换回PIL图像
    result_img = Image.fromarray(result_np)
    
    # 清理临时变量以节省内存
    del img_np, result_np
    
    return result_img


