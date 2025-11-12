from modules.upscaler import Upscaler, UpscalerData
from waifu2x.main import processImageWithSplitter, getModel, getCarnV2Model, processImageWithRealCUGAN


class Waifu2xFields():
    def __init__(self, style: str, noise: int):
        self.style = style
        self.noise = noise

    def getName(self):
        noiseStr = ['none', 'low', 'medium', 'hight'][self.noise]
        return f'Waifu2x {self.style.lower()} denoise {self.noise} ({noiseStr})'


data = [
    Waifu2xFields('Anime', 0),
    Waifu2xFields('Anime', 1),
    Waifu2xFields('Anime', 2),
    Waifu2xFields('Anime', 3),
    Waifu2xFields('Photo', 0),
    Waifu2xFields('Photo', 1),
    Waifu2xFields('Photo', 2),
    Waifu2xFields('Photo', 3),
]


class BaseClass(Upscaler):
    def __init__(self, dirname, waifu2xFields: Waifu2xFields = None):
        if waifu2xFields is None:
            self.scalers = []
            return
        self.waifu2xFields = waifu2xFields
        self.name = "Waifu2x"
        self.scalers = [UpscalerData(self.waifu2xFields.getName(), None, self, 2)]
        super().__init__()

    def do_upscale(self, img, selected_model):
        model = getModel(self.waifu2xFields.noise, self.waifu2xFields.style)
        return processImageWithSplitter(model, img)


class Class0(BaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, data[0])
class Class1(BaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, data[1])
class Class2(BaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, data[2])
class Class3(BaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, data[3])
class Class4(BaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, data[4])
class Class5(BaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, data[5])
class Class6(BaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, data[6])
class Class7(BaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, data[7])



class CarnV2Upscaler(Upscaler):
    def __init__(self, dirname):
        self.name = "Waifu2x"
        self.scalers = [UpscalerData("Waifu2x+ model CarnV2", None, self, 2)]
        super().__init__()

    def do_upscale(self, img, selected_model):
        model = getCarnV2Model()
        return processImageWithSplitter(model, img)


# RealCUGAN超分辨率模型支持 - 使用与waifu2x相同的模式

# 为了与waifu2x的实现保持一致，我们为每个RealCUGAN配置创建独立的类
# 首先定义一个RealCUGANFields类来存储配置信息，类似于waifu2x的实现
class RealCUGANFields():
    def __init__(self, scale, noise_level, model_type):
        self.scale = scale
        self.noise_level = noise_level
        self.model_type = model_type
    
    def getName(self):
        noise_str_map = {
            -1: 'conservative',
            0: 'no_denoise',
            1: 'denoise1x',
            2: 'denoise2x',
            3: 'denoise3x'
        }
        noise_str = noise_str_map.get(self.noise_level, 'no_denoise')
        model_type_str = 'Pro' if self.model_type == 'pro' else 'Standard'
        return f"RealCUGAN {model_type_str} {self.scale}x {noise_str}"

# 创建RealCUGAN配置数据
realcugan_data = [
    # 标准模型 - 2x缩放
    RealCUGANFields(2, 0, 'standard'),   # no_denoise
    RealCUGANFields(2, 1, 'standard'),   # denoise1x
    RealCUGANFields(2, 2, 'standard'),   # denoise2x
    RealCUGANFields(2, 3, 'standard'),   # denoise3x
    RealCUGANFields(2, -1, 'standard'),  # conservative
    
    # 标准模型 - 3x缩放
    RealCUGANFields(3, 0, 'standard'),   # no_denoise
    RealCUGANFields(3, 3, 'standard'),   # denoise3x
    RealCUGANFields(3, -1, 'standard'),  # conservative
    
    # 标准模型 - 4x缩放
    RealCUGANFields(4, 0, 'standard'),   # no_denoise
    RealCUGANFields(4, 3, 'standard'),   # denoise3x
    RealCUGANFields(4, -1, 'standard'),  # conservative
    
    # Pro模型 - 2x缩放
    RealCUGANFields(2, 0, 'pro'),        # no_denoise
    RealCUGANFields(2, 3, 'pro'),        # denoise3x
    RealCUGANFields(2, -1, 'pro'),       # conservative
    
    # Pro模型 - 3x缩放
    RealCUGANFields(3, 0, 'pro'),        # no_denoise
    RealCUGANFields(3, 3, 'pro'),        # denoise3x
    RealCUGANFields(3, -1, 'pro')        # conservative
]

# 定义RealCUGAN的基类，类似于waifu2x的BaseClass
class RealCUGANBaseClass(Upscaler):
    def __init__(self, dirname, realcugan_fields=None):
        if realcugan_fields is None:
            self.scalers = []
            return
        self.realcugan_fields = realcugan_fields
        self.name = "RealCUGAN"
        self.scalers = [UpscalerData(self.realcugan_fields.getName(), None, self, realcugan_fields.scale)]
        super().__init__()
    
    def do_upscale(self, img, selected_model):
        # 使用默认的分块参数以平衡速度和显存使用
        return processImageWithRealCUGAN(
            img=img,
            scale=self.realcugan_fields.scale,
            noise_level=self.realcugan_fields.noise_level,
            model_type=self.realcugan_fields.model_type,
            tile_mode=4,  # 默认使用4x4分块
            cache_mode=0,  # 默认使用缓存
            alpha=1.0      # 默认锐化程度
        )


# 创建RealCUGAN模型的各种配置实例，使用与waifu2x相同的模式
# 标准模型 - 2x缩放
class RealCUGAN_2x_NoDenoise(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[0])

class RealCUGAN_2x_Denoise1x(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[1])

class RealCUGAN_2x_Denoise2x(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[2])

class RealCUGAN_2x_Denoise3x(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[3])

class RealCUGAN_2x_Conservative(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[4])

# 标准模型 - 3x缩放
class RealCUGAN_3x_NoDenoise(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[5])

class RealCUGAN_3x_Denoise3x(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[6])

class RealCUGAN_3x_Conservative(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[7])

# 标准模型 - 4x缩放
class RealCUGAN_4x_NoDenoise(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[8])

class RealCUGAN_4x_Denoise3x(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[9])

class RealCUGAN_4x_Conservative(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[10])

# Pro模型 - 2x缩放
class RealCUGAN_Pro_2x_NoDenoise(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[11])

class RealCUGAN_Pro_2x_Denoise3x(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[12])

class RealCUGAN_Pro_2x_Conservative(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[13])

# Pro模型 - 3x缩放
class RealCUGAN_Pro_3x_NoDenoise(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[14])

class RealCUGAN_Pro_3x_Denoise3x(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[15])

class RealCUGAN_Pro_3x_Conservative(RealCUGANBaseClass, Upscaler):
    def __init__(self, dirname):
        super().__init__(dirname, realcugan_data[16])

# 创建所有上采样器实例并导出，使WebUI能够发现它们
def get_upscalers():
    # 这里的dirname参数在实际使用中会被webui传递
    # 为了模块初始化时不报错，我们提供一个默认值
    dirname = ""
    
    # 创建并返回所有上采样器实例的列表
    return [
        # Waifu2x模型
        Class0(dirname),
        Class1(dirname),
        Class2(dirname),
        Class3(dirname),
        Class4(dirname),
        Class5(dirname),
        Class6(dirname),
        Class7(dirname),
        CarnV2Upscaler(dirname),
        
        # RealCUGAN Standard模型 - 2x缩放
        RealCUGAN_2x_NoDenoise(dirname),
        RealCUGAN_2x_Denoise1x(dirname),
        RealCUGAN_2x_Denoise2x(dirname),
        RealCUGAN_2x_Denoise3x(dirname),
        RealCUGAN_2x_Conservative(dirname),
        
        # RealCUGAN Standard模型 - 3x缩放
        RealCUGAN_3x_NoDenoise(dirname),
        RealCUGAN_3x_Denoise3x(dirname),
        RealCUGAN_3x_Conservative(dirname),
        
        # RealCUGAN Standard模型 - 4x缩放
        RealCUGAN_4x_NoDenoise(dirname),
        RealCUGAN_4x_Denoise3x(dirname),
        RealCUGAN_4x_Conservative(dirname),
        
        # RealCUGAN Pro模型 - 2x缩放
        RealCUGAN_Pro_2x_NoDenoise(dirname),
        RealCUGAN_Pro_2x_Denoise3x(dirname),
        RealCUGAN_Pro_2x_Conservative(dirname),
        
        # RealCUGAN Pro模型 - 3x缩放
        RealCUGAN_Pro_3x_NoDenoise(dirname),
        RealCUGAN_Pro_3x_Denoise3x(dirname),
        RealCUGAN_Pro_3x_Conservative(dirname)
    ]

# WebUI会自动查找并实例化所有继承自Upscaler的类
# 但为了确保所有模型都能被正确发现，我们需要创建一个额外的导出机制
# 创建一个列表来存储所有上采样器类，这是WebUI查找的标准方式
# 注意：WebUI会自动实例化这些类，不需要我们手动创建实例
# 我们只需要确保这些类被导出

# 直接导出所有上采样器类，这样WebUI就能找到它们
# 在WebUI中，会自动搜索并实例化所有继承自Upscaler的类
# 但为了确保兼容性，我们显式地导出它们
__all__ = [
    # Waifu2x模型类
    'Class0', 'Class1', 'Class2', 'Class3', 
    'Class4', 'Class5', 'Class6', 'Class7',
    'CarnV2Upscaler',
    
    # RealCUGAN Standard模型类
    'RealCUGAN_2x_NoDenoise', 'RealCUGAN_2x_Denoise1x', 
    'RealCUGAN_2x_Denoise2x', 'RealCUGAN_2x_Denoise3x', 
    'RealCUGAN_2x_Conservative',
    'RealCUGAN_3x_NoDenoise', 'RealCUGAN_3x_Denoise3x', 
    'RealCUGAN_3x_Conservative',
    'RealCUGAN_4x_NoDenoise', 'RealCUGAN_4x_Denoise3x', 
    'RealCUGAN_4x_Conservative',
    
    # RealCUGAN Pro模型类
    'RealCUGAN_Pro_2x_NoDenoise', 'RealCUGAN_Pro_2x_Denoise3x', 
    'RealCUGAN_Pro_2x_Conservative',
    'RealCUGAN_Pro_3x_NoDenoise', 'RealCUGAN_Pro_3x_Denoise3x', 
    'RealCUGAN_Pro_3x_Conservative'
]

# 另外，WebUI也会查找名为upscalers的变量
# 为了兼容不同版本的WebUI，我们也定义这个变量
# 但这里我们不直接实例化它们，而是让WebUI自己处理
upscalers = [
    Class0,
    Class1,
    Class2,
    Class3,
    Class4,
    Class5,
    Class6,
    Class7,
    CarnV2Upscaler,
    RealCUGAN_2x_NoDenoise,
    RealCUGAN_2x_Denoise1x,
    RealCUGAN_2x_Denoise2x,
    RealCUGAN_2x_Denoise3x,
    RealCUGAN_2x_Conservative,
    RealCUGAN_3x_NoDenoise,
    RealCUGAN_3x_Denoise3x,
    RealCUGAN_3x_Conservative,
    RealCUGAN_4x_NoDenoise,
    RealCUGAN_4x_Denoise3x,
    RealCUGAN_4x_Conservative,
    RealCUGAN_Pro_2x_NoDenoise,
    RealCUGAN_Pro_2x_Denoise3x,
    RealCUGAN_Pro_2x_Conservative,
    RealCUGAN_Pro_3x_NoDenoise,
    RealCUGAN_Pro_3x_Denoise3x,
    RealCUGAN_Pro_3x_Conservative
]


