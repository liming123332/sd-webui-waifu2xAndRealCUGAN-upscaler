import gradio as gr
from modules import scripts_postprocessing, shared, script_callbacks
import copy
from waifu2x.main import getModel, processImageWithSplitter, getCarnV2Model, processImageWithRealCUGAN

if hasattr(scripts_postprocessing.ScriptPostprocessing, 'process_firstpass'):  # webui >= 1.7
    from modules.ui_components import InputAccordion
else:
    InputAccordion = None


class Waifu2xExtras(scripts_postprocessing.ScriptPostprocessing):
    name = "Waifu2x Upscale"
    order = 1010

    def ui(self):
        global METHODS
        with (
            InputAccordion(False, label=self.name) if InputAccordion
            else gr.Accordion(self.name, open=False)
            as enable
        ):
            if not InputAccordion:
                enable = gr.Checkbox(False, label="Enable")
            
            # 添加RealCUGAN到样式选项
            style = gr.Radio(value="Anime", choices=["Anime", "Photo", "CarnV2", "RealCUGAN"], label="Style")
            
            # 原始的降噪选项
            noise = gr.Radio(value='Medium', choices=["None", "Low", "Medium", "High"],
                        label="Noise reduction", type="index")
            
            # 用于RealCUGAN的特殊选项
            realcugan_noise = gr.Radio(value='No Denoise', choices=["No Denoise", "Denoise 1x", "Denoise 2x", "Denoise 3x", "Conservative"],
                        label="RealCUGAN Noise reduction", type="index")
            
            realcugan_scale = gr.Radio(value='2x', choices=["2x", "3x", "4x"],
                        label="RealCUGAN Scale", type="index")
            
            realcugan_model_type = gr.Radio(value='Standard', choices=["Standard", "Pro"],
                        label="RealCUGAN Model Type")
            
            # 原始的缩放选项（用于非RealCUGAN模型）
            scale = gr.Radio(value='4x', choices=["2x", "4x", "8x", "16x"],
                        label="Scale", type="index")

        # 根据选择的样式显示或隐藏相应的选项
        def update_controls(style_value):
            # 原始选项的可见性
            noise_visible = style_value not in ["CarnV2", "RealCUGAN"]
            scale_visible = style_value not in ["RealCUGAN"]
            # RealCUGAN特定选项的可见性
            realcugan_noise_visible = style_value == "RealCUGAN"
            realcugan_scale_visible = style_value == "RealCUGAN"
            realcugan_model_type_visible = style_value == "RealCUGAN"
            
            return [
                gr.update(visible=noise_visible),
                gr.update(visible=scale_visible),
                gr.update(visible=realcugan_noise_visible),
                gr.update(visible=realcugan_scale_visible),
                gr.update(visible=realcugan_model_type_visible)
            ]
        
        style.change(
            fn=update_controls,
            inputs=[style],
            outputs=[noise, scale, realcugan_noise, realcugan_scale, realcugan_model_type],
            show_progress=False
        )

        args = {
            'enable': enable,
            'noise': noise,
            'scale': scale,
            'style': style,
            'realcugan_noise': realcugan_noise,
            'realcugan_scale': realcugan_scale,
            'realcugan_model_type': realcugan_model_type
        }
        return args

    def process_firstpass(self, pp: scripts_postprocessing.PostprocessedImage, **args):
        if args['enable']:
            if args['style'] == "RealCUGAN":
                # RealCUGAN的缩放比例直接对应选择的值
                realcugan_scales = [2, 3, 4]
                scale_factor = realcugan_scales[args['realcugan_scale']]
                pp.shared.target_width = pp.image.width * scale_factor
                pp.shared.target_height = pp.image.height * scale_factor
            else:
                # 原始处理逻辑
                pp.shared.target_width = pp.image.width * 2 ** (args['scale'] + 1)
                pp.shared.target_height = pp.image.height * 2 ** (args['scale'] + 1)

    def process(self, pp: scripts_postprocessing.PostprocessedImage, **args):
        if args['enable'] == False:
            return

        info = copy.copy(args)
        del info['enable']
        
        if args['style'] == "RealCUGAN":
            # 处理RealCUGAN的情况
            # 映射RealCUGAN的降噪级别选项到内部值
            realcugan_noise_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: -1}  # No Denoise, Denoise 1x, Denoise 2x, Denoise 3x, Conservative
            noise_level = realcugan_noise_map[args['realcugan_noise']]
            
            # 获取缩放比例
            realcugan_scales = [2, 3, 4]
            scale_factor = realcugan_scales[args['realcugan_scale']]
            
            # 获取模型类型
            model_type = 'pro' if args['realcugan_model_type'] == 'Pro' else 'standard'
            
            # 使用RealCUGAN处理图像
            pp.image = processImageWithRealCUGAN(
                img=pp.image,
                scale=scale_factor,
                noise_level=noise_level,
                model_type=model_type,
                tile_mode=4,  # 默认使用4x4分块
                cache_mode=0,  # 默认使用缓存
                alpha=1.0      # 默认锐化程度
            )
            
            # 更新info信息
            info['realcugan_noise'] = ["No Denoise", "Denoise 1x", "Denoise 2x", "Denoise 3x", "Conservative"][info['realcugan_noise']]
            info['realcugan_scale'] = ["2x", "3x", "4x"][info['realcugan_scale']]
            # 删除不需要的字段
            del info['noise']
            del info['scale']
        else:
            # 原始处理逻辑
            if args['style'] == "CarnV2":
                model = getCarnV2Model()
                del info['noise']
            else:
                model = getModel(args['noise'], args['style'])
                info['noise'] = ["None", "Low", "Medium", "High"][info['noise']]
            
            info['scale'] = ["2x", "4x", "8x", "16x"][info['scale']]
            
            for _ in range(args['scale'] + 1):
                pp.image = processImageWithSplitter(model, pp.image)
                if shared.state.interrupted: break
            
            # 删除RealCUGAN相关的字段
            if 'realcugan_noise' in info:
                del info['realcugan_noise']
                del info['realcugan_scale']
                del info['realcugan_model_type']
        
        pp.info[self.name] = str(info)


def on_ui_settings():
    shared.opts.add_option(
        "show_waifu2x_accordion",
        shared.OptionInfo(
            False,
            "Show Waifu2x accordion in extras tab",
            gr.Checkbox,
            section=('upscaling', "Upscaling")
        ).needs_reload_ui()
    )

script_callbacks.on_ui_settings(on_ui_settings)

if not shared.opts.data.get('show_waifu2x_accordion', False):
    del Waifu2xExtras

