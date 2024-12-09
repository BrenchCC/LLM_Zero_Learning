from transformers import PretrainedConfig

class MLLMConfig(PretrainedConfig):
    '''
    定义了一个名为 MLLMConfig 的类，它继承自 PretrainedConfig 类，用于配置多模态语言模型（MLLM）的参数。

    参数:
    - llm_model_path: LLM 模型的路径。
    - vision_model_path: 视觉模型的路径。
    - image_pad_num: 图像填充的数量。
    - freeze_vision_model: 是否冻结视觉模型。
    - kwargs: 其他参数。

    方法:
    - __init__: 初始化方法，设置模型的参数。
    '''
    model_type = "mllm"

    def __init__(
        self,
        llm_model_path = '/mnt/bn/brench-lq1/mllm_self_training/mllm_building/base_models/llm_model_qwen2.5_1.5b',
        vision_model_path = '/mnt/bn/brench-lq1/mllm_self_training/mllm_building/base_models/vision_model_siglip_16_384',
        image_pad_num = 81,
        freeze_vision_model = False,
        **kwargs
    ):
        '''
        初始化函数，用于设置模型的参数。

        参数:
        - llm_model_path: LLM 模型的路径。
        - vision_model_path: 视觉模型的路径。
        - image_pad_num: 图像填充的数量。
        - freeze_vision_model: 是否冻结视觉模型。
        - kwargs: 其他参数。
        '''
        self.llm_model_path = llm_model_path
        self.vision_model_path = vision_model_path
        self.image_pad_num = image_pad_num
        self.freeze_vision_model = freeze_vision_model
        super().__init__(**kwargs)
