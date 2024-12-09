import io
import os
import json
import requests
from PIL import Image
from typing import List, Dict, Union, Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoProcessor, AutoModel



# PretrainedDataset 数据集处理类 返回经过tokenizer和VisionProcessor处理过的input_ids, labels, pixel_values
# 返回每一条数据的输入文本token标识符, 目标描述文本的token序列, 预处理后的图像张量

class PretrainedDataset(Dataset):
    '''
    预训练数据集类,用于加载和处理图像和文本数据

    参数:
    - images_path: 图像文件路径
    - annotations_path: 注释文件路径
    - config: 配置对象,包含模型路径等信息

    方法:
    - __init__: 初始化方法,加载配置、tokenizer和processor,并读取注释文件
    - __len__: 返回数据集长度
    - __getitem__: 根据索引返回数据集中的样本,包括图像的像素值、输入文本的token标识符和目标描述文本的token序列
    '''
    def __init__(self, images_path, annotations_path, config):
        '''
        初始化函数,用于加载配置、tokenizer和processor,并读取注释文件

        参数:
        - images_path: 图像文件路径
        - annotations_path: 注释文件路径
        - config: 配置对象,包含模型路径等信息
        '''
        # 保存配置对象
        self.config = config
        # 保存图像文件路径
        self.images_path = images_path
        # 保存注释文件路径
        self.annotations_path = annotations_path
        # 从配置中获取llm模型路径,并初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
        # 从配置中获取vision模型路径,并初始化processor
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        # 定义系统提示
        self.system_prompt = {
        "role": "system",
        "content": "你叫Flash,你是为一位专门为Brench服务的多模态AI助手"
        }

        # 读取注释文件
        with open(self.annotations_path, 'r', encoding='utf-8') as f:
            self.processor_data = json.load(f)
    
    def __len__(self):
        '''
        返回数据集长度

        返回值:
        - 数据集长度
        '''
        return len(self.processor_data)
    
    def __getitem__(self, index):
        '''
        根据索引返回数据集中的样本, 包括图像的像素值、输入文本的token标识符和目标描述文本的token序列

        参数:
        - index: 样本索引

        返回值:
        - 包含'input_ids', 'labels', 'pixel_values'的字典
        '''
        # 获取数据集中的样本
        data_sample = self.processor_data[index]
        try:
            # 获取图像文件名和对话数据
            image_file_name = data_sample['image']
            conversations = data_sample['conversations']
            # 打开图像并转换为RGB格式
            image = Image.open(os.path.join(self.images_path, image_file_name)).convert('RGB')
            # 使用processor处理图像,获取像素值
            pixel_values = self.processor(text=None, images=image)['pixel_values']
            # 构建用户提示
            user_prompt = {
                "role": "user",
                "content": conversations[0]['value']
            }
            # 构建查询文本
            query_text = [self.system_prompt, user_prompt]
            # 使用tokenizer应用聊天模板,生成查询输入
            query_input = self.tokenizer.apply_chat_template(
                query_text,
                tokenize=False,
                add_generation_prompt=True
            ).replace('<image>','<|image_pad|>'*self.config.image_pad_num)
            # 构建响应文本
            response_text = conversations[1]['value'] + self.tokenizer.eos_token
            # 获取查询输入和响应输入的token标识符
            query_input_ids = self.tokenizer(query_input)['input_ids']
            response_input_ids = self.tokenizer(response_text)['input_ids']
            # 拼接输入标识符和标签标识符
            input_ids = query_input_ids + response_input_ids
            labels = [self.tokenizer.pad_token_id] * len(query_input_ids) + response_input_ids
            # 去除最后一个token
            input_ids = input_ids[:-1]
            labels = labels[1:]
        except:
            # 如果处理过程中出现异常,使用默认图像和提示
            default_image = Image.new('RGB',(224,224),color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            user_prompt = {
                "role": "user",
                "content":"这张图片描述的内容是什么\n<image>"
            }
            query_text = [self.system_prompt, user_prompt]
            query_input = self.tokenizer.apply_chat_template(
                query_text,
                tokenize=False,
                add_generation_prompt=True
            ).replace('<image>','<|image_pad|>'*self.config.image_pad_num)
            response_text = "图片内容为空,无法生成相关的回复\n" + self.tokenizer.eos_token
            query_input_ids = self.tokenizer(query_input)['input_ids']
            response_input_ids = self.tokenizer(response_text)['input_ids']
            input_ids = query_input_ids + response_input_ids
            labels = [self.tokenizer.pad_token_id] * len(query_input_ids) + response_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }

            

class  DatasetCollator: 
    '''
    数据集整理器,用于将数据集转换为模型可接受的输入格式

    参数:
    - config: 配置对象,包含模型路径等信息

    方法:
    - __init__: 初始化方法,加载配置和tokenizer
    - __call__: 调用方法,处理输入的features列表,返回拼接后的字典
    '''
    def __init__(self, config):
        '''
        初始化函数,用于加载配置和tokenizer

        参数:
        - config: 配置对象,包含模型路径等信息
        '''
        self.config = config
        # 从配置中获取llm模型路径
        self.llm_model_path = self.config.llm_model_path
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path)
    
    def __call__(self, features: List[Dict[str,Any]])->Dict[str,torch.Tensor]:
        '''
        调用函数,处理输入的features列表,返回拼接后的字典

        参数:
        - features: 包含多个样本的列表,每个样本是一个字典,包含'input_ids', 'labels', 'pixel_values'

        返回值:
        - 拼接后的字典,包含'input_ids', 'labels', 'pixel_values'
        '''
        # 计算输入序列的最大长度
        max_length = max(len(feature['input_ids']) for feature in features)
        # 初始化输入id列表、标签列表和像素值列表
        input_ids = []
        labels = []
        pixel_values = []
        # 遍历每个样本
        for feature in features:
            # 将当前样本的输入id和标签填充到最大长度
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_length - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_length - len(feature['labels'])))
            # 收集当前样本的像素值
            pixel_values.append(feature['pixel_values'])
        # 将输入id、标签和像素值转换为张量并返回
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pixel_values': torch.cat(pixel_values, dim=0)
        }

