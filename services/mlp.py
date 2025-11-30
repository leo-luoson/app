import os
import sys
import json
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


class TradeModel(nn.Cell):
    def __init__(self, 
                 country_vocab_size: int, 
                 reg_place_vocab_size: int, 
                 product_code_vocab_size: int, 
                 unit_vocab_size: int,  
                 one_hot_feature_count: int,
                 numerical_feature_count: int,
                 country_embed_dim: int = 64, 
                 reg_place_embed_dim: int = 16, 
                 product_code_embed_dim: int = 128, 
                 unit_embed_dim: int = 16):
          
        super(TradeModel, self).__init__()
        
        # --- 定义嵌入层 ---
        self.embedding_country = nn.Embedding(vocab_size=country_vocab_size, embedding_size=country_embed_dim)
        self.embedding_reg_place = nn.Embedding(vocab_size=reg_place_vocab_size, embedding_size=reg_place_embed_dim)
        self.embedding_product_code = nn.Embedding(vocab_size=product_code_vocab_size, embedding_size=product_code_embed_dim)
        self.embedding_unit = nn.Embedding(vocab_size=unit_vocab_size, embedding_size=unit_embed_dim)

        mlp_input_dim = (country_embed_dim + reg_place_embed_dim + product_code_embed_dim + 
                         unit_embed_dim + numerical_feature_count + one_hot_feature_count)
        
        self.mlp = nn.SequentialCell(
            nn.Dense(mlp_input_dim, 512),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Dense(512, 256),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Dense(256, 128),
            nn.ReLU(),
            nn.Dense(128, 64),
            nn.ReLU(),
            nn.Dense(64, 1),
        )

    def construct(self, X: ms.Tensor):
        # X : (batch_size, feature_count)
        # 0:国家, 1:注册地, 2:商品编码, 3:单位, 4...:数值+独热
        
        country_idx = X[:, 0].astype(ms.int32)
        reg_place_idx = X[:, 1].astype(ms.int32)
        product_code_idx = X[:, 2].astype(ms.int32)
        unit_idx = X[:, 3].astype(ms.int32)
        
        numerical_and_onehot = X[:, 4:]
        
        country_embed = self.embedding_country(country_idx)
        reg_place_embed = self.embedding_reg_place(reg_place_idx)
        product_code_embed = self.embedding_product_code(product_code_idx)
        unit_embed = self.embedding_unit(unit_idx)
        
        combined_features = ops.concat((country_embed, reg_place_embed, product_code_embed, unit_embed, numerical_and_onehot), axis=1)
        output = self.mlp(combined_features)
        return output


def predict(country, reg_place, product_code, unit, year, trade_method):
    try:
        # 1. 加载映射表
        assets_dir = os.path.join(project_root, 'json')
        json_dir = os.path.join(assets_dir, 'mapping')
        with open(os.path.join(json_dir, 'country_to_index.json'), 'r', encoding='utf-8') as f:
            country_mapper = json.load(f)
        with open(os.path.join(json_dir, 'province_to_index.json'), 'r', encoding='utf-8') as f:
            province_mapper = json.load(f)
        # with open(os.path.join(json_dir, 'product_code_to_index.json'), 'r', encoding='utf-8') as f:
        #     product_code_mapper = json.load(f)
        
        
        with open(os.path.join(json_dir, 'unit_to_index.json'), 'r', encoding='utf-8') as f:
            unit_mapper = json.load(f)
        with open(os.path.join(json_dir, 'trade_to_onehot.json'), 'r', encoding='utf-8') as f:
            trade_mapper = json.load(f)
    except Exception as e:
        print(f"[Error] 映射表加载失败: {e}")
    # 2. 计算维度
    vocab_size_country = len(country_mapper)
    vocab_size_reg = len(province_mapper)

    vocab_size_product = 8170

    vocab_size_unit = len(unit_mapper)
    one_hot_dim = len(next(iter(trade_mapper.values())))
    # 3. 初始化模型
    model = TradeModel(
        country_vocab_size=vocab_size_country,
        reg_place_vocab_size=vocab_size_reg,
        product_code_vocab_size=vocab_size_product,
        unit_vocab_size=vocab_size_unit,
        one_hot_feature_count=one_hot_dim,
        numerical_feature_count=1,
        country_embed_dim=64,
        reg_place_embed_dim=16,
        product_code_embed_dim=128,
        unit_embed_dim=16
    )
    # 4. 加载权重
    ckpt_path = os.path.join(project_root, 'ckpt', 'mlp.ckpt')
    param_dict = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)
    # 5. 预处理输入
    country_idx = country_mapper.get(country, 0)
    reg_place_idx = province_mapper.get(reg_place, 0)
    # 临时随机处理
    product_code_idx = random.randint(0, vocab_size_product - 1)  

    unit_idx = unit_mapper.get(unit, 0)
    year_normalized = (year - 2012) / (2021 - 2012)
    trade_onehot = trade_mapper.get(trade_method, [0]*one_hot_dim)

    input_list = [
        float(country_idx),float(reg_place_idx), float(product_code_idx), float(unit_idx),
        year_normalized
    ]
    input_list.extend(trade_onehot)
    input_tensor = Tensor(np.array([input_list], dtype=np.float32))

    # 6. 执行推理
    output = model(input_tensor)
    pred_log_val = output.asnumpy()[0][0]
    mean = 4.0703
    std = 2.0955
    pred_value = np.exp(pred_log_val * std + mean)
    return pred_value

# # --- 本地测试用 ---
# if __name__ == "__main__":
#     country = "美国"
#     reg_place = "广东省"
#     product_code = "XX" #这块是有问题的，我在代码里面加了把这块随机变成0-8000的数字逻辑，传商品名称即可
#     unit = "千克"
#     year = 2020
#     trade_method = "一般贸易"
    
#     pred_value = predict(country, reg_place, product_code, unit, year, trade_method)
#     print(f"预测值: {pred_value}")