import os
import json
import mindspore as ms
import mindspore.dataset.vision as vision
from mindcv.models import create_model
import numpy as np
from PIL import Image

# === 配置 ===
ckpt_path = "../ckpt/resnet.ckpt"  # 模型权重路径
# 验证能用即可，二分类，实际最终更改模型和映射即可
class_map = {
    0: "第20章-蔬菜、水果、坚果或植物其他部分的制品",
    1: "第48章-纸及纸板;纸浆、纸或纸板制品"
}
NUM_CLASSES = 2


def preprocess_image(image_path):

    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    transforms = [
        vision.Decode(),           
        vision.Resize(256),        
        vision.CenterCrop(224),    
        vision.Normalize(
            mean=[0.485*255, 0.456*255, 0.406*255],
            std=[0.229*255, 0.224*255, 0.225*255]
        ),
        vision.HWC2CHW()          
    ]
    
    for op in transforms[1:]:  
        img_array = op(img_array)

    img_tensor = ms.Tensor(np.expand_dims(img_array, axis=0), ms.float32)
    return img_tensor

def predict_image(image_path):

    # 这里可后续更改
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    network = create_model(
        model_name='resnet50',
        num_classes=NUM_CLASSES,
        pretrained=False  
    )

    param_dict = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(network, param_dict)
    network.set_train(False)  
    
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在 - {image_path}")
        return None
    
    input_tensor = preprocess_image(image_path)
    # 推理
    logits = network(input_tensor)
    probs = ms.ops.softmax(logits, axis=1)
    pred_idx = int(probs.argmax(1).asnumpy()[0])
    confidence = float(probs.max(1).asnumpy()[0])

    # result = {
    #     'class_id': pred_idx,
    #     'class_name': class_map.get(pred_idx, "未知类别"),
    #     'confidence': confidence,
    #     'all_probs': {class_map[i]: float(probs[0][i].asnumpy()) 
    #                   for i in range(NUM_CLASSES)}
    # }
    
    # print("\n" + "="*50)
    # print(f"预测结果: {result['class_name']}")
    # print(f"置信度: {result['confidence']*100:.2f}%")
    # print("\n所有类别概率:")
    # for name, prob in result['all_probs'].items():
    #     print(f"  {name}: {prob*100:.2f}%")
    # print("="*50 + "\n")

    return class_map.get(pred_idx, "未知类别")


# if __name__ == "__main__":

    
#     test_img = "../test/app (8).jpg"  
#     print(predict_image(test_img))
    

    
