from openai import OpenAI
from anthropic import Anthropic
import base64
import os
import sys
# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  
sys.path.append(project_root)

from config import settings
import io
from PIL import ImageGrab

def encode_image_from_bytes(image_bytes):
   
    return base64.b64encode(image_bytes).decode('utf-8')


def encode_image_from_path(image_path):

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_media_type(image_data, image_type):
    if image_type == "path":
        ext = os.path.splitext(image_data)[1].lower()
        types = {'.png': 'image/png', '.jpg': 'image/jpeg', 
                '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.webp': 'image/webp'}
        return types.get(ext, 'image/jpeg')  # 默认
    
    elif image_type in ["bytes", "base64"]:
        if image_type == "base64":
            
            image_bytes = base64.b64decode(image_data[:100])
        else:
            image_bytes = image_data[:20]  
        
        # 检查文件签名
        if image_bytes.startswith(b'\x89PNG'):
            return 'image/png'
        elif image_bytes.startswith(b'\xff\xd8\xff'):
            return 'image/jpeg'
        elif image_bytes.startswith(b'GIF87a') or image_bytes.startswith(b'GIF89a'):
            return 'image/gif'
        elif image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes[:20]:
            return 'image/webp'
        else:
            return 'image/jpeg'  
    return 'image/jpeg'  

def call_LLM(text_prompt, image_data=None, image_type=None):

    system_prompt = """你是一个贸易数据分析专家，基于用户传入的商品的贸易特点进行分析，最后给出专业建议，回答要简洁明了，避免赘述。"""
    api_key = settings.API_KEY
    base_url = getattr(settings, "BASE_URL", None)
    client = Anthropic(api_key=api_key, base_url=base_url)
    if image_data :
        if image_type == "path":
            encoded_image = encode_image_from_path(image_data)
            media_type = get_image_media_type(image_data, image_type)
        elif image_type == "bytes":
            encoded_image = encode_image_from_bytes(image_data)
            media_type = get_image_media_type(image_data, image_type)
        elif image_type == "base64":
            encoded_image = image_data
            media_type = get_image_media_type(image_data, image_type)
        else:
            raise ValueError("Unsupported image_type. Use 'path', 'bytes', or 'base64'.")

        user_message = {"role": "user", "content": [
            {"type": "text", "text": text_prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "data": encoded_image,
                    "media_type": media_type
                }
            }
        ]}
    else:
        user_message = {"role": "user", "content": text_prompt}





    for i in range(0, 4):  # 重复4次调用，减少网络干扰,这四次调用，不会抛出异常
        try:
            response = client.messages.create(
                model="claude-opus-4-5-20251101",
                system=system_prompt,
                messages=[
                    user_message
                ],
                max_tokens=1024,
                temperature=0.7,
                stream=False
            )
            if response.content and response.content[0].text:
                return response.content[0].text
        except Exception as e:
            continue

    #如果前四次由于异常返回结果为空，进行第五次调用，如果再次失败，就抛出异常
    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        system=system_prompt,
        messages=[
            user_message
        ],
        max_tokens=1024,
        temperature=0.7,
        stream=False
    )

    #如果调用结果不为空，返回调用结果
    if response.content and response.content[0].text:
        return response.content[0].text
    else:
        raise Exception("LLM调用失败，返回结果为空")

    
# # # 测试代码
# if __name__ == "__main__":
#     image_path = "../test/test.png"
#     text_prompt = "请分析这张图片中的贸易数据特点。"
#     result = call_LLM(text_prompt, image_data=image_path, image_type="path")
#     print(result)
#     print("\n" + "="*50 + "\n")
#     try:
#         input("请先复制一张图片到剪贴板，然后按回车继续...")
#         clipboard_image = ImageGrab.grabclipboard()
#         if clipboard_image:
#             # 将图片转为字节流
#             img_byte_arr = io.BytesIO()
#             clipboard_image.save(img_byte_arr, format='PNG')
#             img_byte_arr = img_byte_arr.getvalue()
            
#             result = call_LLM(text_prompt, image_data=img_byte_arr, image_type="bytes")
#             print("剪贴板图片测试结果:")
#             print(result)
#         else:
#             print("剪贴板中没有图片")
#     except Exception as e:
#         print(f"剪贴板测试失败: {e}")
    
#     print("\n" + "="*50 + "\n")