# 中国进出口贸易数据智能分析与预测系统

## 📋 项目简介

基于华为云技术栈(openGauss + Ascend NPU + ModelArts)的智能贸易分析平台，提供：
- 🎯 **单价预测**：MLP神经网络预测贸易商品单价(MAPE 8.3%)
- 🖼️ **图像识别**：ResNet50自动识别商品类别(准确率92.3%)
- 📊 **数据可视化**：ECharts多维度交互式图表
- 🤖 **AI分析**：Claude多模态大模型智能解读
- 📈 **聚类分析**：K-means贸易伙伴/商品分组

---

## 🛠️ 技术栈

### 后端
- **Web框架**: Flask 2.3.0
- **深度学习**: MindSpore 2.2.0 (华为深度学习框架)
- **数据库**: openGauss (PostgreSQL兼容)
- **数据库驱动**: psycopg2-binary
- **AI模型**: Claude Opus 4.5 (Anthropic)
- **图像处理**: Pillow
- **科学计算**: NumPy

### 前端
- **UI框架**: Bootstrap 5.3.3
- **图表库**: ECharts 5.x
- **JavaScript**: Vanilla ES6 (模块化)

### 华为云服务
- **数据库**: openGauss (123.249.40.133:5432)
- **计算**: 华为云ECS + Ascend NPU
- **存储**: OBS对象存储 (规划中)
- **操作系统**: openEuler 22.03 LTS

---

## 📁 项目结构

```
/home/user/app/
├── app.py                      # Flask主应用 (路由定义)
├── requirements.txt            # Python依赖列表
├── config/
│   └── settings.py            # 数据库和API配置
├── services/                   # 业务逻辑层
│   ├── db_Manager.py          # 数据库操作类
│   ├── mlp.py                 # MLP单价预测模型
│   ├── resnet.py              # ResNet50图像识别
│   └── call_LLM.py            # Claude LLM调用
├── ckpt/                       # 模型权重文件
│   ├── mlp.ckpt               # MLP模型权重 (450MB)
│   └── resnet.ckpt            # ResNet50权重 (98MB)
├── json/                       # 数据资源
│   ├── mapping/               # 特征映射表
│   │   ├── country_to_index.json              # 国家→索引 (191个)
│   │   ├── province_to_index.json             # 省份→索引 (31个)
│   │   ├── unit_to_index.json                 # 单位→索引 (47个)
│   │   ├── trade_to_onehot.json               # 贸易方式→独热编码
│   │   └── category_chapter_product_mapping.json  # 商品分类树
│   ├── total_stats/           # 宏观统计数据
│   │   ├── total_stats_{year}.json            # 年度总体统计
│   │   ├── country_stats_{year}.json          # Top 10国家
│   │   ├── province_stats_{year}.json         # Top 10省份
│   │   └── product_stats_{year}.json          # Top 10商品
│   └── cluster/               # 聚类结果
│       ├── kmeans/            # K-means聚类数据
│       ├── world.json         # 世界地图GeoJSON
│       ├── china.json         # 中国地图GeoJSON
│       └── country_name_mapping.json  # 国家名中英文映射
├── static/                     # 静态资源
│   ├── css/
│   │   └── style.css          # 全局样式
│   └── js/
│       ├── param.js           # 参数选择器 (800+行)
│       ├── chart.js           # 单价预测图表
│       ├── dashboard.js       # 特征大屏逻辑 (500+行)
│       └── llm.js             # AI对话管理 (340行)
└── templates/                  # HTML模板
    ├── index.html             # 单价预测页面
    └── dashboard.html         # 特征大屏页面
```

---

## 🚀 快速开始

### 1. 环境要求

- **Python**: 3.8+
- **操作系统**: Linux (推荐 openEuler 22.03) / macOS / Windows
- **硬件** (可选):
  - CPU推理: 4核8GB内存
  - GPU推理: 华为Ascend NPU / NVIDIA GPU
- **数据库**: openGauss / PostgreSQL 12+

### 2. 安装依赖

```bash
# 克隆项目
git clone <repository_url>
cd app

# 安装Python依赖
pip install -r requirements.txt

# 如果使用Ascend NPU，需安装MindSpore Ascend版本
# pip install mindspore-ascend
```

### 3. 配置数据库

编辑 `config/settings.py`:

```python
# 数据库配置
DB_HOST = "123.249.40.133"      # openGauss主机地址
DB_PORT = 5432
DB_USER = "ltb"
DB_PASSWORD = "your_password"   # 修改为你的密码
DB_NAME = "postgres"
TABLE_NAME = "dboper.imports_master"

# API密钥配置
API_KEY = "your_anthropic_api_key"  # 修改为你的Claude API Key
BASE_URL = "https://aicanapi.com"
```

### 4. 启动应用

```bash
# 开发模式
python app.py

# 生产模式 (使用Gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 app:app --timeout 300
```

访问: `http://localhost:5000`

---

## 📡 API接口文档

### 🔵 图一：单价预测相关接口

#### 1. 单价预测

**接口**: `POST /api/predict`

**功能**: 使用MLP模型预测贸易商品单价

**请求参数**:
```json
{
  "country": "United States",           // 贸易国家
  "reg_place": "广东省",                // 商品注册地
  "product_code": "2005999990",        // 商品编码 (HS编码)
  "unit": "千克",                      // 计价单位
  "year": 2023,                        // 年份
  "trade_method": "一般贸易"           // 贸易方式
}
```

**响应示例**:
```json
{
  "success": true,
  "predicted_price": 2.3456,          // 预测单价 (元/单位)
  "unit": "千克",
  "year": 2023
}
```

**代码位置**: `app.py:150-180`, `services/mlp.py:100-150`

---

#### 2. 获取历史真实数据

**接口**: `POST /api/get_real_data`

**功能**: 查询历史交易的平均单价 (用于对比验证预测结果)

**请求参数**:
```json
{
  "country": "United States",
  "province": "广东省",
  "trade_type": "一般贸易",
  "name": "番茄罐头",                  // 商品名称
  "unit": "千克",
  "start_year": 2012,                 // 起始年份
  "end_year": 2021                    // 结束年份
}
```

**响应示例**:
```json
[
  {"year": 2012, "avg_price": 2.15},
  {"year": 2013, "avg_price": 2.23},
  ...
  {"year": 2021, "avg_price": 2.48}
]
```

**说明**:
- 2012-2021年返回真实数据(从openGauss数据库查询)
- 图表中2012-2021为蓝色实线,2025-2030为红色虚线(模型外推预测)

**代码位置**: `app.py:200-230`, `services/db_Manager.py:50-85`

---

#### 3. 获取商品分类映射

**接口**: `GET /api/product_mapping`

**功能**: 获取完整的商品分类树 (类→章→商品)

**响应示例**:
```json
{
  "第01类-活动物;动物产品": {
    "第01章-活动物": [
      "活马",
      "活牛",
      "活猪",
      ...
    ],
    "第02章-肉及食用杂碎": [...]
  },
  "第02类-植物产品": {...}
}
```

**用途**: 用于前端下拉菜单的动态加载

**代码位置**: `app.py:250-260`

**数据源**: `json/mapping/category_chapter_product_mapping.json`

---

#### 4. 获取参数选项列表

**接口**: `GET /api/country_options`

**功能**: 获取所有贸易国家列表

**响应示例**:
```json
["United States", "Japan", "South Korea", "Germany", ...]
```

**类似接口**:
- `GET /api/province_options` - 获取省份列表
- `GET /api/trade_type_options` - 获取贸易方式列表
- `GET /api/unit_options` - 获取单位列表

**代码位置**: `app.py:270-320`

**数据源**:
- `json/mapping/country_to_index.json`
- `json/mapping/province_to_index.json`
- `json/mapping/trade_to_onehot.json`
- `json/mapping/unit_to_index.json`

---

#### 5. AI大模型分析

**接口**: `POST /api/llm_analyze`

**功能**: 使用Claude多模态大模型分析图表和数据

**请求参数**:
```json
{
  "text_prompt": "请分析这张图表的趋势",
  "image_data": "data:image/png;base64,iVBORw0KGgo...",  // Base64编码图片
  "image_type": "base64"                                // 类型: base64/path/bytes
}
```

**响应示例**:
```json
{
  "success": true,
  "analysis": "从图表可以看出，2018年前单价稳定在2.5元/kg，2018年后骤降至1.8元/kg(下降28%)，推测与中美贸易摩擦导致的关税增加有关..."
}
```

**支持的输入方式**:
- 本地图片上传
- 粘贴剪贴板图片
- Base64编码图片

**代码位置**: `app.py:330-360`, `services/call_LLM.py:68-95`

---

### 🟢 图二：特征大屏相关接口

#### 6. 商品图像识别

**接口**: `POST /api/recognize_product`

**功能**: 上传商品图片，ResNet50自动识别类别

**请求参数**:
```
Content-Type: multipart/form-data
file: <image_file>  // 图片文件 (JPG/PNG，最大10MB)
```

**响应示例**:
```json
{
  "success": true,
  "chapter_name": "第20章-蔬菜、水果、坚果或植物其他部分的制品"
}
```

**支持的输入方式**:
- 本地图片上传
- 摄像头拍照识别

**代码位置**: `app.py:380-410`, `services/resnet.py:45-85`

---

#### 7. 获取折线图数据

**接口**: `POST /api/get_line_data`

**功能**: 获取指定商品章节的10年趋势数据

**请求参数**:
```json
{
  "chapter_name": "第20章-蔬菜、水果、坚果或植物其他部分的制品",
  "param": "单价"  // 参数: 金额 | 贸易条数 | 单价
}
```

**响应示例**:
```json
[
  {"year": 2012, "value": 2.15},       // 单价(元/kg)
  {"year": 2013, "value": 2.23},
  ...
  {"year": 2021, "value": 2.48}
]
```

**说明**:
- 图像识别后自动调用此接口更新折线图
- 默认展示"单价"参数,可切换到"金额"或"贸易条数"

**代码位置**: `app.py:420-450`, `services/db_Manager.py:120-160`

---

#### 8. 获取饼图数据

**接口**: `POST /api/get_pie_data`

**功能**: 获取Top 5贸易国家/省份的占比数据

**请求参数**:
```json
{
  "chapter_name": "第20章-蔬菜、水果、坚果或植物其他部分的制品",
  "relation": "国家",              // 维度: 国家 | 省份
  "year": 2021,
  "param": "单价"                  // 参数: 金额 | 贸易条数 | 单价
}
```

**响应示例**:
```json
[
  {"name": "United States", "value": 3.52, "proportion": 25.3},
  {"name": "Japan", "value": 2.81, "proportion": 20.1},
  {"name": "South Korea", "value": 2.15, "proportion": 15.2},
  {"name": "Germany", "value": 1.82, "proportion": 13.0},
  {"name": "其他", "value": 3.68, "proportion": 26.4}
]
```

**说明**:
- 图像识别后自动调用此接口更新饼图
- 默认展示"国家 + 2021 + 单价",三个参数均可选择

**代码位置**: `app.py:460-490`, `services/db_Manager.py:180-230`

---

#### 9. 宏观统计数据

**接口**: `GET /api/macro_stats?year=2021`

**功能**: 获取年度宏观统计指标

**响应示例**:
```json
{
  "total_amount": 449580000000000,     // 总贸易额(元)
  "total_transactions": 1267120,       // 总交易次数
  "num_partners": 191,                 // 贸易伙伴数量
  "num_provinces": 31,                 // 进口省份数量
  "num_products": 9594,                // 商品种类数量
  "avg_price": 3.54                    // 平均单价(元/kg)
}
```

**展示方式**: 卡片式布局,6个指标分别显示

**代码位置**: `app.py:500-520`

**数据源**: `json/total_stats/total_stats_{year}.json`

---

#### 10. 宏观条形图数据

**接口**: `POST /api/macro_bar_data`

**功能**: 获取Top 10国家/省份/商品的条形图数据

**请求参数**:
```json
{
  "relation": "国家",              // 维度: 国家 | 省份 | 商品
  "param": "金额",                // 参数: 金额 | 贸易次数 | 单价
  "year": 2021
}
```

**响应示例**:
```json
[
  {"name": "United States", "value": 1.2e11},
  {"name": "Japan", "value": 9.5e10},
  ...
  {"name": "France", "value": 3.2e10}
]
```

**代码位置**: `app.py:530-560`

**数据源**:
- `json/total_stats/country_stats_{year}.json`
- `json/total_stats/province_stats_{year}.json`
- `json/total_stats/product_stats_{year}.json`

---

#### 11. 聚类分析

**接口**: `POST /api/cluster_analysis`

**功能**: 获取K-means聚类结果 (用于散点图可视化)

**请求参数**:
```json
{
  "year": 2020,
  "node_type": "贸易国家",              // 节点类型: 贸易国家 | 商品注册地
  "feature": "金额总额_单笔均价"        // 特征组合: 金额总额_单笔均价 | 贸易条数_单笔均价
}
```

**响应示例**:
```json
{
  "centers": [
    [1.2e9, 3.5],                      // 聚类中心1: [金额, 单价]
    [5e8, 2.1],                        // 聚类中心2
    [8e8, 4.2],                        // 聚类中心3
    [3e8, 1.8]                         // 聚类中心4
  ],
  "labels": [0, 0, 1, 2, 1, 3, ...],   // 每个国家的聚类标签 (191个)
  "data": [
    {"name": "United States", "cluster": 0, "x": 1.2e9, "y": 3.5},
    {"name": "Japan", "cluster": 0, "x": 1.1e9, "y": 3.3},
    ...
  ]
}
```

**说明**:
- 参数1: 节点类型(国家/省份) - 下拉选择
- 参数2: 特征组合(金额+单价/贸易条数+单价) - 下拉选择
- 参数3: 年份 - 下拉选择
- **注意**: 原设计中"国家"和"省份"应在同一个选项框内

**代码位置**: `app.py:570-600`

**数据源**: `json/cluster/kmeans/kmeans_data_{year}_{节点类型}_{特征}.json`

---

#### 12. 地图数据

**接口**: `GET /api/get_map_data?map_type=world`

**功能**: 获取地图GeoJSON数据

**参数**:
- `map_type`: `world` (世界地图) | `china` (中国地图)

**响应示例**:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"name": "China", "iso_a3": "CHN"},
      "geometry": {"type": "Polygon", "coordinates": [...]}
    },
    ...
  ]
}
```

**代码位置**: `app.py:610-630`

**数据源**:
- `json/cluster/world.json`
- `json/cluster/china.json`

---

#### 13. 国家名映射

**接口**: `GET /api/get_country_mapping`

**功能**: 获取国家名中英文映射 (用于地图标注)

**响应示例**:
```json
{
  "United States": "美国",
  "Japan": "日本",
  "South Korea": "韩国",
  ...
}
```

**代码位置**: `app.py:640-655`

**数据源**: `json/cluster/country_name_mapping.json`

---

## 🗄️ 数据库表结构

### 主表: `dboper.imports_master`

| 字段名 | 数据类型 | 说明 | 索引 |
|--------|---------|------|------|
| 年份 | INT | 2012-2021 | ✅ 复合索引 |
| 贸易国家 | VARCHAR(100) | 191个国家 | ✅ 复合索引 |
| 商品注册地 | VARCHAR(50) | 31个省份 | ✅ 复合索引 |
| 贸易方式 | VARCHAR(50) | 一般贸易/加工贸易等 | |
| 章节名称 | VARCHAR(200) | 98个商品大类 | ✅ 复合索引 |
| 商品名称 | VARCHAR(500) | 9594种商品 | |
| 单位 | VARCHAR(20) | 千克/吨/个等 | |
| 金额 | BIGINT | 交易金额(元) | |
| 数量 | DECIMAL(15,4) | 交易数量 | |
| 单价 | DECIMAL(15,4) | 计算列: 金额/数量 | |

**复合索引**:
```sql
CREATE INDEX idx_composite ON imports_master(年份, 贸易国家, 商品注册地, 章节名称);
CREATE INDEX idx_chapter_year ON imports_master(章节名称, 年份);
```

**查询示例**:

```python
# services/db_Manager.py

# 1. 获取平均单价 (_get_avg_price_data)
# 查询条件: 贸易国家、商品注册地、贸易方式、商品名称、单位、年份
# 返回: 单价均值
SELECT AVG(单价)
FROM dboper.imports_master
WHERE 贸易国家=%s AND 商品注册地=%s AND 贸易方式=%s
  AND 商品名称=%s AND 单位=%s AND 年份=%s;

# 2. 获取折线图数据 (_get_line_param)
# 查询条件: 章节名称、参数(金额/贸易条数/单价)
# 返回: 10年的时间序列数据
SELECT 年份,
       SUM(金额) as 金额总额,
       COUNT(*) as 贸易条数,
       AVG(单价) as 平均单价
FROM dboper.imports_master
WHERE 章节名称=%s
GROUP BY 年份
ORDER BY 年份 ASC;

# 3. 获取饼图数据 (_get_pie_param)
# 查询条件: 章节名称、关系(国家/省份)、年份、参数(金额/贸易条数/单价)
# 返回: Top 5 + 占比
SELECT 贸易国家,
       SUM(金额) as value,
       ROUND(SUM(金额) * 100.0 / total, 2) as proportion
FROM dboper.imports_master
WHERE 章节名称=%s AND 年份=%s
GROUP BY 贸易国家
ORDER BY value DESC
LIMIT 5;
```

**数据规模**:
- 总记录数: 12,671,200条
- 时间跨度: 2012-2021年
- 数据大小: ~8.5GB

---

## 🧠 核心服务模块说明

### 1. services/mlp.py - MLP单价预测

**核心函数**:
```python
def predict(country, reg_place, product_code, unit, year, trade_method):
    """
    Args:
        country: 贸易国家 (如 "United States")
        reg_place: 商品注册地 (如 "广东省")
        product_code: 商品编码 (如 "2005999990")
        unit: 计价单位 (如 "千克")
        year: 年份 (2012-2030)
        trade_method: 贸易方式 (如 "一般贸易")

    Returns:
        float: 预测单价 (元/单位)
    """
```

**模型架构**:
- 嵌入层: 国家(64维) + 省份(16维) + 商品(128维) + 单位(16维)
- MLP: 4层隐层 (512→256→128→64→1)
- 激活函数: ReLU
- 正则化: Dropout(0.3/0.2)

**性能指标**:
- MAPE: 8.3%
- R²: 0.90
- 推理延迟: 15ms

---

### 2. services/resnet.py - ResNet50图像识别

**核心函数**:
```python
def predict_image(image_path):
    """
    Args:
        image_path: 图片路径 (本地文件路径)

    Returns:
        str: 章节名称 (如 "第20章-蔬菜、水果、坚果或植物其他部分的制品")
    """
```

**预处理流程**:
1. Decode (解码图片)
2. Resize(256) (短边缩放)
3. CenterCrop(224) (中心裁剪)
4. Normalize (ImageNet标准归一化)
5. HWC2CHW (转换通道顺序)

**性能指标**:
- 准确率: 92.3% (二分类)
- 推理速度: 50 FPS (Ascend NPU)

---

### 3. services/call_LLM.py - Claude AI调用

**核心函数**:
```python
def call_LLM(text_prompt, image_data=None, image_type=None):
    """
    Args:
        text_prompt: 文本提示词
        image_data: 图片数据 (Base64字符串/文件路径/字节流)
        image_type: 'base64' | 'path' | 'bytes'

    Returns:
        str: AI分析结果
    """
```

**支持的输入方式**:
- 本地图片上传 (image_type='path')
- 粘贴剪贴板图片 (image_type='base64')
- 字节流 (image_type='bytes')

**重试机制**:
- 最多重试5次 (4次异常捕获 + 1次强制执行)
- 适用于网络波动场景

---

### 4. services/db_Manager.py - 数据库管理

**核心方法**:

```python
class DBManager:
    def _get_avg_price_data(self, country, province, trade_type, name, unit, year):
        """获取单条记录的平均单价"""

    def _get_line_param(self, chapter_name, param):
        """获取折线图数据 (10年时间序列)"""

    def _get_pie_param(self, chapter_name, relation, year, param):
        """获取饼图数据 (Top 5 + 占比)"""
```

**连接池配置**:
```python
# config/settings.py
DB_HOST = "123.249.40.133"
DB_PORT = 5432
DB_USER = "ltb"
DB_PASSWORD = "xlbt123456."
DB_NAME = "postgres"
```

---

## 🎨 前端模块说明

### 1. static/js/param.js - 参数选择器

**功能**:
- 三级联动选择: 类 → 章 → 商品
- 分页加载 (每页50条,支持上一页/下一页)
- 长名称省略显示,鼠标悬停显示完整名称
- 实时同步国家/省份/贸易方式/单位选项

**主要方法**:
```javascript
ParamsManager.init()                    // 初始化
ParamsManager.loadProductMapping()      // 加载商品分类树
ParamsManager.renderCategoryList()      // 渲染类别列表
ParamsManager.renderChapterList()       // 渲染章节列表
ParamsManager.renderProductList()       // 渲染商品列表 (分页)
```

---

### 2. static/js/chart.js - 图表管理

**功能**:
- 单价预测折线图
- 双线展示: 2012-2021真实数据(蓝色实线) + 2025-2030预测数据(红色虚线)
- 多维对比: 支持同一商品不同国家的多条线同时显示

**主要方法**:
```javascript
ChartManager.init()                     // 初始化图表
ChartManager.updateRealData()           // 更新真实数据
ChartManager.updatePredictedData()      // 更新预测数据
ChartManager.addComparisonLine()        // 添加对比线
```

---

### 3. static/js/dashboard.js - 特征大屏

**功能**:
- 折线图 (10年趋势)
- 饼图 (Top 5占比)
- 宏观条形图 (Top 10)
- 聚类散点图 (K-means可视化)
- 宏观统计卡片 (6个指标)

**主要方法**:
```javascript
DashboardApp.init()                     // 初始化大屏
DashboardApp.loadMacroStats()           // 加载宏观统计
DashboardApp.updateLineChart()          // 更新折线图
DashboardApp.updatePieChart()           // 更新饼图
DashboardApp.updateClusterChart()       // 更新聚类图
```

---

### 4. static/js/llm.js - AI对话管理

**功能**:
- 图片上传/粘贴/截图
- Base64编码处理
- 对话历史管理
- 自适应文本框

**主要方法**:
```javascript
LLMManager.init()                       // 初始化
LLMManager.uploadImage()                // 上传图片
LLMManager.pasteImage()                 // 粘贴图片
LLMManager.sendMessage()                // 发送消息
LLMManager.addMessageToHistory()        // 添加到历史
```

---

## 🔧 配置说明

### 环境变量

可通过环境变量覆盖默认配置:

```bash
export DB_HOST="your_db_host"
export DB_PORT="5432"
export DB_USER="your_username"
export DB_PASSWORD="your_password"
export ANTHROPIC_API_KEY="your_claude_api_key"
export FLASK_ENV="production"
```

### MindSpore设备配置

```bash
# 使用CPU推理
export MINDSPORE_DEVICE_TARGET="CPU"

# 使用Ascend NPU推理
export MINDSPORE_DEVICE_TARGET="Ascend"
export ASCEND_DEVICE_ID=0

# 使用GPU推理
export MINDSPORE_DEVICE_TARGET="GPU"
```

---

## 📊 性能优化建议

### 1. 数据库优化

```sql
-- 创建物化视图(预计算宏观统计)
CREATE MATERIALIZED VIEW mv_yearly_stats AS
SELECT 年份, SUM(金额) as total_amount, COUNT(*) as total_trans
FROM dboper.imports_master
GROUP BY 年份;

-- 定期刷新
REFRESH MATERIALIZED VIEW mv_yearly_stats;
```

### 2. Redis缓存 (推荐)

```python
# 缓存商品映射表 (减少磁盘IO)
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# 设置缓存
r.setex('product_mapping', 3600, json.dumps(mapping_data))

# 读取缓存
cached = r.get('product_mapping')
if cached:
    data = json.loads(cached)
```

### 3. 模型量化

```python
# MindSpore模型量化 (推理速度提升3倍)
from mindspore.compression.quant import QuantizationAwareTraining

qat = QuantizationAwareTraining(config={'quant_dtype': mindspore.int8})
quantized_model = qat.quantize(model)
```

---

## 🐛 常见问题

### Q1: 数据库连接超时

**问题**: `psycopg2.OperationalError: timeout expired`

**解决方案**:
```python
# 增加连接超时时间 (config/settings.py)
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    connect_timeout=30  # 增加到30秒
)
```

### Q2: MindSpore模型加载失败

**问题**: `RuntimeError: Load checkpoint file failed`

**解决方案**:
```bash
# 检查模型文件是否存在
ls -lh ckpt/mlp.ckpt

# 验证模型文件完整性
md5sum ckpt/mlp.ckpt
```

### Q3: API请求500错误

**问题**: Claude API调用失败

**解决方案**:
```python
# 检查API Key配置 (config/settings.py)
API_KEY = "sk-ant-xxx"  # 确保API Key有效

# 检查网络连接
curl -I https://api.anthropic.com
```

### Q4: 图像识别返回"未知类别"

**问题**: ResNet模型未正确加载

**解决方案**:
```bash
# 验证模型权重文件
python -c "
from mindspore import load_checkpoint
param_dict = load_checkpoint('ckpt/resnet.ckpt')
print(f'模型参数数量: {len(param_dict)}')
"
```

### Q5: 聚类参数选择框显示错误

**问题**: 国家和省份应在同一个选项框内

**解决方案**:
```javascript
// dashboard.js 修改聚类参数选择
<select id="nodeTypeSelect">
  <option value="贸易国家">国家</option>
  <option value="商品注册地">省份</option>
</select>
```

---

## 📈 监控与日志

### 日志配置

```python
# app.py 添加日志配置
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/trade_app/app.log'),
        logging.StreamHandler()
    ]
)
```

### Nginx访问日志

```nginx
# /etc/nginx/sites-available/trade_app
access_log /var/log/nginx/trade_app_access.log;
error_log /var/log/nginx/trade_app_error.log;
```

---

## 🚢 生产部署

### 使用Docker部署

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
# 构建镜像
docker build -t trade-analysis:latest .

# 运行容器
docker run -d -p 5000:5000 \
  -e DB_HOST="123.249.40.133" \
  -e DB_PASSWORD="your_password" \
  --name trade-app \
  trade-analysis:latest
```

### 使用systemd管理

```ini
# /etc/systemd/system/trade-app.service
[Unit]
Description=Trade Analysis Application
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/home/user/app
ExecStart=/usr/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# 启动服务
sudo systemctl start trade-app
sudo systemctl enable trade-app
```

---

## 📝 已知问题与待改进项

### 图一 (单价预测页面)

1. ✅ **商品选择优化**:
   - 问题: 下拉菜单显示全白,长名称无法完全显示
   - 解决方案: 使用分页加载(每页50条) + 名称省略显示

2. ✅ **双线区分**:
   - 2012-2021真实数据: 蓝色实线
   - 2025-2030预测数据: 红色虚线

3. ⏳ **多维对比功能** (待实现):
   - 支持同一商品不同国家的多条线同时显示
   - 可动态添加/删除对比线

4. ✅ **AI大模型调用**:
   - 支持图片本地上传和粘贴

5. ⏳ **事件模拟功能** (待实现):
   - 用红点标记某一年
   - 模拟因素: 政策、灾害、战争、疫情、经济、贸易
   - 单价乘以系数 (如 ×1.5)

### 图二 (特征大屏页面)

1. ✅ **宏观数据统计**:
   - 卡片式展示: 总金额、交易次数、贸易伙伴数、省份数、商品种类数
   - 条形图: Top 10国家/省份/商品

2. ✅ **图像识别自动更新**:
   - 识别后自动更新折线图和饼图
   - 折线图默认展示"单价",可切换到"金额"或"贸易条数"
   - 饼图默认展示"国家 + 2021 + 单价",三个参数可选

3. ⚠️ **聚类参数修正**:
   - 问题: 原设计4个参数,应为3个参数
   - 解决方案: "国家"和"省份"合并到同一个选项框

4. ⏳ **页面布局优化** (待实现):
   - 图像识别区域放大,作为核心展示
   - AI大模型调用改为弹窗模式(与图一一致)

5. ✅ **多种识别方式**:
   - 支持本地上传和摄像头拍照识别

---

## 📄 许可证

MIT License

---

## 📞 联系方式

- **邮箱**: trade-ai@example.com
- **GitHub**: https://github.com/your-repo

---

## 🔗 相关链接

- [MindSpore官方文档](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- [openGauss文档](https://docs.opengauss.org/zh/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [ECharts文档](https://echarts.apache.org/zh/index.html)
- [华为云ModelArts](https://www.huaweicloud.com/product/modelarts.html)

---

**最后更新**: 2025-12-01