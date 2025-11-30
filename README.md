当前页面有误的地方：
对于图一
1.商品选择有问题，点击出来是类-章-商品，显示有问题是全白看不清的，然后这部分处理东西很多面板有点小了（因为商品很多可以用下一页，为了美观长的名称可省略，挪到上面可全显示）
2.预测部分2012-2021是真实数据，2025-2030是模型外推(两条线不一样颜色)
对于图二：
1.聚类 展示部分有错误，我的意思是三个参数，您写成了四个有点问题，国家省份应该在一个选项框内
2.页面布局有问题，图像识别那块尽量大点核心展示，然后调用AI大模型这块希望和图一那样也是弹出

对于图一：需要增加部分
1.预测要有多维对比的功能，即用户可以多增加同一商品......但是不同国家的线都显示在图上
2.AI大模型调用  图片可本地上传可粘贴，要debug一下
3.增加 事件模拟（这块用红点表示某一年即可，用户可以选择某一年去模拟）因素：政策、灾害、战争、疫情、经济、贸易（显式*1.5即可先有这个功能

对于图二：需要增加部分
1.宏观数据统计部分，第一个总体宏观，卡片的格式展示总金额 、交易次数、 贸易伙伴总数、省份总数、商品种类总数（参数为年份），第二个仍然是条形图，保持现有不变
2.图像识别后（可上传可摄像头拍照识别），自动更新折线图和饼图（如果页面布局够的话，不够就展示折线即可然后手动切换），折线图默认展示单价的折线图（有参数选项可以使用金额和贸易条数），饼图默认展示国家+2021+单价（三个参数可选）

我改动文件：
1.增加ckpt文件夹，保存模型的参数文件
2.增加config文件夹，settings.py远程连接数据库的账号密码以及地址端口以及API
3.增加json文件夹，
——cluster——kmeans/....为聚类json文件（只需选择贸易国家和商品注册地即可
        |——china.json 中国地图映射文件
        |——world.json 世界地图映射文件
        |__country_name_mapping.json 中文国家名称映射为英文名称文件
——mapping——category_chapter_product_mapping.json 商品选择那部分可用这个文件
        |——country_to_index.json、province_to_index.json、trade_to_onehot.json、unit_to_index.json,参数初始化只用键值
——total_stats——宏观数据展示的部分文件（第一个总体宏观total_stats_{year}.json以及 country/province/product_stats_{year}.json文件

4.增加services文件夹
call_LLM.py 调用AI大模型文件 def call_LLM(text_prompt, image_data=None, image_type=None):（实际前端粘贴我不清楚是否可以，不行我再修改）
db_Manager.py (用于返回页面一调用真实数据和页面二返回图像识别后line和pie图)
* # 使用贸易国家、商品注册地、贸易方式、商品名称、单位、年份 作为查询条件 ，返回所有记录单价的均值
    def _get_avg_price_data(self, country, province, trade_type,name, unit, year):
* # 使用章节名称、参数（金额总额、贸易条数、单价） 作为查询条件 ，返回10年的金额sum、贸易条数count、单价avg
    def _get_line_param(self, chapter_name,  param):
* # 使用章节名称、关系连接（国家/省份）、年份、参数（金额总额、贸易条数、单价） 作为查询条件 ，返回金额sum、贸易条数count、单价avg,返回top 5名称及值 
    def _get_pie_param(self, chapter_name, relation, year, param):
mlp.py 单价预测 def predict(country, reg_place, product_code, unit, year, trade_method):
resnet.py 图像识别 def predict_image(image_path):