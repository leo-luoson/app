from flask import Flask, render_template, request, jsonify
import os
import json
import base64
from werkzeug.utils import secure_filename

# 导入服务模块
from services.db_Manager import DBManager
from services.mlp import predict as mlp_predict
from services.resnet import predict_image
from services.call_LLM import call_LLM

# 配置
UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

    # 初始化数据库管理器
    db_manager = DBManager()

    # ========== 页面路由 ==========
    @app.route("/")
    def index():
        """
        Page 1: Trade unit price prediction (static layout only).
        Page 1: Trade unit price prediction.
        """
        return render_template("index.html")

    @app.route("/dashboard")
    def dashboard():
        """
        Page 2: Trade feature dashboard (static layout only).
        Page 2: Trade feature dashboard.
        """
        return render_template("dashboard.html")

    # NOTE:
    # Future API endpoints (for AI model, data queries, recognition, etc.)
    # should be added here, for example:
    #
    # @app.route("/api/predict", methods=["OST"])
    # def api_predict():
    #     # TODO: receive parameters & call ML/LLM services
    #     pass
    #
    # Keep this file lightweight for now since the user only needs static pages.
    # ========== API 接口 ==========

    # === 图一：单价预测相关API ===

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        """
        单价预测接口
        接收参数：country, reg_place, product_code, unit, year, trade_method
        返回：预测的单价值
        """
        try:
            data = request.get_json()
            country = data.get('country')
            reg_place = data.get('reg_place')
            product_code = data.get('product_code', 'XX')  # 商品编码暂时用占位符
            unit = data.get('unit')
            year = data.get('year')
            trade_method = data.get('trade_method')

            # 调用MLP模型预测
            predicted_price = mlp_predict(country, reg_place, product_code, unit, year, trade_method)

            return jsonify({
                'success': True,
                'predicted_price': float(predicted_price)
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route("/api/get_real_data", methods=["POST"])
    def api_get_real_data():
        """
        获取真实历史数据
        接收参数：country, province, trade_type, name, unit, start_year, end_year
        返回：各年份的平均单价数据
        """
        try:
            data = request.get_json()
            country = data.get('country')
            province = data.get('province')
            trade_type = data.get('trade_type')
            name = data.get('name')
            unit = data.get('unit')
            start_year = int(data.get('start_year', 2012))
            end_year = int(data.get('end_year', 2021))

            result = []
            for year in range(start_year, end_year + 1):
                avg_price = db_manager._get_avg_price_data(
                    country, province, trade_type, name, unit, year
                )
                result.append({
                    'year': year,
                    'avg_price': float(avg_price)
                })

            return jsonify({
                'success': True,
                'data': result
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route("/api/llm_analyze", methods=["POST"])
    def api_llm_analyze():
        """
        AI大模型分析接口
        接收参数：text_prompt, image_data(可选), image_type(可选)
        返回：AI分析结果
        """
        try:
            data = request.get_json()
            text_prompt = data.get('text_prompt', '')
            image_data = data.get('image_data')  # base64编码的图片
            image_type = data.get('image_type', 'base64')  # 'base64', 'path', 'bytes'

            # 调用LLM服务
            result = call_LLM(text_prompt, image_data, image_type)

            return jsonify({
                'success': True,
                'result': result
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route("/api/product_mapping", methods=["GET"])
    def api_product_mapping():
        """
        获取商品分类映射
        返回：商品分类结构
        """
        try:
            json_path = os.path.join(
                os.path.dirname(__file__),
                'json/mapping/category_chapter_product_mapping.json'
            )
            with open(json_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)

            return jsonify({
                'success': True,
                'data': mapping
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    @app.route("/api/country_options", methods=["GET"])
    def api_country_options():
        """
        获取贸易国家选项
        返回：国家列表
        """
        try:
            json_path = os.path.join(
                os.path.dirname(__file__),
                'json/mapping/country_to_index.json'
            )
            with open(json_path, 'r', encoding='utf-8') as f:
                options = json.load(f)

            return jsonify({
                'success': True,
                'data': list(options.keys())
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    @app.route("/api/province_options", methods=["GET"])
    def api_province_options():
        """
        获取商品注册地选项
        返回：省份列表
        """
        try:
            json_path = os.path.join(
                os.path.dirname(__file__),
                'json/mapping/province_to_index.json'
            )
            with open(json_path, 'r', encoding='utf-8') as f:
                options = json.load(f)

            return jsonify({
                'success': True,
                'data': list(options.keys())
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    @app.route("/api/trade_type_options", methods=["GET"])    
    def api_trade_type_options():
        """
        获取贸易方式选项
        返回：贸易方式列表
        """
        try:
            json_path = os.path.join(
                os.path.dirname(__file__),
                'json/mapping/trade_to_onehot.json'
            )
            with open(json_path, 'r', encoding='utf-8') as f:
                options = json.load(f)

            return jsonify({
                'success': True,
                'data': list(options.keys())
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    @app.route("/api/unit_options", methods=["GET"])
    def api_unit_options():
        """
        获取单位选项
        返回：单位列表
        """
        try:
            json_path = os.path.join(
                os.path.dirname(__file__),
                'json/mapping/unit_to_index.json'
            )
            with open(json_path, 'r', encoding='utf-8') as f:
                options = json.load(f)

            return jsonify({
                'success': True,
                'data': list(options.keys())
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    # === 图二：特征大屏相关API ===

    @app.route("/api/recognize_product", methods=["POST"])
    def api_recognize_product():
        """
        图像识别接口
        接收：上传的图片文件
        返回：识别出的商品章节名称
        """
        try:
            if 'file' not in request.files:
                return jsonify({
                    'success': False,
                    'error': '没有上传文件'
                }), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': '文件名为空'
                }), 400

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # 调用ResNet模型识别
                chapter_name = predict_image(filepath)

                # 删除临时文件
                os.remove(filepath)

                return jsonify({
                    'success': True,
                    'chapter_name': chapter_name
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '不支持的文件类型'
                }), 400

        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route("/api/get_line_data", methods=["POST"])
    def api_get_line_data():
        """
        获取折线图数据
        接收参数：chapter_name, param（金额/贸易条数/单价）
        返回：10年的数据
        """
        try:
            data = request.get_json()
            chapter_name = data.get('chapter_name')
            param = data.get('param', '单价')

            result = db_manager._get_line_param(chapter_name, param)

            if result is None:
                return jsonify({
                    'success': False,
                    'error': '查询失败'
                }), 500

            # 转换为前端可用的格式
            formatted_result = []
            for row in result:
                formatted_result.append({
                    'year': row['年份'],
                    'value': float(row.get('total_amount') or row.get('trade_count') or row.get('avg_price', 0))
                })

            return jsonify({
                'success': True,
                'data': formatted_result
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route("/api/get_pie_data", methods=["POST"])
    def api_get_pie_data():
        """
        获取饼图数据
        接收参数：chapter_name, relation（国家/省份）, year, param（金额/贸易条数/单价）
        返回：Top 5数据及占比
        """
        try:
            data = request.get_json()
            chapter_name = data.get('chapter_name')
            relation = data.get('relation', '国家')
            year = int(data.get('year'))
            param = data.get('param', '单价')

            result = db_manager._get_pie_param(chapter_name, relation, year, param)

            if result is None:
                return jsonify({
                    'success': False,
                    'error': '查询失败'
                }), 500

            # 转换为前端可用的格式
            formatted_result = []
            for row in result:
                formatted_result.append({
                    'name': row['name'],
                    'value': float(row.get('total_amount') or row.get('trade_count') or row.get('avg_price', 0)),
                    'proportion': float(row['proportion'])
                })

            return jsonify({
                'success': True,
                'data': formatted_result
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route("/api/macro_stats", methods=["GET"])
    def api_macro_stats():
        """
        获取宏观统计数据
        返回：总金额、交易次数、贸易伙伴总数、省份总数、商品种类总数
        """
        try:
            year = int(request.args.get('year'))
            json_path = os.path.join(
                os.path.dirname(__file__),
                f'json/total_stats/total_stats_{year}.json'
            )

            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)

                return jsonify({
                    'success': True,
                    'data': stats
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'{year}年的统计数据不存在'
                }), 404

        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    # 宏观数据特征条形图API X为top 10的国家，Y为对应的贸易额 s为单位
    # 前端调取1.国家/省份/商品 2.金额/贸易次数/单价 3.year
    @app.route("/api/macro_bar_data", methods=["POST"])
    def api_macro_bar_data():
        """
        获取宏观条形图数据
        接收参数：relation（国家/省份/商品）, param（金额/贸易次数/单价）, year
        返回：Top 10数据
        """
        # 国家：country、省份：province、商品：product
        # 金额：stats 贸易次数：trade_count 单价：avg_price
        
        try:
            data = request.get_json()
            # 将国家/省份/商品映射为文件名中的关键字
            relation_map = {
                '国家': 'country',
                '省份': 'province',
                '商品': 'product'
            }
            relation = relation_map.get(data.get('relation', '国家'), 'country')
            param_map = {
                '金额': 'stats',
                '贸易次数': 'trade_count',
                '单价': 'avg_price'
            }
            param = param_map.get(data.get('param', '金额'), 'stats')
            year = int(data.get('year'))
            json_path = os.path.join(
                os.path.dirname(__file__),
                f'json/total_stats/{relation}_{param}_{year}.json'
            )
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    bar_data = json.load(f)

                return jsonify({
                    'success': True,
                    'data': bar_data
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'{year}年的宏观条形图数据不存在'
                }), 404
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    # 聚类分析API 传入参数是year 节点，feature： 贸易方式_金额
    @app.route("/api/cluster_analysis", methods=["POST"])
    def api_cluster_analysis():
        """
        获取聚类分析数据
        接收参数：year, feature
        返回：聚类结果
        """
        try:
            data = request.get_json()
            year = int(data.get('year'))
            node = data.get('node', '国家')
            feature = data.get('feature', '贸易方式_金额')
            
            json_path = os.path.join(
                os.path.dirname(__file__),
                f'json/cluster/kmeans/kmeans_data_{year}_{node}_{feature}.json'
            )
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    cluster_data = json.load(f)

                return jsonify({
                    'success': True,
                    'data': cluster_data
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'{year}年的聚类数据不存在'
                }), 404

        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    return app

# 测试
if __name__ == "__main__":
    # Development entry-point. In production, prefer a WSGI server.
    app = create_app()
    app.run(debug=True)