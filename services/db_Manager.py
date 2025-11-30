# services/db_manager.py

import psycopg2
from psycopg2.extras import RealDictCursor
import sys
import os
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import settings

logging.basicConfig(level=logging.INFO, format='[DB] %(message)s')
logger = logging.getLogger(__name__)

class DBManager:
    def _get_connection(self):
        try:
            
            conn = psycopg2.connect(
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                database=settings.DB_NAME,
                connect_timeout=10
            )
            return conn
        except Exception as e:
            logger.error(f"连接失败! 请检查IP白名单或账号密码。错误详情: {e}")
            return None
# 使用贸易国家、商品注册地、贸易方式、商品名称、单位、年份 作为查询条件 ，返回所有记录单价的均值
    def _get_avg_price_data(self, country, province, trade_type,name, unit, year):

        conn = self._get_connection()
        if not conn:
            return []

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:

                query = f"""
                    SELECT AVG(单价) as avg_price
                    FROM {settings.TABLE_NAME}
                    WHERE 贸易国家 = %s
                    AND 商品注册地 = %s
                    AND 贸易方式 = %s
                    AND 商品名称 = %s
                    AND 单位 = %s
                    AND 年份 = %s
                """
                cursor.execute(query, (country, province, trade_type, name, unit, year))
                result = cursor.fetchone()
                if result and result['avg_price']:
                    return float(result['avg_price'])
                return 0.0
                
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return 0.0
        finally:
            conn.close()
                
# 使用章节名称、参数（金额总额、贸易条数、单价） 作为查询条件 ，返回10年的金额sum、贸易条数count、单价avg

    def _get_line_param(self, chapter_name,  param):
        # 先定parameter映射
        param_map = {  
            '金额': 'SUM(金额) as total_amount',
            '贸易条数': 'COUNT(*) as trade_count',
            '单价': 'AVG(单价) as avg_price'
        }
        if param not in param_map:
            logger.error(f"未知参数: {param}")
            return None
        conn = self._get_connection()
        if not conn:
            return None

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = f"""
                    SELECT 年份, {param_map[param]}
                    FROM {settings.TABLE_NAME}
                    WHERE 章节名称 = %s
                    GROUP BY 年份
                    ORDER BY 年份 ASC
                """
                cursor.execute(query, (chapter_name,))
                results = cursor.fetchall()
                return results
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return None
            
# 使用章节名称、关系连接（国家/省份）、年份、参数（金额总额、贸易条数、单价） 作为查询条件 ，返回金额sum、贸易条数count、单价avg,返回top 5名称及值
    def _get_pie_param(self, chapter_name, relation, year, param):

        param_map = {  
            '金额': 'SUM(金额) as total_amount',
            '贸易条数': 'COUNT(*) as trade_count',
            '单价': 'AVG(单价) as avg_price'
        }
        relation_map = {
            '国家': '贸易国家',
            '省份': '商品注册地'
        }
        if relation not in relation_map:
            logger.error(f"未知关系: {relation}")
            return None
        if param not in param_map:
            logger.error(f"未知参数: {param}")
            return None
        conn = self._get_connection()
        if not conn:
            return None
        #金额sum、贸易条数count、单价avg的top 5的国家/省份名称及值及归一化占比
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = f"""
                    SELECT {relation_map[relation]} as name, {param_map[param]}
                    FROM {settings.TABLE_NAME}
                    WHERE 章节名称 = %s
                    AND 年份 = %s
                    GROUP BY {relation_map[relation]}
                    ORDER BY {param_map[param].split(' as ')[1]} DESC
                    LIMIT 5
                """
                cursor.execute(query, (chapter_name, year))
                results = cursor.fetchall()
                total_value = sum([row[param_map[param].split(' as ')[1]] for row in results])
                for row in results:
                    value = row[param_map[param].split(' as ')[1]]
                    row['proportion'] = (value / total_value) if total_value > 0 else 0
                return results
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return None
        finally:
            conn.close()

            


# # 测试代码
# if __name__ == "__main__":
# db_manager = DBManager()
# avg_price = db_manager._get_avg_price_data("西班牙", "上海市", "一般贸易", "苦参碱", "千克", 2012)
# print(f"平均单价: {avg_price}")
# db_manager = DBManager()
# agg_result = db_manager._get_line_param("第20章-蔬菜、水果、坚果或植物其他部分的制品", "金额")
# print(f"结果: {agg_result}")
# #类型
# print(type(agg_result))
# agg_result = db_manager._get_line_param("第20章-蔬菜、水果、坚果或植物其他部分的制品", "贸易条数")
# print(f"结果: {agg_result}")
# agg_result = db_manager._get_line_param("第20章-蔬菜、水果、坚果或植物其他部分的制品", "单价")
# print(f"结果: {agg_result}")
# db_manager = DBManager()
# pie_result = db_manager._get_pie_param("第20章-蔬菜、水果、坚果或植物其他部分的制品", "国家", 2012, "金额")
# print(f"饼图结果: {pie_result}")
# #类型
# print(type(pie_result))
# pie_result = db_manager._get_pie_param("第20章-蔬菜、水果、坚果或植物其他部分的制品", "省份", 2012, "贸易条数")
# print(f"饼图结果: {pie_result}")
# pie_result = db_manager._get_pie_param("第20章-蔬菜、水果、坚果或植物其他部分的制品", "国家", 2012, "单价")
# print(f"饼图结果: {pie_result}")
    