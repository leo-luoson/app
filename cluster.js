$(document).ready(function() {
    
    // 初始化图表
    var myChart = echarts.init(document.getElementById('mainChart'));
    myChart.showLoading();

    // =========================================================
    // 1. 内置完整的国家名称映射 (防止外部JSON加载失败)
    // Key = GeoJSON里的英文名 (ECharts标准)
    // Value = 数据里的中文名
    // =========================================================
    var globalNameMap = {
    "Somalia": "索马里",
    "Liechtenstein": "列支敦士登",
    "Morocco": "摩洛哥",
    "W. Sahara": "西撒哈拉",
    "Serbia": "塞尔维亚",
    "Afghanistan": "阿富汗",
    "Angola": "安哥拉",
    "Albania": "阿尔巴尼亚",
    "Aland": "奥兰群岛",
    "Andorra": "安道尔",
    "United Arab Emirates": "阿联酋",
    "Argentina": "阿根廷",
    "Armenia": "亚美尼亚",
    "American Samoa": "美属萨摩亚",
    "Fr. S. Antarctic Lands": "法属南部领地",
    "Antigua and Barb.": "安提瓜和巴布达",
    "Australia": "澳大利亚",
    "Austria": "奥地利",
    "Azerbaijan": "阿塞拜疆",
    "Burundi": "布隆迪",
    "Belgium": "比利时",
    "Benin": "贝宁",
    "Burkina Faso": "布基纳法索",
    "Bangladesh": "孟加拉国",
    "Bulgaria": "保加利亚",
    "Bahrain": "巴林",
    "Bahamas": "巴哈马",
    "Bosnia and Herz.": "波黑",
    "Belarus": "白俄罗斯",
    "Belize": "伯利兹",
    "Bermuda": "百慕大",
    "Bolivia": "多民族玻利维亚国",
    "Brazil": "巴西",
    "Barbados": "巴巴多斯",
    "Brunei": "文莱",
    "Bhutan": "不丹",
    "Botswana": "博茨瓦纳",
    "Central African Rep.": "中非",
    "Canada": "加拿大",
    "Switzerland": "瑞士",
    "Chile": "智利",
    "China": "中国",
    "Côte d'Ivoire": "科特迪瓦",
    "Cameroon": "喀麦隆",
    "Dem. Rep. Congo": "刚果(金)",
    "Congo": "刚果(布)",
    "Colombia": "哥伦比亚",
    "Comoros": "科摩罗",
    "Cape Verde": "佛得角",
    "Costa Rica": "哥斯达黎加",
    "Cuba": "古巴",
    "Curaçao": "库拉索",
    "Cayman Is.": "开曼群岛",
    "N. Cyprus": "北塞浦路斯",
    "Cyprus": "塞浦路斯",
    "Czech Rep.": "捷克",
    "Germany": "德国",
    "Djibouti": "吉布提",
    "Dominica": "多米尼克",
    "Denmark": "丹麦",
    "Dominican Rep.": "多米尼加共和国",
    "Algeria": "阿尔及利亚",
    "Ecuador": "厄瓜多尔",
    "Egypt": "埃及",
    "Eritrea": "厄立特里亚",
    "Spain": "西班牙",
    "Estonia": "爱沙尼亚",
    "Ethiopia": "埃塞俄比亚",
    "Finland": "芬兰",
    "Fiji": "斐济",
    "Falkland Is.": "福克兰群岛",
    "France": "法国",
    "Faeroe Is.": "法罗群岛",
    "Micronesia": "密克罗尼西亚联邦",
    "Gabon": "加蓬",
    "United Kingdom": "英国",
    "Georgia": "格鲁吉亚",
    "Ghana": "加纳",
    "Guinea": "几内亚",
    "Gambia": "冈比亚",
    "Guinea-Bissau": "几内亚比绍",
    "Eq. Guinea": "赤道几内亚",
    "Greece": "希腊",
    "Grenada": "格林纳达",
    "Greenland": "格陵兰",
    "Guatemala": "危地马拉",
    "Guam": "关岛",
    "Guyana": "圭亚那",
    "Heard I. and McDonald Is.": "赫德岛和麦克唐纳群岛",
    "Honduras": "洪都拉斯",
    "Croatia": "克罗地亚",
    "Haiti": "海地",
    "Hungary": "匈牙利",
    "Indonesia": "印度尼西亚",
    "Isle of Man": "马恩岛",
    "India": "印度",
    "Br. Indian Ocean Ter.": "英属印度洋领地",
    "Ireland": "爱尔兰",
    "Iran": "伊朗",
    "Iraq": "伊拉克",
    "Iceland": "冰岛",
    "Israel": "以色列",
    "Italy": "意大利",
    "Jamaica": "牙买加",
    "Jersey": "泽西岛",
    "Jordan": "约旦",
    "Japan": "日本",
    "Siachen Glacier": "锡亚琴冰川",
    "Kazakhstan": "哈萨克斯坦",
    "Kenya": "肯尼亚",
    "Kyrgyzstan": "吉尔吉斯斯坦",
    "Cambodia": "柬埔寨",
    "Kiribati": "基里巴斯",
    "Korea": "韩国",
    "Kuwait": "科威特",
    "Lao PDR": "老挝",
    "Lebanon": "黎巴嫩",
    "Liberia": "利比里亚",
    "Libya": "利比亚",
    "Saint Lucia": "圣卢西亚",
    "Sri Lanka": "斯里兰卡",
    "Lesotho": "莱索托",
    "Lithuania": "立陶宛",
    "Luxembourg": "卢森堡",
    "Latvia": "拉脱维亚",
    "Moldova": "摩尔多瓦",
    "Madagascar": "马达加斯加",
    "Mexico": "墨西哥",
    "Macedonia": "北马其顿共和国",
    "Mali": "马里",
    "Malta": "马耳他",
    "Myanmar": "缅甸",
    "Montenegro": "黑山",
    "Mongolia": "蒙古",
    "N. Mariana Is.": "北马里亚纳群岛",
    "Mozambique": "莫桑比克",
    "Mauritania": "毛里塔尼亚",
    "Montserrat": "蒙特塞拉特",
    "Mauritius": "毛里求斯",
    "Malawi": "马拉维",
    "Malaysia": "马来西亚",
    "Namibia": "纳米比亚",
    "New Caledonia": "新喀里多尼亚",
    "Niger": "尼日尔",
    "Nigeria": "尼日利亚",
    "Nicaragua": "尼加拉瓜",
    "Niue": "纽埃",
    "Netherlands": "荷兰",
    "Norway": "挪威",
    "Nepal": "尼泊尔联邦民主共和国",
    "New Zealand": "新西兰",
    "Oman": "阿曼",
    "Pakistan": "巴基斯坦",
    "Panama": "巴拿马",
    "Peru": "秘鲁",
    "Philippines": "菲律宾",
    "Palau": "帕劳",
    "Papua New Guinea": "巴布亚新几内亚",
    "Poland": "波兰",
    "Puerto Rico": "波多黎各",
    "Dem. Rep. Korea": "朝鲜",
    "Portugal": "葡萄牙",
    "Paraguay": "巴拉圭",
    "Palestine": "巴勒斯坦",
    "Fr. Polynesia": "法属波利尼西亚",
    "Qatar": "卡塔尔",
    "Romania": "罗马尼亚",
    "Russia": "俄罗斯联邦",
    "Rwanda": "卢旺达",
    "Saudi Arabia": "沙特阿拉伯",
    "Sudan": "苏丹",
    "S. Sudan": "南苏丹共和国",
    "Senegal": "塞内加尔",
    "Singapore": "新加坡",
    "S. Geo. and S. Sandw. Is.": "南乔治亚和南桑威奇群岛",
    "Saint Helena": "圣赫勒拿",
    "Solomon Is.": "所罗门群岛",
    "Sierra Leone": "塞拉利昂",
    "El Salvador": "萨尔瓦多",
    "St. Pierre and Miquelon": "圣皮埃尔和密克隆",
    "São Tomé and Principe": "圣多美和普林西比",
    "Suriname": "苏里南",
    "Slovakia": "斯洛伐克",
    "Slovenia": "斯洛文尼亚",
    "Sweden": "瑞典",
    "Swaziland": "斯威士兰",
    "Seychelles": "塞舌尔",
    "Syria": "叙利亚",
    "Turks and Caicos Is.": "特克斯和凯科斯群岛",
    "Chad": "乍得",
    "Togo": "多哥",
    "Thailand": "泰国",
    "Tajikistan": "塔吉克斯坦",
    "Turkmenistan": "土库曼斯坦",
    "Timor-Leste": "东帝汶",
    "Tonga": "汤加",
    "Trinidad and Tobago": "特立尼达和多巴哥",
    "Tunisia": "突尼斯",
    "Turkey": "土耳其",
    "Tanzania": "坦桑尼亚",
    "Uganda": "乌干达",
    "Ukraine": "乌克兰",
    "Uruguay": "乌拉圭",
    "United States": "美国",
    "Uzbekistan": "乌兹别克斯坦",
    "St. Vin. and Gren.": "圣文森特和格林纳丁斯",
    "Venezuela": "委内瑞拉",
    "U.S. Virgin Is.": "美属维尔京群岛",
    "Vietnam": "越南",
    "Vanuatu": "瓦努阿图",
    "Samoa": "萨摩亚",
    "Yemen": "也门",
    "South Africa": "南非",
    "Zambia": "赞比亚",
    "Zimbabwe": "津巴布韦"
};

    // =========================================================
    // 2. 预加载地图资源
    // =========================================================
    $.when(
        $.get('/api/data/cluster/world.json'),
        $.get('/api/data/cluster/china.json')
    ).done(function(worldRes, chinaRes) {
        
        echarts.registerMap('world', worldRes[0]);
        echarts.registerMap('china', chinaRes[0]);
        
        myChart.hideLoading();

        $('#btnLoad').click(function() {
            loadAndRenderChart();
        });

        // 自动加载一次
        loadAndRenderChart();

    }).fail(function(err) {
        myChart.hideLoading();
        console.error("地图加载失败:", err);
        alert("严重错误：world.json 或 china.json 加载失败，请按F12查看Console");
    });

    // =========================================================
    // 3. 工具函数
    // =========================================================
    function flattenData(dictData) {
        var result = [];
        // 如果 dictData 已经是数组，直接返回
        if (Array.isArray(dictData)) return dictData;
        
        Object.keys(dictData).forEach(function(key) {
            var list = dictData[key];
            list.forEach(function(item) {
                item.cluster = parseInt(key); 
                result.push(item);
            });
        });
        return result;
    }

    function getTooltipFormatter(params) {
        if (params.data && params.data.raw) {
            var r = params.data.raw;
            var dot = `<span style="display:inline-block;margin-right:5px;border-radius:10px;width:10px;height:10px;background-color:${params.color};"></span>`;
            
            // 安全处理数字
            var amount = r['金额总额'] ? parseFloat(r['金额总额']).toLocaleString() : '0';
            var price = r['单笔均价'] ? parseFloat(r['单笔均价']).toLocaleString() : '0';
            var count = r['贸易条数'] || 0;

            return `
                <div style="font-size:14px; font-weight:bold; margin-bottom:5px; color:#fff;">
                    ${dot} ${r['节点名称']}
                </div>
                <div style="font-size:12px; line-height:1.8; color:#eee;">
                    聚类类别：<b style="color:#fff">${r.cluster}</b><br/>
                    金额总额：<b style="color:#fff">${amount}</b><br/>
                    贸易条数：<b style="color:#fff">${count}</b><br/>
                    单笔均价：<b style="color:#fff">${price}</b>
                </div>
            `;
        }
        return params.name; 
    }

    // =========================================================
    // 4. 主逻辑
    // =========================================================
    function loadAndRenderChart() {
        var year = $('#selectYear').val();
        var target = $('#selectTarget').val();      
        var features = $('#selectFeatures').val();  

        $('#chartTitle').text(`${year}年 ${target} 聚类分析`);
        var filename = `kmeans_data_${year}_${target}_${features}.json`;
        
        myChart.showLoading();

        $.get('/api/data/json/' + filename)
            .done(function(rawDictData) {
                myChart.hideLoading();
                var flatData = flattenData(rawDictData);

                if (target === '贸易国家') {
                    renderWorldMap(flatData);
                } else if (target === '商品注册地') {
                    renderChinaMap(flatData);
                } else {
                    renderScatterPlot(flatData, features, target);
                }
            })
            .fail(function() {
                myChart.hideLoading();
                alert('未找到数据文件：' + filename);
            });
    }

    // =========================================================
    // 渲染器 A: 世界地图 
    // =========================================================
    function renderWorldMap(data) {
        var maxLabel = 0;
        data.forEach(d => { if(d.cluster > maxLabel) maxLabel = d.cluster; });

        // 数据清洗与构造
        var seriesData = data.map(function(item) {
            var name = item['节点名称'] ? item['节点名称'].trim() : "";
            
            // 简单修正：有些数据里叫 "美国"，有些叫 "美国 "，这里去空格
            // 另外，ECharts 映射是从 GeoJSON -> nameMap -> Data
            // 所以这里只需要保证 data.name 和 nameMap 的 value 一致即可
            
            return {
                name: name, 
                value: item.cluster, 
                raw: item
            };
        });
        
        // 调试：打印几个数据看看是否正常
        console.log("世界地图数据准备就绪，共", seriesData.length, "条");

        var option = {
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(50, 50, 50, 0.8)',
                borderColor: '#777',
                borderWidth: 1,
                padding: 10,
                formatter: getTooltipFormatter
            },
            visualMap: {
                type: 'piecewise', 
                splitNumber: maxLabel + 1,
                min: 0,
                max: maxLabel,
                left: 'left',
                bottom: 'bottom',
                inRange: { 
                    color: ['#c23531', '#2f4554', '#61a0a8', '#d48265', '#91c7ae', '#749f83'] 
                },
                text: ['Cluster High', 'Cluster Low'],
                formatter: function (value) { return 'Cluster ' + value; }
            },
            series: [{
                type: 'map',
                map: 'world',
                roam: true,
                // 【核心】直接使用内置的字典
                nameMap: globalNameMap, 
                
                itemStyle: {
                    areaColor: '#f3f3f3', 
                    borderColor: '#ccc'
                },
                emphasis: {
                    label: { show: true },
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                },
                data: seriesData
            }]
        };

        myChart.clear();
        myChart.setOption(option);
    }

    // =========================================================
    // 渲染器 B: 中国地图
    // =========================================================
    function renderChinaMap(data) {
        var maxLabel = 0;
        data.forEach(d => { if(d.cluster > maxLabel) maxLabel = d.cluster; });

        var seriesData = data.map(function(item) {
            var rawName = item['节点名称'];
            return {
                name: rawName,
                value: item.cluster,
                raw: item
            };
        });

        var option = {
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(50, 50, 50, 0.8)',
                borderColor: '#777',
                borderWidth: 1,
                padding: 10,
                formatter: getTooltipFormatter
            },
            visualMap: {
                type: 'piecewise',
                min: 0,
                max: maxLabel,
                left: 'left',
                bottom: 'bottom',
                inRange: { color: ['#c23531', '#2f4554', '#61a0a8', '#d48265', '#91c7ae'] },
                formatter: function (value) { return 'Cluster ' + value; }
            },
            series: [{
                type: 'map',
                map: 'china',
                roam: true,
                label: { show: true, fontSize: 10 },
                itemStyle: {
                    areaColor: '#f3f3f3',
                    borderColor: '#ccc'
                },
                emphasis: {
                    label: { show: true },
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                },
                data: seriesData
            }]
        };

        myChart.clear();
        myChart.setOption(option);
    }

    // =========================================================
    // 渲染器 C: 散点图
    // =========================================================
    function renderScatterPlot(data, features, target) {
        var featureNames = features.split('_'); 
        var xField = featureNames[0]; 
        var yField = featureNames[1]; 
        var maxLabel = 0;
        data.forEach(d => { if(d.cluster > maxLabel) maxLabel = d.cluster; });

        var scatterData = data.map(item => {
            return {
                value: [ item[xField], item[yField], item.cluster ],
                name: item['节点名称'], 
                raw: item
            };
        });

        var option = {
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(50, 50, 50, 0.8)',
                borderColor: '#777',
                borderWidth: 1,
                padding: 10,
                formatter: getTooltipFormatter
            },
            grid: { left: '10%', right: '15%', top: '15%', bottom: '10%' },
            xAxis: { name: xField, type: 'value', scale: true },
            yAxis: { name: yField, type: 'value', scale: true },
            visualMap: {
                type: 'piecewise',
                dimension: 2, 
                categories: Array.from({length: maxLabel + 1}, (_, i) => i),
                inRange: { color: ['#c23531', '#2f4554', '#61a0a8', '#d48265', '#91c7ae'] },
                right: 10,
                top: 'center',
                text: ['Cluster']
            },
            series: [{
                type: 'scatter',
                symbolSize: 15,
                data: scatterData
            }]
        };
        myChart.clear();
        myChart.setOption(option);
    }
    
    window.addEventListener('resize', () => myChart.resize());
});