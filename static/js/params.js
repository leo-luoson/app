/**
 * params.js - 参数选择模块
 * 处理商品、国家、省份、贸易方式、单位等参数的选择逻辑
 */

const ParamsManager = (function() {
    // 存储当前选择的值
    let currentSelection = {
        category: null,      // 当前选中的类
        chapter: null,       // 当前选中的章节
        product: null,       // 当前选中的商品
        country: null,       // 当前选中的国家
        province: null,      // 当前选中的省份
        tradeType: null,     // 当前选中的贸易方式
        unit: null,          // 当前选中的单位
        startYear: null,     // 起始年份
        endYear: null        // 终止年份
    };

    // 存储从API获取的数据
    let apiData = {
        productMapping: null,
        countries: null,
        provinces: null,
        tradeTypes: null,
        units: null
    };

    // 分页配置
    const PAGE_SIZE = 10;
    let currentPages = {
        category: 0,
        chapter: 0,
        product: 0
    };

    /**
     * 初始化参数选择器
     */
    function init() {
        loadAllOptions();
        bindEvents();
    }

    /**
     * 加载所有选项数据
     */
    async function loadAllOptions() {
        try {
            await Promise.all([
                loadProductMapping(),
                loadCountryOptions(),
                loadProvinceOptions(),
                loadTradeTypeOptions(),
                loadUnitOptions()
            ]);
            console.log('所有选项数据加载完成');
        } catch (error) {
            console.error('加载选项数据失败:', error);
            alert('加载数据失败，请刷新页面重试');
        }
    }

    /**
     * 加载商品映射（类-章节-商品）
     */
    async function loadProductMapping() {
        try {
            const response = await fetch('/api/product_mapping');
            const result = await response.json();
            if (result.success) {
                apiData.productMapping = result.data;
                renderCategoryList();
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            console.error('加载商品映射失败:', error);
            throw error;
        }
    }

    /**
     * 加载国家选项
     */
    async function loadCountryOptions() {
        try {
            const response = await fetch('/api/country_options');
            const result = await response.json();
            if (result.success) {
                apiData.countries = result.data;
                renderCountrySelect();
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            console.error('加载国家选项失败:', error);
            throw error;
        }
    }

    /**
     * 加载省份选项
     */
    async function loadProvinceOptions() {
        try {
            const response = await fetch('/api/province_options');
            const result = await response.json();
            if (result.success) {
                apiData.provinces = result.data;
                renderProvinceSelect();
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            console.error('加载省份选项失败:', error);
            throw error;
        }
    }

    /**
     * 加载贸易方式选项
     */
    async function loadTradeTypeOptions() {
        try {
            const response = await fetch('/api/trade_type_options');
            const result = await response.json();
            if (result.success) {
                apiData.tradeTypes = result.data;
                renderTradeTypeSelect();
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            console.error('加载贸易方式选项失败:', error);
            throw error;
        }
    }

    /**
     * 加载单位选项
     */
    async function loadUnitOptions() {
        try {
            const response = await fetch('/api/unit_options');
            const result = await response.json();
            if (result.success) {
                apiData.units = result.data;
                renderUnitSelect();
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            console.error('加载单位选项失败:', error);
            throw error;
        }
    }

    /**
     * 渲染商品类别列表（第一级）
     */
    function renderCategoryList() {
        const categoryListEl = document.getElementById('productCategoryList');
        if (!categoryListEl || !apiData.productMapping) return;

        const categories = Object.keys(apiData.productMapping);
        const totalPages = Math.ceil(categories.length / PAGE_SIZE);
        const currentPage = currentPages.category;
        const start = currentPage * PAGE_SIZE;
        const end = start + PAGE_SIZE;
        const pageCategories = categories.slice(start, end);

        let html = pageCategories.map((category, index) => `
            <li class="list-group-item list-group-item-action ${index === 0 && currentPage === 0 ? 'active' : ''}"
                data-category="${category}">
                ${category}
            </li>
        `).join('');

        // 添加分页按钮
        if (totalPages > 1) {
            html += `
                <li class="list-group-item d-flex justify-content-between">
                    <button class="btn btn-sm btn-outline-secondary" id="btnPrevCategory"
                            ${currentPage === 0 ? 'disabled' : ''}>上一页</button>
                    <span class="align-self-center small">${currentPage + 1} / ${totalPages}</span>
                    <button class="btn btn-sm btn-outline-secondary" id="btnNextCategory"
                            ${currentPage >= totalPages - 1 ? 'disabled' : ''}>下一页</button>
                </li>
            `;
        }

        categoryListEl.innerHTML = html;

        // 绑定分类点击事件
        categoryListEl.querySelectorAll('[data-category]').forEach(li => {
            li.addEventListener('click', (e) => {
                categoryListEl.querySelectorAll('.list-group-item').forEach(el =>
                    el.classList.remove('active'));
                e.currentTarget.classList.add('active');
                currentSelection.category = e.currentTarget.dataset.category;
                currentPages.chapter = 0; // 重置章节页码
                renderChapterList();
            });
        });

        // 绑定分页按钮
        const btnPrev = document.getElementById('btnPrevCategory');
        const btnNext = document.getElementById('btnNextCategory');
        if (btnPrev) {
            btnPrev.addEventListener('click', () => {
                currentPages.category--;
                renderCategoryList();
            });
        }
        if (btnNext) {
            btnNext.addEventListener('click', () => {
                currentPages.category++;
                renderCategoryList();
            });
        }

        // 默认选中第一个
        if (pageCategories.length > 0 && currentPage === 0) {
            currentSelection.category = pageCategories[0];
            renderChapterList();
        }
    }

    /**
     * 渲染章节列表（第二级）
     */
    function renderChapterList() {
        const chapterListEl = document.getElementById('productChapterList');
        if (!chapterListEl || !currentSelection.category || !apiData.productMapping) return;

        const chapters = Object.keys(apiData.productMapping[currentSelection.category] || {});
        const totalPages = Math.ceil(chapters.length / PAGE_SIZE);
        const currentPage = currentPages.chapter;
        const start = currentPage * PAGE_SIZE;
        const end = start + PAGE_SIZE;
        const pageChapters = chapters.slice(start, end);

        let html = `<h6 class="mb-2">商品章节</h6>`;

        html += pageChapters.map((chapter, index) => `
            <li class="list-group-item list-group-item-action ${index === 0 && currentPage === 0 ? 'active' : ''}"
                data-chapter="${chapter}">
                ${chapter}
            </li>
        `).join('');

        // 添加分页按钮
        if (totalPages > 1) {
            html += `
                <li class="list-group-item d-flex justify-content-between">
                    <button class="btn btn-sm btn-outline-secondary" id="btnPrevChapter"
                            ${currentPage === 0 ? 'disabled' : ''}>上一页</button>
                    <span class="align-self-center small">${currentPage + 1} / ${totalPages}</span>
                    <button class="btn btn-sm btn-outline-secondary" id="btnNextChapter"
                            ${currentPage >= totalPages - 1 ? 'disabled' : ''}>下一页</button>
                </li>
            `;
        }

        chapterListEl.innerHTML = html;

        // 绑定章节点击事件
        chapterListEl.querySelectorAll('[data-chapter]').forEach(li => {
            li.addEventListener('click', (e) => {
                chapterListEl.querySelectorAll('.list-group-item').forEach(el =>
                    el.classList.remove('active'));
                e.currentTarget.classList.add('active');
                currentSelection.chapter = e.currentTarget.dataset.chapter;
                currentPages.product = 0; // 重置商品页码
                renderProductList();
            });
        });

        // 绑定分页按钮
        const btnPrev = document.getElementById('btnPrevChapter');
        const btnNext = document.getElementById('btnNextChapter');
        if (btnPrev) {
            btnPrev.addEventListener('click', () => {
                currentPages.chapter--;
                renderChapterList();
            });
        }
        if (btnNext) {
            btnNext.addEventListener('click', () => {
                currentPages.chapter++;
                renderChapterList();
            });
        }

        // 默认选中第一个
        if (pageChapters.length > 0 && currentPage === 0) {
            currentSelection.chapter = pageChapters[0];
            renderProductList();
        }
    }

    /**
     * 渲染商品列表（第三级）
     */
    function renderProductList() {
        const productListEl = document.getElementById('productList');
        if (!productListEl || !currentSelection.category || !currentSelection.chapter) return;

        const products = apiData.productMapping[currentSelection.category][currentSelection.chapter] || [];
        const totalPages = Math.ceil(products.length / PAGE_SIZE);
        const currentPage = currentPages.product;
        const start = currentPage * PAGE_SIZE;
        const end = start + PAGE_SIZE;
        const pageProducts = products.slice(start, end);

        let html = `<h6 class="mb-2">具体商品</h6><div class="row g-2">`;

        html += pageProducts.map(product => `
            <div class="col-6">
                <button type="button" class="btn btn-outline-primary w-100 btn-product-item"
                        data-product="${product}">
                    ${product}
                </button>
            </div>
        `).join('');

        html += `</div>`;

        // 添加分页按钮
        if (totalPages > 1) {
            html += `
                <div class="d-flex justify-content-between mt-3">
                    <button class="btn btn-sm btn-outline-secondary" id="btnPrevProduct"
                            ${currentPage === 0 ? 'disabled' : ''}>上一页</button>
                    <span class="align-self-center small">${currentPage + 1} / ${totalPages}</span>
                    <button class="btn btn-sm btn-outline-secondary" id="btnNextProduct"
                            ${currentPage >= totalPages - 1 ? 'disabled' : ''}>下一页</button>
                </div>
            `;
        }

        productListEl.innerHTML = html;

        // 绑定商品点击事件
        productListEl.querySelectorAll('.btn-product-item').forEach(btn => {
            btn.addEventListener('click', (e) => {
                productListEl.querySelectorAll('.btn-product-item').forEach(b =>
                    b.classList.remove('active'));
                e.currentTarget.classList.add('active');
                currentSelection.product = e.currentTarget.dataset.product;
            });
        });

        // 绑定分页按钮
        const btnPrev = document.getElementById('btnPrevProduct');
        const btnNext = document.getElementById('btnNextProduct');
        if (btnPrev) {
            btnPrev.addEventListener('click', () => {
                currentPages.product--;
                renderProductList();
            });
        }
        if (btnNext) {
            btnNext.addEventListener('click', () => {
                currentPages.product++;
                renderProductList();
            });
        }
    }

    /**
     * 渲染国家下拉框
     */
    function renderCountrySelect() {
        const countrySelect = document.getElementById('countrySelect');
        if (!countrySelect || !apiData.countries) return;

        const countries = Object.keys(apiData.countries);
        countrySelect.innerHTML = `
            <option value="">请选择国家</option>
            ${countries.map(country => `<option value="${country}">${country}</option>`).join('')}
        `;
    }

    /**
     * 渲染省份下拉框
     */
    function renderProvinceSelect() {
        const provinceSelect = document.getElementById('provinceSelect');
        if (!provinceSelect || !apiData.provinces) return;

        const provinces = Object.keys(apiData.provinces);
        provinceSelect.innerHTML = `
            <option value="">全部省份</option>
            ${provinces.map(province => `<option value="${province}">${province}</option>`).join('')}
        `;
    }

    /**
     * 渲染贸易方式下拉框
     */
    function renderTradeTypeSelect() {
        const tradeModeSelect = document.getElementById('tradeModeSelect');
        if (!tradeModeSelect || !apiData.tradeTypes) return;

        tradeModeSelect.innerHTML = `
            <option value="">请选择贸易方式</option>
            ${apiData.tradeTypes.map(type => `<option value="${type}">${type}</option>`).join('')}
        `;
    }

    /**
     * 渲染单位下拉框
     */
    function renderUnitSelect() {
        const unitSelect = document.getElementById('unitSelect');
        if (!unitSelect || !apiData.units) return;

        const units = Object.keys(apiData.units);
        unitSelect.innerHTML = units.map((unit, index) =>
            `<option value="${unit}" ${index === 0 ? 'selected' : ''}>${unit}</option>`
        ).join('');
    }

    /**
     * 绑定事件
     */
    function bindEvents() {
        // 确认商品选择按钮
        const btnConfirmProduct = document.getElementById('btnConfirmProduct');
        if (btnConfirmProduct) {
            btnConfirmProduct.addEventListener('click', () => {
                if (!currentSelection.product) {
                    alert('请先选择具体商品');
                    return;
                }

                // 更新按钮显示文本
                const productBtn = document.querySelector('[data-bs-target="#productModal"]');
                if (productBtn) {
                    productBtn.textContent = currentSelection.product;
                    productBtn.classList.remove('btn-outline-primary');
                    productBtn.classList.add('btn-success');
                }

                // 关闭模态框
                const modalEl = document.getElementById('productModal');
                const modal = bootstrap.Modal.getInstance(modalEl);
                if (modal) modal.hide();
            });
        }

        // 监听下拉框变化
        const countrySelect = document.getElementById('countrySelect');
        if (countrySelect) {
            countrySelect.addEventListener('change', (e) => {
                currentSelection.country = e.target.value;
            });
        }

        const provinceSelect = document.getElementById('provinceSelect');
        if (provinceSelect) {
            provinceSelect.addEventListener('change', (e) => {
                currentSelection.province = e.target.value;
            });
        }

        const tradeModeSelect = document.getElementById('tradeModeSelect');
        if (tradeModeSelect) {
            tradeModeSelect.addEventListener('change', (e) => {
                currentSelection.tradeType = e.target.value;
            });
        }

        const unitSelect = document.getElementById('unitSelect');
        if (unitSelect) {
            unitSelect.addEventListener('change', (e) => {
                currentSelection.unit = e.target.value;
            });
        }

        const startYearSelect = document.getElementById('startYearSelect');
        if (startYearSelect) {
            startYearSelect.addEventListener('change', (e) => {
                currentSelection.startYear = parseInt(e.target.value);
            });
        }

        const endYearSelect = document.getElementById('endYearSelect');
        if (endYearSelect) {
            endYearSelect.addEventListener('change', (e) => {
                currentSelection.endYear = parseInt(e.target.value);
            });
        }
    }

    /**
     * 获取当前选择的所有参数
     */
    function getCurrentSelection() {
        return { ...currentSelection };
    }

    /**
     * 验证参数是否完整
     */
    function validateSelection() {
        const errors = [];

        if (!currentSelection.product) {
            errors.push('请选择商品');
        }
        if (!currentSelection.country) {
            errors.push('请选择贸易国家');
        }
        if (!currentSelection.tradeType) {
            errors.push('请选择贸易方式');
        }
        if (!currentSelection.unit) {
            errors.push('请选择计价单位');
        }
        if (!currentSelection.startYear) {
            errors.push('请选择起始年份');
        }
        if (!currentSelection.endYear) {
            errors.push('请选择终止年份');
        }

        return {
            valid: errors.length === 0,
            errors: errors
        };
    }

    // 对外暴露的接口
    return {
        init,
        getCurrentSelection,
        validateSelection
    };
})();
