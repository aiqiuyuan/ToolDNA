#tool_memory.py
str1 = """\n\n# 工具信息\n## 查询二手车库存\n工具名称：search_used_cars\n工具描述：可以通过查询内容和筛选条件对于库存内的车辆进行筛选，并展示数条搜索结果\n需要填入：\nquery (str)：查询的内容，可以类似“奔驰”，“x3”，“增程式”这类\nfilters (dict)：筛选项，具体可选的筛选项为：\n    price_range (List[int])：价格区间（单位是万元），用[15, 21]表示需要筛选15万到21万的车辆\n    mileage (List[int])：里程区间（单位是万公里），用[1, 6]表示需要筛选1万公里到6万公里里程的车辆\n    car_age (List[int])：车龄（单位是年），用[2, 4]表示需要筛选车龄为两年以上四年以下的车辆\n    energy_type (str)：能源类型，必须是[\"新能源\", \"非新能源\"]之一，纯电/插混/增程均属于\"新能源\"\n    category (List[str])：车辆级别，必须是[\"轿车\", \"SUV\", \"MPV\", \"跑车\"]中的一个或多个\n    emission_standard (List[str])：排放标准，必须是[\"国四\", \"国五\", \"国六\", \"国六b\"]中的一个或多个\npage (int)：当前页码，默认值为1\npage_size (int)：每页检索条数，默认值为5，最大值为30\n使用示例：\n    search_used_cars({\"query\": \"mini\", \"filters\": {\"car_age\": [1, 4]}})\n    search_used_cars({\"query\": \"宝马三系\", \"page_size\": 20})\n    search_used_cars({\"filters\": {\"energy_type\": \"新能源\", \"category\": [\"轿车\", \"跑车\"]}, \"page_size\": 30})\n\n## 查看二手车详情\n工具名称：view_details\n工具描述：查看二手车的某些方面的具体细节情况\n需要填入：\n    sku_ids (List[str])：车辆id的列表，可以从上下文或`search_used_cars`工具的结果中获取，禁止凭空捏造\n    aspects (List[str])：方面的列表，必须是[\"基础信息\",\"优势分析\",\"车况检测\",\"电池信息\",\"易损件\",\"整备清单\",\"参保期限\",\"保险理赔\",\"优惠活动\",\"选配清单\"]中的一个或多个，建议一次性查询多个方面的信息（\"基础信息\"包括钥匙数量等信息）\n使用示例：view_details({\"sku_ids\": [\"10086\"], \"aspects\": [\"车况检测\", \"参保期限\", \"保险理赔\"]})\n\n## 查看二手车辆参配\n工具名称：check_configs\n工具描述：查看二手车的某些方面的官方参数配置\n需要填入：\n    sku_ids (List[str])：车辆id的列表，可以从上下文或`search_used_cars`工具的结果中获取，禁止凭空捏造\n    aspects (List[str])：方面的列表，必须是[\"基本信息\",\"车身\",\"发动机\",\"电动机\",\"电池/充电\",\"变速箱\",\"底盘转向\",\"车轮制动\",\"主动安全\",\"被动安全\",\"辅助操控配置\",\"外部配置\",\"内部配置\",\"舒适/防盗配置\",\"座椅配置\",\"智能互联\",\"影音娱乐\",\"灯光配置\",\"玻璃后视镜\",\"空调冰箱\",\"智能化配置\"]中的一个或多个，建议一次性查询多个方面的信息\n使用示例：check_configs({\"sku_ids\": [\"10086\"], \"aspects\": [\"基本信息\", \"电动机\", \"电池/充电\"]})\n\n## 查询板车托运费\n工具名称：check_delivery_fee\n工具描述：查看从车辆所在城市使用板车托运至上牌城市的费用\n需要填入：\n    source_city (str)：车辆所在门店的所在城市，标准化的城市名，例如\"北京\"、\"上海\"、\"杭州\"等，而不是\"浙江杭州\"、\"上海市\"等\n    registration_city (str)：客户期望的上牌城市，标准化的城市名\n使用示例：check_delivery_fee({\"source_city\": \"武汉\", \"registration_city\": \"北京\"})\n\n## 查询在某城市上牌的迁入政策（限迁）\n工具名称：check_policy\n工具描述：查看需要将车辆在某城市上牌的迁入政策\n需要填入：\n    registration_city (str)：上牌城市，标准化的城市名\n使用示例：check_policy({\"registration_city\": \"海口\"})\n\n## 贷款计算\n工具名称：calculate_loan\n工具描述：计算贷款\n需要填入：\n    total_price (float)：二手车总价（单位万元），特指二手车价格，而不是贷款金额\n    down_payment_ratio (float)：首付比例，必须是[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]中的一个，假如客户要求比例是0.25，则四舍五入为0.3，假如客户要求比例是0.08，则四舍五入为0.1\n    periods (int)：贷款月限（36/48/60）\n使用示例：calculate_loan({\"total_price\": 19.88, \"down_payment_ratio\": 0.3, \"periods\": 36})\n\n## 计算器\n工具名称：calculator\n工具描述：计算器，用于做一些基础的加减乘除多步运算（当前场景下常见的是计算首期应付费用，首期应付费用=首付比例*当前车款价+上牌费+物流费）\n需要填入：\n    expression (str)：基本运算的表达式，例如\"45000+1000+2500\"、\"88000*0.2+1000+1050\"\n使用示例：calculator({\"expression\": \"68000*0.3+1000+2500\"})\n\n## 转人工\n工具名称：call_human\n工具描述：将对话交由人工客服处理\n需要填入：\n    reason (str)：转人工的原因，必须是[\"用户要求\",\"车辆选配\",\"视频看车\",\"征信审核\",\"其他原因\"]中的一个，当用户有上述意图时请及时使用此工具\n使用示例：call_human({\"reason\": \"视频看车\"})\n\n## 生成二手车链接（含下订入口）\n工具名称：create_car_urls\n工具描述：对具体的n辆二手车库存车辆生成相关页面链接，车源详情页链接内包含车辆详情、图片等，用户可自行浏览，下订页面可以直接自助下订，检测报告页面内有详细检测内容和细节图片\n需要填入：\n    sku_ids (List[str])：车辆id的列表，可以从上下文或`search_used_cars`工具的结果中获取，禁止凭空捏造\n    aspects (List[str])：方面的列表，必须是[\"车源详情页\",\"下订页面\",\"检测报告页面\"]中的一个或多个，如果需要则一次性生成多个方面的链接\n使用示例：create_car_urls({\"sku_ids\": [\"10086\"], \"aspects\": [\"车源详情页\", \"检测报告页面\"]})\n\n## 懂咔咔搜索\n工具名称：dcar_search\n工具描述：当有汽车相关知识需要联网搜索相关资料才能回答时，请使用懂咔咔搜索来获取实时知识\n需要填入：\n    query (str)：完整的检索词\n使用示例：\n    dcar_search({\"query\": \"17款的途观有手机app吗\"})\n    dcar_search({\"query\": \"19款理想L9 max和最新款有什么区别\"})\n    dcar_search({\"query\": \"su7 max型号是顶配版本吗\"})\n\n"""
# tool_memory.py
import ray
import os
from threading import Lock
import time

@ray.remote
class StateManager:
    def __init__(self):
        self.initialized = False
        self.tools_ref = None
    
    def set_initialized(self, tools):
        self.initialized = True
        self.tools_ref = ray.put(tools)
        return True
    
    def is_initialized(self):
        return self.initialized
    
    def get_tools_ref(self):
        return self.tools_ref

class ToolMemory:
    _instance = None
    _lock = Lock()
    _state_manager = None  # Ray Actor句柄

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                # 首次创建实例时打印标志（单例实例创建）
                print("[ToolMemory] 单例实例创建（全局唯一实例）")
                # 初始化实例属性（确保后续逻辑可用）
                cls._instance._initialized = False  # 初始化实例属性
        return cls._instance

    def __init__(self, tools_str: str = ""):
        # 检查实例是否已初始化（此时 _initialized 已存在）
        if self._initialized:
            return

        if ray.is_initialized():
            # 首次初始化时创建StateManager Actor（仅主节点执行）
            if self._state_manager is None:
                self._state_manager = StateManager.remote()
            
            current_node_id = ray.get_runtime_context().get_node_id()
            all_nodes = ray.nodes()
            main_node_id = all_nodes[0]["NodeID"] if all_nodes else None

            if current_node_id == main_node_id:
                # 主节点：初始化工具并设置Actor状态
                self.tools = self._parse_tools(tools_str)
                ray.get(self._state_manager.set_initialized.remote(self.tools))
                self._initialized = True  # 标记已初始化（实例属性）
                print(f"[ToolMemory] 主节点初始化完成 | 工具数量: {len(self.tools)}")
            else:
                # 从节点：通过Actor获取状态
                self._wait_for_main_node_initialization()
                print(f"[ToolMemory] 从节点同步完成 | 工具数量: {len(self.tools)}")
        else:
            # 本地环境直接初始化
            self.tools = self._parse_tools(tools_str)
            self._initialized = True  # 标记已初始化（实例属性）
            print(f"[ToolMemory] 本地环境初始化完成 | 工具数量: {len(self.tools)}")

    def _wait_for_main_node_initialization(self):
        while not self._initialized:
            try:
                # 通过Actor检查初始化状态
                initialized = ray.get(self._state_manager.is_initialized.remote())
                if initialized:
                    # 通过Actor获取工具数据引用
                    tools_ref = ray.get(self._state_manager.get_tools_ref.remote())
                    self.tools = ray.get(tools_ref)
                    self._initialized = True  # 标记已初始化（实例属性）
                else:
                    time.sleep(1)
            except ray.exceptions.GetTimeoutError:
                time.sleep(1)
    
    def _parse_tools(self, tools_str: str) -> list:
        """
        解析工具字符串，分割成10个独立工具描述
        :param tools_str: 包含所有工具描述的字符串
        :return: 包含10个工具描述的列表
        """
        # 以"## "作为分割点来分离各个工具
        parts = tools_str.strip().split("## ")[1:]
        # 每个部分前添加"## "还原原始格式
        return ["## " + part.strip() for part in parts]
    
    def update(self, update_str: str):
        """
        更新工具描述
        :param update_str: 包含更新工具描述的字符串
        """
        # 解析更新字符串中的工具
        updated_tools = self._parse_tools(update_str)
        #print("updated_tools:",updated_tools)
        
        # 创建工具名称到索引的映射
        tool_name_to_index = {}
        for i, tool in enumerate(self.tools):
            tool_name = self._extract_tool_name(tool)
            #print("tool_name:",tool_name)
            tool_name_to_index[tool_name] = i
        
        # 更新现有工具
        updated_count = 0
        for tool in updated_tools:
            #print("tool:",tool)
            tool_name = self._extract_tool_name(tool)
            #print("tool_name",tool_name)
            if tool_name in tool_name_to_index:
                self.tools[tool_name_to_index[tool_name]] = tool
                updated_count += 1
            else:
                print(f"未找到匹配的工具{tool_name}")
        print(f"[TOOL UPDATE] 成功更新{updated_count}个工具描述")
    
    def _extract_tool_name(self, tool_str: str) -> str:
        import re
        # 正则说明：
        # - 忽略行首的注释符号（如#、//）或空格
        # - 匹配"工具名称"后接任意空白+冒号（全角/半角）
        # - 提取冒号后的内容（直到行尾或遇到括号/空格）
        pattern = r"^\s*[#/]*\s*工具名称\s*[：:]\s*([^()\s]+)"
        for line in tool_str.split('\n'):
            # 只检查包含"工具名称"的行
            if "工具名称" not in line:
                continue
            # 去除行首注释符号（如#、//）和前后空格
            line_clean = re.sub(r"^[#/]+", "", line).strip()
            match = re.search(pattern, line_clean, re.IGNORECASE)
            if match:
                return match.group(1).strip()  # 提取名称（忽略括号和空格）
        return ""
    
    def get_all_tools(self) -> str:
        """
        获取当前所有工具描述的完整字符串
        :return: 包含所有工具描述的字符串
        """
        return "\n\n".join(self.tools)



# # 测试模块
# import unittest

# class TestToolMemory(unittest.TestCase):
#     def setUp(self):
#         # 初始化内存
#         self.memory = ToolMemory(str1)
        
#         # 准备更新内容
#         self.update_str1 = """## 查询二手车库存
# 工具名称：search_used_cars
# 工具描述：可以通过查询内容和筛选条件对于库存内的车辆进行筛选，并展示数条搜索结果
# 需要填入：
# query (str)：查询的内容，可以类似"奔驰"，"x3"，"增程式"这类
# filters (dict)：筛选项，具体可选的筛选项为：
#     price_range (List[int])：价格区间（单位是万元），用[15, 21]表示需要筛选15万到21万的车辆
#     mileage (List[int])：里程区间（单位是万公里），用[1, 6]表示需要筛选1万公里到6万公里里程的车辆
#     car_age (List[int])：车龄（单位是年），用[2, 4]表示需要筛选车龄为两年以上四年以下的车辆
#     energy_type (str)：能源类型，必须是["新能源", "非新能源"]之一，纯电/插混/增程均属于"新能源"
#     category (List[str])：车辆级别，必须是["轿车", "SUV", "MPV", "跑车"]中的一个或多个
#     emission_standard (List[str])：排放标准，必须是["国四", "国五", "国六", "国六b"]中的一个或多个
# page (int)：当前页码，默认值为1
# page_size (int)：每页检索条数，默认值为5，最大值为30
# 使用示例：
#     search_used_cars({"query": "mini", "filters": {"car_age": [1, 4]}})
#     search_used_cars({"query": "宝马五系", "page_size": 20})
#     search_used_cars({"filters": {"energy_type": "新能源", "category": ["轿车", "跑车"]}, "page_size": 30})
# """
        
#         self.update_str2 = """## 贷款计算
# 工具名称：calculate_loan
# 工具描述：计算贷款（更新版本）
# 需要填入：
#     total_price (float)：二手车总价（单位万元）
#     down_payment_ratio (float)：首付比例，范围0.1-0.6
#     periods (int)：贷款月限（12/24/36/48/60）
# 使用示例：calculate_loan({"total_price": 25.5, "down_payment_ratio": 0.2, "periods": 36})
# """

#     def test_initialization(self):
#         """测试初始化是否正确"""
#         tools = self.memory.tools
#         self.assertEqual(len(tools), 10)
#         self.assertIn("search_used_cars", tools[0])
#         self.assertIn("view_details", tools[1])
#         self.assertIn("check_configs", tools[2])
#         self.assertIn("check_delivery_fee", tools[3])
#         self.assertIn("check_policy", tools[4])
#         self.assertIn("calculate_loan", tools[5])
#         self.assertIn("calculator", tools[6])
#         self.assertIn("call_human", tools[7])
#         self.assertIn("create_car_urls", tools[8])
#         self.assertIn("dcar_search", tools[9])

#     def test_single_update(self):
#         """测试单个工具更新"""
#         # 更新前
#         original_tool = self.memory.tools[0]
        
#         # 执行更新
#         self.memory.update(self.update_str1)
        
#         # 验证更新
#         updated_tool = self.memory.tools[0]
#         self.assertNotEqual(original_tool, updated_tool)
#         self.assertIn("宝马五系", updated_tool)
#         self.assertNotIn("宝马三系", updated_tool)
        
#         # 验证其他工具未变化
#         self.assertIn("view_details", self.memory.tools[1])

#     def test_multi_update(self):
#         """测试多个工具更新"""
#         # 组合两个更新
#         multi_update = self.update_str1 + "\n\n" + self.update_str2
        
#         # 执行更新
#         self.memory.update(multi_update)
        
#         # 验证第一个工具更新
#         self.assertIn("宝马五系", self.memory.tools[0])
#         # 验证第五个工具更新
#         self.assertIn("更新版本", self.memory.tools[5])
#         # 验证其他工具未变化
#         self.assertIn("view_details", self.memory.tools[1])

#     def test_get_all_tools(self):
#         """测试获取所有工具"""
#         all_tools = self.memory.get_all_tools()
#         self.assertIsInstance(all_tools, str)
#         self.assertEqual(all_tools.count("## "), 10)
#         self.assertIn("search_used_cars", all_tools)
#         self.assertIn("dcar_search", all_tools)

# # 运行测试
# if __name__ == "__main__":
#     unittest.main()