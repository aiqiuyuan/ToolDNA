import re
from verl.utils.reward_score.tool_memory import ToolMemory
from typing import List, Dict, Tuple, Optional, Set

str1 = """# 工具信息
## 查询二手车库存
工具名称：search_used_cars
工具描述：可以通过查询内容和筛选条件对于库存内的车辆进行筛选，并展示数条搜索结果
需要填入：
query (str)：查询的内容，可以类似“奔驰”，“x3”，“增程式”这类
filters (dict)：筛选项，具体可选的筛选项为：
    price_range (List[int])：价格区间（单位是万元），用[15, 21]表示需要筛选15万到21万的车辆
    mileage (List[int])：里程区间（单位是万公里），用[1, 6]表示需要筛选1万公里到6万公里里程的车辆
    car_age (List[int])：车龄（单位是年），用[2, 4]表示需要筛选车龄为两年以上四年以下的车辆
    energy_type (str)：能源类型，必须是["新能源", "非新能源"]之一，纯电/插混/增程均属于"新能源"
    category (List[str])：车辆级别，必须是["轿车", "SUV", "MPV", "跑车"]中的一个或多个
    emission_standard (List[str])：排放标准，必须是["国四", "国五", "国六", "国六b"]中的一个或多个
page (int)：当前页码，默认值为1
page_size (int)：每页检索条数，默认值为5，最大值为30
使用示例：
    search_used_cars({"query": "mini", "filters": {"car_age": [1, 4]}})
    search_used_cars({"query": "宝马三系", "page_size": 20})
    search_used_cars({"filters": {"energy_type": "新能源", "category": ["轿车", "跑车"]}, "page_size": 30})

## 查看二手车详情
工具名称：view_details
工具描述：查看二手车的某些方面的具体细节情况
需要填入：
    sku_ids (List[str])：车辆id的列表，可以从上下文或`search_used_cars`工具的结果中获取，禁止凭空捏造
    aspects (List[str])：方面的列表，必须是["基础信息","优势分析","车况检测","电池信息","易损件","整备清单","参保期限","保险理赔","优惠活动","选配清单"]中的一个或多个，建议一次性查询多个方面的信息（"基础信息"包括钥匙数量等信息）
使用示例：view_details({"sku_ids": ["10086"], "aspects": ["车况检测", "参保期限", "保险理赔"]})

## 查看二手车辆参配
工具名称：check_configs
工具描述：查看二手车的某些方面的官方参数配置
需要填入：
    sku_ids (List[str])：车辆id的列表，可以从上下文或`search_used_cars`工具的结果中获取，禁止凭空捏造
    aspects (List[str])：方面的列表，必须是["基本信息","车身","发动机","电动机","电池/充电","变速箱","底盘转向","车轮制动","主动安全","被动安全","辅助操控配置","外部配置","内部配置","舒适/防盗配置","座椅配置","智能互联","影音娱乐","灯光配置","玻璃后视镜","空调冰箱","智能化配置"]中的一个或多个，建议一次性查询多个方面的信息
使用示例：check_configs({"sku_ids": ["10086"], "aspects": ["基本信息", "电动机", "电池/充电"]})

## 查询板车托运费
工具名称：check_delivery_fee
工具描述：查看从车辆所在城市使用板车托运至上牌城市的费用
需要填入：
    source_city (str)：车辆所在门店的所在城市，标准化的城市名，例如"北京"、"上海"、"杭州"等，而不是"浙江杭州"、"上海市"等
    registration_city (str)：客户期望的上牌城市，标准化的城市名
使用示例：check_delivery_fee({"source_city": "武汉", "registration_city": "北京"})

## 查询在某城市上牌的迁入政策（限迁）
工具名称：check_policy
工具描述：查看需要将车辆在某城市上牌的迁入政策
需要填入：
    registration_city (str)：上牌城市，标准化的城市名
使用示例：check_policy({"registration_city": "海口"})

## 贷款计算
工具名称：calculate_loan
工具描述：计算贷款
需要填入：
    total_price (float)：二手车总价（单位万元），特指二手车价格，而不是贷款金额
    down_payment_ratio (float)：首付比例，必须是[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]中的一个，假如客户要求比例是0.25，则四舍五入为0.3，假如客户要求比例是0.08，则四舍五入为0.1
    periods (int)：贷款月限（36/48/60）
使用示例：calculate_loan({"total_price": 19.88, "down_payment_ratio": 0.3, "periods": 36})

## 计算器
工具名称：calculator
工具描述：计算器，用于做一些基础的加减乘除多步运算（当前场景下常见的是计算首期应付费用，首期应付费用=首付比例*当前车款价+上牌费+物流费）
需要填入：
    expression (str)：基本运算的表达式，例如"45000+1000+2500"、"88000*0.2+1000+1050"
使用示例：calculator({"expression": "68000*0.3+1000+2500"})

## 转人工
工具名称：call_human
工具描述：将对话交由人工客服处理
需要填入：
    reason (str)：转人工的原因，必须是["用户要求","车辆选配","视频看车","征信审核","其他原因"]中的一个，当用户有上述意图时请及时使用此工具
使用示例：call_human({"reason": "视频看车"})

## 生成二手车链接（含下订入口）
工具名称：create_car_urls
工具描述：对具体的n辆二手车库存车辆生成相关页面链接，车源详情页链接内包含车辆详情、图片等，用户可自行浏览，下订页面可以直接自助下订，检测报告页面内有详细检测内容和细节图片
需要填入：
    sku_ids (List[str])：车辆id的列表，可以从上下文或`search_used_cars`工具的结果中获取，禁止凭空捏造
    aspects (List[str])：方面的列表，必须是["车源详情页","下订页面","检测报告页面"]中的一个或多个，如果需要则一次性生成多个方面的链接
使用示例：create_car_urls({"sku_ids": ["10086"], "aspects": ["车源详情页", "检测报告页面"]})

## 懂咔咔搜索
工具名称：dcar_search
工具描述：当有汽车相关知识需要联网搜索相关资料才能回答时，请使用懂咔咔搜索来获取实时知识
需要填入：
    query (str)：完整的检索词
使用示例：
    dcar_search({"query": "17款的途观有手机app吗"})
    dcar_search({"query": "19款理想L9 max和最新款有什么区别"})
    dcar_search({"query": "su7 max型号是顶配版本吗"})

# """

def get_format_score(completion: str) -> float:
    """
    格式检查：
    1. 必须包含<thought>和<actions>/<chat>标签
    2. 标签内容去首尾空格后非空
    3. 支持多行内容匹配
    """
    pattern_chat = r"<thought>(.*?)</thought>\s*<chat>(.*?)</chat>"
    pattern_actions = r"<thought>(.*?)</thought>\s*<actions>(.*?)</actions>"
    
    match_chat = re.search(pattern_chat, completion, re.DOTALL)
    match_actions = re.search(pattern_actions, completion, re.DOTALL)
    
    if match_chat:
        thought = match_chat.group(1).strip()
        chat = match_chat.group(2).strip()
        return 1.0 if thought and chat else 0.0
    elif match_actions:
        thought = match_actions.group(1).strip()
        actions = match_actions.group(2).strip()
        return 1.0 if thought and actions else 0.0
    return 0.0

def extract_action_content(text: str) -> str:
    """提取<actions>或<chat>的内容"""
    if "<actions>" in text and "</actions>" in text:
        return text.split("<actions>")[1].split("</actions>")[0].strip()
    elif "<chat>" in text and "</chat>" in text:
        return text.split("<chat>")[1].split("</chat>")[0].strip()
    return ""

def check_tool_usage(solution_str: str, ground_truth: str) -> float:
    """
    检查工具使用是否匹配：
    1. 类型匹配（同时为actions或chat）
    2. 内容完全匹配（仅当类型为actions时）
    """
    # 判断类型是否匹配
    solution_has_actions = "<actions>" in solution_str
    gt_has_actions = "<actions>" in ground_truth
    
    if solution_has_actions != gt_has_actions:
        return 0.0  # 类型不匹配，漏用或多用工具
    
    # 类型匹配但无工具调用时视为正确
    if not solution_has_actions and not gt_has_actions:
        return 1.0
    
    # 提取并比较actions内容
    solution_actions = extract_action_content(solution_str)
    gt_actions = extract_action_content(ground_truth)
    
    return 1.0 if solution_actions == gt_actions else 0.0

def get_tool_call_score(solution_str: str, ground_truth: str) -> float:
    """工具调用评分：类型匹配+内容完全匹配"""
    return check_tool_usage(solution_str, ground_truth)

def extract_tool_info(tool_text: str) -> Dict[str, str]:
    """从工具描述文本中提取工具信息"""
    tool_info = {}
    
    # 提取工具名称
    name_match = re.search(r"工具名称：(.+)", tool_text)
    if name_match:
        tool_info["name"] = name_match.group(1).strip()
    
    # 提取工具描述
    desc_match = re.search(r"工具描述：(.+?)(\n需要填入：|$)", tool_text, re.DOTALL)
    if desc_match:
        tool_info["description"] = desc_match.group(1).strip()
    
    # 提取参数信息
    params_match = re.search(r"需要填入：(.+?)(\n使用示例：|$)", tool_text, re.DOTALL)
    if params_match:
        params_text = params_match.group(1).strip()
        param_lines = params_text.split("\n")
        params = []
        current_param = None
        current_desc = []
        
        for line in param_lines:
            line = line.strip()
            if line.startswith("    ") and current_param:
                # 参数描述的继续行
                current_desc.append(line.strip())
            else:
                # 新参数
                if current_param:
                    params.append({
                        "name": current_param,
                        "description": "\n".join(current_desc),
                        "type": current_type  # 新增参数类型信息
                    })
                param_match = re.search(r"(\w+)\s+\((.+?)\)：(.+)", line)
                if param_match:
                    current_param = param_match.group(1)
                    current_type = param_match.group(2)
                    current_desc = [param_match.group(3)]
                else:
                    current_param = None
                    current_desc = []
        
        # 添加最后一个参数
        if current_param:
            params.append({
                "name": current_param,
                "description": "\n".join(current_desc),
                "type": current_type
            })
        
        tool_info["parameters"] = params
    
    # 提取使用示例
    example_match = re.search(r"使用示例：(.+)", tool_text, re.DOTALL)
    if example_match:
        examples_text = example_match.group(1).strip()
        examples = []
        # 简单分割示例（假设每个示例占一行）
        for line in examples_text.split("\n"):
            line = line.strip()
            if line:
                examples.append(line)
        tool_info["examples"] = examples
    
    return tool_info

def compare_parameters(param_list1: List[Dict], param_list2: List[Dict]) -> bool:
    """比较两个参数列表是否基本匹配（参数名和类型必须存在，描述可以简化）"""
    # 转换为集合进行比较
    params1 = {(p["name"], p["type"]) for p in param_list1}
    params2 = {(p["name"], p["type"]) for p in param_list2}
    
    # 检查是否所有必需的参数都存在且类型匹配
    if params1 != params2:
        return False
    
    # 检查每个参数是否存在于desc中（参数名和类型必须存在）
    for param1 in param_list1:
        found = False
        for param2 in param_list2:
            if param1["name"] == param2["name"] and param1["type"] == param2["type"]:
                # 参数名和类型匹配，认为参数存在
                found = True
                break
        if not found:
            return False
    
    return True

def compare_examples(examples1: List[str], examples2: List[str]) -> bool:
    """比较使用示例是否基本匹配（只检查第一个示例的工具名和参数键）"""
    if not examples1 or not examples2:
        return False
    
    # 提取第一个示例中的工具名和参数
    def extract_example_info(example: str) -> Tuple[str, str]:
        # 简单提取工具名和参数部分
        tool_pattern = r"([a-zA-Z_]+)\("
        param_pattern = r"\((.*)\)"
        
        tool_match = re.search(tool_pattern, example)
        param_match = re.search(param_pattern, example)
        
        tool_name = tool_match.group(1) if tool_match else ""
        params = param_match.group(1) if param_match else ""
        
        return (tool_name, params)
    
    tool1, params1 = extract_example_info(examples1[0])
    tool2, params2 = extract_example_info(examples2[0])
    
    # 工具名必须相同
    if tool1 != tool2:
        return False
    
    # 参数部分只需要包含相同的键（不要求值相同）
    def extract_keys(params_str: str) -> Set[str]:
        keys = set()
        # 简单提取键（假设格式为"key": value）
        key_pattern = r'"([a-zA-Z_]+)"\s*:'
        keys.update(re.findall(key_pattern, params_str))
        return keys
    
    keys1 = extract_keys(params1)
    keys2 = extract_keys(params2)
    
    return keys1 == keys2

def compare_tool_info(tool_info1: Dict[str, str], tool_info2: Dict[str, str]) -> bool:
    """比较两个工具信息是否基本匹配（不要求完全相同）"""
    # 检查工具名称
    if tool_info1.get("name") != tool_info2.get("name"):
        return False
    
    # 检查参数
    params1 = tool_info1.get("parameters", [])
    params2 = tool_info2.get("parameters", [])
    
    if not compare_parameters(params1, params2):
        return False
    
    # 检查使用示例
    examples1 = tool_info1.get("examples", [])
    examples2 = tool_info2.get("examples", [])
    
    if not compare_examples(examples1, examples2):
        return False
    
    return True

def extract_description_content(text: str) -> str:
    """提取<description>标签中的内容"""
    pattern = r"<description>(.*?)</description>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_tool_names_from_actions(actions_text: str) -> List[str]:
    """从actions文本中提取工具名称"""
    tool_pattern = r"\[([a-zA-Z_]+)\("
    matches = re.findall(tool_pattern, actions_text)
    return matches

def extract_tool_text_from_str1(tool_name: str) -> Optional[str]:
    """从str1中提取指定工具的完整文本描述"""
    pattern = r"## (.+?)\n工具名称：" + re.escape(tool_name) + r"\n(.+?)(?=##|$)"
    match = re.search(pattern, str1, re.DOTALL)
    if match:
        return match.group(0)
    return None

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Dict = None) -> float:
    """无需显式传入memory，直接访问单例实例"""
    # 获取单例memory
    memory = ToolMemory()  # 自动获取全局唯一实例
    
    """
    综合评分：
    - 格式分40%：标签完整且内容非空
    - 工具分60%：类型匹配+内容完全匹配
    
    如果总分是1.0，并且description中的工具信息与str1中的匹配，则更新memory内容
    否则不更新，只返回分数
    """
    if not solution_str or not ground_truth:
        return 0.0
    
    format_score = get_format_score(solution_str)
    tool_score = get_tool_call_score(solution_str, ground_truth)
    
    total = format_score * 0.4 + tool_score * 0.6
    print(f"[REWARD SCORE] Format: {format_score:.2f}, Tool: {tool_score:.2f}, Total: {total:.2f}")
    
    # 如果总分不是1.0，直接返回分数
    if total != 1.0:
        return total
    
    # 提取description内容
    description_content = extract_description_content(solution_str)
    #print("description_content:",description_content )
    if not description_content:
        return total
    
    
    # 解析description中的工具块
    all_tool_blocks = [
        f"## {block.strip()}" 
        for block in description_content.strip().split("## ") 
        if block.strip()
    ]
    if not all_tool_blocks:
        return total
    
    # 准备更新的工具块（仅保留信息匹配的工具）
    valid_tool_blocks = []
    
    for block in all_tool_blocks:
        # 提取当前工具块的信息
        new_tool_info = extract_tool_info(block)
        if not new_tool_info.get("name"):
            print(f"[TOOL ERROR] 工具块缺少名称，跳过: {block[:50]}...")
            continue
        
        # 从memory中查找同名原始工具
        original_tool = None
        for tool in memory.tools:
            if new_tool_info["name"] in tool:
                original_tool = tool
                break
        
        if not original_tool:
            print(f"[TOOL ERROR] 未找到原始工具: {new_tool_info['name']}")
            continue
        
        # 提取原始工具信息
        original_tool_info = extract_tool_info(original_tool)
        
        # 比较工具信息（使用用户提供的compare_tool_info函数）
        if compare_tool_info(original_tool_info, new_tool_info):
            valid_tool_blocks.append(block)
            print(f"[TOOL INFO] 工具信息匹配: {new_tool_info['name']}")
        else:
            print(f"[TOOL WARNING] 工具信息不匹配，跳过更新: {new_tool_info['name']}")
    
    # 仅当有匹配的工具块时执行更新
    if valid_tool_blocks:
        update_str = "\n\n".join(valid_tool_blocks)
        memory.update(update_str)
        print(f"[TOOL UPDATE] 成功更新{len(valid_tool_blocks)}个工具")
    else:
        print("[TOOL UPDATE] 无匹配的工具信息，未执行更新")
    
    return total

# ...（前面的函数和之前完全一致，仅修改main测试部分）

# if __name__ == "__main__":
#     # 初始化ToolMemory（str1包含10个工具）
#     original_tools_str = str1  # 保存原始工具字符串
#     memory = ToolMemory(original_tools_str)
#     original_search_tool = None
    
#     # 提取原始search_used_cars工具描述（用于后续对比）
#     for tool in memory.tools:
#         if "工具名称：search_used_cars" in tool:
#             original_search_tool = tool
#             break
#     print("初始search_used_cars工具描述（前50字）:", original_search_tool[:50] if original_search_tool else "未找到")

#     # 测试用例1：完全正确，应更新search_used_cars的描述
#     test_solution1 = """
#     <thought>用户需要查询红旗E-QM5库存</thought>
#     <actions>[search_used_cars({"query":"红旗E-QM5","page_size":8})]</actions>
#     <description>
#     ## 查询二手车库存
#     工具名称：search_used_cars
#     工具描述：【更新后】可以通过查询内容和筛选条件对于库存内的车辆进行筛选，并展示数条搜索结果
#     需要填入：
#     query (str)：查询的内容，可以类似"奔驰"，"x3"，"增程式"这类
#     filters (dict)：筛选项，具体可选的筛选项为：
#         price_range (List[int])：价格区间（单位是万元），用[15, 21]表示需要筛选15万到21万的车辆
#         mileage (List[int])：里程区间（单位是万公里），用[1, 6]表示需要筛选1万公里到6万公里里程的车辆
#         car_age (List[int])：车龄（单位是年），用[2, 4]表示需要筛选车龄为两年以上四年以下的车辆
#         energy_type (str)：能源类型，必须是["新能源", "非新能源"]之一，纯电/插混/增程均属于"新能源"
#         category (List[str])：车辆级别，必须是["轿车", "SUV", "MPV", "跑车"]中的一个或多个
#         emission_standard (List[str])：排放标准，必须是["国四", "国五", "国六", "国六b"]中的一个或多个
#     page (int)：当前页码，默认值为1
#     page_size (int)：每页检索条数，默认值为5，最大值为30
#     使用示例：
#         search_used_cars({"query": "mini", "filters": {"car_age": [1, 4]}})
#         search_used_cars({"query": "宝马五系", "page_size": 20})
#         search_used_cars({"filters": {"energy_type": "新能源", "category": ["轿车", "跑车"]}, "page_size": 30})
#     </description>
#     """
#     test_ground_truth1 = """
#     <thought>用户需要查询红旗E-QM5库存</thought>
#     <actions>[search_used_cars({"query":"红旗E-QM5","page_size":8})]</actions>
#     """
#     score1 = compute_score("/dcar", test_solution1, test_ground_truth1)
#     print(f"Test Case 1 Score: {score1:.2f}, 预期: 1.00")

#     # 验证search_used_cars是否被更新（检查描述是否包含【更新后】关键词）
#     updated_search_tool = None
#     for tool in memory.tools:
#         if "工具名称：search_used_cars" in tool:
#             updated_search_tool = tool
#             break
#     print("更新后的search_used_cars工具描述（前50字）:", updated_search_tool[:50] if updated_search_tool else "未找到")
#     print("是否成功更新（检查是否包含【更新后】）:", "【更新后】" in (updated_search_tool or ""))

#     # 测试用例2：description中缺少工具信息，不更新
#     test_solution2 = """
#     <thought>用户需要查询红旗E-QM5库存</thought>
#     <actions>[search_used_cars({"query":"红旗E-QM5","page_size":8})]</actions>
#     <description>这是一个无效的描述</description>
#     """
#     score2 = compute_score( "/dcar", test_solution2, test_ground_truth1)
#     print(f"Test Case 2 Score: {score2:.2f}, 预期: 1.00")
#     # 验证search_used_cars描述是否保持更新后的状态（包含【更新后】）
#     recheck_tool = next(tool for tool in memory.tools if "工具名称：search_used_cars" in tool)
#     print("测试用例2后search_used_cars是否仍包含【更新后】:", "【更新后】" in recheck_tool)

#     # 测试用例3：工具信息不完整（缺少emission_standard参数），不更新
#     test_solution3 = """
#     <thought>用户需要查询红旗E-QM5库存</thought>
#     <actions>[search_used_cars({"query":"红旗E-QM5","page_size":8})]</actions>
#     <description>
#     ## 查询二手车库存
#     工具名称：search_used_cars
#     工具描述：可以通过查询内容和筛选条件对于库存内的车辆进行筛选，并展示数条搜索结果
#     需要填入：
#     query (str)：查询的内容，可以类似“奔驰”，“x3”，“增程式”这类
#     filters (dict)：筛选项，具体可选的筛选项为：
#         price_range (List[int])：价格区间（单位是万元），用[15, 21]表示需要筛选15万到21万的车辆
#         mileage (List[int])：里程区间（单位是万公里），用[1, 6]表示需要筛选1万公里到6万公里里程的车辆
#         car_age (List[int])：车龄（单位是年），用[2, 4]表示需要筛选车龄为两年以上四年以下的车辆
#         energy_type (str)：能源类型，必须是["新能源", "非新能源"]之一，纯电/插混/增程均属于"新能源"
#         category (List[str])：车辆级别，必须是["轿车", "SUV", "MPV", "跑车"]中的一个或多个
#     page (int)：当前页码，默认值为1
#     page_size (int)：每页检索条数，默认值为5，最大值为30
#     使用示例：
#         search_used_cars({"query": "mini", "filters": {"car_age": [1, 4]}})
#         search_used_cars({"query": "宝马三系", "page_size": 20})
#         search_used_cars({"filters": {"energy_type": "新能源", "category": ["轿车", "跑车"]}, "page_size": 30})
#     </description>
#     """
#     score3 = compute_score( "/dcar", test_solution3, test_ground_truth1)
#     print(f"Test Case 3 Score: {score3:.2f}, 预期: 1.00")
#     # 验证search_used_cars描述是否未被覆盖（仍包含【更新后】）
#     recheck_tool3 = next(tool for tool in memory.tools if "工具名称：search_used_cars" in tool)
#     print("测试用例3后search_used_cars是否仍包含【更新后】:", "【更新后】" in recheck_tool3)

# # 联合测试代码
# if __name__ == "__main__":
#     # 初始化ToolMemory
#     memory = ToolMemory(str1)
#     print("初始工具数量:", len(memory.tools))
    
#     # 测试用例1：完全正确，应更新工具描述
#     test_solution1 = """
#     <thought>
#     1. 用户回复“发给我看看”...
#     </thought>
#     <actions>
#     [search_used_cars({"query":"红旗E-QM5","page_size":8})]
#     </actions>
#     <description>
#     ## 查询二手车库存
#     工具名称：search_used_cars
#     工具描述：可以通过查询内容和筛选条件对于库存内的车辆进行筛选，并展示数条搜索结果
#     需要填入：
#     query (str)：查询的内容，可以类似“奔驰”，“x3”，“增程式”这类
#     filters (dict)：筛选项，具体可选的筛选项为：
#         price_range (List[int])：价格区间（单位是万元），用[15, 21]表示需要筛选15万到21万的车辆
#         mileage (List[int])：里程区间（单位是万公里），用[1, 6]表示需要筛选1万公里到6万公里里程的车辆
#         car_age (List[int])：车龄（单位是年），用[2, 4]表示需要筛选车龄为两年以上四年以下的车辆
#         energy_type (str)：能源类型，必须是["新能源", "非新能源"]之一，纯电/插混/增程均属于"新能源"
#         category (List[str])：车辆级别，必须是["轿车", "SUV", "MPV", "跑车"]中的一个或多个
#         emission_standard (List[str])：排放标准，必须是["国四", "国五", "国六", "国六b"]中的一个或多个
#     page (int)：当前页码，默认值为1
#     page_size (int)：每页检索条数，默认值为5，最大值为30
#     使用示例：
#         search_used_cars({"query": "mini", "filters": {"car_age": [1, 4]}})
#         search_used_cars({"query": "宝马三系", "page_size": 20})
#         search_used_cars({"filters": {"energy_type": "新能源", "category": ["轿车", "跑车"]}, "page_size": 30})

#     </description>
#     """
#     test_ground_truth1 = """
#     <thought>
#     1. 用户回复“发给我看看”...
#     </thought>
#     <actions>
#     [search_used_cars({"query":"红旗E-QM5","page_size":8})]
#     </actions>
#     """
#     score1 = compute_score(memory, "/dcar", test_solution1, test_ground_truth1)
#     print(f"Test Case 1 Score: {score1:.2f}, 预期: 1.00")
#     print("更新后工具数量:", len(memory.tools))
    
#     # 验证工具是否更新（查看search_used_cars工具是否存在）
#     search_tool_found = False
#     for tool in memory.tools:
#         if "search_used_cars" in tool:
#             search_tool_found = True
#             break
#     print("search_used_cars工具存在:", search_tool_found)
    
#     # 测试用例2：description中缺少工具信息，不应更新
#     test_solution2 = """
#     <thought>
#     1. 用户回复“发给我看看”...
#     </thought>
#     <actions>
#     [search_used_cars({"query":"红旗E-QM5"":8})]
#     </actions>
#     <description>
#     这是一个测试描述
#     </description>
#     """
#     score2 = compute_score(memory, "/dcar", test_solution2, test_ground_truth1)
#     print(f"Test Case 2 Score: {score2:.2f}, 预期: 1.00")
#     print("工具数量应保持不变:", len(memory.tools))
    
#     # 测试用例3：工具信息不完整，不应更新
#     test_solution3 = """
#     <thought>
#     1. 用户回复“发给我看看”...
#     </thought>
#     <actions>
#     [search_used_cars({"query":"红旗E-QM5","page_size":8})]
#     </actions>
#     <description>
#     ## 查询二手车库存
#     工具名称：search_used_cars
#     工具描述：可以通过查询内容和筛选条件对于库存内的车辆进行筛选，并展示数条搜索结果
#     需要填入：
#     query (str)：查询的内容，可以类似“奔驰”，“x3”，“增程式”这类
#     filters (dict)：筛选项，具体可选的筛选项为：
#         price_range (List[int])：价格区间（单位是万元），用[15, 21]表示需要筛选15万到21万的车辆
#         mileage (List[int])：里程区间（单位是万公里），用[1, 6]表示需要筛选1万公里到6万公里里程的车辆
#         car_age (List[int])：车龄（单位是年），用[2, 4]表示需要筛选车龄为两年以上四年以下的车辆
#     page (int)：当前页码，默认值为1
#     page_size (int)：每页检索条数，默认值为5，最大值为30
#     使用示例：
#         search_used_cars({"query": "mini", "filters": {"car_age": [1, 4]}})
#         search_used_cars({"query": "宝马三系", "page_size": 20})
#         search_used_cars({"filters": {"energy_type": "新能源", "category": ["轿车", "跑车"]}, "page_size": 30})

#     </description>
#     """
#     score3 = compute_score(memory, "/dcar", test_solution3, test_ground_truth1)
#     print(f"Test Case 3 Score: {score3:.2f}, 预期: 1.00")
#     print("工具数量应保持不变:", len(memory.tools))