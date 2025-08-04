import re
#from verl.utils.reward_score.tool_memory import ToolMemory
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
import json
import ast

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
    """提取<actions>或<chat>的内容并去除所有空格"""
    if "<actions>" in text and "</actions>" in text:
        content = text.split("<actions>")[1].split("</actions>")[0].strip()
        return re.sub(r'\s+', '', content)
    elif "<chat>" in text and "</chat>" in text:
        content = text.split("<chat>")[1].split("</chat>")[0].strip()
        return re.sub(r'\s+', '', content)
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


def parse_actions(actions_text: str) -> Tuple[List[Dict], int]:
    """
    解析actions文本，提取工具名称和参数，支持嵌套JSON，保留格式分数
    改进版：正确处理嵌套的 []，避免错误匹配
    """
    # 移除XML标签和换行符
    clean_text = re.sub(r'<[^>]*>', '', actions_text).replace('\n', '')
    
    # 查找外层的 [ 和 ]，计算括号深度来确定正确的范围
    bracket_depth = 0
    start_index = -1
    end_index = -1
    
    for i, char in enumerate(clean_text):
        if char == '[':
            if bracket_depth == 0:
                start_index = i
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1
            if bracket_depth == 0 and start_index != -1:
                end_index = i
                break
    
    tool_list_str = ""
    if start_index != -1 and end_index != -1:
        tool_list_str = clean_text[start_index + 1:end_index].strip()
    else:
        print(f"[ERROR] 未找到有效的工具调用格式: {clean_text}")
        return [], 0
    
    parsed_actions = []
    tool_format_score = 0
    all_params_valid = True
    
    # 改进的工具调用匹配模式，支持嵌套结构
    # 使用更复杂的逻辑来处理嵌套的 {} 和 ()
    current_pos = 0
    tool_list_length = len(tool_list_str)
    
    while current_pos < tool_list_length:
        # 查找工具名称的开始位置
        while current_pos < tool_list_length and tool_list_str[current_pos].isspace():
            current_pos += 1
            
        # 查找工具名称
        tool_name_start = current_pos
        while current_pos < tool_list_length and (tool_list_str[current_pos].isalnum() or tool_list_str[current_pos] == '_'):
            current_pos += 1
        tool_name_end = current_pos
            
        if tool_name_start >= tool_name_end:
            # 没有找到有效的工具名称
            current_pos += 1
            continue
            
        tool_name = tool_list_str[tool_name_start:tool_name_end].strip()
        
        # 查找参数部分
        if current_pos >= tool_list_length or tool_list_str[current_pos] != '(':
            # 没有参数部分
            parsed_actions.append({
                "name": tool_name,
                "params": {}
            })
            continue
            
        # 跳过 '('
        current_pos += 1
        
        # 查找参数结束位置，处理嵌套的 {} 和 ()
        param_depth = 1  # 当前深度，'(' 为 1
        param_start = current_pos
        while current_pos < tool_list_length and param_depth > 0:
            if tool_list_str[current_pos] == '(':
                param_depth += 1
            elif tool_list_str[current_pos] == ')':
                param_depth -= 1
            current_pos += 1
                
        # 提取参数字符串
        param_str = tool_list_str[param_start:current_pos-1].strip()
        
        # 尝试解析参数
        params = {}
        try:
            # 直接使用 json.loads 解析参数
            params = json.loads(param_str)
        except json.JSONDecodeError:
            print(f"[WARN] json.loads解析失败，尝试ast.literal_eval: {param_str}")
            try:
                params = ast.literal_eval(param_str)
                if not isinstance(params, dict):
                    raise ValueError("解析结果不是字典")
            except (ValueError, SyntaxError) as e:
                print(f"[ERROR] 参数解析失败: {param_str}, 错误: {str(e)}")
                params = {}
                all_params_valid = False
        
        # 确保params是字典
        if not isinstance(params, dict):
            print(f"[WARN] params不是字典类型: {params}")
            params = {}
            all_params_valid = False
        
        parsed_actions.append({
            "name": tool_name,
            "params": params
        })
        
        # 跳过可能的逗号
        while current_pos < tool_list_length and tool_list_str[current_pos].isspace():
            current_pos += 1
        if current_pos < tool_list_length and tool_list_str[current_pos] == ',':
            current_pos += 1
            
    # 更新格式分数
    tool_format_score = 1 if all_params_valid else 0
    
    return parsed_actions, tool_format_score


# 新增函数：比较工具名称
def compare_tool_names(solution_actions: List[Dict], gt_actions: List[Dict]) -> float:
    """比较工具名称（只要存在匹配的工具即算成功）"""
    try:
        solution_tools = {action["name"] for action in solution_actions}
        gt_tools = {action["name"] for action in gt_actions}
      
        
        # 检查是否有交集
        if solution_tools == gt_tools:
            return 1.0
        elif solution_tools & gt_tools:
            return 0.5
        return 0.0
    except Exception as e:
        print(f"[compare_tool_names ERROR] {str(e)}")
        return 0.0

# 新增函数：比较参数名称
def is_fuzzy_match(solution_val: str, gt_val: str) -> bool:
    """实现字符级模糊匹配：只要参考值的任意字符出现在解决方案值中"""
    if not isinstance(solution_val, str) or not isinstance(gt_val, str):
        return solution_val == gt_val
    
    # 统一转小写处理，忽略大小写
    sol_lower = solution_val.lower()
    gt_lower = gt_val.lower()
    
    # 检查参考值中的任意字符是否存在于解决方案值中
    return any(char in sol_lower for char in gt_lower)

def compare_param_names(solution_actions: List[Dict], gt_actions: List[Dict]) -> float:
    """比较参数名称，search_used_cars仅检查query参数是否存在"""
    try:
        # 检查每个参考答案中的工具调用是否能在生成结果中找到匹配
        all_matched = True
        all_perfect = True

        for gt_action in gt_actions:
            gt_name = gt_action["name"]
            gt_params = gt_action["params"]
            
            # 标记是否找到匹配的工具调用
            found_match = False
            perfect_match = False
            
            for sol_action in solution_actions:
                sol_name = sol_action["name"]
                sol_params = sol_action["params"]
                
                # 工具名称不匹配，跳过
                if sol_name != gt_name:
                    continue
                
                # 处理search_used_cars特殊逻辑
                if gt_name == "search_used_cars":
                    # 只要存在query参数即算匹配
                    if "query" in sol_params:
                        found_match = True
                        if set(gt_params.keys()) == set(sol_params.keys()):      
                            perfect_match = True
                        break
                else:
                    # 其他工具要求参数名称完全一致
                    if set(gt_params.keys()) == set(sol_params.keys()):
                        found_match = True
                        perfect_match = True
                        break
            
            # 如果没有找到匹配，返回0分
            if not found_match:
                all_matched = False
                all_perfect = False
            if not perfect_match:
                all_perfect = False
        
        if all_perfect:
            return 1.0
        elif all_matched:
            return 0.5
        return 0.0

    except Exception as e:
        print(f"[compare_param_names ERROR] {str(e)}")
        return 0.0

def compare_param_values(solution_actions: List[Dict], gt_actions: List[Dict]) -> float:
    """比较参数值，支持部分匹配逻辑"""
    try:
        has_fuzzy_match = False
        for gt_action in gt_actions:
            gt_name = gt_action["name"]
            gt_params = gt_action["params"]

            found_match = False
            
            for sol_action in solution_actions:
                sol_name = sol_action["name"]
                sol_params = sol_action["params"]
                
                if sol_name != gt_name:
                    continue
                
                # 处理view_details和check_configs的aspects参数
                if gt_name in ["view_details", "check_configs"] and "aspects" in gt_params:
                    gt_aspects = gt_params["aspects"]
                    sol_aspects = sol_params.get("aspects", [])
                    
                    # 检查生成结果是否包含参考答案的所有aspects
                    if set(gt_aspects) == (set(sol_aspects)):
                        found_match = True
                        break
                    elif set(gt_aspects).issubset(set(sol_aspects)):
                        found_match = True
                        has_fuzzy_match = True
                
                # 处理search_used_cars的query参数
                elif gt_name == "search_used_cars" and "query" in gt_params:
                    sol_query = sol_params.get("query", "")
                    gt_query = gt_params["query"]

                    if sol_query == gt_query:
                        found_match = True
                        break
                    elif is_fuzzy_match(sol_query, gt_query):
                        found_match = True
                        has_fuzzy_match = True
                
                # 处理其他工具的严格匹配
                else:
                    if sol_params == gt_params:
                        found_match = True
                        break
            if not found_match:
                return 0.0
        
        return 0.5 if has_fuzzy_match else 1.0

    except Exception as e:
        print(f"[compare_param_values ERROR] {str(e)}")
        return 0.0

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Dict = None) -> float:
    """无需显式传入memory，直接访问单例实例"""
    # 获取单例memory
    #memory = ToolMemory()  # 自动获取全局唯一实例

    DEBUG_LOG_PATH = ""
    SCORE_LOG_PATH = ""
    
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

    solution_actions, solution_tool_format_score = parse_actions(extract_action_content(solution_str))
    gt_actions, gt_tool_format_score = parse_actions(extract_action_content(ground_truth))

    tool_name_score = compare_tool_names(solution_actions, gt_actions)
    param_name_score = compare_param_names(solution_actions, gt_actions)
    param_value_score = compare_param_values(solution_actions, gt_actions)

    has_description = 1.0 if extract_description_content(solution_str) else 0.0

    total = (format_score - 1) + (solution_tool_format_score - 1) + tool_score * 0.25 + tool_name_score * 0.25 + param_name_score * 0.25 + param_value_score * 0.25
    print(f"[REWARD SCORE] Format: {format_score:.2f}, Tool Format: {solution_tool_format_score:.2f}, Tool Name: {tool_name_score:.2f}, Param Name: {param_name_score:.2f}, Param Value: {param_value_score:.2f}, Tool Call: {tool_score:.2f},  Description: {has_description:.2f}, Total: {total:.2f}")
    if (format_score == 1.0 and solution_tool_format_score == 1.0 and tool_name_score > 0.0 and param_name_score > 0.0 and param_value_score > 0.0):
        resolution = 1.0
    else:
        resolution = 0.0
    total_new = resolution
    
    try:
        # 构造日志条目
        log_entry = {
            "timestamp": datetime.now().isoformat(),  # 时间戳
            "data_source": data_source,             # 数据源标识
            "ground_truth": ground_truth,           # 真实标签
            "solution_str": solution_str,           # 模型输出
            "scores": {                              # 各维度分数
                "format_score": format_score,
                "tool_format_score": solution_tool_format_score,
                "tool_name_score": tool_name_score,
                "param_name_score": param_name_score,
                "param_value_score": param_value_score,
                "tool_score": tool_score,
                "has_description": has_description,
                "total": total
            }
        }

        # 写入JSON Lines文件（每行一个独立JSON对象）
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")  # 换行分隔不同条目

        print(f"[DEBUG LOG] 已记录调试数据到 {DEBUG_LOG_PATH}")
    except Exception as e:
        print(f"[DEBUG LOG ERROR] 日志记录失败: {str(e)}")

    try:
        # 构造日志条目
        score_entry = {                           
                "format_score": format_score,
                "tool_format_score": solution_tool_format_score,
                "tool_name_score": tool_name_score,
                "param_name_score": param_name_score,
                "param_value_score": param_value_score,
                "tool_score": tool_score,
                "has_description": has_description,
                "total": total
            }

        # 写入JSON Lines文件（每行一个独立JSON对象）
        with open(SCORE_LOG_PATH, "a", encoding="utf-8") as f:
            json.dump(score_entry, f, ensure_ascii=False)
            f.write("\n")  # 换行分隔不同条目

        print(f"[DEBUG LOG] 已记录调试数据到 {SCORE_LOG_PATH}")
    except Exception as e:
        print(f"[DEBUG LOG ERROR] 日志记录失败: {str(e)}")

    # #如果总分不是1.0，直接返回分数
    # if total != 1.0:
    #     return total
    
    # # 提取description内容
    # description_content = extract_description_content(solution_str)
    # print("description_content:",description_content )
    # if not description_content:
    #     return total
    
    
    # # 解析description中的工具块
    # all_tool_blocks = [
    #     f"## {block.strip()}" 
    #     for block in description_content.strip().split("## ") 
    #     if block.strip()
    # ]
    # if not all_tool_blocks:
    #     return total
    
    # # 准备更新的工具块（仅保留信息匹配的工具）
    # valid_tool_blocks = []
    
    # for block in all_tool_blocks:
    #     # 提取当前工具块的信息
    #     new_tool_info = extract_tool_info(block)
    #     if not new_tool_info.get("name"):
    #         print(f"[TOOL ERROR] 工具块缺少名称，跳过: {block[:50]}...")
    #         continue
        
    #     # 从memory中查找同名原始工具
    #     original_tool = None
    #     for tool in memory.tools:
    #         if new_tool_info["name"] in tool:
    #             original_tool = tool
    #             break
        
    #     if not original_tool:
    #         print(f"[TOOL ERROR] 未找到原始工具: {new_tool_info['name']}")
    #         continue
        
    #     # 提取原始工具信息
    #     original_tool_info = extract_tool_info(original_tool)
        
    #     # 比较工具信息（使用用户提供的compare_tool_info函数）
    #     if compare_tool_info(original_tool_info, new_tool_info):
    #         valid_tool_blocks.append(block)
    #         print(f"[TOOL INFO] 工具信息匹配: {new_tool_info['name']}")
    #     else:
    #         print(f"[TOOL WARNING] 工具信息不匹配，跳过更新: {new_tool_info['name']}")
    
    # # 仅当有匹配的工具块时执行更新
    # if valid_tool_blocks:
    #     update_str = "\n\n".join(valid_tool_blocks)
    #     memory.update(update_str)
    #     print(f"[TOOL UPDATE] 成功更新{len(valid_tool_blocks)}个工具")
    # else:
    #     print("[TOOL UPDATE] 无匹配的工具信息，未执行更新")
    
    return total_new


###测试用例（按用户思路实现）
# if __name__ == "__main__":
#     # 测试用例1：完全正确
#     test_solution1 = """
#     <thought>
#     1. 用户回复“发给我看看”...
#     </thought>  
#     <actions>\n[search_used_cars({"query": "奔驰C60","filters":{"category":["轿车"],"price_range":[17,3]}})]\n</actions>
#     """
#     test_ground_truth1 = """
#     <thought>
#     1. 用户回复“发给我看看”...
#     </thought>

#     <actions>\n[search_used_cars({"query":"奔驰C260","filters":{"category":["轿车"],"price_range":[17,23]}})]\n</actions>
#     """
#     score1 = compute_score("/dcar", test_solution1, test_ground_truth1)
#     print(f"Test Case 1 Score: {score1:.2f}  # 预期: 1.00")


    #例子：[search_used_cars({"query":"奔驰C","filters":{"category":["轿车"],"price_range":[17,23]}})]
    #例子：[search_used_cars({"query": "奔驰C260", "filters": {"price_range": [20, 21], "category": ["轿车"], "registration_city": "深圳"}}), search_used_cars({"query": "奔驰C200", "filters": {"price_range": [20, 21], "category": ["轿车"], "registration_city": "深圳"}})]
    
