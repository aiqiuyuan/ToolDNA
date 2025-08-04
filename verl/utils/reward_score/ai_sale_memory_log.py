import re
from verl.utils.reward_score.tool_memory import ToolMemory
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
import json

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

    DEBUG_LOG_PATH = ""
    
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
    has_description = 1.0 if extract_description_content(solution_str) else 0.0

    total = format_score * 0.4 + tool_score * 0.6 + has_description
    print(f"[REWARD SCORE] Format: {format_score:.2f}, Tool: {tool_score:.2f}, Description: {has_description:.2f}, Total: {total:.2f}")
    
    try:
        # 构造日志条目
        log_entry = {
            "timestamp": datetime.now().isoformat(),  # 时间戳
            "data_source": data_source,             # 数据源标识
            "ground_truth": ground_truth,           # 真实标签
            "solution_str": solution_str,           # 模型输出
            "scores": {                              # 各维度分数
                "format_score": format_score,
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

    # 如果总分不是1.0，直接返回分数
    if total != 1.0:
        return total
    
    # 提取description内容
    description_content = extract_description_content(solution_str)
    print("description_content:",description_content )
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
