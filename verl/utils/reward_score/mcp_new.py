import re
from typing import List, Dict, Tuple
import json

def get_format_score(completion: str) -> float:
    """
    格式检查：
    1. 必须包含<thought>和<actions>/<chat>标签
    2. 标签内容去首尾空格后非空
    3. 支持多行内容匹配
    """
    pattern = r"<thought>(.*?)</thought>\s*<MCP>(.*?)</MCP>"
    
    match_actions = re.search(pattern, completion, re.DOTALL)
    
    if match_actions:
        thought = match_actions.group(1).strip()
        answer = match_actions.group(2).strip()
        return 1.0 if thought and answer else 0.0
    return 0.0

def extract_answer_content(text: str) -> str:
    if "<MCP>" in text and "</MCP>" in text:
        return text.split("<MCP>")[1].split("</MCP>")[0].strip()
    return ""

def check_answer(solution_str: str, ground_truth: str) -> float:
    
    # 提取并比较actions内容
    solution_actions = extract_answer_content(solution_str)
    gt_actions = ground_truth

    try:
        solution_dict = json.loads(solution_actions)
        gt_dict = json.loads(gt_actions)
    except:
        return 0
    
    return 1.0 if solution_dict == gt_dict else 0.0

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Dict = None) -> float:
    """
    综合评分：
    - 格式分40%：标签完整且内容非空
    - 工具分60%：类型匹配+内容完全匹配
    """
    if not solution_str or not ground_truth:
        return 0.0
    
    format_score = get_format_score(solution_str)
    answer_score = check_answer(solution_str, ground_truth)
    
    total = format_score - 1 + answer_score
    print(f"[REWARD SCORE] Format: {format_score:.2f}, Answer: {answer_score:.2f}, Total: {total:.2f}")
    return total