import re
from typing import List, Dict, Tuple

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

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Dict = None) -> float:
    """
    综合评分：
    - 格式分40%：标签完整且内容非空
    - 工具分60%：类型匹配+内容完全匹配
    """
    if not solution_str or not ground_truth:
        return 0.0
    
    format_score = get_format_score(solution_str)
    tool_score = get_tool_call_score(solution_str, ground_truth)
    
    total = format_score * 0.4 + tool_score * 0.6
    print(f"[REWARD SCORE] Format: {format_score:.2f}, Tool: {tool_score:.2f}, Total: {total:.2f}")
    return total

# # 测试用例（按用户思路实现）
# if __name__ == "__main__":
#     # 测试用例1：完全正确
#     test_solution1 = """
#     <thought>
#     1. 用户回复“发给我看看”...
#     </thought>
#     <actions>
#     [search_used_cars({"query":"红旗E-QM5","page_size":8})]
#     </actions>
#     """
#     test_ground_truth1 = """
#     <thought>
#     1. 用户回复“发给我看看”...
#     </thought>
#     <actions>
#     [search_used_cars({"query":"红旗E-QM5","page_size":8})]
#     </actions>
#     """
#     score1 = compute_score("/dcar", test_solution1, test_ground_truth1)
#     print(f"Test Case 1 Score: {score1:.2f}  # 预期: 1.00")
    
#     # 测试用例2：参数错误（内容不一致）
#     test_solution2 = """
#     <thought>
#     1. 用户回复“发给我看看”...
#     </thought>
#     <actions>
#     [search_used_cars({"query":"红旗E-QM5","page_size":10})]
#     </actions>
#     """
#     score2 = compute_score("/dcar", test_solution2, test_ground_truth1)
#     print(f"Test Case 2 Score: {score2:.2f}  # 预期: 0.40")
    
#     # 测试用例3：漏用工具（类型不匹配）
#     test_solution3 = """
#     <thought>
#     1. 用户回复“发给我看看”...
#     </thought>
#     <chat>
#     好的，马上为您展示...
#     </chat>
#     """
#     score3 = compute_score("/dcar", test_solution3, test_ground_truth1)
#     print(f"Test Case 3 Score: {score3:.2f}  # 预期: 0.00")
    
#     # 测试用例4：格式错误
#     test_solution4 = """
#     <thought>
#     1. 已知用户咨询蔚来EC6...
#     </thought>
#     <actions>
#     [check_policy({"registration_city":"上海"})
#     """
#     score4 = compute_score("/dcar", test_solution4, test_ground_truth1)
#     print(f"Test Case 4 Score: {score4:.2f}  # 预期: 0.00")
    
#     # 测试用例5：多用工具（类型不匹配）
#     test_ground_truth5 = """
#     <thought>
#     1. 客户表示想在上海上牌...
#     </thought>
#     <chat>
#     二手新能源汽车...
#     </chat>
#     """
#     test_solution5 = """
#     <thought>
#     1. 客户表示想在上海上牌...
#     </thought>
#     <actions>
#     [check_policy({"registration_city":"上海"})]
#     </actions>
#     """
#     score5 = compute_score("/dcar", test_solution5, test_ground_truth5)
#     print(f"Test Case 5 Score: {score5:.2f}  # 预期: 0.00")



#     test_solution6 ="<thought>\n1. 用户询问不用抢购吗和总费用多少。首先分析抢购问题，由于当前活动信息中提到小米SU7和ULTRA以及100台准新车官方补贴1万需在3月30日17点 - 3月31日24点抢购，200台包邮车源在3月28日 - 3月31日抢购，而用户意向的是理想L8 2024款Max版，未明确该车型参与上述抢购活动，所以此问题需转人工进一步确认是否需要抢购，故调用call_human工具，理由为“其他原因”。\n2. 接着分析总费用问题，用户是全款自提在深圳上牌，根据购车流程及费用信息，自提且在车源地（佛山）以外城市（深圳）上牌，需支付车款和上牌费，车款因有直降10000元优惠，实际车款为265900元，上牌费为500元，所以总费用为车款加上牌费，即265900 + 500，调用calculator工具进行计算，表达式为\"265900+500\"。\n</thought>\n<actions>\n[calculator({\"expression\":\"265900+500\"}), call_human({\"reason\": \"其他原因\"})]\n</actions>"
#     test_ground_truth_6 ="""<thought>\n1. 用户询问不用抢购吗和总费用多少。首先分析抢购问题，由于当前活动信息中提到小米SU7和ULTRA以及100台准新车官方补贴1万需在3月30日17点 - 3月31日24点抢购，200台包邮车源在3月28日 - 3月31日抢购，而用户意向的是理想L8 2024款Max版，未明确该车型参与上述抢购活动，所以此问题需转人工进一步确认是否需要抢购，故调用call_human工具，理由为“其他原因”。\n2. 接着分析总费用问题，用户是全款自提在深圳上牌，根据购车流程及费用信息，自提且在车源地（佛山）以外城市（深圳）上牌，需支付车款和上牌费，车款因有直降10000元优惠，实际车款为265900元，上牌费为500元，所以总费用为车款加上牌费，即265900 + 500，调用calculator工具进行计算，表达式为\"265900+500\"。\n</thought>\n<actions>\n[calculator({\"expression\":\"265900+500\"}), call_human({\"reason\": \"其他原因\"})]\n</actions>"""
#     score6 = compute_score("/dcar", test_solution6, test_ground_truth_6)
#     print(f"Test Case 6 Score: {score6:.2f}  # 预期: 1.00")


#     test_solution7 ="""<thought>\n1. 用户询问不用抢购吗和总费用多少。首先分析抢购问题，由于当前活动信息中提到小米SU7和ULTRA以及100台准新车官方补贴1万需在3月30日17点 - 3月31日24点抢购，200台包邮车源在3月28日 - 3月31日抢购，而用户意向的是理想L8 2024款Max版，未明确该车型参与上述抢购活动，所以此问题需转人工进一步确认是否需要抢购，故调用call_human工具，理由为“其他原因”。\n2. 接着分析总费用问题，用户是全款自提在深圳上牌，根据购车流程及费用信息，自提且在车源地（佛山）以外城市（深圳）上牌，需支付车款和上牌费，车款因有直降10000元优惠，实际车款为265900元，上牌费为500元，所以总费用为车款加上牌费，即265900 + 500，调用calculator工具进行计算，表达式为\"265900+500\"。\n</thought>\n<actions>\n[calculator({\"expression\":\"265900+500\"}), call_human({\"reason\": \"其他原因\"})]\n</actions>"""
#     test_ground_truth_7 ="""<thought>\n1. 用户询问不用抢购吗和总费用多少。首先分析抢购问题，由于当前活动信息中提到小米SU7和ULTRA以及100台准新车官方补贴1万需在3月30日17点 - 3月31日24点抢购，200台包邮车源在3月28日 - 3月31日抢购，而用户意向的是理想L8 2024款Max版，未明确该车型参与上述抢购活动，所以此问题需转人工进一步确认是否需要抢购，故调用call_human工具，理由为“其他原因”。\n2. 接着分析总费用问题，用户是全款自提在深圳上牌，根据购车流程及费用信息，自提且在车源地（佛山）以外城市（深圳）上牌，需支付车款和上牌费，车款因有直降10000元优惠，实际车款为265900元，上牌费为500元，所以总费用为车款加上牌费，即265900 + 500，调用calculator工具进行计算，表达式为\"265900+500\"。\n</thought>\n<actions>\n[calculator({\"expression\":\"265900+500\"})</actions>"""
#     score7 = compute_score("/dcar", test_solution7, test_ground_truth_7)
#     print(f"Test Case 7 Score: {score7:.2f}  # 预期: 0.4")

#     test_solution6 ="""<thought>\n1. 用户询问不用抢购吗和总费用多少。首先分析抢购问题，由于当前活动信息中提到小米SU7和ULTRA以及100台准新车官方补贴1万需在3月30日17点 - 3月31日24点抢购，200台包邮车源在3月28日 - 3月31日抢购，而用户意向的是理想L8 2024款Max版，未明确该车型参与上述抢购活动，所以此问题需转人工进一步确认是否需要抢购，故调用call_human工具，理由为“其他原因”。\n2. 接着分析总费用问题，用户是全款自提在深圳上牌，根据购车流程及费用信息，自提且在车源地（佛山）以外城市（深圳）上牌，需支付车款和上牌费，车款因有直降10000元优惠，实际车款为265900元，上牌费为500元，所以总费用为车款加上牌费，即265900 + 500，调用calculator工具进行计算，表达式为\"265900+500\"。\n</thought>\n<actions>\n[calculator({\"expression\":\"265900+500\"}), call_human({\"reason\": \"其他原因\"})]\n</actions>"""
#     test_ground_truth_6 ="""<thought>\n1. 用户询问不用抢购吗和总费用多少。首先分析抢购问题，由于当前活动信息中提到小米SU7和ULTRA以及100台准新车官方补贴1万需在3月30日17点 - 3月31日24点抢购，200台包邮车源在3月28日 - 3月31日抢购，而用户意向的是理想L8 2024款Max版，未明确该车型参与上述抢购活动，所以此问题需转人工进一步确认是否需要抢购，故调用call_human工具，理由为“其他原因”。\n2. 接着分析总费用问题，用户是全款自提在深圳上牌，根据购车流程及费用信息，自提且在车源地（佛山）以外城市（深圳）上牌，需支付车款和上牌费，车款因有直降10000元优惠，实际车款为265900元，上牌费为500元，所以总费用为车款加上牌费，即265900 + 500，调用calculator工具进行计算，表达式为\"265900+500\"。\n</thought>\n<chat>\n[calculator({\"expression\":\"265900+500\"}), call_human({\"reason\": \"其他原因\"})]\n</chat>"""
#     score6 = compute_score("/dcar", test_solution6, test_ground_truth_6)
#     print(f"Test Case 6 Score: {score6:.2f}  # 预期: 0.4")
