import json
import re
import os
import logging
import subprocess
from typing import Dict, List, Tuple, Optional, Set
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
#from verl.utils.reward_score.simple_memory import ToolMemory
from verl.utils.reward_score.tool_memory import ToolMemory
from datetime import datetime
# 配置日志
# 打印关键环境变量
print("工具模块环境变量 - CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "未设置"))
print("工具模块环境变量 - LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "未设置"))
print("工具模块环境变量 - PATH:", os.environ.get("PATH", "未设置"))

# 打印PyTorch和CUDA状态
print("工具模块PyTorch版本:", torch.__version__)
print("工具模块CUDA版本:", torch.version.cuda)
print("工具模块CUDA可用:", torch.cuda.is_available())
print("工具模块可用GPU数量:", torch.cuda.device_count())
# 配置日志（程序启动时执行一次）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='tool_update.log',  # 日志文件路径
    filemode='a'  # 追加模式（避免覆盖历史日志）
)
logger = logging.getLogger(__name__)

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# logger.info(f"设置CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def log_update_str(update_str: str):
    """直接写入文件记录更新内容（带时间戳和分隔符）"""
    try:
        # 确保日志目录存在
        DEBUG_PATH = "/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/tool_update_0720.log"
        log_dir = os.path.dirname(DEBUG_PATH)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 生成时间戳（格式：2025-06-17 15:30:00）
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 构建日志内容（带分隔符和时间戳）
        log_content = f"\n===== {timestamp} =====\n{update_str}\n{'=' * 50}\n"
        
        # 追加写入文件（使用with open确保文件正确关闭）
        with open(DEBUG_PATH, "a", encoding="utf-8") as f:
            f.write(log_content)
        
        print(f"更新内容已记录至: {DEBUG_PATH}")
        return True
    except Exception as e:
        print(f"日志写入失败: {str(e)}")
        return False

class ModelSingleton:
    """模型单例类，确保模型仅加载一次"""
    _instance = None
    _model = None
    _tokenizer = None
    _generator = None

    def __new__(cls, model_path: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model_path = model_path
            cls._instance._load_model()  # 首次实例化时加载模型
        return cls._instance

    def _load_model(self):
        """加载模型、分词器和生成管道（仅执行一次）"""
        logger.info(f"加载模型到内存: {self._model_path}（8卡自动分配）")
        try:
            # 加载模型（多卡自动分配+半精度）
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 加载分词器（使用非fast模式避免并行问题）
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            # 创建生成管道（复用模型和分词器）
            self._generator = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=2048,
                temperature=0.2,
                top_p=0.9
            )
            
            logger.info("模型加载完成，设备分布: {}".format(self._model.hf_device_map))
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

    def generate_batch(self, prompts: list, batch_size: int = 8) -> list:
        """批量生成响应（利用Dataset并行处理）"""
        if not self._generator:
            return []
        
        try:
            # 1. 构建Dataset（高效数据结构）
            dataset = Dataset.from_dict({"text": prompts})
            
            # 2. 批量生成（自动分批次处理）
            results = []
            for i in range(0, len(prompts), batch_size):
                batch_prompts = dataset[i:i+batch_size]["text"]
                
                # 3. 使用pipeline批量生成（关键优化点）
                batch_outputs = self._generator(
                    batch_prompts,
                    num_return_sequences=1,
                    return_full_text=False
                )
                
                # 4. 解析结果
                for output in batch_outputs:
                    text = output["generated_text"]
                    # 移除prompt前缀（假设所有prompt都相同或需单独处理）
                    if text.startswith(prompts[len(results)]):
                        text = text[len(prompts[len(results)]):].strip()
                    results.append(text)
            
            return results
        except Exception as e:
            logger.error(f"批量生成失败: {str(e)}")
            return []

    def generate(self, prompt: str) -> Optional[str]:
        """使用单例模型生成响应"""
        if not self._generator:
            logger.error("模型未加载，无法生成响应")
            return None
        
        try:
            response = self._generator(
                prompt,
                num_return_sequences=1,
                return_full_text=False
            )[0]["generated_text"]
            
            # 移除prompt前缀（生成的文本可能包含输入prompt）
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            logger.debug(f"模型响应: {response[:200]}...")
            return response
        except Exception as e:
            logger.error(f"生成失败: {str(e)}")
            return None

# 全局单例实例（必须全局定义）
_model_singleton = None

str1 = """\n\n# 工具信息\n## 查询二手车库存\n工具名称：search_used_cars\n工具描述：可以通过查询内容和筛选条件对于库存内的车辆进行筛选，并展示数条搜索结果\n需要填入：\nquery (str)：查询的内容，可以类似“奔驰”，“x3”，“增程式”这类\nfilters (dict)：筛选项，具体可选的筛选项为：\n    price_range (List[int])：价格区间（单位是万元），用[15, 21]表示需要筛选15万到21万的车辆\n    mileage (List[int])：里程区间（单位是万公里），用[1, 6]表示需要筛选1万公里到6万公里里程的车辆\n    car_age (List[int])：车龄（单位是年），用[2, 4]表示需要筛选车龄为两年以上四年以下的车辆\n    energy_type (str)：能源类型，必须是[\"新能源\", \"非新能源\"]之一，纯电/插混/增程均属于\"新能源\"\n    category (List[str])：车辆级别，必须是[\"轿车\", \"SUV\", \"MPV\", \"跑车\"]中的一个或多个\n    emission_standard (List[str])：排放标准，必须是[\"国四\", \"国五\", \"国六\", \"国六b\"]中的一个或多个\npage (int)：当前页码，默认值为1\npage_size (int)：每页检索条数，默认值为5，最大值为30\n使用示例：\n    search_used_cars({\"query\": \"mini\", \"filters\": {\"car_age\": [1, 4]}})\n    search_used_cars({\"query\": \"宝马三系\", \"page_size\": 20})\n    search_used_cars({\"filters\": {\"energy_type\": \"新能源\", \"category\": [\"轿车\", \"跑车\"]}, \"page_size\": 30})\n\n## 查看二手车详情\n工具名称：view_details\n工具描述：查看二手车的某些方面的具体细节情况\n需要填入：\n    sku_ids (List[str])：车辆id的列表，可以从上下文或`search_used_cars`工具的结果中获取，禁止凭空捏造\n    aspects (List[str])：方面的列表，必须是[\"基础信息\",\"优势分析\",\"车况检测\",\"电池信息\",\"易损件\",\"整备清单\",\"参保期限\",\"保险理赔\",\"优惠活动\",\"选配清单\"]中的一个或多个，建议一次性查询多个方面的信息（\"基础信息\"包括钥匙数量等信息）\n使用示例：view_details({\"sku_ids\": [\"10086\"], \"aspects\": [\"车况检测\", \"参保期限\", \"保险理赔\"]})\n\n## 查看二手车辆参配\n工具名称：check_configs\n工具描述：查看二手车的某些方面的官方参数配置\n需要填入：\n    sku_ids (List[str])：车辆id的列表，可以从上下文或`search_used_cars`工具的结果中获取，禁止凭空捏造\n    aspects (List[str])：方面的列表，必须是[\"基本信息\",\"车身\",\"发动机\",\"电动机\",\"电池/充电\",\"变速箱\",\"底盘转向\",\"车轮制动\",\"主动安全\",\"被动安全\",\"辅助操控配置\",\"外部配置\",\"内部配置\",\"舒适/防盗配置\",\"座椅配置\",\"智能互联\",\"影音娱乐\",\"灯光配置\",\"玻璃后视镜\",\"空调冰箱\",\"智能化配置\"]中的一个或多个，建议一次性查询多个方面的信息\n使用示例：check_configs({\"sku_ids\": [\"10086\"], \"aspects\": [\"基本信息\", \"电动机\", \"电池/充电\"]})\n\n## 查询板车托运费\n工具名称：check_delivery_fee\n工具描述：查看从车辆所在城市使用板车托运至上牌城市的费用\n需要填入：\n    source_city (str)：车辆所在门店的所在城市，标准化的城市名，例如\"北京\"、\"上海\"、\"杭州\"等，而不是\"浙江杭州\"、\"上海市\"等\n    registration_city (str)：客户期望的上牌城市，标准化的城市名\n使用示例：check_delivery_fee({\"source_city\": \"武汉\", \"registration_city\": \"北京\"})\n\n## 查询在某城市上牌的迁入政策（限迁）\n工具名称：check_policy\n工具描述：查看需要将车辆在某城市上牌的迁入政策\n需要填入：\n    registration_city (str)：上牌城市，标准化的城市名\n使用示例：check_policy({\"registration_city\": \"海口\"})\n\n## 贷款计算\n工具名称：calculate_loan\n工具描述：计算贷款\n需要填入：\n    total_price (float)：二手车总价（单位万元），特指二手车价格，而不是贷款金额\n    down_payment_ratio (float)：首付比例，必须是[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]中的一个，假如客户要求比例是0.25，则四舍五入为0.3，假如客户要求比例是0.08，则四舍五入为0.1\n    periods (int)：贷款月限（36/48/60）\n使用示例：calculate_loan({\"total_price\": 19.88, \"down_payment_ratio\": 0.3, \"periods\": 36})\n\n## 计算器\n工具名称：calculator\n工具描述：计算器，用于做一些基础的加减乘除多步运算（当前场景下常见的是计算首期应付费用，首期应付费用=首付比例*当前车款价+上牌费+物流费）\n需要填入：\n    expression (str)：基本运算的表达式，例如\"45000+1000+2500\"、\"88000*0.2+1000+1050\"\n使用示例：calculator({\"expression\": \"68000*0.3+1000+2500\"})\n\n## 转人工\n工具名称：call_human\n工具描述：将对话交由人工客服处理\n需要填入：\n    reason (str)：转人工的原因，必须是[\"用户要求\",\"车辆选配\",\"视频看车\",\"征信审核\",\"其他原因\"]中的一个，当用户有上述意图时请及时使用此工具\n使用示例：call_human({\"reason\": \"视频看车\"})\n\n## 生成二手车链接（含下订入口）\n工具名称：create_car_urls\n工具描述：对具体的n辆二手车库存车辆生成相关页面链接，车源详情页链接内包含车辆详情、图片等，用户可自行浏览，下订页面可以直接自助下订，检测报告页面内有详细检测内容和细节图片\n需要填入：\n    sku_ids (List[str])：车辆id的列表，可以从上下文或`search_used_cars`工具的结果中获取，禁止凭空捏造\n    aspects (List[str])：方面的列表，必须是[\"车源详情页\",\"下订页面\",\"检测报告页面\"]中的一个或多个，如果需要则一次性生成多个方面的链接\n使用示例：create_car_urls({\"sku_ids\": [\"10086\"], \"aspects\": [\"车源详情页\", \"检测报告页面\"]})\n\n## 懂咔咔搜索\n工具名称：dcar_search\n工具描述：当有汽车相关知识需要联网搜索相关资料才能回答时，请使用懂咔咔搜索来获取实时知识\n需要填入：\n    query (str)：完整的检索词\n使用示例：\n    dcar_search({\"query\": \"17款的途观有手机app吗\"})\n    dcar_search({\"query\": \"19款理想L9 max和最新款有什么区别\"})\n    dcar_search({\"query\": \"su7 max型号是顶配版本吗\"})\n\n"""


memory = ToolMemory(str1)
print(memory)  # 初始化时传入工具字符串

def extract_tool_info(tool_text: str) -> Dict[str, str]:
    """从工具描述文本中提取工具信息"""
    tool_info = {}
    
    name_match = re.search(r"工具名称：(.+)", tool_text)
    if name_match:
        tool_info["name"] = name_match.group(1).strip()
    
    desc_match = re.search(r"工具描述：(.+?)(\n需要填入：|$)", tool_text, re.DOTALL)
    if desc_match:
        tool_info["description"] = desc_match.group(1).strip()
    
    params_match = re.search(r"需要填入：(.+?)(\n使用示例：|$)", tool_text, re.DOTALL)
    if params_match:
        params_text = params_match.group(1).strip()
        param_lines = params_text.split("\n")
        params = []
        current_param = None
        current_desc = []
        current_type = ""
        
        for line in param_lines:
            line = line.strip()
            if line.startswith("    ") and current_param:
                current_desc.append(line.strip())
            else:
                if current_param:
                    params.append({
                        "name": current_param,
                        "description": "\n".join(current_desc),
                        "type": current_type
                    })
                param_match = re.search(r"(\w+)\s+\((.+?)\)：(.+)", line)
                if param_match:
                    current_param = param_match.group(1)
                    current_type = param_match.group(2)
                    current_desc = [param_match.group(3)]
                else:
                    current_param = None
                    current_desc = []
        
        if current_param:
            params.append({
                "name": current_param,
                "description": "\n".join(current_desc),
                "type": current_type
            })
        
        tool_info["parameters"] = params
    
    example_match = re.search(r"使用示例：(.+)", tool_text, re.DOTALL)
    if example_match:
        examples_text = example_match.group(1).strip()
        examples = []
        for line in examples_text.split("\n"):
            line = line.strip()
            if line:
                examples.append(line)
        tool_info["examples"] = examples
    
    return tool_info

def compare_parameters(param_list1: List[Dict], param_list2: List[Dict]) -> bool:
    params1 = {(p["name"], p["type"]) for p in param_list1}
    params2 = {(p["name"], p["type"]) for p in param_list2}
    if params1 != params2:
        return False
    for param1 in param_list1:
        found = False
        for param2 in param_list2:
            if param1["name"] == param2["name"] and param1["type"] == param2["type"]:
                found = True
                break
        if not found:
            return False
    return True

def compare_examples(examples1: List[str], examples2: List[str]) -> bool:
    if not examples1 or not examples2:
        return False
    
    def extract_example_info(example: str) -> Tuple[str, str]:
        tool_pattern = r"([a-zA-Z_]+)\("
        param_pattern = r"\((.*)\)"
        tool_match = re.search(tool_pattern, example)
        param_match = re.search(param_pattern, example)
        tool_name = tool_match.group(1) if tool_match else ""
        params = param_match.group(1) if param_match else ""
        return (tool_name, params)
    
    tool1, params1 = extract_example_info(examples1[0])
    tool2, params2 = extract_example_info(examples2[0])
    if tool1 != tool2:
        return False
    
    def extract_keys(params_str: str) -> Set[str]:
        keys = set()
        # 去除转义符（将\"替换为"）
        clean_params = params_str.replace('\\"', '"')
        # 原始正则匹配
        key_pattern = r'"([a-zA-Z_]+)"\s*:'
        keys.update(re.findall(key_pattern, clean_params))
        return keys
        
    keys1 = extract_keys(params1)  # 原始示例的参数键集合（如{"reason"}）
    keys2 = extract_keys(params2)  # 新示例的参数键集合（如{"reason"}）
    return keys1 == keys2  # 参数键必须完全一致（数量和名称）

def compare_tool_info(tool_info1: Dict[str, str], tool_info2: Dict[str, str]) -> bool:
    """比较两个工具信息是否基本匹配（不要求完全相同）"""
    # 检查工具名称
    if tool_info1.get("name") != tool_info2.get("name"):
        print("名称不匹配")
        return False
    
    # 检查参数
    params1 = tool_info1.get("parameters", [])
    params2 = tool_info2.get("parameters", [])
    
    if not compare_parameters(params1, params2):
        print("参数不匹配")
        return False
    
    # 检查使用示例
    examples1 = tool_info1.get("examples", [])
    examples2 = tool_info2.get("examples", [])
    
    if not compare_examples(examples1, examples2):
        print("示例不匹配")
        return False
    
    return True

def read_jsonl_file(file_path: str) -> List[Dict]:
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.error(f"第{line_num}行JSON解析失败: {line.strip()[:50]}...")
        logger.info(f"成功读取{len(data)}条数据")
        return data
    except Exception as e:
        logger.error(f"读取文件出错: {str(e)}")
        return []

def extract_description_content(text: str) -> str:
    pattern = r"<description>(.*?)</description>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_tool_name_from_actions(actions_str: str) -> Optional[str]:
    """从<actions>标签中提取工具名称"""
    if not actions_str:
        return None
    
    # 匹配[工具名({...})]格式
    match = re.search(r"\[(\w+) *\({", actions_str)
    if match:
        return match.group(1)
    return None


def build_prompt(ground_truth: str, solution_str: str, original_tool_desc: str) -> str:
    prompt = f"""你需要扮演工具描述优化专家，根据提供的ground_truth和solution_str分析工具描述缺陷并生成结构化修改方案。请严格按以下要求输出：
### 注意点（必须保持不变的内容）：
1. 工具参数的类型（如List[str]、int）和必填要求必须与原始定义一致
2. 工具名称必须与原始名称完全一致，不得修改
3. 参数名称（如sku_ids、aspects）必须保持不变，不得重命名
4. **使用示例的第一条必须保持原状**，后面可以新增，但是需要换行
5. 参数描述中原本可以填入的字段不要删除。
6. 不要只根据当前的一条案例给出过于细节的要求，影响到对其他任务的处理，而是要根据工具的使用场景和案例，给出全面的指导。

### 输出格式要求：
<description>
## 工具名称：[工具名]
工具描述：[修改后的工具描述，需包含调用场景、避免错误的提示，用**加粗关键规则**]
需要填入：
    [参数1名称] ([类型])：[参数说明，补充错误案例中的缺失约束]
    [参数2名称] ([类型])：[参数说明，修正模糊点]
使用示例：[包含错误修复逻辑的调用示例，参数值需符合业务规则]
</description>

### 数据输入：
{{"ground_truth": "{ground_truth.replace('"', '\\"').replace('\n', '\\n')}", 
"solution_str": "{solution_str.replace('"', '\\"').replace('\n', '\\n')}",
"original_tool_desc": "{original_tool_desc.replace('"', '\\"').replace('\n', '\\n')}"}}

### 输出示例：
<description>
## 查看二手车详情
工具名称：view_details
工具描述：查看二手车的具体细节情况，**当用户询问事故历史时必须调用此工具，禁止转人工**
需要填入：
    sku_ids (List[str])：车辆id列表，需从search_used_cars结果中获取，**禁止虚构**
    aspects (List[str])：必须包含["车况检测"]以获取事故记录，**新能源车型需强制包含电池信息**
使用示例：view_details({{"sku_ids": ["18749633"], "aspects": ["车况检测", "电池信息"]}})
</description>
"""
    #print(prompt)
    return prompt

def call_local_large_model(prompt: str, model_path: str) -> Optional[str]:
    """调用本地大模型（单例模式，避免重复加载）"""
    global _model_singleton
    
    # 首次调用时初始化单例（后续调用直接复用）
    if not _model_singleton:
        try:
            _model_singleton = ModelSingleton(model_path)
            logger.info("模型单例初始化完成")
        except Exception as e:
            logger.error(f"单例初始化失败: {str(e)}")
            return None
    
    # 使用单例生成响应
    return _model_singleton.generate(prompt)

def get_tool_from_memory(tool_name: str) -> Optional[str]:
    """从ToolMemory中获取指定工具描述"""
    all_tools = memory.get_all_tools()
    tool_blocks = all_tools.split("## ")
    
    for block in tool_blocks:
        if block.strip() and tool_name in block:
            return "## " + block.strip()
    
    return None

def process_tool_update(llm_response: str) -> bool:
    
    if not llm_response:
        logger.warning("无模型响应，跳过更新")
        return False
    
    description_content = extract_description_content(llm_response)
    if not description_content:
        logger.warning("未找到<description>标签内容")
        return False
    
    all_tool_blocks = [
        f"## {block.strip()}" 
        for block in description_content.strip().split("## ") 
        if block.strip()
    ]
    if not all_tool_blocks:
        logger.warning("未解析到工具块")
        return False
    
    valid_tool_blocks = []
    for block in all_tool_blocks:
        new_tool_info = extract_tool_info(block)
        tool_name = new_tool_info.get("name")
        if not tool_name:
            logger.error(f"工具块缺少名称，跳过: {block[:50]}...")
            continue
        
        # 从memory获取原始工具（适配ToolMemory接口）
        original_tool = get_tool_from_memory(tool_name)
        if not original_tool:
            logger.error(f"未找到原始工具: {tool_name}")
            continue
        
        original_tool_info = extract_tool_info(original_tool)
        
        if compare_tool_info(original_tool_info, new_tool_info):
            valid_tool_blocks.append(block)
            logger.info(f"工具信息匹配，准备更新: {tool_name}")
        else:
            logger.warning(f"工具信息不匹配，跳过更新: {tool_name}")
    
    if valid_tool_blocks:
        update_str = "\n\n".join(valid_tool_blocks)
        memory.update(update_str)
        # ------------------- 新增：文件写入日志 -------------------
        log_update_str(update_str)  # 调用文件写入函数
        # -------------------------------------------------------
        logger.info(f"成功更新{len(valid_tool_blocks)}个工具")
        return True
    else:
        logger.info("无匹配工具块，未执行更新")
        return False


def process_data(file_path: str, model_path: str) -> None:
    data_list = read_jsonl_file(file_path)
    if not data_list:
        logger.warning("无数据需要处理")
        return

     # 从后往前取最后的320条数据
    start_idx = max(0, len(data_list) - 320)
    selected_data_list = data_list[start_idx:]
    
    for idx, data in enumerate(selected_data_list):
        logger.info(f"处理第 {start_idx+idx+1}/{len(data_list)} 条数据")
        logger.debug(f"原始数据字段: {list(data.keys())}")
        
        ground_truth = data.get("ground_truth", "")
        solution_str = data.get("solution_str", "")
        
        if not all([ground_truth, solution_str]):
            logger.warning("ground_truth或solution_str缺失，跳过处理")
            continue
        
        # 从ground_truth中提取工具名称
        ground_truth_actions = re.search(r"<actions>(.*?)</actions>", ground_truth, re.DOTALL)
        ground_truth_actions = ground_truth_actions.group(1).strip() if ground_truth_actions else ""
        tool_name = extract_tool_name_from_actions(ground_truth_actions)
        if not tool_name:
            logger.warning("未提取到工具名称，跳过处理")
            continue
        
        logger.info(f"提取到工具名称: {tool_name}")
        
        # 从memory中获取原始工具描述
        original_tool = get_tool_from_memory(tool_name)
        if not original_tool:
            logger.warning(f"memory中未找到工具: {tool_name}")
            continue
        
        logger.debug(f"原始工具描述长度: {len(original_tool)}")
        
        # 构建提示词
        prompt = build_prompt(ground_truth_actions, solution_str, original_tool)
        logger.debug(f"提示词长度: {len(prompt)}")
        
        # 调用模型（测试时返回模拟响应）
        llm_response = call_local_large_model(prompt, model_path)
        if not llm_response:
            logger.warning("模型响应为空，跳过更新")
            continue
        
        # 处理工具更新
        process_tool_update(llm_response)

# def main():
#     file_path = "/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/debug_records_0616.jsonl"
#     model_path = "/mnt/bn/motor-nlp-team/models/LLM/base_models/DeepSeek-R1-Distill-Qwen-32B"
    
#     # 初始化Ray（ToolMemory需要）
#     # if not ray.is_initialized():
#     #     ray.init(address="auto")  # 自动连接到Ray集群，或使用ray.init()本地模式
    
#     if not os.path.exists(file_path):
#         logger.error(f"数据文件不存在: {file_path}")
#         return
    
#     logger.info("开始工具优化流程...")
#     process_data(file_path, model_path)
#     logger.info("工具优化流程完成")

#     current_tool_descs_str = memory.get_all_tools()
#     print("\n\n所有工具信息：")
#     print(current_tool_descs_str)

# if __name__ == "__main__":
#     main()