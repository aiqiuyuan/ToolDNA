from threading import Lock

class ToolMemory:
    _instance = None
    _lock = Lock()  # 直接初始化锁对象
    tools = None

    def __new__(cls, *args, **kwargs):
        with cls._lock:  # 现在_lock是Lock实例，支持上下文管理器
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, tools_str: str = ""):
        if self._initialized:
            return
        
        self.tools = self._parse_tools(tools_str)
        self._initialized = True
        print(f"[ToolMemory] 本地简化版初始化完成 | 工具数量: {len(self.tools)}")
    
    def _parse_tools(self, tools_str: str) -> list:
        parts = tools_str.strip().split("## ")[1:]
        return ["## " + part.strip() for part in parts]
    
    def update(self, update_str: str):
        updated_tools = self._parse_tools(update_str)
        tool_name_to_index = {self._extract_tool_name(tool): i for i, tool in enumerate(self.tools)}
        
        updated_count = 0
        for tool in updated_tools:
            tool_name = self._extract_tool_name(tool)
            if tool_name in tool_name_to_index:
                self.tools[tool_name_to_index[tool_name]] = tool
                updated_count += 1
        print(f"[TOOL UPDATE] 成功更新{updated_count}个工具描述")
    
    def _extract_tool_name(self, tool_str: str) -> str:
        import re
        pattern = r"^\s*工具名称\s*[：:]\s*([^()\s]+)"
        for line in tool_str.split('\n'):
            if "工具名称" in line:
                line_clean = re.sub(r"^[#/]+", "", line).strip()
                match = re.search(pattern, line_clean, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        return ""
    
    def get_all_tools(self) -> str:
        return "\n\n".join(self.tools)

