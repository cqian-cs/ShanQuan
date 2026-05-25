# SPDX-FileCopyrightText: 2026-present cqian-cs <cqian.cs@qq.com>
#
# SPDX-License-Identifier: MIT
__version__ = "1.1.0"

import asyncio
import os
import time
import random
import traceback
import copy
from typing import Dict, Any, List, Optional, AsyncIterator,AsyncGenerator, NotRequired, TypedDict, Literal, Callable, Union
import concurrent.futures
import aiohttp
import orjson
import ast
import inspect

class FunctionCall(TypedDict):
    name: str
    arguments: str          # JSON string

class ToolCall(TypedDict):
    id: str
    index: int
    type: Literal['function']
    function: FunctionCall

class TaskData(TypedDict):
    key: str                # 任务ID
    f: str                  # 函数名
    kwargs: dict            # 函数参数

class MultimodalContentAttachment(TypedDict):
    url: str

class MultimodalContent(TypedDict):
    type: Literal['text','image_url','video_url','file_url']
    text: NotRequired[str]
    image_url: NotRequired[MultimodalContentAttachment]
    video_url: NotRequired[MultimodalContentAttachment]
    file_url: NotRequired[MultimodalContentAttachment]


class Message(TypedDict):
    role: Literal['system','user','assistant','tool']
    content: Union[str,List[MultimodalContent]]
    tool_calls: NotRequired[list[ToolCall]] # role='assistant' 且调用工具时
    thinking_content: NotRequired[str]      # role='assistant' 且有思考时
    tool_call_id: NotRequired[str]         # role='tool' 时

class LLMCustomBodyThinking(TypedDict):
    type: Literal['enabled', 'disabled']

class LLMCustomBodyResponseFormat(TypedDict):
    type: Literal['text', 'json_object']

class LLMCustomBody(TypedDict):
    thinking: LLMCustomBodyThinking # OpenAI格式
    enable_thinking: bool           # 硅基流动格式
    response_format: LLMCustomBodyResponseFormat


class LLMData(TypedDict):
    message: Message
    id: str
    model: str
    created: int
    finish_reason: Literal['stop', 'tool_calls', 'length', 'sensitive', '']
    completion_tokens: int
    total_tokens: int
    cached_tokens: int

class ActData(TypedDict):
    all_success: bool
    delta_messages: list[Message]

class AgentData(TypedDict):
    messages: list[Message]
    finish_reason: Literal['stop', 'tool_calls', 'length', 'sensitive', '']
    completion_tokens: int
    total_tokens: int
    cached_tokens: int

class _Result(TypedDict):
    suc: bool               # 任务是否成功

class TaskResult(_Result,TaskData):
    data: Any               # 任务函数返回值，或报错信息

class LLMResult(_Result, total=False):
    data: LLMData           # suc=True 时
    msg: str                # suc=False 时

class ActResult(_Result, total=False):
    data: ActData
    msg: str

class AgentResult(_Result, total=False):
    data: AgentData
    msg: str

class FetchResult(_Result, total=False):
    data: Any
    msg: str

#-----------------------------------------------------------#
# 名空间工具（用于导入导出python函数）
#-----------------------------------------------------------#
# ns_json = ns_dumps(func1,func2,...) 把func1,func2,...打包成JSON
# ns = ns_loads(ns_json) 导入JSON，ns["函数名"]就是原python函数
# ns["func1"](*args,**kwargs) 调用func1
# ns["func2"](*args,**kwargs) 调用func2
#-----------------------------------------------------------#
def ns_dumps(*items):
    """导出函数到名空间（json字符串）
    可以自动导入函数调用的全局变量和全局函数，但函数内不可以有全局模块，模块要在函数内import。
    Returns:
        dict: namespace
    """    
    ns_info = {
        'functions': {},
        'globals': {}
    }
    def add_function(item):
        source = inspect.getsource(item)
        tree = ast.parse(source)
        function_def = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        
        function_name = function_def.name
        if function_name in ns_info['functions']:
            return
        args = []
        defaults = []
        for arg in function_def.args.args:
            args.append(arg.arg)
        for default in function_def.args.defaults:
            defaults.append(ast.literal_eval(default))
        function_body = ast.unparse(function_def.body)
        ns_info['functions'][function_name] = {
            'name': function_name,
            'args': args,
            'defaults': defaults,
            'body': function_body,
        }
        # 收集函数中使用的全局变量
        global_vars = {}
        for name, value in item.__globals__.items():
            if name in function_body and not name.startswith('__'):
                if callable(value):
                    add_function(value)
                elif not inspect.ismodule(value):
                    global_vars[name] = value
        ns_info['globals'].update(global_vars)
    for item in items:
        if callable(item):
            add_function(item)
        elif isinstance(item, tuple) and len(item) == 2:
            name, value = item
            if not inspect.ismodule(value):
                ns_info['globals'][name] = value
    return orjson.dumps(ns_info)

def ns_loads(json_str):
    """导入名空间（json字符串）
    Returns:
        dict: namespace
    """
    ns_info = orjson.loads(json_str)
    namespace = ns_info['globals'].copy()
    for func_name, func_data in ns_info['functions'].items():
        params = []
        for i, arg in enumerate(func_data['args']):
            if i >= len(func_data['args']) - len(func_data['defaults']):
                default_value = repr(func_data['defaults'][i - (len(func_data['args']) - len(func_data['defaults']))])
                params.append(f"{arg}={default_value}")
            else:
                params.append(arg)
        param_str = ", ".join(params)
        func_code = f"def {func_data['name']}({param_str}):\n"
        body_lines = func_data['body'].strip().split('\n')
        func_code += '\n'.join(f"    {line}" for line in body_lines)
        exec(func_code, namespace, namespace)
    return namespace
#-----------------------------------------------------------#



#-----------------------------------------------------------#
# 单机异步进程池工具（以异步的方式，批量调用同步函数）
#-----------------------------------------------------------#
# async def main():
#    cpool = ProcessExecutor( # 创建进程池
#        [func1,func2,...],   # 进程池允许的函数
#        n_workers=5          # 子进程数量
#    )
#    ret = await cpool.submit(
#        key='task1',          # 任务ID
#        func_name='func1',    # 函数名
#        kwargs={'a':1,'b':2}  # 函数参数
#    )
#-----------------------------------------------------------#
# 返回格式：
#     ret['suc']   (bool): 是否没有报错，正常运行
#     ret['data']   (any): 执行结果，或者报错提示
#     ret['key']    (str): 任务ID
#     ret['f']      (str): 函数名
#-----------------------------------------------------------#
_global_ns = None
def _init_in_process(ns_json):
    global _global_ns
    _global_ns = ns_loads(ns_json)
    
def _run_in_process(key, func_name, kwargs):
    try:
        f = _global_ns[func_name]
        ret = f(**kwargs)
        return {'suc': True, 'data': ret, 'key': key, 'f': func_name}
    except Exception as e:
        return {'suc': False, 'data': traceback.format_exc(), 'key': key, 'f': func_name}

class ProcessExecutor:
    def __init__(self, funcs, n_workers=None):
        if n_workers is None:
            n_workers = max(os.cpu_count()//2,1)
        self._pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_in_process,
            initargs=(ns_dumps(*funcs),)
        )
    

    async def submit(self, key, func_name, kwargs):
        cfuture = self._pool.submit(_run_in_process, key, func_name, kwargs)
        afuture = asyncio.wrap_future(cfuture, loop=asyncio.get_running_loop())
        try:
            return await afuture
        except asyncio.CancelledError:
            if not cfuture.done():
                cfuture.cancel()
            raise
    def __del__(self, wait=True):
        self._pool.shutdown(wait=wait)

#-----------------------------------------------------------#



#-----------------------------------------------------------#
# 单机异步“协程-进程”池工具（同步函数→多进程; 异步函数→多协程）
#-----------------------------------------------------------#
# 注册API函数: @api(qps=None, limit=None)
# 初始化池配置: init_pool(n_workers=1, global_limit=5000) 
# 批量调用（中途退出则自动取消未完成的任务）：
# async with Batch([
#     {'f': str（API名称）, 'kwargs': dict（API参数）}  # <- payload
# ]) as futures:
#     async for ret in futures:
#         ret['suc']     (bool) 是否成功
#         ret['data']    (any)  结果
#         ret['f']       (str)  调用函数
#         ret['kwargs']  (dict) 调用参数
#         ret[...]       payload中的其他字段
# 批量调用: batch(key_payloads)
# 取消调用: abort(func_name, key_prefix)
#-----------------------------------------------------------#
# 示例代码:
#-----------------------------------------------------------#
# @api(qps=None, limit=3)
# def test_task(x): # <-- 注册一个同步函数（由进程池执行）
#     import time
#     time.sleep(1)
#     return x * x
# 
# @api(qps=None, limit=3)
# async def fast_task(name): # <-- 注册一个异步函数（由协程池执行）
#     if name == 1:
#         raise ValueError("Simulated Error in Fast Task")
#     print(f"[{time.strftime('%H:%M:%S')}] 🚀 Start Fast {name}")
#     await asyncio.sleep(2)
#     print(f"[{time.strftime('%H:%M:%S')}] ✅ End Fast {name}")
#     return f'Fast-{name}'
# 
# @api(qps=3, limit=None)
# async def slow_task(name):  # <-- 注册一个异步函数（由协程池执行）
#     print(f"[{time.strftime('%H:%M:%S')}] 🐌 Start Slow {name}")
#     await asyncio.sleep(2) 
#     print(f"[{time.strftime('%H:%M:%S')}] ✅ End Slow {name}")
#     return f'Slow-{name}'
# 
# @api(qps=3, limit=None)
# async def trigger_abort_after_delay(func_name, prefix, delay=0.5):
#     await asyncio.sleep(delay)
#     print(f"\n>>> [Monitor] 时间到！触发 Abort: 取消所有 {prefix} 开头的任务 <<<\n")
#     abort(func_name, prefix)
# 
# async def main():
#    init_pool(n_workers=4,global_limit=30) # <-- 初始化池配置(在所有@api注册后调用)
#    async for ret in batch({               # <-- 批量调用(哪个任务先完成，就立即返回)
#        "X1": {                              # <-- 任务ID
#            'f':'trigger_abort_after_delay', # <-- 任务函数名
#            'kwargs':{                       # <-- 任务函数参数
#                'func_name': 'slow_task',
#                'prefix': 'S',
#                'delay': 1.0
#            }
#        },
#        "X2": {'f':'trigger_abort_after_delay', 'kwargs':{
#            'func_name': 'fast_task',
#            'prefix': 'F',
#            'delay': 1.0
#        }},
#        "X3": {'f':'trigger_abort_after_delay', 'kwargs':{
#            'func_name': 'test_task',
#            'prefix': 'T',
#            'delay': 1.0
#        }},
#        **{f"T{i}": {'f':'test_task', 'kwargs':{'x':i}} for i in range(6)},
#        **{f"F{i}": {'f':'fast_task', 'kwargs':{'name':i}} for i in range(6)},
#        **{f"S{i}": {'f':'slow_task', 'kwargs':{'name':i}} for i in range(6)},
#    }):
#        print(ret)
#        # ret['suc']   (bool): 是否没有报错，正常运行
#        # ret['data']   (any): 执行结果，或者报错提示
#        # ret['key']    (str): 任务ID
#        # ret['f']      (str): 函数名
# 
# if __name__ == '__main__':
#     asyncio.run(main())
#-----------------------------------------------------------#
GLOBAL_STATE: Dict[str, Any] = {
    'api_table': {},          # 存储 API 注册信息
    'active_tasks': {},       # 存储正在运行的任务 用于 abort
    'api_timers': {},         # 存储虚拟时间轴，用于跨 batch 的 QPS 控制
    'global_limit': 5000,     # 全局并发上限
    'global_semaphore': None, # 全局并发信号量
    'state_lock': None,       # 状态锁
    '_process_executor': None, # 进程池执行器
}

def func_to_tool(func):
    """函数装饰器，用于将一个普通Python函数转换为Function Calling的JSON Schema定义。"""
    from typing import get_origin, get_args, Literal
    properties = {}
    required_params = []
    for name, param in inspect.signature(func).parameters.items():
        param_schema = {}
        py_type = param.annotation
        param_schema["type"] = {
            int:'integer',
            float:'number',
            bool:'boolean',
            str:'string',
            inspect._empty:'string'
        }.get(py_type)
        if not param_schema["type"]:
            origin = get_origin(py_type)
            args = get_args(py_type)
            if origin is Literal:
                param_schema["type"] = type(args[0]).__name__
                param_schema["enum"] = list(args)
            elif origin is list:
                param_schema["type"] = "array"
                item_type = args[0] if args else str
                if item_type is int: param_schema["items"] = {"type": "integer"}
                elif item_type is float: param_schema["items"] = {"type": "number"}
                elif item_type is bool: param_schema["items"] = {"type": "boolean"}
                else: param_schema["items"] = {"type": "string"}
            else:
                param_schema["type"] = "object"
        if param.default is not inspect._empty:
            param_schema["default"] = param.default
        else:
            required_params.append(name)
        properties[name] = param_schema
    input_schema = {
        "type": "object",
        "properties": properties,
    }
    if required_params:
        input_schema["required"] = required_params
    #mcp_definition = {
    #    "name": func.__name__,
    #    "description": func.__doc__,
    #    "inputSchema": input_schema
    #}
    tool_definition = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__,
            "parameters": input_schema
        }
    }
    return tool_definition

def init_pool(n_workers=1, global_limit=5000):
    """初始化全局池配置"""
    GLOBAL_STATE['global_limit'] = global_limit
    GLOBAL_STATE['global_semaphore'] = asyncio.Semaphore(global_limit)
    GLOBAL_STATE['state_lock'] = asyncio.Lock()
    GLOBAL_STATE['_process_executor'] = ProcessExecutor(
        funcs = [
            x['f'] 
            for x in GLOBAL_STATE['api_table'].values()
            if not inspect.iscoroutinefunction(x['f'])
        ],
        n_workers=n_workers
    )

def api(qps=None, limit=None):
    """
    装饰器：注册 API 到全局状态
    """
    def _command(func):
        interval = 1.0 / qps if qps and qps > 0 else 0
        semaphore = asyncio.Semaphore(limit) if limit else None
        
        GLOBAL_STATE['api_table'][func.__name__] = {
            'f': func,
            'local_semaphore': semaphore,
            'limit': limit,
            'qps_interval': interval,
            'tool': func_to_tool(func),
        }
        
        # 初始化该 API 的虚拟时间锚点
        if func.__name__ not in GLOBAL_STATE['api_timers']:
            GLOBAL_STATE['api_timers'][func.__name__] = 0.0
            
        return func
    return _command

async def _worker_wrapper(key, func_name, f, kwargs, local_semaphore, initial_delay):
    """包装执行函数"""
    global_semaphore = GLOBAL_STATE['global_semaphore']
    try:
        if initial_delay > 0:
            try:
                await asyncio.wait_for(asyncio.sleep(initial_delay), timeout=initial_delay + 3600)
            except asyncio.CancelledError:
                return {'suc': False, 'data': 'Cancelled in Waiting', 'key': key, 'f': func_name}
        
        async with global_semaphore:
            try:
                if local_semaphore:
                    async with local_semaphore:
                        return await _execute(key, func_name, f, kwargs)
                else:
                    return await _execute(key, func_name, f, kwargs)
            except asyncio.CancelledError:
                return {'suc': False, 'data': 'Cancelled in Queueing', 'key': key, 'f': func_name}
    except asyncio.CancelledError:
        return {'suc': False, 'data': 'Cancelled in Global Queueing', 'key': key, 'f': func_name}

async def _execute(key, func_name, f, kwargs):
    """执行实际业务逻辑"""
    try:
        if inspect.iscoroutinefunction(f):
            res = await f(**kwargs)
            return {'suc': True, 'data': res, 'key': key, 'f': func_name}
        else:
            res = await GLOBAL_STATE['_process_executor'].submit(key,func_name,kwargs)
            return res
    except asyncio.CancelledError:
        return {'suc': False, 'data': 'Cancelled in Executing', 'key': key, 'f': func_name}
    except Exception as e:
        return {'suc': False, 'data': traceback.format_exc(), 'key': key, 'f': func_name}

async def batch(key_payloads: Dict[str, dict]) -> AsyncIterator[Dict[str, Any]]:
    """ 批量执行任务 {key: {f:str, kwargs: dict}} -> {suc:bool, data:Any, key:str, f:str}"""
    api_table = GLOBAL_STATE['api_table']
    active_tasks = GLOBAL_STATE['active_tasks']
    api_timers = GLOBAL_STATE['api_timers']
    global_limit = GLOBAL_STATE['global_limit']
    state_lock = GLOBAL_STATE['state_lock']

    is_overload = False
    async with state_lock:
        current_global_active = sum(len(tasks) for tasks in active_tasks.values())
        total_incoming = len(key_payloads)
        if current_global_active + total_incoming > global_limit:
            is_overload = True
    
    if is_overload:
        for key in key_payloads.keys():
            yield {'suc': False, 'data': 'Global Overload', 'key': key}
        return

    planned_tasks = {} 
    for key, payload in key_payloads.items():
        func_name = payload.get('f')
        if not func_name or func_name not in api_table:
            yield {'suc': False, 'data': 'Function not found', 'key': key}
            continue
        if func_name not in planned_tasks:
            planned_tasks[func_name] = []
        planned_tasks[func_name].append((key, payload))

    tasks_to_wait = []
    task_map = {} 

    now = time.time()
    
    for func_name, tasks in planned_tasks.items():
        func_info = api_table[func_name]
        interval = func_info['qps_interval']
        last_time = max(now, api_timers.get(func_name, 0.0))
        
        if func_name not in active_tasks:
            active_tasks[func_name] = {}
            
        for key, payload in tasks:
            delay = 0
            if interval > 0:
                next_time = last_time + interval
                delay = next_time - now
                api_timers[func_name] = next_time
                last_time = next_time
            
            coro = _worker_wrapper(
                key, 
                func_name, 
                func_info['f'],
                payload.get('kwargs', {}),
                func_info['local_semaphore'],
                delay
            )
            task = asyncio.create_task(coro)
            tasks_to_wait.append(task)
            task_map[task] = key
            active_tasks[func_name][key] = task
            
    for future in asyncio.as_completed(tasks_to_wait):
        ret = await future
        f_name = ret.get('f')
        k = ret.get('key')
        if f_name and k and active_tasks.get(f_name, {}).get(k):
            del active_tasks[f_name][k]
        yield ret

def abort(func_name, key_prefix):
    """取消指定前缀的任务 (全局函数版本)"""
    active_tasks = GLOBAL_STATE['active_tasks']
    if func_name not in active_tasks:
        return 0
    count = 0
    for key, task in list(active_tasks[func_name].items()):
        if key.startswith(key_prefix):
            task.cancel()
            count += 1
    print(f"[Abort] 已向 {count} 个“{func_name}”任务发送取消信号")
    return count

class Batch:
    def __init__(self,payloads,batch_id:Optional[str]=None):
        self.batch_id = batch_id or f"{time.time_ns()}_{''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',k=3))}"
        self.key_payloads = {
            f"{self.batch_id}#{i}": payload
            for i,payload in enumerate(payloads)
        }
        self.func_names = set(x['f'] for x in payloads)
        self.done = False
    async def __aenter__(self):
        return self
    async def __aexit__(self,*args):
        self.abort()
    def abort(self):
        if not self.done:
            for func_name in self.func_names:
                abort(func_name,self.batch_id)
    async def __aiter__(self) -> AsyncGenerator[TaskResult, None]:
        async for ret in batch(self.key_payloads):
            payload = self.key_payloads[ret['key']]
            ret.update(payload)
            yield ret
        self.done = True


#-----------------------------------------------------------#
# 内置异步工具
#-----------------------------------------------------------#



_session: Optional[aiohttp.ClientSession] = None

@api()
async def fetch_text(
    url: str,
    method: str = "GET",
    *,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    data: Any = None,
    headers: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    timeout: float = 30.0,
    url_transformer: Optional[Callable[[str], str]] = None
) -> FetchResult:
    global _session
    if _session is None or _session.closed:
        connector = aiohttp.TCPConnector(
            limit=1000,                 # 全局最大连接数
            limit_per_host=200,         # 单域名最大连接数 (防止单点拖垮)
            force_close=False,          # 保持 Keep-Alive 复用
            enable_cleanup_closed=True  # 自动清理已关闭的连接
        )
        _session = aiohttp.ClientSession(connector=connector)
    async def request_context():
        final_url = url_transformer(url) if url_transformer else url
        final_headers = headers
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        try:
            async with _session.request(
                method,
                final_url,
                params=params,
                json=json,
                data=data,
                headers=final_headers,
                timeout=client_timeout
            ) as response:
                try:
                    if 200 <= response.status < 300:
                        resp_data = await response.text()
                        return {'suc': True, 'data': resp_data, 'msg': 'ok'}
                    elif response.status in (429, 403, 500, 502, 503, 504):
                        text = await response.text()
                        return {'suc': False, 'retry': True, 'data': '', 'msg': f'HTTP {response.status}: {text[:50]}'}
                    else:
                        text = await response.text()
                        return {'suc': False, 'retry': False, 'data': '', 'msg': f'HTTP {response.status}: {text[:100]}'}
                except Exception:
                    response.close()
                    raise
        except asyncio.TimeoutError:
            return {'suc': False, 'retry': True, 'data': '', 'msg': 'Timeout'}
        except aiohttp.ClientError as e:
            return {'suc': False, 'retry': True, 'data': '', 'msg': f'Network Error: {str(e)[:50]}'}
        except Exception as e:
            return {'suc': False, 'retry': False, 'data': '', 'msg': f'Unexpected: {str(e)[:50]}'}
    last_error = ""
    for attempt in range(max_retries + 1):
        ret = await request_context()
        if ret['suc']:
            return ret
        if not ret.get('retry', False):
            return ret
        last_error = ret['msg']
        if attempt < max_retries:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(wait_time)
    return {'suc': False, 'data': '', 'msg': f'Failed after retries: {last_error}'}

@api()
async def fetch_json(
    url: str,
    method: str = "GET",
    *,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    data: Any = None,
    headers: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    timeout: float = 30.0,
    url_transformer: Optional[Callable[[str], str]] = None
) -> FetchResult:
    ret = await fetch_text(
        url=url,
        method=method,
        params=params,
        json=json,
        data=data,
        headers=headers,
        max_retries=max_retries,
        timeout=timeout,
        url_transformer=url_transformer
    )
    if ret['suc']:
        raw_data = ret.get('data')
        try:
            parsed_data = orjson.loads(raw_data)
            ret['data'] = parsed_data
        except:
            ret['suc'] = False
            ret['msg'] = f'JSON解析失败: {raw_data[:100]}...' 
    return ret



async def fetch_stream(
    url: str,
    method: str = "GET",
    *,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    data: Any = None,
    headers: Optional[Dict[str, Any]] = None,
    timeout: float = 180.0,
    url_transformer: Optional[Callable[[str], str]] = None,
) -> AsyncGenerator[bytes, None]:
    """
    流式获取 HTTP 响应，按行 yield 数据
    
    Args:
        url: 请求 URL
        method: HTTP 方法
        params: URL 查询参数
        json: JSON 请求体
        data: 请求体
        headers: 请求头
        timeout: 超时时间(秒)
        url_transformer: URL 转换函数
    
    Yields:
        bytes: 每一行的数据（包含换行符 \\n）
    """
    
    global _session
    if _session is None or _session.closed:
        connector = aiohttp.TCPConnector(
            limit=1000,              # 全局最大连接数
            limit_per_host=200,      # 单域名最大连接数 (防止单点拖垮)
            force_close=False,       # 保持 Keep-Alive 复用
            enable_cleanup_closed=True  # 自动清理已关闭的连接
        )
        _session = aiohttp.ClientSession(connector=connector)
    final_url = url_transformer(url) if url_transformer else url
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with _session.request(
        method,
        final_url,
        params=params,
        json=json,
        data=data,
        headers=headers,
        timeout=client_timeout
    ) as response:
        async for line in response.content:
            yield line

async def close_pool():
    global _session
    if _session and not _session.closed:
        await _session.close()
    _session = None

def new_llm_msg(role:Literal['system','user','assistant','tool'],content:str,**kwargs) -> Message:
    return {'role':role,'content':content,**{k:v for k,v in kwargs.items() if v}}


GLOBAL_STATE['llm_zhipu_host'] = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLOBAL_STATE['llm_zhipu_api_key'] = os.getenv("ZAI_API_KEY")


def get_tool_ctx(tool_name,ctx_param_names:Optional[set]=None):
    if tool_name not in GLOBAL_STATE['api_table']:
        return {'suc':False,'msg':f"Tool api not found: {tool_name}"}
    if 'tool' not in GLOBAL_STATE['api_table'][tool_name]:
        return {'suc':False,'msg':f"Tool description not found: {tool_name}"}
    if ctx_param_names is None:
        return {'suc':True, 'data':set()}
    raw_tool = GLOBAL_STATE['api_table'][tool_name]['tool']
    f_param_names = set(raw_tool['function']['parameters']['properties'].keys())
    masked_param_names = f_param_names & ctx_param_names
    return {'suc':True, 'data':masked_param_names}

def get_tool_description(tool_name,ctx_param_names:Optional[set]=None):
    if tool_name not in GLOBAL_STATE['api_table']:
        return {'suc':False,'msg':f"Tool api not found: {tool_name}"}
    if 'tool' not in GLOBAL_STATE['api_table'][tool_name]:
        return {'suc':False,'msg':f"Tool description not found: {tool_name}"}
    raw_tool = GLOBAL_STATE['api_table'][tool_name]['tool']
    if ctx_param_names is None:
        return {'suc':True, 'data':raw_tool}
    f_param_names = set(raw_tool['function']['parameters']['properties'].keys())
    masked_param_names = f_param_names & ctx_param_names
    if not masked_param_names:
        return {'suc':True, 'data':raw_tool}
    masked_tool = copy.deepcopy(raw_tool)
    for param in masked_param_names:
        masked_tool['function']['parameters']['properties'].pop(param, None)
        if param in masked_tool['function']['parameters']['required']:
            masked_tool['function']['parameters']['required'].remove(param)
    return {'suc':True, 'data':masked_tool}


async def f_write(x,mode:Literal['system','user','assistant','tool','think'],last_mode=['assistant']):
    if mode!=last_mode[0]:
        x = '\n' + x
        last_mode[0] = mode
    if mode=='assistant':
        print(x, end='')
    elif mode=='think':
        print(f"\033[33m{x}\033[0m", end='')
    elif mode=='tool':
        print(f"\033[36m{x}\033[0m", end='')
    elif mode=='system':
        print(f"\033[31m{x}\033[0m", end='')
    elif mode=='user':
        print(f"\033[32m{x}\033[0m", end='')
    

async def llm(
    messages: list[Message],
    tool_names: Optional[list[str]] = None,
    ctx_kwargs:Optional[Dict]=None, # llm不可见、工具调用时才会填充的参数。
    url=GLOBAL_STATE['llm_zhipu_host'],
    api_key=GLOBAL_STATE['llm_zhipu_api_key'],
    model="glm-4.7-flash",
    temperature=0.2,
    top_p=0.7,
    max_tokens=65536,
    custom_body:Optional[LLMCustomBody] = None,
    f_write:Optional[Callable[[str,Literal['system','user','assistant','tool','think']],None]]=f_write,
)-> LLMResult:
    json_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True,
        **(custom_body or {})
    }
    ctx_param_names = set(ctx_kwargs.keys()) if ctx_kwargs else None
    tools = []
    for tool_name in (tool_names or []):
        ret = get_tool_description(tool_name,ctx_param_names)
        if ret['suc']:
            tools.append(ret['data'])
        else:
            await f_write(ret['data'],'system')
    if tools:
        json_payload['tools'] = tools
    full_response = ""
    thinking_content = ""
    tool_calls = []
    first_chunk = True
    meta_info = {
        "id": "",
        "model": "",
        "created": 0,
        "finish_reason": "",
        "usage": {}
    }
    try:
        async for line in fetch_stream(
            url,
            method="POST",
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            },
            json=json_payload
        ):            
            if not line.startswith(b"data: "):
                continue
            data_bytes = line[6:].strip()
            if data_bytes == b"[DONE]":
                if f_write:
                    await f_write('\n', 'assistant')
                break
            try:
                chunk_data = orjson.loads(data_bytes)
                if first_chunk:
                    meta_info["id"] = chunk_data.get('id', '')
                    meta_info["model"] = chunk_data.get('model', '')
                    meta_info["created"] = chunk_data.get('created', 0)
                    first_chunk = False
                if 'usage' in chunk_data:
                    meta_info["usage"] = chunk_data['usage']
                if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                    choice = chunk_data['choices'][0]
                    current_reason = choice.get('finish_reason')
                    if current_reason:
                        meta_info["finish_reason"] = current_reason
                    delta = choice.get('delta', {})
                    if 'content' in delta:
                        if not full_response:
                            delta['content'] = delta['content'].lstrip('\n')
                        content = delta['content']
                        if content:
                            full_response += content
                            if f_write:
                                await f_write(content, 'assistant')
                    if 'reasoning_content' in delta:
                        thinking_content += delta['reasoning_content']
                        if f_write:
                            await f_write(delta['reasoning_content'], 'think')
                    if 'tool_calls' in delta and delta['tool_calls']:
                        for tool_call in delta['tool_calls']:
                            tool_calls.append(tool_call)
            except orjson.JSONDecodeError:
                continue
        return {
            'suc': True,
            'data': {
                'message': new_llm_msg('assistant', full_response, thinking_content=thinking_content, tool_calls=tool_calls),
                "id": meta_info['id'],
                "model": meta_info['model'],
                "created": meta_info['created'],
                "finish_reason": meta_info['finish_reason'],
                "completion_tokens": meta_info['usage'].get('completion_tokens',0),
                "total_tokens": meta_info['usage'].get('total_tokens',0),
                "cached_tokens": meta_info['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0),
            },
        }
    except Exception as e:
        return {
            'suc': False,
            'msg': f'Unexpected Error: {str(e)}\n{traceback.format_exc()}'
        }

async def act(
    tool_calls:List[ToolCall],
    ctx_kwargs:Optional[Dict]=None, # llm可见、工具调用时才会填充的参数。
    batch_id:Optional[str]=None,
    f_write:Optional[Callable[[str,Literal['system','user','assistant','tool','think']],None]]=f_write,
)->ActResult:
    if not tool_calls:
        return {'suc': False, 'msg': "No tool_calls found."}
    ctx_param_names = set(ctx_kwargs.keys()) if ctx_kwargs else None
    payloads = []
    index_functions = {}
    for tool_call in tool_calls:
        if tool_call['type'] == 'function':
            index_functions[tool_call['index']] = tool_call
        elif tool_call['type'] == None:
            target_tool_call = index_functions.get(tool_call['index'])
            for k,v in tool_call['function'].items():
                if k in target_tool_call['function']:
                    target_tool_call['function'][k] += v
                else:
                    target_tool_call['function'][k] = v
    for tool_call in index_functions.values():
        if tool_call['type'] != 'function':
            return {'suc': False, 'msg': f"Invalid tool_call type. {tool_call}"}
        f_name = tool_call['function']['name']
        ret = get_tool_ctx(f_name,ctx_param_names)
        if not ret['suc']:
            return {'suc': False, 'msg': ret['msg']}
        fill_param_names = ret['data']
        f_args = orjson.loads(tool_call['function']['arguments'])
        for param in fill_param_names:
            f_args[param] = ctx_kwargs[param]
        payloads.append({
            'key':tool_call['id'],
            #'tool_call': tool_call['function'],
            'f': f_name,
            'kwargs': f_args,
        })
    # tool_calls正确，批量执行，统一返回结果
    all_success = True
    delta_messages = []
    try:
        async with Batch(payloads,batch_id) as futures:
            progress = 0
            async for ret_tool in futures:
                progress += 1
                if not ret_tool['suc']:
                    all_success = False
                tool_call_id = ret_tool.pop('key',None)
                ret_tool.pop('kwargs',None)
                tool_name = ret_tool.pop('f',None)
                try:
                    tool_result_str = orjson.dumps(ret_tool).decode()
                except Exception as e:
                    tool_result_str = str(ret_tool)
                delta_messages.append(new_llm_msg('tool', tool_result_str, tool_call_id = tool_call_id))
                if f_write:
                    await f_write(f"{'✓' if ret_tool['suc'] else '✗'} [{progress}/{len(payloads)}] {tool_name}\n",'tool')
    except Exception as e:
        return {
            'suc': False, 
            'data': {'all_success':False, 'delta_messages': delta_messages}, 
            'msg': f'Unexpected Error in util_pool.act_run: {str(e)}'
        }
    return {'suc': True, 'data': {'all_success':all_success, 'delta_messages': delta_messages}}



async def agent(
    messages:List[Message],
    tool_names:Optional[List[str]]=None,
    ctx_kwargs:Optional[Dict]=None, # llm不可见、工具调用时才会填充的参数。
    n_steps = 1,
    temperature=0.2,
    top_p=0.7,
    max_tokens=65536,
    custom_body:Optional[LLMCustomBody] = None,
    api_key=GLOBAL_STATE['llm_zhipu_api_key'],
    url=GLOBAL_STATE['llm_zhipu_host'],
    model="glm-4.7-flash",
    f_write:Optional[Callable[[str,Literal['system','user','assistant','tool','think']],None]]=f_write,
)->AgentResult:
    finish_reason = None
    completion_tokens = 0
    total_tokens = 0
    cached_tokens = 0
    for i_step in range(-1,n_steps):
        # act
        tool_calls = messages[-1].get('tool_calls')
        if i_step >= 0 and not tool_calls:
            break
        if tool_calls:
            ret_act = await act(tool_calls,ctx_kwargs,f_write=f_write)
            if not ret_act['suc']:
                return {'suc': False, 'msg': ret_act['msg']}
            for message in ret_act['data']['delta_messages']:
                messages.append(message)
        # llm
        ret_llm = await llm(messages,tool_names if i_step<n_steps-1 else [],ctx_kwargs,url,api_key,model,temperature,top_p,max_tokens,custom_body,f_write)
        if not ret_llm['suc']:
            return {'suc': False, 'msg': ret_llm['msg']}
        messages.append(ret_llm['data']['message'])
        finish_reason = ret_llm['data']['finish_reason']
        completion_tokens += ret_llm['data']['completion_tokens']
        total_tokens += ret_llm['data']['total_tokens']
        cached_tokens += ret_llm['data']['cached_tokens']
    return {
        'suc': True, 
        'data': {
            'messages':messages,
            'finish_reason': finish_reason,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'cached_tokens': cached_tokens,
        }
    }
    
#
#@api()
#async def get_weather(city: str,user_id:str) -> dict:
#    """获取{city}城市的天气数据"""
#    import random
#    await asyncio.sleep(random.random())
#    weather_data = {
#        "city": city,
#        "user_id": user_id,
#        "temperature": f"{random.randint(15,30)}°C",
#        "condition": f"{random.choice(['晴天', '阴天', '雨天'])}",
#        "humidity": f"{random.randint(70,90)}%",
#        "wind_speed": f"{random.randint(0,10)} km/h"
#    }
#    return weather_data
#
#
#async def main():
#    init_pool()
#    messages = [
#        new_llm_msg('system', f'你是一个Function Agent，请使用工具进行回答。'),
#        new_llm_msg('user', "比较一下广州和深圳和青岛的天气"),
#    ]
#    ret = await agent(
#        messages, 
#        tool_names=['get_weather'],
#        ctx_kwargs={'user_id': 'test_user_123'},
#        n_steps=3,
#        model="glm-4.5-flash",
#        temperature=0.2,
#        top_p=0.7,
#        max_tokens=65536,
#        f_write=f_write,
#    )
#    from rich import print
#    print(ret)
#    if ret['suc']:
#        print(f"""
#Final Content:
#    {ret['data']['messages'][-1]['content']}
#Token Usage:
#    completion={ret['data']['completion_tokens']}
#    total={ret['data']['total_tokens']}
#    cached={ret['data']['cached_tokens']}
#        """)
#    else:
#        print(f"Error: {ret['msg']}")
#    await close_pool()
#
#if __name__ == '__main__':
#    asyncio.run(main())

