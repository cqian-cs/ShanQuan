# ShanQuan (山泉) ⛰️

ShanQuan 是一个多工具并行的 AI 智能体框架。


## 🧭 工具定位

|          | LangChain                            | ShanQuan                   |
| ----     | ----------------------               | -------------------        |
| Agent    | 对象，有生命周期                     | 普通python函数，无状态        |
| 工具     | 对象，继承BaseTool类                 | 普通python函数，注册到全局字典 |
| 上下文   | 内置于 Agent、Memory、ToolRunTime 等 | 外置于控制流，显式参数注入     |
| 工具调用 | 默认串行（一个一个执行）              | 自动并行（同时执行）           |
| 控制流   | 由框架编排                           | 你的函数，你的流程            |



## ✨ 特性

- **多工具并行**: LLM 返回多个工具调用时，自动并发执行。
- **协程 + 进程混合池**: `async def` 走协程池，普通 `def` 走进程池。
- **内置 QPS / 并发限流**: `@api(qps=5, limit=10)` 一行声明，全局生效，防止打爆上游 API。
- **上下文参数注入**: 上下文参数对 LLM 不可见，工具调用时自动填充。
- **极度轻量**: 本框架基于 `aiohttp` + `orjson`，[总共约1000行代码](./src/shanquan.py)。



## 🚀 快速上手

### 1. 安装
```bash
pip install shanquan
```


### 2. 基础示例
```python
import asyncio
from shanquan import api, agent, init_pool, f_write, close_pool
import os

@api(qps=5,limit=10) # <-- 通过@api来注册工具，同时设置限流：QPS=5、并发=10。
async def get_weather(city: str) -> dict:
    """获取{city}城市的天气数据"""
    import random  # <-- 函数内 import，按需加载
    import time
    print(f"[{time.strftime('%H:%M:%S')}] Start {city}")
    await asyncio.sleep(1) # 延迟1秒，用来判断3次函数调用是同时执行的。
    weather_data = {
        "city": city,
        "temperature": f"{random.randint(15,30)}°C",
        "condition": f"{random.choice(['晴天', '阴天', '雨天'])}",
        "humidity": f"{random.randint(70,90)}%",
        "wind_speed": f"{random.randint(0,10)} km/h"
    }
    print(f"[{time.strftime('%H:%M:%S')}] finish {city}")
    return weather_data


async def main():
    init_pool()
    ret = await agent(
        messages = [
            {'role':'system', 'content': '你是一个Function Agent，请使用工具进行回答。'},
            {'role':'user', 'content':'比较一下广州和深圳和青岛的天气'},
        ], 
        tool_names=['get_weather'], # <-- 工具函数名
        n_steps=3, # <-- 设置“推理-行动”循环次数上限
        model="glm-4.5-flash",
        temperature=0.2,
        top_p=0.7,
        max_tokens=65536,
        url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
        api_key=os.getenv("ZAI_API_KEY"), # <-- 此处填写智谱大模型的API_Key
        custom_body = {
            # 此处可以修改请求体，以适配不同大模型平台
            "thinking": {"type": "disabled"},
            # disabled=关闭思考 enabled=开启思考
            "response_format": {"type": "text"},
            # text=回复文字 json_object=回复JSON
        },
        f_write=f_write, # <-- f_write是流式输出的回显函数
    )
    if ret['suc']:
        print(f"""
Token Usage:
    completion={ret['data']['completion_tokens']}
    total={ret['data']['total_tokens']}
    cached={ret['data']['cached_tokens']}
        """)
    else:
        print(f"Error: {ret['msg']}")
    await close_pool()

if __name__ == '__main__':
    asyncio.run(main())
```

<details>
<summary>展开程序输出</summary>
<pre><code>
我来帮您比较广州、深圳和青岛这三个城市的天气情况。

[17:37:45] Start 广州
[17:37:45] Start 深圳
[17:37:45] Start 青岛
[17:37:46] finish 广州

✓ [1/3] get_weather
[17:37:46] finish 深圳
✓ [2/3] get_weather
[17:37:46] finish 青岛
✓ [3/3] get_weather

根据最新的天气数据，以下是广州、深圳和青岛三个城市的天气对比：

## 🌧️ 天气状况对比

| 城市 | 温度 | 天气状况 | 湿度 | 风速 |
|------|------|----------|------|------|
| **广州** | 29°C | 雨天 | 76% | 6 km/h |
| **深圳** | 22°C | 雨天 | 89% | 6 km/h |
| **青岛** | 17°C | 雨天 | 74% | 7 km/h |

## 📊 详细分析

### 🌡️ **温度对比**
- **广州最热**：29°C，属于较温暖的温度
- **深圳次之**：22°C，温度适中
- **青岛最凉爽**：17°C，相对较冷

### 💧 **湿度对比**
- **深圳最潮湿**：89%，湿度很高，感觉会比较闷热
- **广州和青岛湿度相近**：76%和74%，湿度适中

### 🌬️ **风力对比**
- **青岛风力稍强**：7 km/h
- **广深风力相同**：6 km/h

### 🌧️ **共同特点**
三个城市目前都是**雨天**，都建议携带雨具出行。

## 🎯 总结建议
- 如果喜欢温暖天气，选择广州
- 如果偏好凉爽气候，青岛更合适
- 深圳虽然温度适中，但湿度较高，可能会感觉比较闷热
- 三个城市都需要注意防雨，建议携带雨伞或雨衣

Token Usage:
    completion=402
    total=942
    cached=441
</code></pre>
</details>

### 3. 其他示例

| 示例程序 | 示例内容 |
| ------- | --------|
| [1-hello-world.py](./example/1-hello-world.py)   | 同时调用`def`同步工具与`async def`异步工具   |
| [2-reason-act.py](./example/2-reason-act.py)     | `agent()`的分解步骤，即`llm()`计划，`act()`执行 |
| [3-ctx_kwargs.py](./example/3-ctx_kwargs.py)     | 通过`agent(ctx_kwargs={...})`来注入上下文参数 |
| [4-manual-batch.py](./example/4-manual-batch.py) | 通过创建任务列表来并行调用工具函数 |

