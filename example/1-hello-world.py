"""本程序演示如何注册异步工具与同步，并让AI Agent来调用工具回答问题"""
import asyncio
from shanquan import api, agent, init_pool, f_write, close_pool
import os

@api()
async def get_weather(city: str) -> dict:
    """获取{city}城市的天气数据"""
    import random
    import time
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: get_weather({city}) Start")
    await asyncio.sleep(1) # 延迟1秒，用来判断3次函数调用是同时执行的。
    weather_data = {
        "city": city,
        "temperature": f"{random.randint(15,30)}°C",
        "condition": f"{random.choice(['晴天', '阴天', '雨天'])}",
        "humidity": f"{random.randint(70,90)}%",
        "wind_speed": f"{random.randint(0,10)} km/h"
    }
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: get_weather({city}) Finish")
    return weather_data


@api()
def get_network_latency(city_A: str,city_B:str) -> dict:
    """获取{city_A}城市到{city_B}城市之间的网络延迟"""
    import random
    import time
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: get_network_latency({city_A},{city_B}) Start")
    time.sleep(1)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: get_network_latency({city_A},{city_B}) Finish")
    return f"{random.randint(0,999)}ms"


async def main():
    init_pool(n_workers=5) # <-- 创建5个工人进程
    ret = await agent(
        messages = [
            {'role':'system', 'content': '你是一个Function Agent，请使用工具进行回答。'},
            {'role':'user', 'content':'比较一下广州和深圳和青岛的天气，同时获取他们两两之间的网络延迟，直接输出JSON'},
        ], 
        tool_names=['get_weather','get_network_latency'],
        n_steps=3,
        model="glm-4.5-flash",
        temperature=0.2,
        top_p=0.7,
        max_tokens=65536,
        url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
        api_key=os.getenv("ZAI_API_KEY"),
        custom_body = {
            "thinking": {"type": "disabled"},    # or enabled
            "response_format": {"type": "text"}, # or json_object
        },
        f_write=f_write,
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

    """
我来获取广州、深圳和青岛的天气数据以及它们之间的网络延迟信息。

2026-05-09 17:56:12: get_weather(广州) Start
2026-05-09 17:56:12: get_weather(深圳) Start
2026-05-09 17:56:12: get_weather(青岛) Start
2026-05-09 17:56:13: get_network_latency(广州,深圳) Start
2026-05-09 17:56:13: get_network_latency(广州,青岛) Start
2026-05-09 17:56:13: get_network_latency(深圳,青岛) Start
2026-05-09 17:56:13: get_weather(广州) Finish
2026-05-09 17:56:13: get_weather(深圳) Finish
2026-05-09 17:56:13: get_weather(青岛) Finish

✓ [1/6] get_weather
✓ [2/6] get_weather
✓ [3/6] get_weather
2026-05-09 17:56:14: get_network_latency(深圳,青岛) Finish
2026-05-09 17:56:14: get_network_latency(广州,青岛) Finish
2026-05-09 17:56:14: get_network_latency(广州,深圳) Finish
✓ [4/6] get_network_latency
✓ [5/6] get_network_latency
✓ [6/6] get_network_latency

```json
{
  "weather_comparison": {
    "广州": {
      "temperature": "29°C",
      "condition": "晴天",
      "humidity": "73%",
      "wind_speed": "1 km/h"
    },
    "深圳": {
      "temperature": "27°C",
      "condition": "晴天",
      "humidity": "71%",
      "wind_speed": "10 km/h"
    },
    "青岛": {
      "temperature": "23°C",
      "condition": "雨天",
      "humidity": "76%",
      "wind_speed": "6 km/h"
    }
  },
  "network_latency": {
    "广州_深圳": "241ms",
    "广州_青岛": "471ms",
    "深圳_青岛": "74ms"
  }
}
```

Token Usage:
    completion=308
    total=1155
    cached=86
        
    """