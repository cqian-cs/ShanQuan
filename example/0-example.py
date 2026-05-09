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
    ret = await agent( # <-- 调用一次智能体对话
        messages = [
            {'role':'system', 'content': '你是一个Function Agent，请使用工具进行回答。'},
            {'role':'user', 'content':'比较一下广州和深圳和青岛的天气'},
        ], 
        tool_names=['get_weather'], # <-- 工具函数名
        n_steps=3, # <-- 设置“推理-行动”循环上限
        model="glm-4.5-flash",
        temperature=0.2,
        top_p=0.7,
        max_tokens=65536,
        url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
        api_key=os.getenv("ZAI_API_KEY"), # <-- 此处填写智谱大模型的API_Key
        custom_body = {
            "thinking": {"type": "disabled"},    # disabled=关闭思考 enabled=开启思考
            "response_format": {"type": "text"}, # text=回复文字 json_object=回复JSON
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