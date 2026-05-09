"""本程序演示如何在工具调用中传递上下文参数"""
import asyncio
from shanquan import api, agent, init_pool, f_write, close_pool
import os


# 假如有个工具函数，用到上下文参数（如：会话，数据库对象），不应该由LLM指定其值
# 只要在agent()中指定ctx_kwargs参数，即可向LLM隐藏该参数，执行时自动填充。


@api()
async def get_weather(Session, city: str) -> dict:
    """获取{city}城市的天气数据"""
    import random
    await asyncio.sleep(1)
    # --- Session Operation ---
    print(Session) 
    # -------------------------
    weather_data = {
        "city": city,
        "temperature": f"{random.randint(15,30)}°C",
        "condition": f"{random.choice(['晴天', '阴天', '雨天'])}",
        "humidity": f"{random.randint(70,90)}%",
        "wind_speed": f"{random.randint(0,10)} km/h"
    }
    return weather_data


async def main():
    init_pool()
    ret = await agent(
        messages = [
            {'role':'system', 'content': '你是一个Function Agent，请使用工具进行回答。'},
            {'role':'user', 'content':'比较一下广州和深圳和青岛的天气'},
        ], 
        tool_names=['get_weather'],
        ctx_kwargs={'Session': 'Session(user_001)'}, # 自动填充工具参数
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
