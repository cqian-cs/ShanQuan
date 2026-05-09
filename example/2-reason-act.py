"""本程序演示如何使用llm()进行仅对话，以及act()并行调用工具函数"""
import asyncio
from shanquan import api, llm, act, init_pool, close_pool
import os
from rich import print
@api()
async def get_weather(city: str) -> dict:
    """获取{city}城市的天气数据"""
    import random
    await asyncio.sleep(1) # 延迟1秒，用来证明3次函数调用是同时执行的。
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
    ret_llm = await llm(
        messages = [
            {'role':'system', 'content': '你是一个Function Agent，请使用工具进行回答。'},
            {'role':'user', 'content':'比较一下广州和深圳和青岛的天气'},
        ],
        tool_names=['get_weather'],
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
        f_write=None,
    )
    print(ret_llm)
    """
    {
        'suc': True,
        'data': {
            'message': {
                'role': 'assistant',
                'content': '我来帮您比较广州、深圳和青岛这三个城市的天气情况。\n',
                'tool_calls': [
                    {
                        'id': 'call_8296a4c4536841fcb67c4ee1',
                        'index': 0,
                        'type': 'function',
                        'function': {
                            'name': 'get_weather',
                            'arguments': '{"city":"广州"}'
                        }
                    },
                    {
                        'id': 'call_a0ffc09b87d04644a0f2126a',
                        'index': 1,
                        'type': 'function',
                        'function': {
                            'name': 'get_weather',
                            'arguments': '{"city":"深圳"}'
                        }
                    },
                    {
                        'id': 'call_7fab9d2d0ae34ab6955c0932',
                        'index': 2,
                        'type': 'function',
                        'function': {
                            'name': 'get_weather',
                            'arguments': '{"city":"青岛"}'
                        }
                    }
                ]
            },
            'id': '2026050822210710146f6665fe4bfb',
            'model': 'glm-4.5-flash',
            'created': 1778250067,
            'finish_reason': 'tool_calls',
            'completion_tokens': 57,
            'total_tokens': 241,
            'cached_tokens': 43
        }
    }
    """
    if not ret_llm['suc']:
        print(f"Error: {ret_llm}")
        exit()
    tool_calls = ret_llm['data']['message'].get('tool_calls')
    if tool_calls:
        ret_act = await act(tool_calls,f_write=None)
        print(ret_act)
        """
    {
        'suc': True,
        'data': {
            'all_success': True,
            'delta_messages': [
                {
                    'role': 'tool',
                    'content': 
    '{"suc":true,"data":{"city":"广州","temperature":"27°C","condition":"雨天","humidity":"76%","wind_speed":"2 km/h"}}',
                    'tool_call_id': 'call_e1504117f9c842988a25234e'
                },
                {
                    'role': 'tool',
                    'content': 
    '{"suc":true,"data":{"city":"深圳","temperature":"16°C","condition":"晴天","humidity":"72%","wind_speed":"6 km/h"}}',
                    'tool_call_id': 'call_b58381ef048b417a8e2e4377'
                },
                {
                    'role': 'tool',
                    'content': 
    '{"suc":true,"data":{"city":"青岛","temperature":"15°C","condition":"阴天","humidity":"85%","wind_speed":"2 km/h"}}',
                    'tool_call_id': 'call_da61fd2951e84f4780ebf3c5'
                }
            ]
        }
    }
        """
    await close_pool()

if __name__ == '__main__':
    asyncio.run(main())
