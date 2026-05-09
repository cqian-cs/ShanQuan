"""本程序演示如何创建任务列表来并行调用工具函数"""
import asyncio
from shanquan import api, init_pool,close_pool, Batch
import time

@api(qps=None, limit=3)
def test_task(name): # <-- 注册一个同步函数（由进程池执行）
    import time
    time.sleep(1)
    return f'Test-{name}'

@api(qps=None, limit=3)
async def fast_task(name): # <-- 注册一个异步函数（由协程池执行）
    await asyncio.sleep(1)
    return f'Fast-{name}'

@api(qps=3, limit=None)
async def slow_task(name):  # <-- 注册一个异步函数（由协程池执行）
    await asyncio.sleep(1) 
    if name == 4:
        raise ValueError("没事，这是报错测试。Don't panic,  this is an error test.")
    return f'Slow-{name}'

async def main():
    init_pool(n_workers=4,global_limit=30) # <-- 初始化池配置(在所有@api注册后调用)
    payloads = [
        {'key':'T1', 'f':'test_task', 'kwargs':{'name':1}},
        {'key':'T2', 'f':'test_task', 'kwargs':{'name':2}},
        {'key':'T3', 'f':'test_task', 'kwargs':{'name':3}},
        {'key':'T4', 'f':'test_task', 'kwargs':{'name':4}},
        {'key':'T5', 'f':'test_task', 'kwargs':{'name':5}},
        {'key':'T6', 'f':'test_task', 'kwargs':{'name':6}},
        {'key':'F1', 'f':'fast_task', 'kwargs':{'name':1}},
        {'key':'F2', 'f':'fast_task', 'kwargs':{'name':2}},
        {'key':'F3', 'f':'fast_task', 'kwargs':{'name':3}},
        {'key':'F4', 'f':'fast_task', 'kwargs':{'name':4}},
        {'key':'F5', 'f':'fast_task', 'kwargs':{'name':5}},
        {'key':'F6', 'f':'fast_task', 'kwargs':{'name':6}},
        {'key':'S1', 'f':'slow_task', 'kwargs':{'name':1}},
        {'key':'S2', 'f':'slow_task', 'kwargs':{'name':2}},
        {'key':'S3', 'f':'slow_task', 'kwargs':{'name':3}},
        {'key':'S4', 'f':'slow_task', 'kwargs':{'name':4}},
        {'key':'S5', 'f':'slow_task', 'kwargs':{'name':5}},
        {'key':'S6', 'f':'slow_task', 'kwargs':{'name':6}},
    ]
    async with Batch(payloads) as futures:
        async for ret in futures: # <-- 哪个先做完，哪个先返回
            print(f"""{time.strftime('%H:%M:%S')}|{'✓' if ret['suc'] else '✗'}|{ret['key']}|{ret['f']}|{ret['kwargs']}|{ret['data']}""")
            if not ret['suc']: # <-- 遇到报错，跳出循环，自动取消剩余任务
                break
    await close_pool()
   
if __name__ == '__main__':
    asyncio.run(main())