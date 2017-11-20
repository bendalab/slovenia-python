import concurrent.futures
import time

def wait_a_bit():
    time.sleep(5)
    print('Hello')

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
for idx in range(10):
    executor.submit(wait_a_bit)
