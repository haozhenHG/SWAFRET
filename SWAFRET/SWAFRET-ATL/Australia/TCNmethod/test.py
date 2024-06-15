from datetime import datetime

# 获取当前时间
current_time = datetime.now().strftime('%Y-%m-%d~%H-%M-%S')
# formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
# 打印当前时间
print("当前时间：", current_time)
