import logging
import time
import os
from logging import handlers

# *用于记录日志的函数
# 日志名称包括训练开始的时间，具体参数在内部展示
class Logger(object):
    args = None
    filename = None
    logger = None
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射
 
    def __init__(self,args, level='info', when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.get_filename(args)
        self.logger = logging.getLogger(self.filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=self.filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)
    
    def get_filename(self, args):
        logs_path = args.logs_path
        # 初始化时要确保路径存在，不存在则创建
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        # 日志文件名称,精确到时分秒
        name_log_file_date = args.log_name + time.strftime(".%Y-%m-%d-%H:%M:%S", time.localtime()) + '.log'
        name_log_file_path = os.path.join(logs_path,name_log_file_date)
        
        self.filename = name_log_file_path
        return