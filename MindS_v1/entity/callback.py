import numpy as np
from mindspore import nn
from mindspore import dataset as ds
from mindspore.train import Model, Callback
from mindspore import save_checkpoint
import mindspore as ms 
import os, stat, copy

class Traincallback(Callback):
    def __init__(self,loss_history):
        super(Traincallback, self).__init__()
        self.loss_history = loss_history

    def train_on_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()
        self.loss_history.append(loss)

        
        
# 测试并记录模型在测试集的loss和accuracy，每个epoch结束时进行模型测试并记录结果，跟踪并保存准确率最高的模型的网络参数
class Evalcallback(Callback):
    #保存accuracy最高的网络参数
    best_param = None
    
    def __init__(self, args,model, net,p_history, r_history, f_history,eval_data,test_data,dev_out,test_out):
        super(Evalcallback, self).__init__()
        self.args =args 
        self.model = model
        self.net = net
        self.p_history = p_history
        self.r_history = r_history
        self.f_history = f_history
        self.eval_data = eval_data
        self.test_data = test_data
        self.dev_out =  dev_out
        self.test_out = test_out
    # 每个epoch结束时执行
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        res = self.model.eval(self.eval_data, dataset_sink_mode=False)
        
        if len(self.f_history)==0:
            self.best_param = copy.deepcopy(cb_params.network)
        elif res['PRF'][2]>=max(self.f_history):
            self.best_param = copy.deepcopy(cb_params.network)

        self.p_history.append(res['PRF'][0])
        self.r_history.append(res['PRF'][1])
        self.f_history.append(res['PRF'][2])        
    
    # 训练结束后执行
    def end(self, run_context):
        # 保存最优网络参数
        if os.path.exists('best_param.ckpt'):
            os.chmod('best_param.ckpt', stat.S_IWRITE)

        save_path = os.path.join(self.args.output_dir,"best_param.ckpt")
        save_checkpoint(self.best_param, save_path)
        print(f"best_param.ckpt已保存至{save_path}")

        print("验证集测试结果")
        self.dev_out.eval()
        print("测试集测试结果")
        self.test_out.eval()
