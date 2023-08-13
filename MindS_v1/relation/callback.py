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

class Evalcallback(Callback):
    best_param = None
    def __init__(self, args,model, p_history, r_history, f_history,eval_data,test_data):
        super(Evalcallback, self).__init__()
        self.args = args
        self.model = model
        self.p_history = p_history
        self.r_history = r_history
        self.f_history = f_history
        self.eval_data = eval_data
        self.test_data = test_data
    # 每个epoch结束时执行
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        res = self.model.eval(self.eval_data, dataset_sink_mode=False)
        
        if len(self.f_history)==0:
            self.best_param = copy.deepcopy(cb_params.network)
        elif res['PRF'][1]>=max(self.f_history):
            self.best_param = copy.deepcopy(cb_params.network)

        self.p_history.append(res['PRF'][0])
        self.r_history.append(res['PRF'][1])
        self.f_history.append(res['PRF'][2])        
    
    # 训练结束后执行
    def end(self, run_context):
        # 保存最优网络参数
        if os.path.exists('best_param_re.ckpt'):
            os.chmod('best_param_re.ckpt', stat.S_IWRITE)

        save_path = os.path.join(self.args.output_dir,"best_param_re.ckpt")
        save_checkpoint(self.best_param, save_path)
        cb_params = run_context.original_args()

        print("*"*50)
        print("最终测试结果")
        self.model.eval(self.eval_data, dataset_sink_mode=False)
        self.model.eval(self.test_data, dataset_sink_mode=False)


