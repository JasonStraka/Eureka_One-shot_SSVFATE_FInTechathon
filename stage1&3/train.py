import torch as t
from torch import nn
import logging
logger = logging.getLogger(__name__)

'''
第一阶段，进行SSL：采用异质网络结构，将Top model作为平均汇聚处，Loss计算各方数据的差值，只训练bottom
第二阶段，本地进行训练，与第一阶段模型进行蒸馏
第三阶段，训练异质模型，bottom取Bl，先加载权重
'''
######################################################
# 自监督训练类覆写
######################################################
from ssvfate import HeteroNNTrainerGuestSSL, HeteroNNTrainerHost, HeteroNNModelGuestSSL, HeteroNNModelHostSSL, SSLAggLayerArgument, TopSsl

######################################################
# 自监督训练流程覆写
######################################################
from models.ctr_model import DNNFM
from models.model import MLP2, projection_mlp
from ssl_utils import NegCosineSimilarityLoss, NTXentLoss
from fate.arch import Context
from fate.ml.nn.trainer.trainer_base import HeteroTrainerBase, TrainingArguments
from fate.ml.nn.dataset.table import TableDataset
import pickle

def train(ctx: Context, 
          dataset = None, 
          model = None, 
          optimizer = None, 
          loss_func = None, 
          loss_fn_ssl = None, 
          args: TrainingArguments = None, 
          ):
    scheduler = t.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, )
    # scheduler = None

    if ctx.is_on_guest:
        trainer = HeteroNNTrainerGuestSSL(ctx=ctx,
                                       model=model,
                                       train_set=dataset,
                                       optimizer=optimizer,
                                       loss_fn=loss_func,
                                       loss_fn_ssl=loss_fn_ssl,
                                       training_args=args,
                                       scheduler = scheduler
                                       )
    else:
        trainer = HeteroNNTrainerHost(ctx=ctx,
                                      model=model,
                                      train_set=dataset,
                                      optimizer=optimizer,
                                      training_args=args,
                                      scheduler = scheduler
                                    )

    trainer.train()
    return trainer

def predict(trainer, dataset):
    return trainer.predict(dataset)


def get_setting(ctx, _dim = 512):
    with open("./data/ctr_avazu2party/feature_list.data", "rb") as f:
        feature_list = pickle.load(f)
    # prepare data
    if ctx.is_on_guest:
        ds = TableDataset(to_tensor=True)
        # ds.load("./data/ctr_avazu2party/train_guest.csv") # 50
        ds.load("./data/ctr_avazu2party/train_host.csv") # 70
        feature_list = feature_list[0]
        bottom_model = nn.Sequential(
                            DNNFM(feature_list, feature_list, dnn_hidden_units=[_dim, _dim]),
                            projection_mlp(_dim, _dim, _dim, 3)
                        )
        # top_model = t.nn.Sequential(
        #     t.nn.Linear(_dim, 1),
        #     t.nn.Sigmoid()
        # )
        top_model = TopSsl()
        model = HeteroNNModelGuestSSL(
            top_model=top_model,
            bottom_model=bottom_model,
            agglayer_arg=SSLAggLayerArgument()
        )

        # optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        # loss = t.nn.BCELoss()
        loss = None
        loss_fn_ssl = NegCosineSimilarityLoss()
        # loss_fn_ssl = NTXentLoss(memory_bank_size=2048)

    else:
        ds = TableDataset(to_tensor=True)
        # ds.load("./data/ctr_avazu2party/train_host.csv") # 70
        ds.load("./data/ctr_avazu2party/train_guest.csv") # 50
        feature_list = feature_list[1]
        bottom_model = nn.Sequential(
                            DNNFM(feature_list, feature_list, dnn_hidden_units=[_dim, _dim]),
                            projection_mlp(_dim, _dim, _dim, 3)
                        )
        model = HeteroNNModelHostSSL(
            bottom_model=bottom_model,
            agglayer_arg=SSLAggLayerArgument()
        )
        loss = None
        loss_fn_ssl = None

    args = TrainingArguments(
        num_train_epochs=10,
        per_device_train_batch_size=512,
    )
    optimizer = t.optim.Adam(model.parameters(), lr=0.01)
    return ds, model, optimizer, loss, loss_fn_ssl, args

def run(ctx):
    ds, model, optimizer, loss, loss_fn_ssl, args = get_setting(ctx)
    trainer = train(ctx, ds, model, optimizer, loss, loss_fn_ssl, args)
    # 训练并保存各方子监督预训练结果
    t.save(trainer.model.state_dict(), 'guest-rank('+str(ctx.rank)+').pt' if ctx.is_on_guest else 'host-rank('+str(ctx.rank)+').pt')
    # pred = predict(trainer, ds)
    # if ctx.is_on_guest:
    #     # print("pred:", pred)
    #     # compute auc here
    #     from sklearn.metrics import roc_auc_score
    #     # print('auc is')
    #     # print(roc_auc_score(pred.label_ids, pred.predictions))
    #     # top_out, x_g, x_h, _mean
    #     top_out, top_out_h, x_g, x_h, _mean = pred.predictions
    #     print('auc is', roc_auc_score(pred.label_ids, top_out))

# python train.py --parties guest:9999 host:10000 --log_level INFO
# python train.py --parties guest:9999 host:10000 host:10001 --log_level INFO
if __name__ == '__main__':
    from fate.arch.launchers.multiprocess_launcher import launch
    launch(run)

# https://github.com/FederatedAI/FATE/blob/feature-2.0.0-rc-doc_updatec/doc/2.0/fate/ml/hetero_nn_tutorial.md