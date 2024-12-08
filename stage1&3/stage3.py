import torch as t
from fate.arch import Context
from fate.ml.nn.hetero.hetero_nn import HeteroNNTrainerGuest, HeteroNNTrainerHost, TrainingArguments
from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost
from fate.ml.nn.model_zoo.hetero_nn_model import SSHEArgument, FedPassArgument, TopModelStrategyArguments, StdAggLayerArgument
'''
第一阶段，进行SSL：采用异质网络结构，将Top model作为平均汇聚处，Loss计算各方数据的差值，只训练bottom
第二阶段，本地进行训练，与第一阶段模型进行蒸馏
第三阶段，训练异质模型，bottom取Bl，先加载权重
'''
import pickle
from models.ctr_model import DNNFM, TopDNNFM, BottomDNNFM
from models.model import MLP2, projection_mlp
from fate.ml.nn.dataset.table import TableDataset
import logging
logger = logging.getLogger(__name__)
from sklearn.metrics import roc_auc_score, f1_score

def train(ctx: Context, 
          dataset = None, 
          model = None, 
          optimizer = None, 
          loss_func = None, 
          args: TrainingArguments = None, 
          ):
    scheduler = t.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20)
    scheduler = None

    if ctx.is_on_guest:
        ds = TableDataset(to_tensor=True)
        ds.load("./data/ctr_avazu2party/test_host1.csv") # 50
        trainer = HeteroNNTrainerGuest(ctx=ctx,
                                       model=model,
                                       train_set=dataset,
                                       optimizer=optimizer,
                                       loss_fn=loss_func,
                                       training_args=args,
                                       scheduler = scheduler,
                                       val_set=ds
                                       )
    else:
        ds = TableDataset(to_tensor=True)
        ds.load("./data/ctr_avazu2party/test_guest1.csv") # 50
        trainer = HeteroNNTrainerHost(ctx=ctx,
                                      model=model,
                                      train_set=dataset,
                                      optimizer=optimizer,
                                      training_args=args,
                                      scheduler = scheduler,
                                      val_set=ds
                                    )

    trainer.train()
    return trainer, ds


def predict(trainer, pred):
    # ds = TableDataset(to_tensor=True)
    # ds.load("./data/ctr_avazu2party/test_host1.csv") # 50
    # pred = trainer.predict(ds)
    # print(pred.predictions.shape)
    auc = roc_auc_score(pred.label_ids, pred.predictions)
    return auc

def get_setting(ctx, _dim = 512):
    with open("./data/ctr_avazu2party/feature_list.data", "rb") as f:
        feature_list = pickle.load(f)

    # prepare data
    if ctx.is_on_guest:
        ds = TableDataset(to_tensor=True)
        ds.load("./data/ctr_avazu2party/train_host.csv") # 70
        # ds.load("./data/ctr_avazu2party/train_host1.csv") # 70
        feature_list = feature_list[0]
        _t = TopDNNFM(hidden_dims=[_dim, _dim])
        _t.load_state_dict(t.load('./premodels/model_encoder_local_top-dnnfm-0.pth', weights_only=True, map_location=t.device('cpu')))
        _b = BottomDNNFM(feature_list, feature_list, dnn_hidden_units=[_dim])
        _b.load_state_dict(t.load('./premodels/model_encoder_local_bottom-dnnfm-0.pth', weights_only=True, map_location=t.device('cpu')))
        bottom_model = t.nn.Sequential(
            _b,
            _t,
        )
        top_model = t.nn.Sequential(
            # t.nn.Linear(int(_dim*2), 1),
            t.nn.Linear(int(_dim/4), 1),
            t.nn.Sigmoid()
        )
        model = HeteroNNModelGuest(
            top_model=top_model,
            bottom_model=bottom_model,
            # agglayer_arg=StdAggLayerArgument(merge_type='concat')
            # agglayer_arg=SSHEArgument(
            #     guest_in_features=_dim,
            #     host_in_features=_dim,
            #     out_features=_dim,
            #     # out_features=int(_dim*2),
            #     layer_lr=0.01
            # )
            agglayer_arg=FedPassArgument(
                merge_type='sum',
                layer_type='linear',
                in_channels_or_features=_dim,
                hidden_features=_dim,
                out_channels_or_features=int(_dim/4),
                passport_mode='single',
                passport_distribute='gaussian'
            )
        )

        # optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        loss = t.nn.BCELoss()

    else:
        ds = TableDataset(to_tensor=True)
        ds.load("./data/ctr_avazu2party/train_guest.csv") # 50
        # ds.load("./data/ctr_avazu2party/train_guest1.csv") # 50
        feature_list = feature_list[1]
        _t = TopDNNFM(hidden_dims=[_dim, _dim])
        _t.load_state_dict(t.load('./premodels/model_encoder_local_top-dnnfm-1.pth', weights_only=True, map_location=t.device('cpu')))
        _b = BottomDNNFM(feature_list, feature_list, dnn_hidden_units=[_dim])
        _b.load_state_dict(t.load('./premodels/model_encoder_local_bottom-dnnfm-1.pth', weights_only=True, map_location=t.device('cpu')))
        bottom_model = t.nn.Sequential(
            _b,
            _t,
        )

        model = HeteroNNModelHost(
            bottom_model=bottom_model,
            # agglayer_arg=StdAggLayerArgument(merge_type='concat')
            # agglayer_arg=SSHEArgument(
            #     guest_in_features=_dim,
            #     host_in_features=_dim,
            #     out_features=_dim,
            #     # out_features=int(_dim*2),
            #     layer_lr=0.01
            # )
            agglayer_arg=FedPassArgument(
                merge_type='sum',
                layer_type='linear',
                in_channels_or_features=_dim,
                hidden_features=int(_dim/2),
                out_channels_or_features=int(_dim/4),
                passport_mode='single',
                passport_distribute='gaussian'
            )
        )
        loss = None

    optimizer = t.optim.Adam(model.parameters(), lr=0.001)
    args = TrainingArguments(
        num_train_epochs=5,
        per_device_train_batch_size=256
    )

    return ds, model, optimizer, loss, args


def run(ctx):
    ds, model, optimizer, loss, args = get_setting(ctx)
    trainer, test_ds = train(ctx, ds, model, optimizer, loss, args)
    t.save(trainer.model.state_dict(), 'stage3_guest-rank('+str(ctx.rank)+').pt' if ctx.is_on_guest else 'stage3_host-rank('+str(ctx.rank)+').pt')
    # if not ctx.is_on_guest: t.save(trainer._top_model.state_dict(), 'stage3_host-top-rank('+str(ctx.rank)+').pt')
    pred = trainer.predict(test_ds)
    if ctx.is_on_guest:
        auc = predict(trainer, pred)
        logger.info(f'==============auc is {auc}================')


if __name__ == '__main__':
    from fate.arch.launchers.multiprocess_launcher import launch
    launch(run)
    # python stage3.py --parties guest:9999 host:10000 --log_level INFO