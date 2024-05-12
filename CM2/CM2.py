import pdb
import os

from . import constants
from .modeling_CM2 import CM2Classifier, CM2FeatureExtractor, CM2FeatureProcessor, CM2Regression
from .modeling_CM2 import CM2ForCL, CM2ForMask, CM2ForSup
from .modeling_CM2 import CM2InputEncoder, CM2Model
from .evaluator import predict, evaluate
from .trainer import Trainer
from .trainer_ds import Trainer_ds
from .trainer_utils import CM2CollatorForCL
from .trainer_utils import random_seed

dev = 'cuda'
def build_classifier(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    feature_extractor=None,
    num_class=2,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    vocab_freeze=False,
    use_bert=True,
    device=dev,
    checkpoint=None,
    **kwargs) -> CM2Classifier:
    
    model = CM2Classifier(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        feature_extractor = feature_extractor,
        num_class=num_class,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        activation=activation,
        vocab_freeze=vocab_freeze,
        use_bert=use_bert,
        device=device,
        **kwargs,
    )
    
    if checkpoint is not None:
        model.load(checkpoint)

    return model


def build_regression(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    feature_extractor=None,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    vocab_freeze=False,
    device=dev,
    checkpoint=None,
    **kwargs) ->  CM2Regression:
    
    model =  CM2Regression(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        feature_extractor = feature_extractor,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        activation=activation,
        vocab_freeze=vocab_freeze,
        device=device,
        **kwargs,
    )
    
    if checkpoint is not None:
        model.load(checkpoint)

    return model


def build_contrastive_learner(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    projection_dim=128,
    num_partition=3,
    overlap_ratio=0.5,
    supervised=True,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    device=dev,
    checkpoint=None,
    ignore_duplicate_cols=True,
    vocab_freeze=True,
    **kwargs,
    ): 
    model = CM2ForCL(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        num_partition= num_partition,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        supervised=supervised,
        ffn_dim=ffn_dim,
        projection_dim=projection_dim,
        overlap_ratio=overlap_ratio,
        activation=activation,
        vocab_freeze=vocab_freeze,
        device=device,
    )
    if checkpoint is not None:
        model.load(checkpoint)
    
    # build collate function for contrastive learning
    collate_fn = CM2CollatorForCL(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        overlap_ratio=overlap_ratio,
        num_partition=num_partition,
        ignore_duplicate_cols=ignore_duplicate_cols
    )
    if checkpoint is not None:
        collate_fn.feature_extractor.load(os.path.join(checkpoint, constants.EXTRACTOR_STATE_DIR))

    return model, collate_fn


def build_mask_features_learner(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    mlm_probability=0.15,
    projection_dim=128,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    vocab_freeze=True,
    pretrain_table_num=0,
    device=dev,
    checkpoint=None,
    **kwargs,
    ): 
    model = CM2ForMask(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        mlm_probability=mlm_probability,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        projection_dim=projection_dim,
        activation=activation,
        vocab_freeze=vocab_freeze,
        pretrain_table_num=pretrain_table_num,
        device=device,
    )
    if checkpoint is not None:
        model.load(checkpoint)

    return model

def build_vanilla_sup_learner(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    feature_extractor=None,
    num_class=None,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    vocab_freeze=False,
    use_bert=True,
    device=dev,
    checkpoint=None,
    **kwargs) -> CM2ForSup:
    
    model = CM2ForSup(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        feature_extractor = feature_extractor,
        num_class=num_class,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        activation=activation,
        vocab_freeze=vocab_freeze,
        use_bert=use_bert,
        device=device,
        **kwargs,
    )
    
    if checkpoint is not None:
        model.load(checkpoint)

    return model


def train(
    model, 
    trainset, 
    valset=None,
    cmd_args=None,
    num_epoch=10,
    batch_size=64,
    eval_batch_size=256,
    lr=1e-4,
    weight_decay=0,
    patience=5,
    warmup_ratio=None,
    warmup_steps=None,
    eval_metric='auc',
    output_dir='./ckpt',
    collate_fn=None,
    num_workers=0,
    balance_sample=False,
    load_best_at_last=True,
    ignore_duplicate_cols=True,
    eval_less_is_better=False,
    flag=0,
    regression_task=False,
    train_method='normal',
    device=None,
    data_weight=None,
    **kwargs,
    ):
    if isinstance(trainset, tuple): trainset = [trainset]

    train_args = {
        'num_epoch': num_epoch,
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'lr': lr,
        'weight_decay':weight_decay,
        'patience':patience,
        'warmup_ratio':warmup_ratio,
        'warmup_steps':warmup_steps,
        'eval_metric':eval_metric,
        'output_dir':output_dir,
        'collate_fn':collate_fn,
        'num_workers':num_workers,
        'balance_sample':balance_sample,
        'load_best_at_last':load_best_at_last,
        'ignore_duplicate_cols':ignore_duplicate_cols,
        'eval_less_is_better':eval_less_is_better,
        'flag':flag,
        'regression_task':regression_task,
        'device':device,
        'data_weight':data_weight,
    }

    if train_method=='deepspeed':
        trainer = Trainer_ds(
            model,
            trainset,
            valset,
            cmd_args,
            **train_args,
        )
    else:
        trainer = Trainer(
            model,
            trainset,
            valset,
            **train_args,
        )
    return trainer
