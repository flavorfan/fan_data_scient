import os
import time
import numpy as np
import pandas as pd
import sys


from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names
from deepctr.models import DeepFM

from deepctr.estimator import DeepFMEstimator
from deepctr.estimator.inputs import input_fn_pandas

import os
import time
import logging

import tensorflow as tf
from tensorflow import feature_column as fc
from comm import ACTION_LIST, STAGE_END_DAY, FEA_COLUMN_LIST
from evaluation import uAUC, compute_weighted_score


flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_checkpoint_dir', './data/model', 'model dir')
flags.DEFINE_string('root_path', './data/', 'data dir')
flags.DEFINE_integer('batch_size', 128, 'batch_size')
flags.DEFINE_integer('embed_dim', 10, 'embed_dim')
flags.DEFINE_float('learning_rate', 0.1, 'learning_rate')
flags.DEFINE_float('embed_l2', None, 'embedding l2 reg')

SEED = 2021



class DeepCtrModel(object):

    def __init__(self, linear_feature_columns, dnn_feature_columns, stage, action):
        """
        :param linear_feature_columns: List of tensorflow feature_column
        :param dnn_feature_columns: List of tensorflow feature_column
        :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
        :param action: String. Including "read_comment"/"like"/"click_avatar"/"favorite"/"forward"/"comment"/"follow"
        """
        super(DeepCtrModel, self).__init__()
        self.num_epochs_dict = {"read_comment": 1, "like": 1, "click_avatar": 1, "favorite": 1, "forward": 1,
                                "comment": 1, "follow": 1}
        self.estimator = None
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.stage = stage
        self.action = action

        # model_dir
        if self.stage in ["evaluate", "offline_train"]:
            stage_path = "offline_train"
        else:
            stage_path = "online_train"
        self.model_dir = os.path.join(FLAGS.model_checkpoint_dir, stage_path, self.action + '_deepctr')

    def build_estimator(self):
        #         if self.stage in ["evaluate", "offline_train"]:
        #             stage = "offline_train"
        #         else:
        #             stage = "online_train"

        #         model_checkpoint_stage_dir = os.path.join(FLAGS.model_checkpoint_dir, stage, self.action + '_deepctr')
        #         print(f'model_checkpoint_stage_dir: {model_checkpoint_stage_dir}')

        if not os.path.exists(self.model_dir):
            # 如果模型目录不存在，则创建该目录
            os.makedirs(self.model_dir)
        elif self.stage in ["online_train", "offline_train"]:
            # 训练时如果模型目录已存在，则清空目录
            del_file(self.model_dir)

        #         optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999,
        #                                            epsilon=1)
        config = tf.estimator.RunConfig(model_dir=self.model_dir, tf_random_seed=SEED)
        self.estimator = DeepFMEstimator(
            self.linear_feature_columns,
            self.dnn_feature_columns,
            task='binary',
            model_dir=self.model_dir,
            config=config)

    def df_to_dataset(self, df, stage, action, shuffle=True, batch_size=128, num_epochs=1):
        '''
        把DataFrame转为tensorflow dataset
        :param df: pandas dataframe.
        :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
        :param action: String. Including "read_comment"/"like"/"click_avatar"/"favorite"/"forward"/"comment"/"follow"
        :param shuffle: Boolean.
        :param batch_size: Int. Size of each batch
        :param num_epochs: Int. Epochs num
        :return: tf.data.Dataset object.
        '''
        print(df.shape)
        print(df.columns)
        print("batch_size: ", batch_size)
        print("num_epochs: ", num_epochs)
        if stage != "submit":
            label = df[action]
            ds = tf.data.Dataset.from_tensor_slices((dict(df), label))
        else:
            ds = tf.data.Dataset.from_tensor_slices((dict(df)))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df), seed=SEED)
        ds = ds.batch(batch_size)
        if stage in ["online_train", "offline_train"]:
            ds = ds.repeat(num_epochs)
        return ds

    def input_fn_train(self, df, stage, action, num_epochs):
        return self.df_to_dataset(df, stage, action, shuffle=True, batch_size=FLAGS.batch_size,
                                  num_epochs=num_epochs)

    def input_fn_predict(self, df, stage, action):
        return self.df_to_dataset(df, stage, action, shuffle=False, batch_size=len(df), num_epochs=1)

    def train(self):
        """
        训练单个行为的模型
        """
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action=self.action,
                                                                       day=STAGE_END_DAY[self.stage])
        stage_dir = os.path.join(FLAGS.root_path, self.stage, file_name)
        df = pd.read_csv(stage_dir)
        self.estimator.train(
            input_fn=lambda: self.input_fn_train(df, self.stage, self.action, self.num_epochs_dict[self.action])
        )

    def evaluate(self):
        """
        评估单个行为的uAUC值
        """
        if self.stage in ["online_train", "offline_train"]:
            # 训练集，每个action一个文件
            action = self.action
        else:
            # 测试集，所有action在同一个文件
            action = "all"
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action=action,
                                                                       day=STAGE_END_DAY[self.stage])
        evaluate_dir = os.path.join(FLAGS.root_path, self.stage, file_name)
        df = pd.read_csv(evaluate_dir)
        userid_list = df['userid'].astype(str).tolist()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(df, self.stage, self.action)
        )
        # print(file_name)
        # print(predicts)
        predicts_df = pd.DataFrame.from_dict(predicts)
        # logits = predicts_df["logistic"].map(lambda x: x[0])
        print(predicts_df.head())
        logits = predicts_df["logits"].map(lambda x: x[0])
        labels = df[self.action].values
        uauc = uAUC(labels, logits, userid_list)
        return df[["userid", "feedid"]], logits, uauc

    def predict(self):
        '''
        预测单个行为的发生概率
        '''
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action="all",
                                                                       day=STAGE_END_DAY[self.stage])
        submit_dir = os.path.join(FLAGS.root_path, self.stage, file_name)
        df = pd.read_csv(submit_dir)
        t = time.time()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(df, self.stage, self.action)
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        # logits = predicts_df["logistic"].map(lambda x: x[0])
        logits = predicts_df["logits"].map(lambda x: x[0])
        # 计算2000条样本平均预测耗时（毫秒）
        ts = (time.time() - t) * 1000.0 / len(df) * 2000.0
        return df[["userid", "feedid"]], logits, ts




def get_feature_columns():
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    dense_features = ['videoplayseconds', 'device', 'read_commentsum',
                      'likesum', 'click_avatarsum', 'forwardsum', 'commentsum', 'followsum',
                      'favoritesum', 'read_commentsum_user', 'likesum_user',
                      'click_avatarsum_user', 'forwardsum_user', 'commentsum_user',
                      'followsum_user', 'favoritesum_user'
                      ]

    dnn_feature_columns = []
    linear_feature_columns = []

    for feat in dense_features:
        dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
        linear_feature_columns.append(tf.feature_column.numeric_column(feat))

    user_embedding = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket('userid', 40000, tf.int64), 10)
    feed_embedding = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket('feedid', 240000, tf.int64), 10)
    author_embedding = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket('authorid', 40000, tf.int64), 10)
    bgm_singer_embedding = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket('bgm_singer_id', 40000, tf.int64), 10)
    bgm_song_embedding = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket('bgm_song_id', 60000, tf.int64), 10)

    dnn_feature_columns.append(user_embedding)
    dnn_feature_columns.append(feed_embedding)
    dnn_feature_columns.append(author_embedding)
    dnn_feature_columns.append(bgm_singer_embedding)
    dnn_feature_columns.append(bgm_song_embedding)

    linear_feature_columns.append(tf.feature_column.categorical_column_with_hash_bucket('userid', 40000, tf.int64))
    linear_feature_columns.append(tf.feature_column.categorical_column_with_hash_bucket('feedid', 240000, tf.int64))
    linear_feature_columns.append(tf.feature_column.categorical_column_with_hash_bucket('authorid', 40000, tf.int64))
    linear_feature_columns.append(
        tf.feature_column.categorical_column_with_hash_bucket('bgm_singer_id', 40000, tf.int64))
    linear_feature_columns.append(tf.feature_column.categorical_column_with_hash_bucket('bgm_song_id', 60000, tf.int64))

    return dnn_feature_columns, linear_feature_columns


def del_file(path):
    '''
    删除path目录下的所有内容
    '''
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            print("del: ", c_path)
            os.remove(c_path)


def main(argv):
    t = time.time()
    dnn_feature_columns, linear_feature_columns = get_feature_columns()
    stage = argv[1]
    print('Stage: %s'%stage)
    eval_dict = {}
    predict_dict = {}
    predict_time_cost = {}
    ids = None
    for action in ACTION_LIST:
        print("Action:", action)
        # Change only
        model = DeepCtrModel(linear_feature_columns, dnn_feature_columns, stage, action)
        model.build_estimator()

        if stage in ["online_train", "offline_train"]:
            # 训练 并评估
            model.train()
            ids, logits, action_uauc = model.evaluate()
            eval_dict[action] = action_uauc

        if stage == "evaluate":
            # 评估线下测试集结果，计算单个行为的uAUC值，并保存预测结果
            ids, logits, action_uauc = model.evaluate()
            eval_dict[action] = action_uauc
            predict_dict[action] = logits

        if stage == "submit":
            # 预测线上测试集结果，保存预测结果
            ids, logits, ts = model.predict()
            predict_time_cost[action] = ts
            predict_dict[action] = logits

    if stage in ["evaluate", "offline_train", "online_train"]:
        # 计算所有行为的加权uAUC
        print(eval_dict)
        weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                       "comment": 1, "follow": 1}
        weight_auc = compute_weighted_score(eval_dict, weight_dict)
        print("Weighted uAUC: ", weight_auc)


    if stage in ["evaluate", "submit"]:
        # 保存所有行为的预测结果，生成submit文件
        actions = pd.DataFrame.from_dict(predict_dict)
        print("Actions:", actions)
        ids[["userid", "feedid"]] = ids[["userid", "feedid"]].astype(int)
        res = pd.concat([ids, actions], sort=False, axis=1)
        # 写文件
        file_name = "submit_" + 'deepctr_' + str(int(time.time())) +".csv"
        submit_file = os.path.join(FLAGS.root_path, stage, file_name)
        print('Save to: %s'%submit_file)
        res.to_csv(submit_file, index=False)

    if stage == "submit":
        print('不同目标行为2000条样本平均预测耗时（毫秒）：')
        print(predict_time_cost)
        print('单个目标行为2000条样本平均预测耗时（毫秒）：')
        print(np.mean([v for v in predict_time_cost.values()]))
    print('Time cost: %.2f s'%(time.time()-t))

if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)
    print("Total arguments passed:", n)
    # Arguments passed
    print("\nName of Python script:", sys.argv[0])

    print("\nArguments passed:", end=" ")
    for i in range(1, n):
        print(sys.argv[i], end=" ")


    main(sys.argv)
