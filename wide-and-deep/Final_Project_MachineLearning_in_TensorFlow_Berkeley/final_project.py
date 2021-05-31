import numpy as np
from pandas import read_csv
import pandas as pd
# import tensorflow as tf  # use the tf1 not tf2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
import os
from absl import flags
from sklearn.metrics import confusion_matrix

# set learning rate
learning_rate = 0.0000000001

# set file path
dataset = "epi_r.csv"
datapath = "documents/ml_tf/final_project"
# datapath = "documents/ml_tf/final_project/"

def download_and_clean_file(dataset, datapath):
#     set working directory and download csv file; save as data
    os.chdir(datapath)
    data = read_csv(dataset)

    # clean and order file
    data.dropna(inplace=True) # remove NAs
    data.reset_index(inplace=True) # resetting index
    data.drop('index', axis='columns', inplace=True)

    return(data)


tf.reset_default_graph()
g = tf.Graph()

with g.as_default():
    def input_columns(dataset, datapath):
        data = download_and_clean_file(dataset, datapath)

        # Preparing columns
        data.head(1)  # reads the first line
        rows = len(data)  # counts the number of rows in the file
        shape = data.shape  # shows the shape

        # set empty feature and label
        features = []
        label = []

        rating = data['rating']  # label

        # features
        calories = data['calories']
        protein = data['protein']
        fat = data['fat']
        sodium = data['sodium']

        dessert = data['dessert']  # possible label #2
        peanut_free = data['peanut free']
        soy_free = data['soy free']
        tree_nut_free = data['tree nut free']
        vegetarian = data['vegetarian']
        gourmet = data['gourmet']
        kosher = data['kosher']
        pescatarian = data['pescatarian']
        quick_easy = data['quick & easy']
        wheat_gluten_free = data['wheat/gluten-free']
        bake = data['bake']
        summer = data['summer']
        dairy_free = data['dairy free']
        side = data['side']
        no_sugar_added = data['no sugar added']
        winter = data['winter']
        fall = data['fall']
        dinner = data['dinner']
        sugar_conscious = data['sugar conscious']
        healthy = data['healthy']
        kidney_friendly = data['kidney friendly']
        onion = data['onion']
        tomato = data['tomato']
        vegetable = data['vegetable']
        milk_cream = data['milk/cream']
        fruit = data['fruit']
        vegan = data['vegan']
        kid_friendly = data['kid-friendly']
        egg = data['egg']
        spring = data['spring']
        herb = data['herb']
        garlic = data['garlic']
        salad = data['salad']
        dairy = data['dairy']
        thanksgiving = data['thanksgiving']
        appetizer = data['appetizer']
        lunch = data['lunch']
        cheese = data['cheese']
        chicken = data['chicken']
        roast = data['roast']
        no_cook = data['no-cook']
        soup_stew = data['soup/stew']
        cocktail_party = data['cocktail party']
        ginger = data['ginger']
        potato = data['potato']
        chill = data['chill']
        grill_barbecue = data['grill/barbecue']
        lemon = data['lemon']
        drink = data['drink']
        sauce = data['sauce']
        low_cal = data['low cal']
        christmas = data['christmas']
        high_fiber = data['high fiber']
        food_processor = data['food processor']

        for k in range(rows):  # use loop to put it in the expected format
            # appending features
            features.append([calories[k], protein[k], fat[k], sodium[k], peanut_free[k], soy_free[k], tree_nut_free[k],
                             vegetarian[k], gourmet[k], kosher[k], pescatarian[k], quick_easy[k], wheat_gluten_free[k],
                             bake[k], summer[k],
                             dairy_free[k], side[k], no_sugar_added[k], winter[k], fall[k], dinner[k],
                             sugar_conscious[k],
                             healthy[k], kidney_friendly[k], onion[k], tomato[k], vegetable[k], milk_cream[k], fruit[k],
                             vegan[k],
                             kid_friendly[k], egg[k], spring[k], herb[k], garlic[k], salad[k], dairy[k],
                             thanksgiving[k], appetizer[k], lunch[k],
                             cheese[k], chicken[k], roast[k], no_cook[k], soup_stew[k], cocktail_party[k], ginger[k],
                             potato[k],
                             chill[k], grill_barbecue[k], lemon[k], drink[k], sauce[k], low_cal[k], christmas[k],
                             high_fiber[k], food_processor[k]])

            # creating classes for labels into 5 buckets, i.e less than 1 star rating is class 0, between 1 & 2 star rating is class 1
            label.append([dessert[k]])

        return np.array(label), np.array(features)


    def input_data(dataset, datapath):
        # splitting data 70% train 30% test
        label, features = input_columns(dataset, datapath)
        train_len = int(len(features) * 0.7)
        train_label, train_data = label[:train_len], features[:train_len]
        test_label, test_data = label[train_len:], features[train_len:]
        return train_label, train_data, test_label, test_data


    def build_estimator(model_type, model_dir):
        # Build 3 layer DNN with 100, 75, 50, 25 units respectively.

        hidden_units = [100, 75, 50, 25]

        feature_columns = [tf.feature_column.numeric_column("x", shape=[57])]
        deep_columns = [tf.feature_column.numeric_column("deep", shape=[4])]
        wide_columns = [tf.feature_column.numeric_column("wide", shape=[53])]

        if model_type == 'wide':
            # return tf.estimator.LinearClassifier(feature_columns=feature_columns,
            #                                      n_classes=5,
            #                                      model_dir=model_dir)
            return tf.estimator.LinearRegressor (feature_columns=feature_columns,
                                             # n_classes=5,
                                             model_dir=model_dir)

        elif model_type == 'deep':
            return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                              n_classes=5,
                                              hidden_units=hidden_units,
                                              model_dir=model_dir)

        else:
            return tf.estimator.DNNLinearCombinedClassifier(
                n_classes=5,
                linear_feature_columns=wide_columns,
                dnn_feature_columns=deep_columns,
                dnn_hidden_units=hidden_units,
                model_dir=model_dir)


    def train_input(model_type, train_label, train_data):

        # train_label, train_data, test_label, test_data = input_data(dataset, datapath)
        if model_type == 'wide' or model_type == 'deep':
            return tf.estimator.inputs.numpy_input_fn(
                x={"x": np.array(train_data)},
                y=np.array(train_label),
                num_epochs=None,
                shuffle=True)

        else:
            # Define the training inputs
            return tf.estimator.inputs.numpy_input_fn(
                x={"deep": np.array(train_data[:, 0:4]), "wide": np.array(train_data[:, 4:])},
                y=np.array(train_label),
                num_epochs=None,
                shuffle=True)


    def test_input(model_type):

        train_label, train_data, test_label, test_data = input_data(dataset, datapath)
        if model_type == 'wide' or model_type == 'deep':
            return tf.estimator.inputs.numpy_input_fn(
                x={"x": np.array(test_data)},
                y=np.array(test_label),
                num_epochs=1,
                shuffle=True)

        else:
            # Define the training inputs
            return tf.estimator.inputs.numpy_input_fn(
                x={"deep": np.array(test_data[:, 0:4]), "wide": np.array(test_data[:, 4:])},
                y=np.array(test_label),
                num_epochs=1,
                shuffle=True)


    def train_model(model_type, model_dir):

        train_label, train_data, test_label, test_data = input_data(dataset, datapath)
        classifier = build_estimator(model_type, model_dir)
        train_input_fn = train_input(model_type, train_label, train_data)

        classifier.train(input_fn=train_input_fn, steps=4000)


    def test_model_accuracy(model_type, model_dir):

        train_model(model_type, model_dir)
        train_label, train_data, test_label, test_data = input_data(dataset, datapath)
        classifier = build_estimator(model_type, model_dir)
        test_input_fn = test_input(model_type)

        # getting accuracy score
        accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
        print("\nTest Accuracy: {0:f}\n".format(accuracy_score), "model_type: ", model_type)

        # getting predictions and plotting confusion matrix
        prediction = classifier.predict(input_fn=test_input_fn)
        predicted_classes = [p["class_ids"] for p in prediction]

        # prepare loop to ensure format align
        row = len(predicted_classes)
        prediction_array = []

        for k in range(row):  # use loop to put it in the expected format
            # features
            prediction_array.append(int(predicted_classes[k]))

        # define labels
        labels = ['not dessert', 'dessert']
        cm = confusion_matrix(test_label, prediction_array)
        # print(cm)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


    init = tf.global_variables_initializer()

# Run types of model and set model_dir to run in Tensorboard

with tf.Session(graph=g) as sess:
    sess.graph.as_graph_def()

    # modify model type: [wide, deep, wide+deep]
    test_model_accuracy(model_type="wide", model_dir='fv4')
    #     test_model_accuracy(model_type = "deep", model_dir = 'fv5')
    #     test_model_accuracy(model_type = "wide + deep", model_dir = 'fv6')

    merged_summaries = tf.summary.merge_all()

    # initializations
    sess.run(init)

