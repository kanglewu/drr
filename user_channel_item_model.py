# -*- coding: UTF-8 -*-
import numpy
import tensorflow as tf
from absl import flags
from absl import app

max_seq_len = 10

# 1. 最基本的特征：

# Continuous columns. Wide和Deep组件都会用到。
# 'is_ka','is_standard','match_type','match_score','image_cnt','item_price','jfy_exp_cate_pv_14d'
is_ka = tf.feature_column.numeric_column('is_ka')
is_standard = tf.feature_column.numeric_column('is_standard')
match_score = tf.feature_column.numeric_column('match_score')
image_cnt = tf.feature_column.numeric_column('image_cnt')
item_price = tf.feature_column.numeric_column('item_price')
item_id = tf.feature_column.categorical_column_with_hash_bucket('item_id', hash_bucket_size=10000)
ids_columns = [item_id]
for i in range(max_seq_len):
    ids_columns.append(tf.feature_column.categorical_column_with_hash_bucket('sid_' + str(i), hash_bucket_size=10000))
ids_columns = tf.feature_column.shared_embedding_columns(ids_columns, dimension=8)

sf1_list = []
for i in range(max_seq_len):
    sf1_list.append(tf.feature_column.numeric_column('sf1_' + str(i)))
sf2_list = []
for i in range(max_seq_len):
    sf2_list.append(tf.feature_column.numeric_column('sf2_' + str(i)))
sf3_list = []
for i in range(max_seq_len):
    sf3_list.append(tf.feature_column.numeric_column('sf3_' + str(i)))

# 离散特征
match_type = tf.feature_column.categorical_column_with_vocabulary_list(
    'match_type', ['cntyhot', 'OLI2I', 'OLS2I', 'RTS2I', 'compl', 'RTI2I', 'RTB2I', 'OLC2I', 'RTC2I', 'OLB2I'])
#
# marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
#     'marital_status', [
#         'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
#         'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])
#
# relationship = tf.feature_column.categorical_column_with_vocabulary_list(
#     'relationship', [
#         'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
#         'Other-relative'])
#
# workclass = tf.feature_column.categorical_column_with_vocabulary_list(
#     'workclass', [
#         'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
#         'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

# 展示一下这个API
# occupation = tf.feature_column.categorical_column_with_hash_bucket(
#     'occupation', hash_bucket_size=1000
# )

# Transformations
is_ka_buckets = tf.feature_column.bucketized_column(
    is_ka, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
)
match_score_buckets = tf.feature_column.bucketized_column(
    match_score, boundaries=[0., 0.1, 0.2, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
)

# 2. The Wide Model: Linear Model with CrossedFeatureColumns
"""
The wide model is a linear model with a wide set of *sparse and crossed feature* columns
Wide部分用了一个规范化后的连续特征is_ka_buckets，其他的连续特征没有使用
"""
base_columns = [
    # 全是离散特征
    match_type,
    # marital_status, relationship, workclass, occupation,
    is_ka_buckets,
]

crossed_columns = [
    tf.feature_column.crossed_column(
        ['match_type', match_score_buckets], hash_bucket_size=1000)
]

# 3. The Deep Model: Neural Network with Embeddings
"""
1. Sparse Features -> Embedding vector -> 串联(Embedding vector, 连续特征) -> 输入到Hidden Layer
2. Embedding Values随机初始化
3. 另外一种处理离散特征的方法是：one-hot or multi-hot representation. 但是仅仅适用于维度较低的，embedding是更加通用的做法
4. embedding_column(embedding);indicator_column(multi-hot);
"""
deep_columns = [
    is_ka,
    is_standard,
    match_score,
    image_cnt,
    item_price,
    tf.feature_column.indicator_column(match_type),
    match_score_buckets
    # To show an example of embedding
    # tf.feature_column.embedding_column(occupation, dimension=8)
]

model_dir = './model/dnn'

# 4. Combine Wide & Deep


def my_model(features, labels, mode, params):
    """DNN with three hidden layers and learning_rate=0.1."""
    # Create three fully connected layers.
    import tensorflow as tf
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels=tf.reshape(labels,(-1,1))
        print "labels.get_shape=", labels.get_shape().as_list()
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, 1, activation=None)
    print "logits.get_shape=",logits.get_shape().as_list()
    # Compute predictions.
    prop=tf.nn.sigmoid(logits)
    predicted_classes = tf.greater_equal(prop,0.5)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.sigmoid(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss =tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))


    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


params = {
    'feature_columns': deep_columns,
    # Two hidden layers of 10 nodes each.
    'hidden_units': [10, 10],
    # The model must choose between 3 classes.
    'n_classes': 2,
}

model = tf.estimator.Estimator(
    model_fn=my_model,
    config=tf.estimator.RunConfig(
        model_dir=model_dir,
        save_checkpoints_steps=100
    ),
    params=params
)


# 5. Train & Evaluate
_CSV_COLUMNS = []
_CSV_COLUMN_DEFAULTS = []

for i in range(max_seq_len):
    _CSV_COLUMNS.append('sf1_' + str(i))
    _CSV_COLUMNS.append('sf2_' + str(i))
    _CSV_COLUMNS.append('sf3_' + str(i))
    _CSV_COLUMNS.append('sid_' + str(i))
    _CSV_COLUMN_DEFAULTS.extend([[0.], [0.], [0.], [0.]])

_CSV_COLUMNS.extend([
    'item_id', 'channel_id', 'user',
    'is_ka', 'is_standard', 'match_type',
    'match_score', 'image_cnt', 'item_price',
    'jfy_exp_cate_pv_14d', 'label'
])
print _CSV_COLUMNS
_CSV_COLUMN_DEFAULTS.extend([[0.], [0.], [0.], [0.], [0.], [''], [0.], [0.], [0.], [0.], [0.]])

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}




def input_fn(data_file, num_epochs, shuffle, batch_size):
    """为Estimator创建一个input function"""
    assert tf.gfile.Exists(data_file), "{0} not found.".format(data_file)

    def parse_csv(line):
        print("Parsing", data_file)
        # tf.decode_csv会把csv文件转换成很a list of Tensor,一列一个。record_defaults用于指明每一列的缺失值用什么填充
        columns = tf.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')
        return features, labels  # tf.equal(x, y) 返回一个bool类型Tensor， 表示x == y, element-wise

    dataset = tf.data.TextLineDataset(data_file) \
        .map(parse_csv, num_parallel_calls=5)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'] + _NUM_EXAMPLES['validation'])

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


# Train + Eval
train_epochs = 6
epochs_per_eval = 2
batch_size = 40
train_file = './data/channel.data'
test_file = './data/channel.test'

for n in range(train_epochs // epochs_per_eval):
    model.train(input_fn=lambda: input_fn(train_file, epochs_per_eval, True, batch_size))
    results = model.evaluate(input_fn=lambda: input_fn(
        test_file, 1, False, batch_size))

    # Display Eval results
    print("Results at epoch {0}".format((n + 1) * epochs_per_eval))
    print('-' * 30)

    for key in sorted(results):
        print("{0:20}: {1:.4f}".format(key, results[key]))

preds = model.predict(input_fn=lambda: input_fn(test_file, 1, False, batch_size), predict_keys=None)
with open("./data/pred.txt", "w") as fo:
    for prob in preds:
        print prob
        #fo.write("label=%f,score=%f\n" % (numpy.argmax(prob['class_ids']), numpy.max(prob['probabilities'])))
        #fo.write("%s\n" % (prob))

