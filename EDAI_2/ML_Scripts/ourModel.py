import util as util
import random
import tensorflow as tf
import time

def loadML():
    # Set file names
    file_train_instances = r"C:\Users\91846\OneDrive\Desktop\EDAI_2\CSVs\train_stances.csv"
    file_train_bodies = r"C:\Users\91846\OneDrive\Desktop\EDAI_2\CSVs\train_bodies.csv"
    file_test_instances = r"C:\Users\91846\OneDrive\Desktop\EDAI_2\CSVs\test_stances_unlabeled.csv"
    file_test_bodies = r"C:\Users\91846\OneDrive\Desktop\EDAI_2\CSVs\test_bodies.csv"
    file_predictions = r"C:\Users\91846\OneDrive\Desktop\EDAI_2\CSVs\predictions_test.csv"

    # Initialise hyperparameters
    r = random.Random()
    lim_unigram = 5000
    target_size = 4
    hidden_size = 100
    train_keep_prob = 0.6
    l2_alpha = 0.00001
    learn_rate = 0.01
    clip_ratio = 5
    batch_size_train = 500
    epochs = 90


    # Load data sets
    raw_train = util.FNCData(file_train_instances, file_train_bodies)
    raw_test = util.FNCData(file_test_instances, file_test_bodies)

    # Process data sets - THIS TAKES 17 SECONDS!
    train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = util.pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
    # feature_size = len(train_set[0])
    feature_size = 10001

    # Define model

    # Disable eager execution
    tf.compat.v1.disable_eager_execution()

    # Create placeholders
    features_pl = tf.compat.v1.placeholder(tf.float32, [None, feature_size], 'features')
    stances_pl = tf.compat.v1.placeholder(tf.int64, [None], 'stances')
    keep_prob_pl = tf.compat.v1.placeholder(tf.float32)

    # Infer batch size
    batch_size = tf.shape(features_pl)[0]

    # Define multi-layer perceptron
    hidden_layer = tf.nn.dropout(tf.nn.relu(tf.keras.layers.Dense(hidden_size)(features_pl)), rate=1-keep_prob_pl)
    logits_flat = tf.nn.dropout(tf.keras.layers.Dense(target_size)(hidden_layer), rate=1-keep_prob_pl)
    logits = tf.reshape(logits_flat, [batch_size, target_size])

    # Define L2 loss
    tf_vars = tf.compat.v1.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

    # Define overall loss
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(stances_pl, logits) + l2_loss)

    # Define prediction
    softmaxed_logits = tf.nn.softmax(logits)
    predict = tf.argmax(softmaxed_logits, 1)
    sess = tf.compat.v1.Session()
    util.load_model(sess)
    return sess, keep_prob_pl, predict, features_pl, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer
    
def runModel(sess, keep_prob_pl, predict, features_pl, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    start_time = time.time()
    print("Now running predictions...")
    # THIS is the info from Henry
    userClaims = r"C:\Users\91846\OneDrive\Desktop\EDAI_2\CSVs\claims.csv"
    userBodies = r"C:\Users\91846\OneDrive\Desktop\EDAI_2\CSVs\bodies.csv"
    # parse that info
    raw_test = util.FNCData(userClaims, userBodies)
    # need more stuff for this
    test_set = util.pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    # idk what this does really
    test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
    # run predictions
    try:
        # Run predictions
        test_pred = sess.run(predict, feed_dict=test_feed_dict)
        print("Predictions complete.")
    except Exception as e:
        print("An error occurred during prediction:", e)
        test_pred = None
    # timing
    print("ML 'runModel': --- %s seconds ---" % (time.time() - start_time))
    print("Preditions complete.")
    return test_pred
