# Import relevant packages and modules
from util import *
import random
import tensorflow as tf

def mlPred():
    # Prompt for mode
    mode = input('mode (load / train)? ')

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
    raw_train = FNCData(file_train_instances, file_train_bodies)
    raw_test = FNCData(file_test_instances, file_test_bodies)
    n_train = len(raw_train.instances)


    # Process data sets
    train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
        pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
    feature_size = len(train_set[0])
    test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)


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

    # Load model
    if mode == 'load':
        with tf.compat.v1.Session() as sess:
            load_model(sess)
            print("Model loaded.")
            print("Now running predictions...")
            # Predict
            test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
            # run predictions
            test_pred = sess.run(predict, feed_dict=test_feed_dict)

            print("Test_pred:", test_pred)
            print("Preditions complete.")

    # Train model
    if mode == 'train':

        # Define optimiser
        opt_func = tf.compat.v1.train.AdamOptimizer(learn_rate)
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
        opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

        # Add ops to save and restore all the variables.
        saver = tf.compat.v1.train.Saver()

        # Perform training
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            for epoch in range(epochs):
                total_loss = 0
                indices = list(range(n_train))
                r.shuffle(indices)

                for i in range(n_train // batch_size_train):
                    batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
                    batch_features = [train_set[i] for i in batch_indices]
                    batch_stances = [train_stances[i] for i in batch_indices]

                    batch_feed_dict = {features_pl: batch_features, stances_pl: batch_stances, keep_prob_pl: train_keep_prob}
                    _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
                    total_loss += current_loss

            # save model to disk
            save_path = saver.save(sess, r"C:\Users\91846\OneDrive\Desktop\EDAI_2\ML_scripts\model.ckpt")
            print("Model saved in file: %s" % save_path)

            # Predict
            test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
            test_pred = sess.run(predict, feed_dict=test_feed_dict)


    # Save predictions
    save_predictions(test_pred, file_predictions)

    return test_pred

if __name__ == '__main__':
    mlPred()
