import os

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from collections import Counter

import Summarizer
import summarizer_data_utils
import summarizer_model_utils

def main():

    file_path = './Data.csv'
    data = pd.read_csv(file_path)
    data.shape

    raw_texts = []
    raw_summaries = []

    for text, summary in zip(data.Text, data.Summary):
        if 100< len(text) < 2000:
            raw_texts.append(text)
            raw_summaries.append(summary)

    processed_texts, processed_summaries, words_counted = summarizer_data_utils.preprocess_texts_and_summaries(
        raw_texts,
        raw_summaries,
        keep_most=False
    )

    #for t,s in zip(processed_texts[:1], processed_summaries[:1]):
    #    print('Text\n:', t, '\n')
    #    print('Summary:\n', s, '\n\n\n')

    specials = ["<EOS>", "<SOS>","<PAD>","<UNK>"]
    word2ind, ind2word,  missing_words = summarizer_data_utils.create_word_inds_dicts(words_counted,
                                                                           specials = specials)
    print(len(word2ind), len(ind2word), len(missing_words))

    embed = hub.Module("https://tfhub.dev/google/Wiki-words-250/1")
    emb = embed([key for key in word2ind.keys()])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        embedding = sess.run(emb)

    np.save('./tf_hub_embedding_headlines.npy', embedding)

    converted_texts, unknown_words_in_texts = summarizer_data_utils.convert_to_inds(processed_texts,
                                                                                    word2ind,
                                                                                    eos = False)

    converted_summaries, unknown_words_in_summaries = summarizer_data_utils.convert_to_inds(processed_summaries,
                                                                                            word2ind,
                                                                                            eos = True,
                                                                                            sos = True)

    #print( summarizer_data_utils.convert_inds_to_text(converted_texts[0], ind2word),
    #       summarizer_data_utils.convert_inds_to_text(converted_summaries[0], ind2word))

    # model hyperparameters
    num_layers_encoder = 4
    num_layers_decoder = 4
    rnn_size_encoder = 250
    rnn_size_decoder = 250

    batch_size = 10
    epochs = 2
    clip = 2
    keep_probability = 0.8
    learning_rate = 0.0005
    max_lr=0.005
    learning_rate_decay_steps = 100
    learning_rate_decay = 0.90


    pretrained_embeddings_path = './tf_hub_embedding_headlines.npy'
    summary_dir = os.path.join('./tensorboard/headlines')

    use_cyclic_lr = True
    inference_targets=True

    # build graph and train the model
    summarizer_model_utils.reset_graph()
    summarizer = Summarizer.Summarizer(word2ind,
                                       ind2word,
                                       save_path='./models/headlines/my_model',
                                       mode='TRAIN',
                                       num_layers_encoder = num_layers_encoder,
                                       num_layers_decoder = num_layers_decoder,
                                       rnn_size_encoder = rnn_size_encoder,
                                       rnn_size_decoder = rnn_size_decoder,
                                       batch_size = batch_size,
                                       clip = clip,
                                       keep_probability = keep_probability,
                                       learning_rate = learning_rate,
                                       max_lr=max_lr,
                                       learning_rate_decay_steps = learning_rate_decay_steps,
                                       learning_rate_decay = learning_rate_decay,
                                       epochs = epochs,
                                       pretrained_embeddings_path = pretrained_embeddings_path,
                                       use_cyclic_lr = use_cyclic_lr,)
    #                                    summary_dir = summary_dir)

    summarizer.build_graph()
    summarizer.train(converted_texts,
                     converted_summaries)

    summarizer_model_utils.reset_graph()
    summarizer = Summarizer.Summarizer(word2ind,
                                       ind2word,
                                       './models/headlines/my_model',
                                       'INFER',
                                       num_layers_encoder = num_layers_encoder,
                                       num_layers_decoder = num_layers_decoder,
                                       batch_size = len(converted_texts[:5]),
                                       clip = clip,
                                       keep_probability = 1.0,
                                       learning_rate = 0.0,
                                       beam_width = 5,
                                       rnn_size_encoder = rnn_size_encoder,
                                       rnn_size_decoder = rnn_size_decoder,
                                       inference_targets = False,
                                       pretrained_embeddings_path = pretrained_embeddings_path)

    summarizer.build_graph()
    preds = summarizer.infer(inputs = converted_texts[:5],
                             restore_path =  './models/headlines/my_model',
                             targets = converted_summaries[:5])

    # show results
    summarizer_model_utils.sample_results(preds,
                                          ind2word,
                                          word2ind,
                                          converted_summaries[:5],
                                          converted_texts[:5])


if __name__ == "__main__":
    main()
