#!/bin/bash
function run_ft {
    ft=$1
    ds=$2
    suf=$3
    # python run2.py --train_features_to_sentiment_model --tweet_to_features_model fasttext_word_embedding --tweet_to_features_model_file_prefix $ft --features_to_sentiment_model lstm_features_to_sentiment --dataset $ds --features_to_sentiment_model_file_prefix trained_models/"$ds"_lstm_"$suf"_256_64 --features_to_sentiment_model_params '{"lstm_layer_sizes": [256, 64], "batch_size": 256}' 
    python run2.py --train_features_to_sentiment_model --tweet_to_features_model fasttext_word_embedding --tweet_to_features_model_file_prefix $ft --features_to_sentiment_model gru_features_to_sentiment --dataset $ds --features_to_sentiment_model_file_prefix trained_models/"$ds"_gru_"$suf"_256_64 --features_to_sentiment_model_params '{"gru_layer_sizes": [256, 64], "batch_size": 256}' 
    python run2.py --train_features_to_sentiment_model --tweet_to_features_model fasttext_word_embedding --tweet_to_features_model_file_prefix $ft --features_to_sentiment_model clstm_features_to_sentiment --dataset $ds --features_to_sentiment_model_file_prefix trained_models/"$ds"_cnn_lstm_"$suf" --features_to_sentiment_model_params '{}'
    python run2.py --train_features_to_sentiment_model --tweet_to_features_model fasttext_word_embedding --tweet_to_features_model_file_prefix $ft --features_to_sentiment_model cnn_features_to_sentiment --dataset $ds --features_to_sentiment_model_file_prefix trained_models/"$ds"_cnn_"$suf" --features_to_sentiment_model_params '{}'
}

function run_full {
    ds=$1
    python run2.py --train_full_model --full_model full_lstm --dataset $ds --full_model_file_prefix trained_models/"$ds"_lstm_full_256_64 --full_model_params '{"lstm_layer_sizes": [256, 64]}' 
    python run2.py --train_full_model --full_model full_lstm --dataset $ds --full_model_file_prefix trained_models/"$ds"_lstm_full_512_128 --full_model_params '{"lstm_layer_sizes": [512, 128]}' 
    python run2.py --train_full_model --full_model full_clstm --dataset $ds --full_model_file_prefix trained_models/"$ds"_cnn_lstm --full_model_params '{}'
    python run2.py --train_full_model --full_model full_gru --dataset $ds --full_model_file_prefix trained_models/"$ds"_gru_full_256_64 --full_model_params '{"gru_layer_sizes": [256, 64]}'
    python run2.py --train_full_model --full_model full_cnn --dataset $ds --full_model_file_prefix trained_models/"$ds"_cnn --full_model_params '{}'
}

function run_ft_fix {
    ft=$1
    ds=$2
    suf=$3
    # python run2.py --train_features_to_sentiment_model --tweet_to_features_model fasttext_word_embedding --tweet_to_features_model_file_prefix $ft --features_to_sentiment_model gru_features_to_sentiment --dataset $ds --features_to_sentiment_model_file_prefix trained_models/"$ds"_gru_"$suf"_256_64 --features_to_sentiment_model_params '{"gru_layer_sizes": [256, 64], "batch_size": 256}' 
    python run2.py --train_features_to_sentiment_model --tweet_to_features_model fasttext_word_embedding --tweet_to_features_model_file_prefix $ft --features_to_sentiment_model cnn_features_to_sentiment --dataset $ds --features_to_sentiment_model_file_prefix trained_models/"$ds"_cnn_"$suf" --features_to_sentiment_model_params '{}'
    # python run2.py --train_features_to_sentiment_model --tweet_to_features_model fasttext_word_embedding --tweet_to_features_model_file_prefix $ft --features_to_sentiment_model lstm_features_to_sentiment --dataset $ds --features_to_sentiment_model_file_prefix trained_models/"$ds"_lstm_"$suf"_512_128 --features_to_sentiment_model_params '{"lstm_layer_sizes": [512, 128], "batch_size": 256}' 
}

function run_ft_fix_cnn {
    ft=$1
    ds=$2
    suf=$3
    python run2.py --train_features_to_sentiment_model --tweet_to_features_model fasttext_word_embedding --tweet_to_features_model_file_prefix $ft --features_to_sentiment_model cnn_features_to_sentiment --dataset $ds --features_to_sentiment_model_file_prefix trained_models/"$ds"_cnn_"$suf" --features_to_sentiment_model_params '{}'
}

function run_full_fix_2 {
    ds=$1
    python run2.py --train_full_model --full_model full_lstm --dataset $ds --full_model_file_prefix trained_models/"$ds"_lstm_full_512_128 --full_model_params '{"lstm_layer_sizes": [512, 128]}' 
    python run2.py --train_full_model --full_model full_gru --dataset $ds --full_model_file_prefix trained_models/"$ds"_gru_full_256_64 --full_model_params '{"gru_layer_sizes": [256, 64]}'
}

function run_full_fix_3 {
    ds=$1
    python run2.py --train_full_model --full_model full_lstm --dataset $ds --full_model_file_prefix trained_models/"$ds"_lstm_full_512_128 --full_model_params '{"lstm_layer_sizes": [512, 128]}' 
}

function run_full_percentage {
    ds=$1
    # for share in {"0.01","0.025","0.05","0.1","0.25","0.5","0.75","1.0"}
    for share in {"0.5","0.75","1.0"}
    do
        python run2.py --train_full_model --full_model full_lstm --dataset $ds/$share --full_model_file_prefix trained_models/"$ds"_"$share"_lstm_full_256_64 --full_model_params '{"lstm_layer_sizes": [256, 64]}' 
    done
}

function run_doc2vec_ru { 
    ds=$1
    python run2.py --train_features_to_sentiment_model --tweet_to_features_model doc2vec_embedding --tweet_to_features_model_file_prefix trained_models/mydoc2vec/ru_doc2vec --features_to_sentiment_model dense_features_to_sentiment --dataset my_rated --features_to_sentiment_model_file_prefix trained_models/"$ds"_dense_doc2vec --features_to_sentiment_model_params '{}'
}

# run_ft pretrained_models/wiki.ru.bin my_rated ru_ftwiki
# run_ft pretrained_models/wiki.en.bin sentiment140 en_ftwiki
# run_full sentiment140
# run_full my_rated
# run_full my_eng_plus_sentiment140
# run_ft ../fastText/tweet_fasttext_model_300_epoch30.vec.bin my_rated ru_fttweet30
# run_ft ../fastText/my_en_tweet_fasttext_300_epoch30.bin my_eng_plus_sentiment140 my_eng_plus_sentiment140_fttweet30
# run_full mokoron
# run_full_percentage my_rated
# run_doc2vec_ru

# run_full_fix_2 sentiment140
# run_full_fix_3 mokoron
# run_full_fix_3 my_rated
# run_full_fix_3 my_eng_plus_sentiment140
run_ft ../fastText/my_en_tweet_fasttext_300_epoch30.bin my_eng_plus_sentiment140 my_eng_plus_sentiment140_fttweet30
run_ft pretrained_models/wiki.en.bin my_eng_plus_sentiment140 my_eng_plus_sentiment140_ftwiki
# run_ft_fix pretrained_models/wiki.ru.bin my_rated ru_ftwiki
# run_ft_fix ../fastText/tweet_fasttext_model_300_epoch30.vec.bin my_rated ru_fttweet30
# run_ft_fix ../fastText/my_en_tweet_fasttext_300_epoch30.bin my_eng_plus_sentiment140 my_eng_plus_sentiment140_fttweet30
# run_ft_fix pretrained_models/wiki.en.bin my_eng_plus_sentiment140 en_ftwiki

# python run2.py --train_full_model --full_model full_lstm --dataset sentiment140 --full_model_file_prefix trained_models/en_lstm_full_256_32 --full_model_params '{"lstm_layer_sizes": [256, 32]}' 
# python run2.py --train_full_model --full_model full_lstm --dataset sentiment140 --full_model_file_prefix trained_models/en_lstm_full_512_128 --full_model_params '{"lstm_layer_sizes": [512, 128]}' 
# python run2.py --train_full_model --full_model full_clstm --dataset sentiment140 --full_model_file_prefix trained_models/en_cnn_lstm_full_256_128 --full_model_params '{}'
# python run2.py --train_full_model --full_model full_gru --dataset sentiment140 --full_model_file_prefix trained_models/en_cnn_lstm_full_256_32 --full_model_params '{"gru_ayer_sizes": [256, 64]}'
# python run2.py --train_full_model --full_model full_cnn --dataset sentiment140 --full_model_file_prefix trained_models/en_cnn --full_model_params '{}'
