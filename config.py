dl_train_params = dict(
    max_length  = 128,
    batch_size = 32,
    n_classes = 3, 
    shuffle = True
)

dl_eval_params = dict(
    max_length  = 128,
    batch_size = 1,
    n_classes = 3, 
    shuffle = False
)

training_params = dict(
  epochs_1 = 4, # In total: 2x epochs 
  epochs_2 = 3,
  multiprocessing = True, 
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = ["acc"], 
  transformers_model_name="bert-base-german-cased"
  #transformers_model_name="bert-base-uncased"
)

eval_params = dict(
  loss = "categorical_crossentropy", 
  metrics = ["acc", "sparse_categorical_accuracy"], 
  labels = ["Lauterbach", "Oezdemir", "Spahn"],
  transformers_model_name="bert-base-german-cased"
)

sagemaker_endpoint = dict(
  service_name='sagemaker-runtime',
  region_name='eu-west-1', 
  EndpointName='XXXXXXX',
  ContentType='application/json',
  ACCESS_KEY = 'XXXXXXX',
  SECRET_KEY = 'XXXXXXX'
)


dirs = dict(
  results_path = "/content/drive/MyDrive/BERT/tweet_class/results_4e",
  training_csv_file = "/content/drive/MyDrive/BERT/tweet_class/data_cleaned/twitter_oez_laut_spa_training.csv",
  validation_csv_file = "/content/drive/MyDrive/BERT/tweet_class/data_cleaned/twitter_oez_laut_spa_valid.csv",
  test_csv_file = "/content/drive/MyDrive/BERT/tweet_class/data_cleaned/twitter_oez_laut_spa_test.csv",
  #save_model = "/content/drive/MyDrive/carpal_bones_segmentation/results/models/model1",
  #model_dir = results_path + "/models",
)
