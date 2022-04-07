import numpy as np
import boto3, json, time
import transformers
transformers.logging.set_verbosity_error()
import config


# Load tokenizer from transformers
tokenizer = transformers.BertTokenizer.from_pretrained(
    "bert-base-german-cased", 
    do_lower_case=True
)

# Load boto3 client to access AWS server
sm_rt = boto3.client(service_name=config.sagemaker_endpoint["service_name"], 
                     region_name=config.sagemaker_endpoint["region_name"], 
                     aws_access_key_id=config.sagemaker_endpoint["ACCESS_KEY"],
                     aws_secret_access_key=config.sagemaker_endpoint["SECRET_KEY"]
                     )

def getPredictionFromEndpoint(tweet, 
    boto_client, 
    tokenizer, 
    max_length = 128
):  
    # Step 1: Create tokenized item (unfortunately I haven't  found a way  
    # to include a tokenizer layer into the deployed Keras model)
    tweet = np.array([tweet])
    encoded = tokenizer.batch_encode_plus(
        tweet,
        add_special_tokens=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=True,
        pad_to_max_length=True,
        return_tensors="tf",
    )

    input_ids = np.array(encoded["input_ids"], dtype="int32")
    attention_masks = np.array(encoded["attention_mask"], dtype="int32")
    token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

    # Ste 2: Create compatible server request
    request = {               
                    "inputs": {
                      "input_ids": input_ids.tolist(),
                      "attention_masks": attention_masks.tolist(), 
                      "token_type_ids": token_type_ids.tolist()   
                    }
    }
    data = json.dumps(request)

    # Step 3: Send request to server endpoint and wait for response
    tic = time.time()
    response = boto_client.invoke_endpoint(
                EndpointName=config.sagemaker_endpoint['EndpointName'],
                Body=data,
                ContentType=config.sagemaker_endpoint['ContentType']
    )
    tac = time.time()
    print("Total server response time: ", tac - tic)

    # Step 4: Process the response data to numpy
    response = response["Body"].read()
    response = json.loads(response)
    preds = response['outputs']
    preds =  np.asarray(preds)
    return preds


test_string = "Das ist ein Test!"
preds = getPredictionFromEndpoint(test_string, 
    sm_rt, 
    tokenizer, 
    max_length = config.dl_eval_params["max_length"]
)

print(preds)
print("Predicted label: ", config.eval_params["labels"][np.argmax(preds[0])])






      
      




