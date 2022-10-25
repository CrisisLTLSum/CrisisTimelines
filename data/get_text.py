import tweepy
import json 
from tqdm import tqdm

# assign the values accordingly
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""
  
# authorization of consumer key and consumer secret
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  
# set access to user's access key and access secret 
auth.set_access_token(access_token, access_token_secret)
  
# calling the api 
api = tweepy.API(auth, wait_on_rate_limit=True)


for _set in ["train", "test", "dev"]:
    with open(f"{_set}_notext.json", "r") as f:
        data = json.load(f)

    for item in tqdm(data):
        for ind, tweet in enumerate(item['tweet_ids']):
            try:
                status = api.get_status(tweet, tweet_mode='extended')
                # fetching the text attribute
                text = status.full_text
                item['tweets'].append(text)
                if ind == 0:
                    item['seed'] = text
            except KeyboardInterrupt:
                exit(0)
            except Exception as e:
                print(e)
                print(f'the id {tweet} is passed')
                if ind == 0:
                    print("Seed is Removed!!")
                    item['seed'] = "Seed is Removed!"
                elif item['labels'][ind-1] == "1":
                    print("Important tweet removed!")
                else:
                    print("non-important tweet removed!")

                item['tweets'].append("Error in Retreiving the Tweet!")

    with open(f"{_set}.json", "w") as f:
        json.dump(data, f)
        
        