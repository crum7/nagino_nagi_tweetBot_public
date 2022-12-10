import random
import tweepy
import csv
from transformers import T5Tokenizer, AutoModelForCausalLM ,pipeline,AutoModelForSequenceClassification, BertJapaneseTokenizer
import schedule
from time import sleep
import twitter



#Twitter各種設定
#取得したtwitterAPIを貼り付ける
CONSUMER_KEY = ''
CONSUMER_SECRET = ''
ACCESS_TOKEN = ''
ACCESS_SECRET = ''


#twitter各種設定
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth,wait_on_rate_limit=True)


# OAuthの認証オブジェクトの作成
oauth = twitter.OAuth(ACCESS_TOKEN,
                      ACCESS_SECRET,
                      CONSUMER_KEY,
                      CONSUMER_SECRET)
# Twitter REST APIを利用するためのオブジェクト
twitter_api = twitter.Twitter(auth=oauth)
# Twitter Streaming APIを利用するためのオブジェクト
twitter_stream = twitter.TwitterStream(auth=oauth)

def tweet():



    #csvから単語のリストを読み込む
    csv_file = open("japanese_words.csv", "r", encoding="shift_jis")
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    header = next(f)

    #単語を１文字取る
    japanese_word=[]
    for row in f:
        #rowはList
        japanese_word.append(row[2])


    tweet_seed = random.choice(japanese_word)

    
        # トークナイザーとモデルの準備
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
    model = AutoModelForCausalLM.from_pretrained("output\checkpoint-5500")



        #.transrate()で全角を半角に直す
        #.lower()でアルファベットを小文字に直す
    word = tweet_seed.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})).lower()

    # 推論の実行
    input = tokenizer.encode(word,return_tensors="pt",)
    #do_sample=True、ランダムな自然な文を生成するため。repetition_penalty = 1.1、同じ言葉が反復して出てこないようにするため。num_return_sequences、文の数
    output = model.generate(input, do_sample=True, max_length=100, num_return_sequences=3, repetition_penalty =1.0)
    #skip_special_tokens、いらない文字を無くすため
    become_voice_list=[]
    become_voice_list = tokenizer.batch_decode(output,skip_special_tokens=True)

    #推論された文字を調整
    #pic.twitterが入ってる場合は、次の配列に移る
    for i in become_voice_list:
        
        if 'pic' in i:
            print('スキップ:'+i)
            continue
        else:
            become_voice_str = "".join(i)
            #become_voice_str=become_voice_str.replace(word,'')

            if become_voice_str == '':
                become_voice_str = '...'


    #文章的に直すところ
    voice = become_voice_str.replace("[" , "").replace("@" , "").replace("ω" , "").replace("_","").replace('&amp;','')

    #リストに含まれるツイート内容をランダムでツイート
    print(voice)
    api.update_status(voice)


tweet()