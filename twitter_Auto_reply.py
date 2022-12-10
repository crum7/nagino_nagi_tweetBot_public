from requests_oauthlib import OAuth1Session #pip install requests-oauthlib
from datetime import datetime, timezone, timedelta
import tweepy
import twitter
import random
import csv
from transformers import T5Tokenizer, AutoModelForCausalLM ,pipeline,AutoModelForSequenceClassification, BertJapaneseTokenizer
import schedule
from time import sleep



# 認証キーの設定
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

# OAuth認証
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
# APIのインスタンスを生成
api = tweepy.API(auth,wait_on_rate_limit=True)


# OAuthの認証オブジェクトの作成
oauth = twitter.OAuth(access_token,
                      access_token_secret,
                      consumer_key,
                      consumer_secret)

# Twitter REST APIを利用するためのオブジェクト
twitter_api = twitter.Twitter(auth=oauth)




def NaginoNagi_GPT(word):
     #huggingface--------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # トークナイザーとモデルの準備
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
    model = AutoModelForCausalLM.from_pretrained("output/checkpoint-5500/")
    # 推論の実行
    input = tokenizer.encode(word,return_tensors="pt",)
    #do_sample=True、ランダムな自然な文を生成するため。repetition_penalty = 1.1、同じ言葉が反復して出てこないようにするため。num_return_sequences、文の数
    output = model.generate(input, do_sample=True, max_length=100, num_return_sequences=5, repetition_penalty =1.0)
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
    voice = become_voice_str.replace("[" , "").replace("ω" , "").replace('&amp;','').replace(word,'')
    print(voice)

    return voice










#リプに対してリプする
def search_tweets():
    #ユーザーに表示されるタイムライン

    # enumerateを使っているのは、取得件数の確認のためです。
    for i, tweet in enumerate(tweepy.Cursor(api.home_timeline).items(3200)):

        #.find()が-1のときはその中に無い
        if tweet.text.find('@アカウント名') != -1:
            #print("ツイート本文:" +tweet.text+'投稿者:'+str(tweet.user.name)+str(tweet.user.screen_name)+"ツイートid:"+str(tweet.id))

            before_seed = tweet.text.split(" ")
            tweet_seed = before_seed[-1].replace("\n","")

            #.transrate()で全角を半角に直す
            #.lower()でアルファベットを小文字に直す
            word = tweet_seed.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}))
            print(word)



            #.txtからすでにリプしていないかを調べる
            with open('/auto_rep.txt',encoding='utf-8') as temp_f:
                datafile = temp_f.readlines()
                print(datafile)
                print(word+'\n' in datafile)

            #テキストファイルにあるかどうか
            if word+'\n' in datafile:
                print('済み')
                
            else:
                print('リプライ処理')

                #.txtに書き込み
                
                f = open('auto_rep.txt', 'a',encoding='utf-8')
                f.write(word+'\n')
                f.close()

                voice = NaginoNagi_GPT(word)
                # 特定のツイートへのリプライの場合
                reply_text = '@'+tweet.user.screen_name +str(before_seed[0:-2]).replace("[",' ').replace("]",'').replace("'",'')+ '\n'
                reply_text += voice
                print("reply->>>>"+reply_text)
                
                #リプライ
                twitter_api.statuses.update(status=reply_text, in_reply_to_status_id=tweet.id)
        else:
            print("タイムライン上に無い")
                
            

            





                    

        

if __name__ == '__main__':
    search_tweets()

    #02 スケジュール登録
    #schedule.every(2).minutes.do(search_tweets)

    