from flask import Flask, render_template, request, session
from flask_cors import CORS, cross_origin
from flask_session import Session
import xuli
import pickle
import os
import json
import yake
from langdetect import detect

MODEL_PATH = "models"

app = Flask(__name__)
sess = Session()

labelInfo = ['cong_nghe', 'du_lich', 'giao_duc', 'giai_tri', 'kinh_doanh', 'nhip_song', 'phim_anh', 'phap_luat', 'song_tre',
             'suc_khoe', 'the_gioi', 'the_thao', 'thoi_su', 'thoi_trang', 'xe_360', 'xuat_ban', 'am_nhạc', 'am_thực']

model = pickle.load(
    open(os.path.join(MODEL_PATH, "linear_classifier.pkl"), 'rb'))

stopword = set()
for line in open('stopwords.txt', encoding='utf8'):
    word = line.strip().split()
    stopword.add(word[0])


def takeSecond(elem):
    return elem[1]


def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
    return ' '.join(words)


def result_cheked(probality):
    rsArray = list(zip(probality.flatten(), labelInfo))

    def takeFirst(elem):
        return elem[0]

    rsArray.sort(key=takeFirst, reverse=True)
    return rsArray


def checkKeyWord(list):
    listWord = []
    listPhares = []
    for kw in list:
        if kw[0].find(' ') != -1:
            listPhares.append(kw[0])
        else:
            listWord.append(kw[0])
    result = {
        "texts": listWord,
        "phrases": listPhares,
        "listOrigin": list,
    }
    return result


# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/', methods=['GET'])
@cross_origin(origin='*')
def home():
    return 'HOME'


@app.route('/submit', methods=['POST', 'GET'])
@cross_origin(origin='*')
def submit():
    title = request.form.get('title')
    content = request.form.get('content')
    document = xuli(content)
    document = remove_stopwords(document)
    # label = model.predict([document])
    probality = model.predict_proba([document])
    rs = result_cheked(probality)

    if title:
        probalityTitle = model.predict_proba([title])
        rsTitle = result_cheked(probalityTitle)
        return {
            "rsContent": rs,
            "rsTitle": rsTitle,
        }
    else:
        return {
            "rsContent": rs,
        }


@app.route('/keyword', methods=['GET'])
@cross_origin(origin='*')
def getKeyWord():
    title = request.form.get('title')
    # title = request.json['title']
    lan = detect(title)

    if(lan == 'vi'):
        kw_extractor = yake.KeywordExtractor(
            lan='vi', n=2, stopwords=stopword)
    else:
        kw_extractor = yake.KeywordExtractor()

    keywords = kw_extractor.extract_keywords(title)
    keywords.sort(key=takeSecond, reverse=True)

    result_keywords = checkKeyWord(keywords)

    result = {
        "keywords": result_keywords
    }
    return json.dumps(result)


if __name__ == '__main__':
    sess.init_app(app)
    app.debug = True
    app.run(host='0.0.0.0', port='6868')
