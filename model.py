import xuli
import pickle
import os

MODEL_PATH = "models"

stopword = set()
for line in open('stopwords.txt', encoding='utf8'):
    word = line.strip().split()
    stopword.add(word[0])


def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
    return ' '.join(words)


document = """
React (hay ReactJS, React.js) là một thư viện Javascript mã nguồn mở để xây dựng các thành phần giao diện có thể tái sử dụng. Nó được tạo ra bởi Jordan Walke, một kỹ sư phần mềm tại Facebook. Người bị ảnh hưởng bởi XHP (Một nền tảng thành phần HTML cho PHP). React lần đầu tiên được triển khai cho ứng dụng Newsfeed của Facebook năm 2011, sau đó được triển khai cho Instagram năm 2012. Nó được mở mã nguồn (open-sourced) tại JSConf US tháng 5 năm 2013
"""

# document = document.decode('utf8', 'ignore').encode("utf8")

document = xuli(document)
document = remove_stopwords(document)

model = pickle.load(
    open(os.path.join(MODEL_PATH, "linear_classifier.pkl"), 'rb'))
label = model.predict([document])

print(document)
print(label)
