import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import jieba
import pandas as pd

# 加载新闻文档数据
with open('news_corpus.pkl', 'rb') as file:
    news_documents = pickle.load(file)

# 定义停用词列表
stop_words = list(pd.read_csv('停词库.txt', names=['w'], sep='aaa', encoding='utf-8', engine='python').w)

# 分词并过滤停用词
segmented_documents = []
for doc in news_documents:
    # 使用jieba进行分词
    words = jieba.cut(doc)
    # 过滤掉停用词
    filtered_words = [word for word in words if word not in stop_words]
    # 将过滤后的词语添加到结果列表中
    segmented_documents.append(" ".join(filtered_words))

class PLSAModel:
    def __init__(self, num_topics, num_iterations):
        self.num_topics = num_topics
        self.num_iterations = num_iterations
        self.vocab = None
        self.num_words = None
        self.num_documents = None
        self.word_count_matrix = None
        self.P_z_given_d = None
        self.P_w_given_z = None
        self.P_z_given_dw = None

    def fit(self, documents):
        # 使用CountVectorizer将文本数据转换为词袋向量
        vectorizer = CountVectorizer(max_df=0.8, min_df=0.02)
        count_matrix = vectorizer.fit_transform(documents)
        self.vocab = vectorizer.get_feature_names_out()
        self.num_words = len(self.vocab)
        self.num_documents = count_matrix.shape[0]
        self.word_count_matrix = count_matrix.toarray()

        # 初始化参数
        self.P_z_given_d = np.random.rand(self.num_documents, self.num_topics)
        self.P_w_given_z = np.random.rand(self.num_topics, self.num_words)
        self.P_z_given_dw = np.zeros((self.num_documents, self.num_words, self.num_topics))

        # 运行EM算法
        for i in range(self.num_iterations):
            print(f'Iteration {i + 1}')
            # E步
            for d in range(self.num_documents):
                for w in range(self.num_words):
                    denominator = np.sum(self.P_w_given_z[:, w] * self.P_z_given_d[d])
                    for z in range(self.num_topics):
                        numerator = self.P_w_given_z[z, w] * self.P_z_given_d[d, z]
                        self.P_z_given_dw[d, w, z] = numerator / denominator

            # M步
            for z in range(self.num_topics):
                for w in range(self.num_words):
                    numerator = 0
                    for d in range(self.num_documents):
                        numerator += self.word_count_matrix[d, w] * self.P_z_given_dw[d, w, z]
                    self.P_w_given_z[z, w] = numerator / np.sum(self.word_count_matrix)

            for d in range(self.num_documents):
                for z in range(self.num_topics):
                    numerator = 0
                    for w in range(self.num_words):
                        numerator += self.word_count_matrix[d, w] * self.P_z_given_dw[d, w, z]
                    self.P_z_given_d[d, z] = numerator / np.sum(self.word_count_matrix[d, :])

    def high_freq_words_output(self):
        # 输出高频词
        for k in range(self.num_topics):
            print(f"High frequency words for topic {k}:")
            high_freq_words_5 = self.vocab[np.argsort(-self.P_w_given_z[k, :])][:5]
            print(high_freq_words_5)

# 设置主题数量和迭代次数
num_topics_list = [3, 6, 9]
num_iterations = 3

# 训练和输出不同主题数量下的高频词
for num_topics in num_topics_list:
    plsa_model = PLSAModel(num_topics=num_topics, num_iterations=num_iterations)
    plsa_model.fit(segmented_documents)
    print(f'Topics: {num_topics}')
    plsa_model.high_freq_words_output()
