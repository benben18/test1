# -- encoding:utf-8 --
"""
Create by on 2019/3/30
根据tfidf模型的相似度
"""
import jieba
from gensim import corpora, models, similarities

doc0 = "我不喜欢上海"
doc1 = "上海是一个好地方"
doc2 = "北京是一个好地方"
doc3 = "上海好吃的在哪里"
doc4 = "上海好玩的在哪里"
doc5 = "上海是好地方"
doc6 = "上海路和上海人"
doc7 = "喜欢小吃"
doc8 = "我不喜欢上海的小吃"
doc_test = "我喜欢上海的小吃"

all_doc = []
all_doc.append(doc0)
all_doc.append(doc1)
all_doc.append(doc2)
all_doc.append(doc3)
all_doc.append(doc4)
all_doc.append(doc5)
all_doc.append(doc6)
all_doc.append(doc7)
all_doc.append(doc8)

all_doc_list = []
# 1.jieba 分词
# 2.dictionary
# 3.tfidf
# 4.similarity


# 1.jieba 分词
for doc in all_doc:
    doc_list = [word for word in jieba.cut(doc)]
    all_doc_list.append(doc_list)

doc_test_list = [word for word in jieba.cut(doc_test)]

# 2.dictionary
dictionary = corpora.Dictionary(all_doc_list)

print(dictionary.keys())
# [1, 14, 11, 16, 12, 13, 2, 9, 8, 17, 0, 4, 5, 3, 7, 15, 6, 10]
print(dictionary.token2id)
# {'不': 1, '人': 14, '好吃': 11, '路': 16, '的': 12, '好玩': 13, '喜欢': 2, '哪里': 9, '北京': 8, '小吃': 17, '上海': 0, '一个': 4, '地方': 5, '我': 3, '是': 7, '和': 15, '好': 6, '在': 10}

corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]

print(corpus)
# [[(0, 1), (1, 1), (2, 1), (3, 1)], [(0, 1), (4, 1), (5, 1), (6, 1), (7, 1)], [(4, 1), (5, 1), (6, 1), (7, 1), (8, 1)], [(0, 1), (9, 1), (10, 1), (11, 1), (12, 1)], [(0, 1), (9, 1), (10, 1), (12, 1), (13, 1)], [(0, 1), (5, 1), (6, 1), (7, 1)], [(0, 2), (14, 1), (15, 1), (16, 1)], [(2, 1), (17, 1)], [(0, 1), (1, 1), (2, 1), (3, 1), (12, 1), (17, 1)]]

doc_test_vec = dictionary.doc2bow(doc_test_list)
print(doc_test_vec)
# [(0, 1), (2, 1), (3, 1), (12, 1), (17, 1)]
# 3.tfidf

tfidf = models.TfidfModel(corpus)
print(tfidf)
# TfidfModel(num_docs=9, num_nnz=40)
print(tfidf[doc_test_vec])
# [(0, 0.09497738018646304), (2, 0.4151903164975543), (3, 0.5684247089220512), (12, 0.4151903164975543), (17, 0.5684247089220512)]

# 4.similarity
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
sim = index[tfidf[doc_test_vec]]
sorted(enumerate(sim), key=lambda item: -item[1])
print(sorted(enumerate(sim), key=lambda item: -item[1]))
# [(8, 0.8693658), (7, 0.70391023), (0, 0.5545687), (3, 0.14727601), (4, 0.14727601), (5, 0.012435907), (6, 0.012435907), (1, 0.009788494), (2, 0.0)]
# 可以看出tfidf模型的基于字面的相似度 并非语义的相似度。
# 从分析结果来看，测试文档与doc7相似度最高，其次是doc0，与doc2的相似度为零。大家可以根据TF-IDF的原理，看看是否符合预期。