# textRNN_pytorch_master
ues lstm network to get a Text Classification

1.数据来源:
中文数据是从https://github.com/brightmart/nlp_chinese_corpus下载的。具体是第3个，百科问答Json版，因为感觉大小适中，适合用来学习。下载下来得到两个文件：baike_qa_train.json和baike_qa_valid.json。内容形式如下:




{"qid": "qid_1815059893214501395", "category": "烦恼-恋爱", "title": "请问深入骨髓地喜欢一个人怎么办我不能确定对方是不是喜欢我，我却想 ", "desc": "我不能确定对方是不是喜欢我，我却想分分秒秒跟他在一起，有谁能告诉我如何能想他少一点", "answer": "一定要告诉他你很喜欢他 很爱他!! 虽然不知道你和他现在的关系是什么！但如果真的觉得很喜欢就向他表白啊！！起码你努力过了！ 女生主动多少占一点优势的！！呵呵 只愿曾经拥有！ 到以后就算感情没现在这么强烈了也不会觉得遗憾啊~！ 与其每天那么痛苦的想他 恋他 还不如直接告诉他 ！ 不要怕回破坏你们现有的感情！因为如果不告诉他 你可能回后悔一辈子！！ "}




2.样本选取
下下来的数据类别非常多，为了简化，我从中帅选了少量的样本进行学习。具体来说，我只选择了标题前2个字为教育、健康、生活、娱乐和游戏五个类别，同时各个类别各5000个.


3.执行过程
1>执行get_my_trainData.py(从元数据中分出我们要拿来做训练的训练集,和测试集,文件中只进行了训练集的划分,请自行通过修改参数划分测试集)===========> 2>执行get_wordlist.py(把我们拿到的数据集进行分词生成词表)=========> 3>执行sen2ind(通过与词表中的词进行对应,将每一句话生成一个句子向量,并和类别存放到对应的文件中,该文件之给出了一个测试集,训练集自行修改代码生成) [textRNN_data.py写了一部分数据处理的方法]============> 4>model.py中写了lstm的模型============> 5>执行train.py(用我们的训练集的数据对我们的model中的参书进行训练,在得到一个很好的结果时保存模型)============>
6>执行test.py(对测试集数据进行测试验证准确率)





经过无数次的实验对模型选择最终分类结果的准确率已达到99.86%
