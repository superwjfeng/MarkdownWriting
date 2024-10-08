# GPT

GPT, Generative Pre-trained Transformer 生成式预训练 Transformer

Pre-trained 是指模型经历了从大量数据中学习的过程，Pre 还暗示了模型能够针对具体任务，通过额外的训练来进行微调

## *Architecture of GPT-3*

MLP层用于训练词汇的语义信息（也就是这个词汇本身是什么意思），注意力层则用于训练词汇之间（上下文）的影响

### Embedding Matrix

模型有一个预设的词汇库，或者说是 token 库。所有的 token 列向量组合起来就形成了 Embedding Matrix $W_E$，它被随机初始化。关于词嵌入的思想在 *DL.md* 的RNN中介绍过

GPT-3中词嵌入的维度为12,288（4\*4096），一共有50,257个tokens

当模型在训练阶段调整权重，以确定不同单词将如何嵌入向量时，训练得到的最终的嵌入向量在空间中的方向，往往会具有某种语义信息

### Attention Block

Attention 模块每次只能处理特定数量的向量，这个数量称为 Context Size 上下文长度，GPT-3 的上下文长度为2048

它组成了 Unembedding Matrix 解嵌入矩阵 $W_U$，它的每一行都应一个词库中的一个 token，列则是token的维度，即 50257*12288

### GPT-3权重参数总览



Total weights: 175,181,291,520（一千七百亿+）. They are organized into 27,938 matrices

| Embedding       | $d_{embed}(12,288)*n_{vocab}(50,257)=617,558,016$            |
| --------------- | ------------------------------------------------------------ |
| Key             | $d_{query}(128)*d_{embed}(12,288)*n_{heads}(96)*n_{layers}(96)=14,495,514,624$ |
| Query           | $d_{query}(128)*d_{embed}(12,288)*n_{heads}(96)*n_{layers}(96)=14,495,514,624$ |
| Value           | $d_{value}(128)*d_{embed}(12,288)*n_{heads}(96)*n_{layers}(96)=14,495,514,624$ |
| Output          | $d_{embed}(12,288)*d_{value}(128)*n_{heads}(96)*n_{layers}(96)=14,495,514,624$ |
| Up-projection   | $n_{neurons}(49,152)*d_{embed}(12,288)*n_{layers}(96)=57,982,058,496$ |
| Down-projection | $d_{embed}(12,288)*n_{neurons}(49,152)*n_{layers}(96)=57,982,058,496$ |
| Unembedding     | $n_{vocab}(50,257)*d_{embed}(12,288)=617,558,016$            |

## *训练*

训练中更实际的做法是利用最终层的每一个向量，同时预测紧随着这个向量的下一个向量（token）

### softmax with temprature

$$
e^{x_1/T}/\sum^{N-1}_{n=0}{e^{x_n/T}}
$$

T小的时候，原来输出概率较大的数值就会更占优势，即更加容易被选中；T大的时候，则原来输出概率较小的数值会更占优势。比如当T大的时候，数字小的单词更容易被选中了，这样AI就更有创造性了

GPT的API不允许选择大于2的T，这没有数学依据，只是一个人为选定的超参数，防止生成的内容过于发散

# 多模态

多模态 LLM 将文本和其他模态的信息结合起来，比如图像、视频、音频和其他感官数据，多模态 LLM 接受了多种类型的数据训练，有助于 transformer 找到不同模态之间的关系，完成一些新的 LLM 不能完成的任务，比如图片描述，音乐解读，视频理解等

模态 Modality 指的是数据或者信息的某种表现形式

# LongChain

system prompt 系统提示词