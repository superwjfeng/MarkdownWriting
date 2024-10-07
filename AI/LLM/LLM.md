# GPT

GPT, Generative Pre-trained Transformer 生成式预训练 Transformer

Pre-trained 是指模型经历了从大量数据中学习的过程，Pre 还暗示了模型能够针对具体任务，通过额外的训练来进行微调



system prompt 系统提示词



模型有一个预设的词汇库

Embedding Matrix



### GPT-3权重参数总览

Total weights: 175,181,291,520. They are organized into 27,938 matrices

| Embedding       | $d_{embed}(12,288)*n_{vocab}(50,257)=617,558,016$            |
| --------------- | ------------------------------------------------------------ |
| Key             | $d_{query}(128)*d_{embed}(12,288)*n_{heads}(96)*n_{layers}(96)=14,495,514,624$ |
| Query           | $d_{query}(128)*d_{embed}(12,288)*n_{heads}(96)*n_{layers}(96)=14,495,514,624$ |
| Value           | $d_{value}(128)*d_{embed}(12,288)*n_{heads}(96)*n_{layers}(96)=14,495,514,624$ |
| Output          | $d_{embed}(12,288)*d_{value}(128)*n_{heads}(96)*n_{layers}(96)=14,495,514,624$ |
| Up-projection   | $n_{neurons}(49,152)*d_{embed}(12,288)*n_{layers}(96)=57,982,058,496$ |
| Down-projection | $d_{embed}(12,288)*n_{neurons}(49,152)*n_{layers}(96)=57,982,058,496$ |
| Unembedding     | $n_{vocab}(50,257)*d_{embed}(12,288)=617,558,016$            |



# 多模态

多模态 LLM 将文本和其他模态的信息结合起来，比如图像、视频、音频和其他感官数据，多模态 LLM 接受了多种类型的数据训练，有助于 transformer 找到不同模态之间的关系，完成一些新的 LLM 不能完成的任务，比如图片描述，音乐解读，视频理解等

模态 Modality 指的是数据或者信息的某种表现形式