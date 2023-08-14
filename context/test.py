### Prompts
# short  << 2k
short_prompt = "Here is an introduction of the city LA:"

# long 
middle_prompt ="""
Text
---
Large language models (LLMs) typically come with a pre-defined context window size. For example, inputs to LLaMA models (Touvron et al., 2023) must be fewer than 2048 tokens. This pre-set
context window limit is frequently exceeded in applications such as conducting long conversations,
summarizing long documents, or executing long-term planning. For these applications, LLMs with
longer context windows are preferred. However, training an LLM from scratch with long context
windows requires significant investments. This naturally leads to a question: Can we extend the
context window of an existing pre-trained LLM?
One straightforward approach is to fine-tune an existing pre-trained Transformer with a longer context window. However, empirically, we found that models trained this way adapt to long context
windows very slowly. After training for more than 10000 batches, the effective context window
saw a minimal increase, moving from 2048 to 2560 (Table 4). This suggests that such method is
inefficient for extending to substantially longer context windows.
While certain techniques such as ALiBi (Press et al., 2022) and LeX (Sun et al., 2022) enable length
extrapolation of Transformers, i.e. train on short context windows and inference on longer ones,
many existing pre-trained LLMs, including LLaMA (Touvron et al., 2023), use positional encodings
that have weak extrapolation properties (e.g., RoPE (Su et al., 2021)). Therefore, the applicability
of these techniques for extending the context window sizes of such LLMs remains limited.
In this work, we introduce Position Interpolation to enable context window extensions for certain
existing pre-trained LLMs, including LLaMA. The key idea is, instead of extrapolation, we directly
down-scale the position indices so that the maximum position index matches the previous context
window limit in the pre-training stage. See Figure 1 for an illustration. In other words, to accommodate more input tokens, we interpolate the position encodings at neighboring integer positions,
utilizing the fact that position encodings can be applied on non-integer positions, as opposed to
extrapolating outside the trained positions, which may lead to catastrophic values. We verify our
approach theoretically, by showing that the interpolated attention score has a much smaller upper
bound (∼ 600× smaller in LLaMA 7B setting) than the extrapolated one, and is thus much more
stable. Therefore, interpolated position encodings are easier for the model to adapt.
Empirically, we found that Position Interpolation is highly effective and efficient, requiring only a
very short period of fine-tuning for the model to fully adapt to greatly extended context windows.
We present experimental results for extending the context window to up to 32768 from the initial
2048 across 7B to 65B LLaMA models using Position Interpolation. Our results show that

1. Position Interpolation can easily enable very long context windows (e.g. 32768), requiring
only fine-tuning for 1000 steps on the Pile (Gao et al., 2020) to achieve a good quality.
The cost of fine-tuning is negligible compared to the pre-training costs. This confirms
our hypothesis that it is relatively easy for the models to adapt to interpolated position
encodings.
2. Position Interpolation generates strong models that can effectively make use of much extended context window. We show that models extended by Position Interpolation enjoy
significant perplexity gains from greatly extended context windows for text modeling, and
we show that the perplexity reduces graceful with the enlargement of context windows.
We also applied Position Interpolation in a long text summarization task, and demonstrate
competitive performances.
3. Position Interpolation preserves model quality relatively well for tasks within its original
context window sizes. We present a variety of evaluation results for the extended LLaMA
models on the original LLaMA benchmark. Compared with original LLaMA models, the
extended LLaMA models saw a minor degradation on several standard benchmarks within
a 2048 token limit.
Our results highlight the innate ability of Transformer models to “extrapolate to sequence lengths longer than the ones encountered during training” as hypothesized in the seminal work of Vaswani et al. (2017). We reaffirm this hypothesis and suggest that the previously known weakness of extrapolating to longer sequences for language modeling (Press et al., 2022) may be due to direct extrapolation of positional encodings and it can be largely mitigated by interpolating position encodings instead.
Concurrent work. Right before our release, we are informed with a concurrent blogpost (SuperHOT kaiokendev (2023)) that also interpolates positional encoding in RoPE to extend the context window from 2K to 8K. Recently, open source community picks it up in Reddit post 1
and Github Issues 2, which shows that fine-tuning with LoRA (Hu et al., 2021) also seems to work well. Our paper shows a full fine-tuning with up to 65B model work well with Position Interpolation, and we also give theoretical explanations why interpolation achieves much more stable results than extrapolation, by showing that the upper bound of interplated attention score is much lower than that of extrapolated ones
---
Here is the summary of the article:
"""

middle_prompt2 = """
Text
---
LLaMA-2-7B-32K
Model Description

LLaMA-2-7B-32K is an open-source, long context language model developed by Together, fine-tuned from Meta's original Llama-2 7B model. This model represents our efforts to contribute to the rapid progress of the open-source ecosystem for large language models. The model has been extended to a context length of 32K with position interpolation, allowing applications on multi-document QA, long text summarization, etc.
What's new?

This model introduces several improvements and new features:

    Extended Context: The model has been trained to handle context lengths up to 32K, which is a significant improvement over the previous versions.

    Pre-training and Instruction Tuning: We have shared our data recipe, which consists of a mixture of pre-training and instruction tuning data.

    Fine-tuning Examples: We provide examples of how to fine-tune the model for specific applications, including book summarization and long context question and answering.

    Software Support: We have updated both the inference and training stack to allow efficient inference and fine-tuning for 32K context.

Model Architecture

The model follows the architecture of Llama-2-7B and extends it to handle a longer context. It leverages the recently released FlashAttention-2 and a range of other optimizations to improve the speed and efficiency of inference and training.
Training and Fine-tuning

The model has been trained using a mixture of pre-training and instruction tuning data.

    In the first training phase of continued pre-training, our data mixture contains 25% RedPajama Book, 25% RedPajama ArXiv (including abstracts), 25% other data from RedPajama, and 25% from the UL2 Oscar Data, which is a part of OIG (Open-Instruction-Generalist), asking the model to fill in missing chunks, or complete the text. To enhance the long-context ability, we exclude data shorter than 2K word. The inclusion of UL2 Oscar Data is effective in compelling the model to read and utilize long-range context.
    We then fine-tune the model to focus on its few shot capacity under long context, including 20% Natural Instructions (NI), 20% Public Pool of Prompts (P3), 20% the Pile. We decontaminated all data against HELM core scenarios . We teach the model to leverage the in-context examples by packing examples into one 32K-token sequence. To maintain the knowledge learned from the first piece of data, we incorporate 20% RedPajama-Data Book and 20% RedPajama-Data ArXiv.

Next, we provide examples of how to fine-tune the model for specific applications. The example datasets are placed in togethercomputer/Long-Data-Collections You can use the OpenChatKit to fine-tune your own 32K model over LLaMA-2-7B-32K. Please refer to OpenChatKit for step-by-step illustrations.

    Long Context QA.

    We take as an example the multi-document question answering task from the paper “Lost in the Middle: How Language Models Use Long Contexts”. The input for the model consists of (i) a question that requires an answer and (ii) k documents, which are passages extracted from Wikipedia. Notably, only one of these documents contains the answer to the question, while the remaining k − 1 documents, termed as "distractor" documents, do not. To successfully perform this task, the model must identify and utilize the document containing the answer from its input context.

    With OCK, simply run the following command to fine-tune:

    bash training/finetune_llama-2-7b-32k-mqa.sh

Summarization.

Another example is BookSum, a unique dataset designed to address the challenges of long-form narrative summarization. This dataset features source documents from the literature domain, including novels, plays, and stories, and offers human-written, highly abstractive summaries. We here focus on chapter-level data. BookSum poses a unique set of challenges, necessitating that the model comprehensively read through each chapter.

With OCK, simply run the following command to fine-tune:

bash training/finetune_llama-2-7b-32k-booksum.sh

Inference

You can use the Together API to try out LLaMA-2-7B-32K for inference. The updated inference stack allows for efficient inference.

To run the model locally, we strongly recommend to install Flash Attention V2, which is necessary to obtain the best performance:

# Please update the path of `CUDA_HOME`
export CUDA_HOME=/usr/local/cuda-11.8
pip install transformers==4.31.0
pip install sentencepiece
pip install ninja
pip install flash-attn --no-build-isolation
pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary

You can use this model directly from the Hugging Face Model Hub or fine-tune it on your own data using the OpenChatKit.

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K", trust_remote_code=True, torch_dtype=torch.float16)

input_context = "Your text here"
input_ids = tokenizer.encode(input_context, return_tensors="pt")
output = model.generate(input_ids, max_length=128, temperature=0.7)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)

Alternatively, you can set trust_remote_code=False if you prefer not to use flash attention.
Limitations and Bias

As with all language models, LLaMA-2-7B-32K may generate incorrect or biased content. It's important to keep this in mind when using the model.
Community

Join us on Together Discord
---
Here is the summary of the text above:

"""

long_prompt = """
Text
---
Transformers

State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX.

Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. These models support common tasks in different modalities, such as:

Natural Language Processing: text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation.
Computer Vision: image classification, object detection, and segmentation.
Audio: automatic speech recognition and audio classification.
Multimodal: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.

Transformers support framework interoperability between PyTorch, TensorFlow, and JAX. This provides the flexibility to use a different framework at each stage of a model’s life; train a model in three lines of code in one framework, and load it for inference in another. Models can also be exported to a format like ONNX and TorchScript for deployment in production environments.

Join the growing community on the Hub, forum, or Discord today!
If you are looking for custom support from the Hugging Face team
HuggingFace Expert Acceleration Program
Contents

The documentation is organized into five sections:

    GET STARTED provides a quick tour of the library and installation instructions to get up and running.

    TUTORIALS are a great place to start if you’re a beginner. This section will help you gain the basic skills you need to start using the library.

    HOW-TO GUIDES show you how to achieve a specific goal, like finetuning a pretrained model for language modeling or how to write and share a custom model.

    CONCEPTUAL GUIDES offers more discussion and explanation of the underlying concepts and ideas behind models, tasks, and the design philosophy of 🤗 Transformers.

    API describes all classes and functions:
        MAIN CLASSES details the most important classes like configuration, model, tokenizer, and pipeline.
        MODELS details the classes and functions related to each model implemented in the library.
        INTERNAL HELPERS details utility classes and functions used internally.

Supported models

    ALBERT (from Google Research and the Toyota Technological Institute at Chicago) released with the paper ALBERT: A Lite BERT for Self-supervised Learning of Language Representations, by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
    ALIGN (from Google Research) released with the paper Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision by Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig.
    AltCLIP (from BAAI) released with the paper AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities by Chen, Zhongzhi and Liu, Guang and Zhang, Bo-Wen and Ye, Fulong and Yang, Qinghong and Wu, Ledell.
    Audio Spectrogram Transformer (from MIT) released with the paper AST: Audio Spectrogram Transformer by Yuan Gong, Yu-An Chung, James Glass.
    Autoformer (from Tsinghua University) released with the paper Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting by Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long.
    Bark (from Suno) released in the repository suno-ai/bark by Suno AI team.
    BART (from Facebook) released with the paper BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer.
    BARThez (from École polytechnique) released with the paper BARThez: a Skilled Pretrained French Sequence-to-Sequence Model by Moussa Kamal Eddine, Antoine J.-P. Tixier, Michalis Vazirgiannis.
    BARTpho (from VinAI Research) released with the paper BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese by Nguyen Luong Tran, Duong Minh Le and Dat Quoc Nguyen.
    BEiT (from Microsoft) released with the paper BEiT: BERT Pre-Training of Image Transformers by Hangbo Bao, Li Dong, Furu Wei.
    BERT (from Google) released with the paper BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
    BERT For Sequence Generation (from Google) released with the paper Leveraging Pre-trained Checkpoints for Sequence Generation Tasks by Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
    BERTweet (from VinAI Research) released with the paper BERTweet: A pre-trained language model for English Tweets by Dat Quoc Nguyen, Thanh Vu and Anh Tuan Nguyen.
    BigBird-Pegasus (from Google Research) released with the paper Big Bird: Transformers for Longer Sequences by Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
    BigBird-RoBERTa (from Google Research) released with the paper Big Bird: Transformers for Longer Sequences by Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
    BioGpt (from Microsoft Research AI4Science) released with the paper BioGPT: generative pre-trained transformer for biomedical text generation and mining by Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon and Tie-Yan Liu.
    BiT (from Google AI) released with the paper Big Transfer (BiT): General Visual Representation Learning by Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby.
    Blenderbot (from Facebook) released with the paper Recipes for building an open-domain chatbot by Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
    BlenderbotSmall (from Facebook) released with the paper Recipes for building an open-domain chatbot by Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
    BLIP (from Salesforce) released with the paper BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation by Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.
    BLIP-2 (from Salesforce) released with the paper BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models by Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi.
    BLOOM (from BigScience workshop) released by the BigScience Workshop.
    BORT (from Alexa) released with the paper Optimal Subarchitecture Extraction For BERT by Adrian de Wynter and Daniel J. Perry.
    BridgeTower (from Harbin Institute of Technology/Microsoft Research Asia/Intel Labs) released with the paper BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning by Xiao Xu, Chenfei Wu, Shachar Rosenman, Vasudev Lal, Wanxiang Che, Nan Duan.
    ByT5 (from Google Research) released with the paper ByT5: Towards a token-free future with pre-trained byte-to-byte models by Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel.
    CamemBERT (from Inria/Facebook/Sorbonne) released with the paper CamemBERT: a Tasty French Language Model by Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
    CANINE (from Google Research) released with the paper CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation by Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting.
    Chinese-CLIP (from OFA-Sys) released with the paper Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese by An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, Chang Zhou.
    CLAP (from LAION-AI) released with the paper Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation by Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov.
    CLIP (from OpenAI) released with the paper Learning Transferable Visual Models From Natural Language Supervision by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.
    CLIPSeg (from University of Göttingen) released with the paper Image Segmentation Using Text and Image Prompts by Timo Lüddecke and Alexander Ecker.
    CodeGen (from Salesforce) released with the paper A Conversational Paradigm for Program Synthesis by Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, Caiming Xiong.
    Conditional DETR (from Microsoft Research Asia) released with the paper Conditional DETR for Fast Training Convergence by Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, Jingdong Wang.
    ConvBERT (from YituTech) released with the paper ConvBERT: Improving BERT with Span-based Dynamic Convolution by Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan.
    ConvNeXT (from Facebook AI) released with the paper A ConvNet for the 2020s by Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie.
    ConvNeXTV2 (from Facebook AI) released with the paper ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders by Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie.
    CPM (from Tsinghua University) released with the paper CPM: A Large-scale Generative Chinese Pre-trained Language Model by Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.
    CPM-Ant (from OpenBMB) released by the OpenBMB.
    CTRL (from Salesforce) released with the paper CTRL: A Conditional Transformer Language Model for Controllable Generation by Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong and Richard Socher.
    CvT (from Microsoft) released with the paper CvT: Introducing Convolutions to Vision Transformers by Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang.
    Data2Vec (from Facebook) released with the paper Data2Vec: A General Framework for Self-supervised Learning in Speech, Vision and Language by Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli.
    DeBERTa (from Microsoft) released with the paper DeBERTa: Decoding-enhanced BERT with Disentangled Attention by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
    DeBERTa-v2 (from Microsoft) released with the paper DeBERTa: Decoding-enhanced BERT with Disentangled Attention by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
    Decision Transformer (from Berkeley/Facebook/Google) released with the paper Decision Transformer: Reinforcement Learning via Sequence Modeling by Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch.
    Deformable DETR (from SenseTime Research) released with the paper Deformable DETR: Deformable Transformers for End-to-End Object Detection by Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai.
    DeiT (from Facebook) released with the paper Training data-efficient image transformers & distillation through attention by Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou.
    DePlot (from Google AI) released with the paper DePlot: One-shot visual language reasoning by plot-to-table translation by Fangyu Liu, Julian Martin Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, Yasemin Altun.
    DETA (from The University of Texas at Austin) released with the paper NMS Strikes Back by Jeffrey Ouyang-Zhang, Jang Hyun Cho, Xingyi Zhou, Philipp Krähenbühl.
    DETR (from Facebook) released with the paper End-to-End Object Detection with Transformers by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko.
    DialoGPT (from Microsoft Research) released with the paper DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation by Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
    DiNAT (from SHI Labs) released with the paper Dilated Neighborhood Attention Transformer by Ali Hassani and Humphrey Shi.
    DistilBERT (from HuggingFace), released together with the paper DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter by Victor Sanh, Lysandre Debut and Thomas Wolf. The same method has been applied to compress GPT2 into DistilGPT2, RoBERTa into DistilRoBERTa, Multilingual BERT into DistilmBERT and a German version of DistilBERT.
    DiT (from Microsoft Research) released with the paper DiT: Self-supervised Pre-training for Document Image Transformer by Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei.
    Donut (from NAVER), released together with the paper OCR-free Document Understanding Transformer by Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park.
    DPR (from Facebook) released with the paper Dense Passage Retrieval for Open-Domain Question Answering by Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih.
    DPT (from Intel Labs) released with the paper Vision Transformers for Dense Prediction by René Ranftl, Alexey Bochkovskiy, Vladlen Koltun.
    EfficientFormer (from Snap Research) released with the paper EfficientFormer: Vision Transformers at MobileNetSpeed by Yanyu Li, Geng Yuan, Yang Wen, Ju Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, Jian Ren.
    EfficientNet (from Google Brain) released with the paper EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks by Mingxing Tan, Quoc V. Le.
    ELECTRA (from Google Research/Stanford University) released with the paper ELECTRA: Pre-training text encoders as discriminators rather than generators by Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.
    EnCodec (from Meta AI) released with the paper High Fidelity Neural Audio Compression by Alexandre Défossez, Jade Copet, Gabriel Synnaeve, Yossi Adi.
    EncoderDecoder (from Google Research) released with the paper Leveraging Pre-trained Checkpoints for Sequence Generation Tasks by Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
    ERNIE (from Baidu) released with the paper ERNIE: Enhanced Representation through Knowledge Integration by Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, Hua Wu.
    ErnieM (from Baidu) released with the paper ERNIE-M: Enhanced Multilingual Representation by Aligning Cross-lingual Semantics with Monolingual Corpora by Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang.
    ESM (from Meta AI) are transformer protein language models. ESM-1b was released with the paper Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences by Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma, and Rob Fergus. ESM-1v was released with the paper Language models enable zero-shot prediction of the effects of mutations on protein function by Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu and Alexander Rives. ESM-2 and ESMFold were released with the paper Language models of protein sequences at the scale of evolution enable accurate structure prediction by Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Sal Candido, Alexander Rives.
    Falcon (from Technology Innovation Institute) by Almazrouei, Ebtesam and Alobeidli, Hamza and Alshamsi, Abdulaziz and Cappelli, Alessandro and Cojocaru, Ruxandra and Debbah, Merouane and Goffinet, Etienne and Heslow, Daniel and Launay, Julien and Malartic, Quentin and Noune, Badreddine and Pannier, Baptiste and Penedo, Guilherme.
    FLAN-T5 (from Google AI) released in the repository google-research/t5x by Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei
    FLAN-UL2 (from Google AI) released in the repository google-research/t5x by Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei
    FlauBERT (from CNRS) released with the paper FlauBERT: Unsupervised Language Model Pre-training for French by Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.
    FLAVA (from Facebook AI) released with the paper FLAVA: A Foundational Language And Vision Alignment Model by Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela.
    FNet (from Google Research) released with the paper FNet: Mixing Tokens with Fourier Transforms by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon.
    FocalNet (from Microsoft Research) released with the paper Focal Modulation Networks by Jianwei Yang, Chunyuan Li, Xiyang Dai, Lu Yuan, Jianfeng Gao.
    Funnel Transformer (from CMU/Google Brain) released with the paper Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.
    GIT (from Microsoft Research) released with the paper GIT: A Generative Image-to-text Transformer for Vision and Language by Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, Lijuan Wang.
    GLPN (from KAIST) released with the paper Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth by Doyeon Kim, Woonghyun Ga, Pyungwhan Ahn, Donggyu Joo, Sehwan Chun, Junmo Kim.
    GPT (from OpenAI) released with the paper Improving Language Understanding by Generative Pre-Training by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
    GPT Neo (from EleutherAI) released in the repository EleutherAI/gpt-neo by Sid Black, Stella Biderman, Leo Gao, Phil Wang and Connor Leahy.
    GPT NeoX (from EleutherAI) released with the paper GPT-NeoX-20B: An Open-Source Autoregressive Language Model by Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, Samuel Weinbach
    GPT NeoX Japanese (from ABEJA) released by Shinya Otani, Takayoshi Makabe, Anuj Arora, and Kyo Hattori.
    GPT-2 (from OpenAI) released with the paper Language Models are Unsupervised Multitask Learners by Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodeiand Ilya Sutskever.
    GPT-J (from EleutherAI) released in the repository kingoflolz/mesh-transformer-jax by Ben Wang and Aran Komatsuzaki.
    GPT-Sw3 (from AI-Sweden) released with the paper Lessons Learned from GPT-SW3: Building the First Large-Scale Generative Language Model for Swedish by Ariel Ekgren, Amaru Cuba Gyllensten, Evangelia Gogoulou, Alice Heiman, Severine Verlinden, Joey Öhman, Fredrik Carlsson, Magnus Sahlgren.
    GPTBigCode (from BigCode) released with the paper SantaCoder: don’t reach for the stars! by Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, Logesh Kumar Umapathi, Carolyn Jane Anderson, Yangtian Zi, Joel Lamy Poirier, Hailey Schoelkopf, Sergey Troshin, Dmitry Abulkhanov, Manuel Romero, Michael Lappert, Francesco De Toni, Bernardo García del Río, Qian Liu, Shamik Bose, Urvashi Bhattacharyya, Terry Yue Zhuo, Ian Yu, Paulo Villegas, Marco Zocca, Sourab Mangrulkar, David Lansky, Huu Nguyen, Danish Contractor, Luis Villa, Jia Li, Dzmitry Bahdanau, Yacine Jernite, Sean Hughes, Daniel Fried, Arjun Guha, Harm de Vries, Leandro von Werra.
    GPTSAN-japanese released in the repository tanreinama/GPTSAN by Toshiyuki Sakamoto(tanreinama).
    Graphormer (from Microsoft) released with the paper Do Transformers Really Perform Bad for Graph Representation? by Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, Tie-Yan Liu.
    GroupViT (from UCSD, NVIDIA) released with the paper GroupViT: Semantic Segmentation Emerges from Text Supervision by Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, Xiaolong Wang.
    Hubert (from Facebook) released with the paper HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units by Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed.
    I-BERT (from Berkeley) released with the paper I-BERT: Integer-only BERT Quantization by Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer.
    ImageGPT (from OpenAI) released with the paper Generative Pretraining from Pixels by Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, Ilya Sutskever.
    Informer (from Beihang University, UC Berkeley, Rutgers University, SEDD Company) released with the paper Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting by Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.
    InstructBLIP (from Salesforce) released with the paper InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning by Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi.
    Jukebox (from OpenAI) released with the paper Jukebox: A Generative Model for Music by Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, Ilya Sutskever.
    LayoutLM (from Microsoft Research Asia) released with the paper LayoutLM: Pre-training of Text and Layout for Document Image Understanding by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou.
    LayoutLMv2 (from Microsoft Research Asia) released with the paper LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding by Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou.
    LayoutLMv3 (from Microsoft Research Asia) released with the paper LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking by Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei.
    LayoutXLM (from Microsoft Research Asia) released with the paper LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding by Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei.
    LED (from AllenAI) released with the paper Longformer: The Long-Document Transformer by Iz Beltagy, Matthew E. Peters, Arman Cohan.
    LeViT (from Meta AI) released with the paper LeViT: A Vision Transformer in ConvNet’s Clothing for Faster Inference by Ben Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, Matthijs Douze.
    LiLT (from South China University of Technology) released with the paper LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding by Jiapeng Wang, Lianwen Jin, Kai Ding.
    LLaMA (from The FAIR team of Meta AI) released with the paper LLaMA: Open and Efficient Foundation Language Models by Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample.
    Llama2 (from The FAIR team of Meta AI) released with the paper Llama2: Open Foundation and Fine-Tuned Chat Models by Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushka rMishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing EllenTan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom.

---

"""

long1_prompt = """
Text
---
Transformers

State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX.

Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. These models support common tasks in different modalities, such as:

Natural Language Processing: text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation.
Computer Vision: image classification, object detection, and segmentation.
Audio: automatic speech recognition and audio classification.
Multimodal: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.

Transformers support framework interoperability between PyTorch, TensorFlow, and JAX. This provides the flexibility to use a different framework at each stage of a model’s life; train a model in three lines of code in one framework, and load it for inference in another. Models can also be exported to a format like ONNX and TorchScript for deployment in production environments.

Join the growing community on the Hub, forum, or Discord today!
If you are looking for custom support from the Hugging Face team
HuggingFace Expert Acceleration Program
Contents

The documentation is organized into five sections:

    GET STARTED provides a quick tour of the library and installation instructions to get up and running.

    TUTORIALS are a great place to start if you’re a beginner. This section will help you gain the basic skills you need to start using the library.

    HOW-TO GUIDES show you how to achieve a specific goal, like finetuning a pretrained model for language modeling or how to write and share a custom model.

    CONCEPTUAL GUIDES offers more discussion and explanation of the underlying concepts and ideas behind models, tasks, and the design philosophy of 🤗 Transformers.

    API describes all classes and functions:
        MAIN CLASSES details the most important classes like configuration, model, tokenizer, and pipeline.
        MODELS details the classes and functions related to each model implemented in the library.
        INTERNAL HELPERS details utility classes and functions used internally.

Supported models

    ALBERT (from Google Research and the Toyota Technological Institute at Chicago) released with the paper ALBERT: A Lite BERT for Self-supervised Learning of Language Representations, by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
    ALIGN (from Google Research) released with the paper Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision by Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig.
    AltCLIP (from BAAI) released with the paper AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities by Chen, Zhongzhi and Liu, Guang and Zhang, Bo-Wen and Ye, Fulong and Yang, Qinghong and Wu, Ledell.
    Audio Spectrogram Transformer (from MIT) released with the paper AST: Audio Spectrogram Transformer by Yuan Gong, Yu-An Chung, James Glass.
    Autoformer (from Tsinghua University) released with the paper Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting by Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long.
    Bark (from Suno) released in the repository suno-ai/bark by Suno AI team.
    BART (from Facebook) released with the paper BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer.
    BARThez (from École polytechnique) released with the paper BARThez: a Skilled Pretrained French Sequence-to-Sequence Model by Moussa Kamal Eddine, Antoine J.-P. Tixier, Michalis Vazirgiannis.
    BARTpho (from VinAI Research) released with the paper BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese by Nguyen Luong Tran, Duong Minh Le and Dat Quoc Nguyen.
    BEiT (from Microsoft) released with the paper BEiT: BERT Pre-Training of Image Transformers by Hangbo Bao, Li Dong, Furu Wei.
    BERT (from Google) released with the paper BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
    BERT For Sequence Generation (from Google) released with the paper Leveraging Pre-trained Checkpoints for Sequence Generation Tasks by Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
    BERTweet (from VinAI Research) released with the paper BERTweet: A pre-trained language model for English Tweets by Dat Quoc Nguyen, Thanh Vu and Anh Tuan Nguyen.
    BigBird-Pegasus (from Google Research) released with the paper Big Bird: Transformers for Longer Sequences by Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
    BigBird-RoBERTa (from Google Research) released with the paper Big Bird: Transformers for Longer Sequences by Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
    BioGpt (from Microsoft Research AI4Science) released with the paper BioGPT: generative pre-trained transformer for biomedical text generation and mining by Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon and Tie-Yan Liu.
    BiT (from Google AI) released with the paper Big Transfer (BiT): General Visual Representation Learning by Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby.
    Blenderbot (from Facebook) released with the paper Recipes for building an open-domain chatbot by Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
    BlenderbotSmall (from Facebook) released with the paper Recipes for building an open-domain chatbot by Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
    BLIP (from Salesforce) released with the paper BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation by Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.
    BLIP-2 (from Salesforce) released with the paper BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models by Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi.
    BLOOM (from BigScience workshop) released by the BigScience Workshop.
    BORT (from Alexa) released with the paper Optimal Subarchitecture Extraction For BERT by Adrian de Wynter and Daniel J. Perry.
    BridgeTower (from Harbin Institute of Technology/Microsoft Research Asia/Intel Labs) released with the paper BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning by Xiao Xu, Chenfei Wu, Shachar Rosenman, Vasudev Lal, Wanxiang Che, Nan Duan.
    ByT5 (from Google Research) released with the paper ByT5: Towards a token-free future with pre-trained byte-to-byte models by Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel.
    CamemBERT (from Inria/Facebook/Sorbonne) released with the paper CamemBERT: a Tasty French Language Model by Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
    CANINE (from Google Research) released with the paper CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation by Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting.
    Chinese-CLIP (from OFA-Sys) released with the paper Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese by An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, Chang Zhou.
    CLAP (from LAION-AI) released with the paper Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation by Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov.
    CLIP (from OpenAI) released with the paper Learning Transferable Visual Models From Natural Language Supervision by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.
    CLIPSeg (from University of Göttingen) released with the paper Image Segmentation Using Text and Image Prompts by Timo Lüddecke and Alexander Ecker.
    CodeGen (from Salesforce) released with the paper A Conversational Paradigm for Program Synthesis by Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, Caiming Xiong.
    Conditional DETR (from Microsoft Research Asia) released with the paper Conditional DETR for Fast Training Convergence by Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, Jingdong Wang.
    ConvBERT (from YituTech) released with the paper ConvBERT: Improving BERT with Span-based Dynamic Convolution by Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan.
    ConvNeXT (from Facebook AI) released with the paper A ConvNet for the 2020s by Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie.
    ConvNeXTV2 (from Facebook AI) released with the paper ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders by Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie.
    CPM (from Tsinghua University) released with the paper CPM: A Large-scale Generative Chinese Pre-trained Language Model by Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.
    CPM-Ant (from OpenBMB) released by the OpenBMB.
    CTRL (from Salesforce) released with the paper CTRL: A Conditional Transformer Language Model for Controllable Generation by Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong and Richard Socher.
    CvT (from Microsoft) released with the paper CvT: Introducing Convolutions to Vision Transformers by Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang.
    Data2Vec (from Facebook) released with the paper Data2Vec: A General Framework for Self-supervised Learning in Speech, Vision and Language by Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli.
    DeBERTa (from Microsoft) released with the paper DeBERTa: Decoding-enhanced BERT with Disentangled Attention by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
    DeBERTa-v2 (from Microsoft) released with the paper DeBERTa: Decoding-enhanced BERT with Disentangled Attention by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
    Decision Transformer (from Berkeley/Facebook/Google) released with the paper Decision Transformer: Reinforcement Learning via Sequence Modeling by Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch.
    Deformable DETR (from SenseTime Research) released with the paper Deformable DETR: Deformable Transformers for End-to-End Object Detection by Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai.
    DeiT (from Facebook) released with the paper Training data-efficient image transformers & distillation through attention by Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou.
    DePlot (from Google AI) released with the paper DePlot: One-shot visual language reasoning by plot-to-table translation by Fangyu Liu, Julian Martin Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, Yasemin Altun.
    DETA (from The University of Texas at Austin) released with the paper NMS Strikes Back by Jeffrey Ouyang-Zhang, Jang Hyun Cho, Xingyi Zhou, Philipp Krähenbühl.
    DETR (from Facebook) released with the paper End-to-End Object Detection with Transformers by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko.
    DialoGPT (from Microsoft Research) released with the paper DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation by Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
    DiNAT (from SHI Labs) released with the paper Dilated Neighborhood Attention Transformer by Ali Hassani and Humphrey Shi.
    DistilBERT (from HuggingFace), released together with the paper DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter by Victor Sanh, Lysandre Debut and Thomas Wolf. The same method has been applied to compress GPT2 into DistilGPT2, RoBERTa into DistilRoBERTa, Multilingual BERT into DistilmBERT and a German version of DistilBERT.
    DiT (from Microsoft Research) released with the paper DiT: Self-supervised Pre-training for Document Image Transformer by Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei.
    Donut (from NAVER), released together with the paper OCR-free Document Understanding Transformer by Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park.
    DPR (from Facebook) released with the paper Dense Passage Retrieval for Open-Domain Question Answering by Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih.
    DPT (from Intel Labs) released with the paper Vision Transformers for Dense Prediction by René Ranftl, Alexey Bochkovskiy, Vladlen Koltun.
    EfficientFormer (from Snap Research) released with the paper EfficientFormer: Vision Transformers at MobileNetSpeed by Yanyu Li, Geng Yuan, Yang Wen, Ju Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, Jian Ren.
    EfficientNet (from Google Brain) released with the paper EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks by Mingxing Tan, Quoc V. Le.
    ELECTRA (from Google Research/Stanford University) released with the paper ELECTRA: Pre-training text encoders as discriminators rather than generators by Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.
    EnCodec (from Meta AI) released with the paper High Fidelity Neural Audio Compression by Alexandre Défossez, Jade Copet, Gabriel Synnaeve, Yossi Adi.
    EncoderDecoder (from Google Research) released with the paper Leveraging Pre-trained Checkpoints for Sequence Generation Tasks by Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
    ERNIE (from Baidu) released with the paper ERNIE: Enhanced Representation through Knowledge Integration by Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, Hua Wu.
    ErnieM (from Baidu) released with the paper ERNIE-M: Enhanced Multilingual Representation by Aligning Cross-lingual Semantics with Monolingual Corpora by Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang.
    ESM (from Meta AI) are transformer protein language models. ESM-1b was released with the paper Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences by Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma, and Rob Fergus. ESM-1v was released with the paper Language models enable zero-shot prediction of the effects of mutations on protein function by Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu and Alexander Rives. ESM-2 and ESMFold were released with the paper Language models of protein sequences at the scale of evolution enable accurate structure prediction by Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Sal Candido, Alexander Rives.
    Falcon (from Technology Innovation Institute) by Almazrouei, Ebtesam and Alobeidli, Hamza and Alshamsi, Abdulaziz and Cappelli, Alessandro and Cojocaru, Ruxandra and Debbah, Merouane and Goffinet, Etienne and Heslow, Daniel and Launay, Julien and Malartic, Quentin and Noune, Badreddine and Pannier, Baptiste and Penedo, Guilherme.
    FLAN-T5 (from Google AI) released in the repository google-research/t5x by Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei
    FLAN-UL2 (from Google AI) released in the repository google-research/t5x by Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei
    FlauBERT (from CNRS) released with the paper FlauBERT: Unsupervised Language Model Pre-training for French by Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.
    FLAVA (from Facebook AI) released with the paper FLAVA: A Foundational Language And Vision Alignment Model by Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela.
    FNet (from Google Research) released with the paper FNet: Mixing Tokens with Fourier Transforms by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon.
    FocalNet (from Microsoft Research) released with the paper Focal Modulation Networks by Jianwei Yang, Chunyuan Li, Xiyang Dai, Lu Yuan, Jianfeng Gao.
    Funnel Transformer (from CMU/Google Brain) released with the paper Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.
    GIT (from Microsoft Research) released with the paper GIT: A Generative Image-to-text Transformer for Vision and Language by Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, Lijuan Wang.
    GLPN (from KAIST) released with the paper Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth by Doyeon Kim, Woonghyun Ga, Pyungwhan Ahn, Donggyu Joo, Sehwan Chun, Junmo Kim.
    GPT (from OpenAI) released with the paper Improving Language Understanding by Generative Pre-Training by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
    GPT Neo (from EleutherAI) released in the repository EleutherAI/gpt-neo by Sid Black, Stella Biderman, Leo Gao, Phil Wang and Connor Leahy.
    GPT NeoX (from EleutherAI) released with the paper GPT-NeoX-20B: An Open-Source Autoregressive Language Model by Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, Samuel Weinbach
    GPT NeoX Japanese (from ABEJA) released by Shinya Otani, Takayoshi Makabe, Anuj Arora, and Kyo Hattori.
    GPT-2 (from OpenAI) released with the paper Language Models are Unsupervised Multitask Learners by Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodeiand Ilya Sutskever.
    GPT-J (from EleutherAI) released in the repository kingoflolz/mesh-transformer-jax by Ben Wang and Aran Komatsuzaki.

---

""" # 6500

long2_prompt = """
Text
---
Transformers

State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX.

Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. These models support common tasks in different modalities, such as:

Natural Language Processing: text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation.
Computer Vision: image classification, object detection, and segmentation.
Audio: automatic speech recognition and audio classification.
Multimodal: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.

Transformers support framework interoperability between PyTorch, TensorFlow, and JAX. This provides the flexibility to use a different framework at each stage of a model’s life; train a model in three lines of code in one framework, and load it for inference in another. Models can also be exported to a format like ONNX and TorchScript for deployment in production environments.

Join the growing community on the Hub, forum, or Discord today!
If you are looking for custom support from the Hugging Face team
HuggingFace Expert Acceleration Program
Contents

The documentation is organized into five sections:

    GET STARTED provides a quick tour of the library and installation instructions to get up and running.

    TUTORIALS are a great place to start if you’re a beginner. This section will help you gain the basic skills you need to start using the library.

    HOW-TO GUIDES show you how to achieve a specific goal, like finetuning a pretrained model for language modeling or how to write and share a custom model.

    CONCEPTUAL GUIDES offers more discussion and explanation of the underlying concepts and ideas behind models, tasks, and the design philosophy of 🤗 Transformers.

    API describes all classes and functions:
        MAIN CLASSES details the most important classes like configuration, model, tokenizer, and pipeline.
        MODELS details the classes and functions related to each model implemented in the library.
        INTERNAL HELPERS details utility classes and functions used internally.

Supported models

    ALBERT (from Google Research and the Toyota Technological Institute at Chicago) released with the paper ALBERT: A Lite BERT for Self-supervised Learning of Language Representations, by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
    ALIGN (from Google Research) released with the paper Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision by Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig.
    AltCLIP (from BAAI) released with the paper AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities by Chen, Zhongzhi and Liu, Guang and Zhang, Bo-Wen and Ye, Fulong and Yang, Qinghong and Wu, Ledell.
    Audio Spectrogram Transformer (from MIT) released with the paper AST: Audio Spectrogram Transformer by Yuan Gong, Yu-An Chung, James Glass.
    Autoformer (from Tsinghua University) released with the paper Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting by Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long.
    Bark (from Suno) released in the repository suno-ai/bark by Suno AI team.
    BART (from Facebook) released with the paper BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer.
    BARThez (from École polytechnique) released with the paper BARThez: a Skilled Pretrained French Sequence-to-Sequence Model by Moussa Kamal Eddine, Antoine J.-P. Tixier, Michalis Vazirgiannis.
    BARTpho (from VinAI Research) released with the paper BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese by Nguyen Luong Tran, Duong Minh Le and Dat Quoc Nguyen.
    BEiT (from Microsoft) released with the paper BEiT: BERT Pre-Training of Image Transformers by Hangbo Bao, Li Dong, Furu Wei.
    BERT (from Google) released with the paper BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
    BERT For Sequence Generation (from Google) released with the paper Leveraging Pre-trained Checkpoints for Sequence Generation Tasks by Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
    BERTweet (from VinAI Research) released with the paper BERTweet: A pre-trained language model for English Tweets by Dat Quoc Nguyen, Thanh Vu and Anh Tuan Nguyen.
    BigBird-Pegasus (from Google Research) released with the paper Big Bird: Transformers for Longer Sequences by Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
    BigBird-RoBERTa (from Google Research) released with the paper Big Bird: Transformers for Longer Sequences by Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
    BioGpt (from Microsoft Research AI4Science) released with the paper BioGPT: generative pre-trained transformer for biomedical text generation and mining by Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon and Tie-Yan Liu.
    BiT (from Google AI) released with the paper Big Transfer (BiT): General Visual Representation Learning by Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby.
    Blenderbot (from Facebook) released with the paper Recipes for building an open-domain chatbot by Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
    BlenderbotSmall (from Facebook) released with the paper Recipes for building an open-domain chatbot by Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
    BLIP (from Salesforce) released with the paper BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation by Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.
    BLIP-2 (from Salesforce) released with the paper BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models by Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi.
    BLOOM (from BigScience workshop) released by the BigScience Workshop.
    BORT (from Alexa) released with the paper Optimal Subarchitecture Extraction For BERT by Adrian de Wynter and Daniel J. Perry.
    BridgeTower (from Harbin Institute of Technology/Microsoft Research Asia/Intel Labs) released with the paper BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning by Xiao Xu, Chenfei Wu, Shachar Rosenman, Vasudev Lal, Wanxiang Che, Nan Duan.
    ByT5 (from Google Research) released with the paper ByT5: Towards a token-free future with pre-trained byte-to-byte models by Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel.
    CamemBERT (from Inria/Facebook/Sorbonne) released with the paper CamemBERT: a Tasty French Language Model by Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
    CANINE (from Google Research) released with the paper CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation by Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting.
    Chinese-CLIP (from OFA-Sys) released with the paper Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese by An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, Chang Zhou.
    CLAP (from LAION-AI) released with the paper Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation by Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov.
    CLIP (from OpenAI) released with the paper Learning Transferable Visual Models From Natural Language Supervision by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.
    CLIPSeg (from University of Göttingen) released with the paper Image Segmentation Using Text and Image Prompts by Timo Lüddecke and Alexander Ecker.
    CodeGen (from Salesforce) released with the paper A Conversational Paradigm for Program Synthesis by Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, Caiming Xiong.
    Conditional DETR (from Microsoft Research Asia) released with the paper Conditional DETR for Fast Training Convergence by Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, Jingdong Wang.
    ConvBERT (from YituTech) released with the paper ConvBERT: Improving BERT with Span-based Dynamic Convolution by Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan.
    ConvNeXT (from Facebook AI) released with the paper A ConvNet for the 2020s by Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie.
    ConvNeXTV2 (from Facebook AI) released with the paper ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders by Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie.
    CPM (from Tsinghua University) released with the paper CPM: A Large-scale Generative Chinese Pre-trained Language Model by Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.
    CPM-Ant (from OpenBMB) released by the OpenBMB.
    CTRL (from Salesforce) released with the paper CTRL: A Conditional Transformer Language Model for Controllable Generation by Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong and Richard Socher.
    CvT (from Microsoft) released with the paper CvT: Introducing Convolutions to Vision Transformers by Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang.
    Data2Vec (from Facebook) released with the paper Data2Vec: A General Framework for Self-supervised Learning in Speech, Vision and Language by Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli.
    DeBERTa (from Microsoft) released with the paper DeBERTa: Decoding-enhanced BERT with Disentangled Attention by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
    DeBERTa-v2 (from Microsoft) released with the paper DeBERTa: Decoding-enhanced BERT with Disentangled Attention by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
    Decision Transformer (from Berkeley/Facebook/Google) released with the paper Decision Transformer: Reinforcement Learning via Sequence Modeling by Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch.
    Deformable DETR (from SenseTime Research) released with the paper Deformable DETR: Deformable Transformers for End-to-End Object Detection by Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai.
    DeiT (from Facebook) released with the paper Training data-efficient image transformers & distillation through attention by Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou.
    DePlot (from Google AI) released with the paper DePlot: One-shot visual language reasoning by plot-to-table translation by Fangyu Liu, Julian Martin Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, Yasemin Altun.
    DETA (from The University of Texas at Austin) released with the paper NMS Strikes Back by Jeffrey Ouyang-Zhang, Jang Hyun Cho, Xingyi Zhou, Philipp Krähenbühl.
    DETR (from Facebook) released with the paper End-to-End Object Detection with Transformers by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko.
    DialoGPT (from Microsoft Research) released with the paper DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation by Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
    DiNAT (from SHI Labs) released with the paper Dilated Neighborhood Attention Transformer by Ali Hassani and Humphrey Shi.
    DistilBERT (from HuggingFace), released together with the paper DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter by Victor Sanh, Lysandre Debut and Thomas Wolf. The same method has been applied to compress GPT2 into DistilGPT2, RoBERTa into DistilRoBERTa, Multilingual BERT into DistilmBERT and a German version of DistilBERT.
    DiT (from Microsoft Research) released with the paper DiT: Self-supervised Pre-training for Document Image Transformer by Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei.
    Donut (from NAVER), released together with the paper OCR-free Document Understanding Transformer by Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park.
    DPR (from Facebook) released with the paper Dense Passage Retrieval for Open-Domain Question Answering by Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih.
    DPT (from Intel Labs) released with the paper Vision Transformers for Dense Prediction by René Ranftl, Alexey Bochkovskiy, Vladlen Koltun.
    EfficientFormer (from Snap Research) released with the paper EfficientFormer: Vision Transformers at MobileNetSpeed by Yanyu Li, Geng Yuan, Yang Wen, Ju Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, Jian Ren.
    EfficientNet (from Google Brain) released with the paper EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks by Mingxing Tan, Quoc V. Le.

---

""" # 4600

long3_prompt = """
Text
---
Transformers

State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX.

Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. These models support common tasks in different modalities, such as:

Natural Language Processing: text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation.
Computer Vision: image classification, object detection, and segmentation.
Audio: automatic speech recognition and audio classification.
Multimodal: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.

Transformers support framework interoperability between PyTorch, TensorFlow, and JAX. This provides the flexibility to use a different framework at each stage of a model’s life; train a model in three lines of code in one framework, and load it for inference in another. Models can also be exported to a format like ONNX and TorchScript for deployment in production environments.

Join the growing community on the Hub, forum, or Discord today!
If you are looking for custom support from the Hugging Face team
HuggingFace Expert Acceleration Program
Contents

The documentation is organized into five sections:

    GET STARTED provides a quick tour of the library and installation instructions to get up and running.

    TUTORIALS are a great place to start if you’re a beginner. This section will help you gain the basic skills you need to start using the library.

    HOW-TO GUIDES show you how to achieve a specific goal, like finetuning a pretrained model for language modeling or how to write and share a custom model.

    CONCEPTUAL GUIDES offers more discussion and explanation of the underlying concepts and ideas behind models, tasks, and the design philosophy of 🤗 Transformers.

    API describes all classes and functions:
        MAIN CLASSES details the most important classes like configuration, model, tokenizer, and pipeline.
        MODELS details the classes and functions related to each model implemented in the library.
        INTERNAL HELPERS details utility classes and functions used internally.

Supported models

    ALBERT (from Google Research and the Toyota Technological Institute at Chicago) released with the paper ALBERT: A Lite BERT for Self-supervised Learning of Language Representations, by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
    ALIGN (from Google Research) released with the paper Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision by Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig.
    AltCLIP (from BAAI) released with the paper AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities by Chen, Zhongzhi and Liu, Guang and Zhang, Bo-Wen and Ye, Fulong and Yang, Qinghong and Wu, Ledell.
    Audio Spectrogram Transformer (from MIT) released with the paper AST: Audio Spectrogram Transformer by Yuan Gong, Yu-An Chung, James Glass.
    Autoformer (from Tsinghua University) released with the paper Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting by Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long.
    Bark (from Suno) released in the repository suno-ai/bark by Suno AI team.
    BART (from Facebook) released with the paper BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer.
    BARThez (from École polytechnique) released with the paper BARThez: a Skilled Pretrained French Sequence-to-Sequence Model by Moussa Kamal Eddine, Antoine J.-P. Tixier, Michalis Vazirgiannis.
    BARTpho (from VinAI Research) released with the paper BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese by Nguyen Luong Tran, Duong Minh Le and Dat Quoc Nguyen.
    BEiT (from Microsoft) released with the paper BEiT: BERT Pre-Training of Image Transformers by Hangbo Bao, Li Dong, Furu Wei.
    BERT (from Google) released with the paper BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
    BERT For Sequence Generation (from Google) released with the paper Leveraging Pre-trained Checkpoints for Sequence Generation Tasks by Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
    BERTweet (from VinAI Research) released with the paper BERTweet: A pre-trained language model for English Tweets by Dat Quoc Nguyen, Thanh Vu and Anh Tuan Nguyen.
    BigBird-Pegasus (from Google Research) released with the paper Big Bird: Transformers for Longer Sequences by Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
    BigBird-RoBERTa (from Google Research) released with the paper Big Bird: Transformers for Longer Sequences by Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
    BioGpt (from Microsoft Research AI4Science) released with the paper BioGPT: generative pre-trained transformer for biomedical text generation and mining by Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon and Tie-Yan Liu.
    BiT (from Google AI) released with the paper Big Transfer (BiT): General Visual Representation Learning by Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby.
    Blenderbot (from Facebook) released with the paper Recipes for building an open-domain chatbot by Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
    BlenderbotSmall (from Facebook) released with the paper Recipes for building an open-domain chatbot by Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
    BLIP (from Salesforce) released with the paper BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation by Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.
    BLIP-2 (from Salesforce) released with the paper BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models by Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi.
    BLOOM (from BigScience workshop) released by the BigScience Workshop.
    BORT (from Alexa) released with the paper Optimal Subarchitecture Extraction For BERT by Adrian de Wynter and Daniel J. Perry.
    BridgeTower (from Harbin Institute of Technology/Microsoft Research Asia/Intel Labs) released with the paper BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning by Xiao Xu, Chenfei Wu, Shachar Rosenman, Vasudev Lal, Wanxiang Che, Nan Duan.
    ByT5 (from Google Research) released with the paper ByT5: Towards a token-free future with pre-trained byte-to-byte models by Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel.
    CamemBERT (from Inria/Facebook/Sorbonne) released with the paper CamemBERT: a Tasty French Language Model by Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
    CANINE (from Google Research) released with the paper CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation by Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting.
    Chinese-CLIP (from OFA-Sys) released with the paper Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese by An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, Chang Zhou.
    CLAP (from LAION-AI) released with the paper Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation by Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov.
    CLIP (from OpenAI) released with the paper Learning Transferable Visual Models From Natural Language Supervision by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.
    CLIPSeg (from University of Göttingen) released with the paper Image Segmentation Using Text and Image Prompts by Timo Lüddecke and Alexander Ecker.
    CodeGen (from Salesforce) released with the paper A Conversational Paradigm for Program Synthesis by Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, Caiming Xiong.
    Conditional DETR (from Microsoft Research Asia) released with the paper Conditional DETR for Fast Training Convergence by Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, Jingdong Wang.
    ConvBERT (from YituTech) released with the paper ConvBERT: Improving BERT with Span-based Dynamic Convolution by Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan.
    ConvNeXT (from Facebook AI) released with the paper A ConvNet for the 2020s by Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie.
    ConvNeXTV2 (from Facebook AI) released with the paper ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders by Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie.
    CPM (from Tsinghua University) released with the paper CPM: A Large-scale Generative Chinese Pre-trained Language Model by Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.
    CPM-Ant (from OpenBMB) released by the OpenBMB.
    CTRL (from Salesforce) released with the paper CTRL: A Conditional Transformer Language Model for Controllable Generation by Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong and Richard Socher.
    CvT (from Microsoft) released with the paper CvT: Introducing Convolutions to Vision Transformers by Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang.
    Data2Vec (from Facebook) released with the paper Data2Vec: A General Framework for Self-supervised Learning in Speech, Vision and Language by Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli.
    DeBERTa (from Microsoft) released with the paper DeBERTa: Decoding-enhanced BERT with Disentangled Attention by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
    DeBERTa-v2 (from Microsoft) released with the paper DeBERTa: Decoding-enhanced BERT with Disentangled Attention by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
    Decision Transformer (from Berkeley/Facebook/Google) released with the paper Decision Transformer: Reinforcement Learning via Sequence Modeling by Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch.
    Deformable DETR (from SenseTime Research) released with the paper Deformable DETR: Deformable Transformers for End-to-End Object Detection by Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai.
    DeiT (from Facebook) released with the paper Training data-efficient image transformers & distillation through attention by Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou.

---
Here is the main content and summary of the article above, including the main points, the last supported model and the list of supported frameworks:

""" # 3800

long4_prompt = """
Text
---
Transformers

State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX.

Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. These models support common tasks in different modalities, such as:

Natural Language Processing: text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation.
Computer Vision: image classification, object detection, and segmentation.
Audio: automatic speech recognition and audio classification.
Multimodal: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.

Transformers support framework interoperability between PyTorch, TensorFlow, and JAX. This provides the flexibility to use a different framework at each stage of a model’s life; train a model in three lines of code in one framework, and load it for inference in another. Models can also be exported to a format like ONNX and TorchScript for deployment in production environments.

Join the growing community on the Hub, forum, or Discord today!
If you are looking for custom support from the Hugging Face team
HuggingFace Expert Acceleration Program
Contents

The documentation is organized into five sections:

    GET STARTED provides a quick tour of the library and installation instructions to get up and running.

    TUTORIALS are a great place to start if you’re a beginner. This section will help you gain the basic skills you need to start using the library.

    HOW-TO GUIDES show you how to achieve a specific goal, like finetuning a pretrained model for language modeling or how to write and share a custom model.

    CONCEPTUAL GUIDES offers more discussion and explanation of the underlying concepts and ideas behind models, tasks, and the design philosophy of 🤗 Transformers.

    API describes all classes and functions:
        MAIN CLASSES details the most important classes like configuration, model, tokenizer, and pipeline.
        MODELS details the classes and functions related to each model implemented in the library.
        INTERNAL HELPERS details utility classes and functions used internally.

Supported models

    ALBERT (from Google Research and the Toyota Technological Institute at Chicago) released with the paper ALBERT: A Lite BERT for Self-supervised Learning of Language Representations, by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
    ALIGN (from Google Research) released with the paper Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision by Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig.
    AltCLIP (from BAAI) released with the paper AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities by Chen, Zhongzhi and Liu, Guang and Zhang, Bo-Wen and Ye, Fulong and Yang, Qinghong and Wu, Ledell.
    Audio Spectrogram Transformer (from MIT) released with the paper AST: Audio Spectrogram Transformer by Yuan Gong, Yu-An Chung, James Glass.
    Autoformer (from Tsinghua University) released with the paper Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting by Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long.
    Bark (from Suno) released in the repository suno-ai/bark by Suno AI team.
    BART (from Facebook) released with the paper BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer.
    BARThez (from École polytechnique) released with the paper BARThez: a Skilled Pretrained French Sequence-to-Sequence Model by Moussa Kamal Eddine, Antoine J.-P. Tixier, Michalis Vazirgiannis.
    BARTpho (from VinAI Research) released with the paper BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese by Nguyen Luong Tran, Duong Minh Le and Dat Quoc Nguyen.
    BEiT (from Microsoft) released with the paper BEiT: BERT Pre-Training of Image Transformers by Hangbo Bao, Li Dong, Furu Wei.
    BERT (from Google) released with the paper BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
    BERT For Sequence Generation (from Google) released with the paper Leveraging Pre-trained Checkpoints for Sequence Generation Tasks by Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
    BERTweet (from VinAI Research) released with the paper BERTweet: A pre-trained language model for English Tweets by Dat Quoc Nguyen, Thanh Vu and Anh Tuan Nguyen.
    BigBird-Pegasus (from Google Research) released with the paper Big Bird: Transformers for Longer Sequences by Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
    BigBird-RoBERTa (from Google Research) released with the paper Big Bird: Transformers for Longer Sequences by Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
    BioGpt (from Microsoft Research AI4Science) released with the paper BioGPT: generative pre-trained transformer for biomedical text generation and mining by Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon and Tie-Yan Liu.
    BiT (from Google AI) released with the paper Big Transfer (BiT): General Visual Representation Learning by Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby.
    Blenderbot (from Facebook) released with the paper Recipes for building an open-domain chatbot by Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
    BlenderbotSmall (from Facebook) released with the paper Recipes for building an open-domain chatbot by Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
    BLIP (from Salesforce) released with the paper BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation by Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.
    BLIP-2 (from Salesforce) released with the paper BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models by Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi.
    BLOOM (from BigScience workshop) released by the BigScience Workshop.
    BORT (from Alexa) released with the paper Optimal Subarchitecture Extraction For BERT by Adrian de Wynter and Daniel J. Perry.
    BridgeTower (from Harbin Institute of Technology/Microsoft Research Asia/Intel Labs) released with the paper BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning by Xiao Xu, Chenfei Wu, Shachar Rosenman, Vasudev Lal, Wanxiang Che, Nan Duan.
    ByT5 (from Google Research) released with the paper ByT5: Towards a token-free future with pre-trained byte-to-byte models by Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel.
    CamemBERT (from Inria/Facebook/Sorbonne) released with the paper CamemBERT: a Tasty French Language Model by Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
    CANINE (from Google Research) released with the paper CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation by Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting.
    Chinese-CLIP (from OFA-Sys) released with the paper Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese by An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, Chang Zhou.
    CLAP (from LAION-AI) released with the paper Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation by Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov.
    CLIP (from OpenAI) released with the paper Learning Transferable Visual Models From Natural Language Supervision by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.
    CLIPSeg (from University of Göttingen) released with the paper Image Segmentation Using Text and Image Prompts by Timo Lüddecke and Alexander Ecker.
    CodeGen (from Salesforce) released with the paper A Conversational Paradigm for Program Synthesis by Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, Caiming Xiong.

---

""" # 2800\

base_prompt = """
Text
---
Transformers

State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX.

Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. These models support common tasks in different modalities, such as:

Natural Language Processing: text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation.
Computer Vision: image classification, object detection, and segmentation.
Audio: automatic speech recognition and audio classification.
Multimodal: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.

Transformers support framework interoperability between PyTorch, TensorFlow, and JAX. This provides the flexibility to use a different framework at each stage of a model’s life; train a model in three lines of code in one framework, and load it for inference in another. Models can also be exported to a format like ONNX and TorchScript for deployment in production environments.

Join the growing community on the Hub, forum, or Discord today!
If you are looking for custom support from the Hugging Face team
HuggingFace Expert Acceleration Program
Contents

The documentation is organized into five sections:

    GET STARTED provides a quick tour of the library and installation instructions to get up and running.

    TUTORIALS are a great place to start if you’re a beginner. This section will help you gain the basic skills you need to start using the library.

    HOW-TO GUIDES show you how to achieve a specific goal, like finetuning a pretrained model for language modeling or how to write and share a custom model.

    CONCEPTUAL GUIDES offers more discussion and explanation of the underlying concepts and ideas behind models, tasks, and the design philosophy of 🤗 Transformers.

    API describes all classes and functions:
        MAIN CLASSES details the most important classes like configuration, model, tokenizer, and pipeline.
        MODELS details the classes and functions related to each model implemented in the library.
        INTERNAL HELPERS details utility classes and functions used internally.

Supported models

    ALBERT (from Google Research and the Toyota Technological Institute at Chicago) released with the paper ALBERT: A Lite BERT for Self-supervised Learning of Language Representations, by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
    ALIGN (from Google Research) released with the paper Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision by Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig.

---

""" # 850

QUESTIONS = [
    "The following is a short summary of the article. It's about a package (or framework) called: ",
    "The following is a short summary of the article. The documentation has five sections: ",
    "The last supported model mentioned in the article is: ",
    "The first supported model mentioned in the article is: ",
]

### Models
import torch
from transformers import LlamaTokenizer

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Run llama-65b")
    parser.add_argument("-p", "--prompt", type=str, default="short", choices=["short", "middle", "long", "long1", "long2", "long3", "long4", "base"])
    parser.add_argument("-m", "--model", type=str, default="65b", choices=["65b", "7b", "13b"])
    parser.add_argument("-f", "--flash", action="store_true", help="Use Flash")
    parser.add_argument("-t", "--together", action="store_true", help="Use Together")
    parser.add_argument("--max_len", default=None)
    parser.add_argument("-i", "--interpolate", type=str, choices=["linear", "dynamic"], default=None)
    parser.add_argument("-s", "--factor", type=int, default=None)
    parser.add_argument("-q", "--question", type=int, default=0, help="Question to ask, needed for long and base prompts")
    return parser   


MAX_LENS = {
    "short": 800,   # 10
    "middle": 2000, # 800
    "long": 12000,  # 8500
    "long1": 8000,  # 6500
    "long2": 6000,  # 4600
    "long3": 4500,  # 3800
    "long4": 3500,  # 2800
    "base": 1500,   # 800
}

MAX_LENGTHS = {
    "short": 2048,   # 10
    "middle": 2048, # 800
    "long": 2048 * 8,  # 8500
    "long1": 2048 * 4,  # 6500
    "long2": 2048 * 4,  # 4600
    "long3": 2048 * 2,  # 3800
    "long4": 2048 * 2,  # 2800
    "base": 2048,   # 800
}

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.interpolate is not None:
        rope_scaling = {
            "type": args.interpolate,
            "factor": args.factor,
        }

    if args.flash:
        if args.together:
            from modeling_flash_llama_raw import LlamaForCausalLM
        else:
            from modeling_flash_llama import LlamaForCausalLM  
    else:
        from transformers import LlamaForCausalLM

    if args.interpolate is not None:
        model = LlamaForCausalLM.from_pretrained(f"huggyllama/llama-{args.model}", torch_dtype=torch.float16, rope_scaling=rope_scaling, device_map="auto")
    else:
        model = LlamaForCausalLM.from_pretrained(f"huggyllama/llama-{args.model}", torch_dtype=torch.float16, device_map="auto")
    
    tokenizer = LlamaTokenizer.from_pretrained(f"huggyllama/llama-{args.model}")

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    prompt = eval(f"{args.prompt}_prompt")

    if args.prompt.startswith("long") or args.prompt == "base":
        prompt += QUESTIONS[args.question]

    print(model.config)

    # max_len
    if args.max_len is None:
        max_len = MAX_LENS[args.prompt]
    else:
        max_len = int(args.max_len)

    ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
    print("Input length: {}".format(len(ids[0])))

    import time
    start = time.time()
    gen_ids = model.generate(ids, max_length=max_len)
    end = time.time()
    # only decode the generated ids, strip the prompt
    print(tokenizer.decode(gen_ids[0][len(ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    print("Time: {}".format(end - start))
    print("Total length: {}".format(len(gen_ids[0])))


if __name__ == "__main__":
    main()