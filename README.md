Find the code in Llama3_2/llama3_summary.py

---
title: "Multilingual Image Corpus – Towards a Multimodal and Multilingual Dataset"
slug: "/lwdozal/paper-summary"
date: 2024-11-02
author: Laura W. Dozal
description: "Paper summary of Multilingual Image Corpus – Towards a Multimodal and Multilingual Dataset"
tags:
  - paper summary
---


## Citation

Svetla Koeva, Ivelina Stoyanova, and Jordan Kralev. 2022. Multilingual Image Corpus – Towards a Multimodal and Multilingual Dataset. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 1509–1518, Marseille, France. European Language Resources Association.


<table>
  <caption>
    Citation summary
  </caption>
  <thead>
  <tr>
    <th></th>
    <th></th>
  </tr>
  </thead>
<tbody>
  <tr>
    <td><b>Paper</b></td>
    <td>Multilingual Image Corpus – Towards a Multimodal and Multilingual Dataset</td>
  </tr>
  <tr>
    <td><b>Authors</b></td>
    <td>Svetla Koeva, Ivelina Stoyanova, Jordan Kralev</td>
  </tr>
  <tr>
    <td><b>Year published</b></td>
    <td>2022</td>
  </tr>
  <tr>
    <td><b>Venue</b></td>
    <td> In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 1509–1518, Marseille, France. European Language Resources Association.</td>
  </tr>
  <tr>
    <td><b>Paper URL</b></td>
    <td>https://aclanthology.org/2022.lrec-1.162/</td>
  </tr>
  <tr>
    <td><b>Code URL</b></td>
    <td>https://live.european-language-grid.eu/catalogue/project/8398</td>
  </tr>
</tbody>
</table>

## Description

_In your own words, what is this paper about?_ 

The paper researched how they could create a multilingual image corpus to enable future exploration on multilingual and multimodal data content.The authors created the dataset created from sources like wikimedia, pexels, pixababy, Flickr, CC Search API, and a few others. These images are setup to provide pixel-level annotations for selected dominant classes within their parent and attribute classes in four thematic domains (Sport, Transport, ARts, and Securty). The deep learning implementation in this paper generates annotations for images. Much of the annotation was done in conjuction with the COCO annotator dataset (which allows for simultaneous work on a project and can track object instances, label objects with disconnected visible parts, etc.). \
The authors implemented an image processing pipeline from a multi-layer network architecture for object detection and object segmentation using the same pre-trained models as the COCO labeling domain. The models were YOLACAT and Resnet50-FPN backbone and Detectron2 while using Fast-RCNN architecture with a Resnet backbone. \
The dataset created from this pipeline enables exploration using models specializing in object detection, segmentation and classificatoin.The dataset has annotated objects and object descriptions in 25 lanaguages set-up in an ontology of visual objects, mirroring that of WordNet.

## Motivation

_Why did you select this paper?_
<!-- NOTE: don't use an LLM to generate this! 🙃 -->

Selfishly (or not?) I'm bulding a mini image-text dataset myself. This dataset will be composed of ~11K Instagram posts that have comments that represent multilingual textual content. I hope this paper can be a good foundation for how I set-up my own data processing and dataset creation using image and text analysis.


## LLM

<table>
  <caption>
    LLM model summary
  </caption>
  <thead>
  <tr>
    <th></th>
    <th></th>
  </tr>
  </thead>
<tbody>
  <tr>
    <th><b>LLM model</b></th>
    <td>Llama-3.2 and Tiny Llama</td>
  </tr>
  <tr>
    <th><b>LLM model version</b></th>
    <td>Llama-3.2-3B-Instruct and PY007/TinyLlama-1.1B-Chat-v0.1, tested: PY007/TinyLlama-1.1B-step-50K-105b</td>
  </tr>
  <tr>
    <th><b>Model/service URL</b></th>
    <td>https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct \n 
    https://huggingface.co/TinyLlama/TinyLlama-1.1B-step-50K-105b \n
    https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.1</td>
  </tr>
  <tr>
    <th><b>Why this model?</b></th>
    <td>Works for multilingual dialogue use cases and works with agentic and retrieval summarization tasks. Somewhat similar to RAG (Retrieval Augmented Generalization). It also has refinement learning with human feedback. I switched to the tiny llama version because it takes up less space on my computer and I was having a difficult time loading and working with the large model, even in the HPC.</td>
  </tr>
</tbody>
</table>


#### Description (LLM)

_In the LLM's words, what is this paper about?_

##### Prompt

```markdown
Please, provide an academic summary for this document.
```
This document discusses the concept of dominant classes within thematic domains, which represent the main "players" or types of entities within a particular field of study. The dominant classes are identified using WordNet sister hyponyms at a specific level, allowing for classification without requiring specialized knowledge of the domain. The document reports on the selection of dominant classes for 137 thematic domains, resulting in 569 attribute classes and 145 parent classes. Additionally, the document describes the recursive extraction of parent classes from the WordNet noun hierarchies to identify the final visual object class. The approach is used to represent the main concepts and entities within a domain in a structured and organized manner.

##### Revised Prompt

```markdown
As a student of Deep Learning and Natural Language Processing doing research in the field, provide a summary of this document
```
Answer: The document discusses the intersection of Deep Learning and Natural Language Processing (NLP) in computer vision tasks, specifically focusing on visual question answering (VQA) and image captioning. It highlights the importance of understanding the nuances of language and its relation to visual information, as well as the need for multimodal fusion of visual and linguistic representations to achieve accurate results. The document also touches on the potential applications of this field, including alignment between images and sentences, automatic question answering, and multilingual definitions of concepts. Overall, the document emphasizes the significance of integrating NLP and computer vision techniques to tackle complex tasks that require understanding and interpretation of both visual and linguistic information.

##### Revised Prompt2
```markdown
What is the main topic discussed in this document?
```
For secondary and terciary runthroughs: 

Answer (Llama3.2): The main topic discussed in this document is the development of an Ontology of Visual Objects, which is a crucial component in the field of Computer Vision and Natural Language Processing. The document outlines the compilation of the ontology based on WordNet noun hierarchies and provides a large collection of high-quality images licensed for commercial and non-commercial use. The document also discusses the automatic prediction of objects and their classes, which facilitates manual annotation.

Answer (Tiny Llama): The main topic discussed in this document is the automatic prediction of objects and their classes.


#### What are the authors proposing?

##### Prompt

```markdown
As a student of Deep Learning and Natural Language Processing doing research in the field, What are the authors proposing in this document?
```

Answer (Llama3.2): Our work is proposing a new approach for the representation and analysis of visual objects in natural language. Our
approach is based on the use of...

For secondary and terciary runthroughs: 

- Answer (TinyLlama Chat): *Could not answer*
- Answer (Tiny Llama Step): *Could not answer*



#### What is the motivation for the work?

Here is where I had to switch to a tiny llama model because it was too big and taking over the HPC, my Google Drive, and crashing my VSCode.
##### Prompt

```markdown
As a student of Deep Learning and Natural Language Processing doing research in the field, What is the motivation for the work?
```
Answer1 (Llama3.2): The work was motivated by the need to provide a large-scale dataset of high-quality images for the purpose of visual object
recognition and computer...

Answer (TinyLlama Chat): Our motivation is to provide a large-scale dataset of high-quality images for the purpose of training deep learning models and
performing natural language...

Answer (Tiny Llama Step): The work is motivated by the need to provide a large collection of high-quality images licensed for commercial and non-commercial use.



#### What is the approach or innovation?

##### Prompt

```markdown
As a student of Deep Learning and Natural Language Processing doing research in the field, What is the approach or innovation?
```

Answer (TinyLlama Chat): We propose a novel approach to automatically extract visual objects from images using deep learning and natural language processing. Our approach consists of two main steps:

Answer (Tiny Llama Step): The approach is to use a large collection of high-quality images licensed for commercial and non-commercial use.




#### What are the results and how do they compare with competing approaches?

##### Prompt

```markdown
As a student of Deep Learning and Natural Language Processing doing research in the field, What are the results and how do they compare with competing approaches?
```
Answer (TinyLlama Chat): *The model could not answer this.*

Answer (Tiny Llama Step): We use a combination of the following approaches:



#### Is the comparison fair?

##### Prompt

```markdown
As a student of Deep Learning and Natural Language Processing doing research in the field, Is the comparison between the results and competing approaches fair?
```

Answer (TinyLlama Chat): Yes, the results are fair. The low relevant results are re-ranking the high relevant results based on their relevance to the query.

```markdown
Is the comparison between the results and competing approaches fair?
```
Answer (Tiny Llama Step): The results are not fair, because the results are not representative of the competing approaches.




#### What are the takeaways according to the authors?

##### Prompt

```markdown
As a student of Deep Learning and Natural Language Processing doing research in the field, What are the takeaways according to the authors?
```

Answer (TinyLlama Chat): The paper provides a high-level overview of the field of Visual Question Answering (VQA) and discusses the challenges and opportunities
Answer (Tiny Llama Step): *Could not answer, the context provided was just references which might have confused the model*




#### What are the takeaways according to you?

##### Prompt

```markdown
As a student of Deep Learning and Natural Language Processing doing research in the field, What are the takeaways according to you?
```
*The model could not answer this.*


#### Would you use this?  If so, how/where would you use this?

##### Prompt

```markdown
As a student of Deep Learning and Natural Language Processing doing research in the field, Would you use this?  If so, how/where would you use this?
```

I would use this to find images that match a specific search query.  I would use this to find images that match a specific search query.


#### What problems remain and what are the next steps?

##### Prompt

```markdown
What problems remain and what are the next steps?
```

*The evaluation of annotation proposals is still ongoing.

*The next steps include:


### Experience using the LLM

_Describe your process for using the LLM.  How did the LLM perform?_
While first asking the model, LLama 3.2, to provide an academic summary, it fell short. This initial summary discussed the different datasets used in this paper and is very vague. There was also, seemingly a bit of hallucination, similar to my classmate's, with regards to the output identifying 'dominant classes'. This might also be because I split the document into large 'chunk' sizes, 1000 characters with 200 character overlap. I had to try again with a new prompt and a chunk size of 300 with an overlap of 100 characters. This seems to be the standard size of paragraphs for academic papers. (source)[https://libguides.hull.ac.uk/writing/paras#:~:text=Length%20of%20a%20paragraph,are%20making%20a%20new%20point.]. Similarly, when implementing sentence transformers to pull semantic relevancies, I used the SentenceTransformer("all-MiniLM-L6-v2") because it should be used with 'short' paragraphs which might fit better for our chunking size. Because the chunk sizes are smaller, asking a quiestion took a long time. So tried in vain to go back and forth between the HPC and my personal machine.

Because I built the model in a Frankenstein fashion, i.e. I used huggingface, online tutorials, stack exchange, chatGPT, etc. I continuously ran into depreciated implementations and weird error messages that prompted me to update and change small aspects of the model. For example, I would get a message: "The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's attention_mask to obtain reliable results.
Setting pad_token_id to eos_token_id:None for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's attention_mask to obtain reliable results." So then I realized I needed to include padding and masking parameters. 
After working within these parameters it initlaly took 23 minutes to answer the first question.

After attempting to run the model for each question, I decided to create a list of each of these questions and send it through the model once. Prompting it separate the answers accordings. 

The one prompt with multiple quesitons began with, *As a student of Deep Learning and Natural Language Processing doing research in the field. Use this document to answer the following questions. Provide each answer in its own paragraph.* 

Each question was supported by the top-most 'relevant' chunk which was identified by getting the highest cosine similarity from the embedded text segments (chunks).

**Further work** can be done to show the top 3 chunks that suppor the quesiton. The set-up is ready to implement within the 'answer_question()' function in the llama3_summary.py script, I just ran out of time to run it here.
 

#### Errors and limitations of the LLM 

_Where did it fall short or make mistakes?_
First I attempted to use install and work with the model in the HPC because I wanted to make sure I had enough memory resources. I began by using Jupyter notebooks but there were many installation problems. I tried using the VSCode GUI but I came across similar download problems. Then I opted to google colab and try to work with GPU there. But I kept getting a runtime error. My last resort was to download and save the modles on my local computer where somehow the models (llama and tokenizer) began to work, albiet extremely slowly.

Working with the space available was tricky and really made me go to different sources.

I ultimately ended up switching to a tiny llama model because it was too big and taking over the HPC, my Google Drive, and crashing my VSCode.

The summarization aspect is described in my process using the LLM.



